"""Standalone BRUCE + NaviGait simulation with an mjviser web viewer.

This is a single-process specialization of the virtual_atalante setup: instead of
a MuJoCo simulator (protobuf client) talking to ``hardware/navigait_controller.py``
(protobuf server) over a socket, this script owns the MuJoCo model + mjviser viewer
*and* calls the NaviGait controller (``ng.get_ctrl``) directly in-process. The
protobuf interface is removed entirely.

Run from the repo root, in a conda env that has both mjviser/viser and the NaviGait
JAX stack (e.g. the ``viser`` env with ``pandas`` installed)::

    python eval/run_navigait_viser.py config/bruce-navigait.yaml

Then open the mjviser URL printed on startup (http://localhost:8080).
"""

import os
import sys

# Allow running as ``python eval/run_navigait_viser.py`` (not just ``-m``): make
# sure the repo root is importable so ``envs``/``learning`` resolve.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import mujoco
import viser
import jax
from mjviser import Viewer

from envs.bruce import interface_westwood as bruce
from envs.bruce.navigait import Bruce
from learning.startup import read_config, create_environment
from learning.inference import load_policy
import utils.geometry as geo

# Control exchange rate (Hz). Matches the controller's ctrl_dt (0.02 s -> 50 Hz).
CTRL_HZ = 50
# mjviser web port.
VISER_PORT = 8080
# Disturbance: default force (N) and how many physics steps it is applied for.
DISTURBANCE_FORCE = 60.0
DISTURBANCE_STEPS = 50

# Number of free-joint DOFs in qpos (7) and qvel (6).
NUM_FREE_POS = geo.FREE3D_POS  # 7
NUM_FREE_VEL = geo.FREE3D_VEL  # 6


class NaviGaitViserSim:
    """Owns the MuJoCo model, the NaviGait controller, and the mjviser viewer."""

    def __init__(self, config):
        # --- NaviGait controller (policy + gait library) -------------------
        # Mirrors hardware/navigait_controller.py: the env provides reset_ctrl/
        # get_ctrl; it never steps physics itself.
        ng, _ = create_environment(config, for_training=False, idealistic=True)
        self.ng: Bruce = ng
        self.policy = jax.jit(load_policy(config))

        # --- Physics model for the viewer ----------------------------------
        self.model = mujoco.MjModel.from_xml_path(bruce.OFFICIAL_XML.as_posix())
        self.data = mujoco.MjData(self.model)

        # Physics steps between control exchanges (e.g. 0.02 / 0.0005 = 40).
        self.steps_per_frame = max(
            1, round((1.0 / CTRL_HZ) / self.model.opt.timestep)
        )
        self._step_count = 0

        # Canonical standing pose (7 free + 10 pitch-mode joints).
        self.qpos_init_pitch = np.array(bruce.DEFAULT_FF + bruce.DEFAULT_JT, dtype=float)

        # Controller state, refreshed on reset.
        self.info = None

        # Commanded velocity, driven by the viser GUI sliders.
        self.cmd_vel = np.zeros(2)   # [vx, vy]
        self.cmd_w = np.zeros(1)     # [yaw rate]

        # Disturbance state.
        self.torso_id = bruce.TORSO_ID
        self._disturbance_force = np.zeros(3)
        self._disturbance_steps_left = 0

    # ------------------------------------------------------------------ state
    def set_initial_state(self):
        """Apply the canonical standing pose to the current MjData."""
        self.data.qpos = bruce.ext(np, bruce.pitch2full, self.qpos_init_pitch.copy(), NUM_FREE_POS)
        self.data.ctrl[:10] = bruce.pitch2bear(np, self.qpos_init_pitch[NUM_FREE_POS:].copy())

    def reset_controller(self):
        """(Re)initialize the NaviGait controller from the current sim state."""
        mujoco.mj_forward(self.model, self.data)
        start_pos = bruce.ext(np, bruce.full2pitch, self.data.qpos.copy(), NUM_FREE_POS)
        gyro = self.data.sensordata[4:7].copy()
        accel = self.data.sensordata[7:10].copy()
        _, self.info = self.ng.reset_ctrl(
            initial_vdes=np.zeros(2),
            w_des_init=np.zeros(1),
            global_hzd_qpos=bruce.ext(np, bruce.pitch2crank, start_pos, geo.FREE3D_POS),
            gyro=gyro,
            accel=accel,
            random_seed=95,
            policy=self.policy,
        )

    # --------------------------------------------------------------- exchange
    def _exchange(self, data):
        """Replaces the protobuf round-trip: read state, call get_ctrl, set ctrl."""
        pitch_qpos = bruce.ext(np, bruce.full2pitch, data.qpos.copy(), NUM_FREE_POS)
        pitch_qvel = bruce.ext(np, bruce.full2pitch, data.qvel.copy(), NUM_FREE_VEL)

        crank_pos = bruce.pitch2crank(np, pitch_qpos[geo.FREE3D_POS:])
        # NOTE: hardware/navigait_controller.py:71 derives crank_vel from qpos
        # (a velocity offset into qpos) -- almost certainly a bug. We use qvel.
        crank_vel = bruce.pitch2crank(np, pitch_qvel[geo.FREE3D_VEL:])

        gyro = data.sensordata[4:7].copy()
        accel = data.sensordata[7:10].copy()

        cmd, self.info = self.ng.get_ctrl(
            time=data.time,
            cmd_vel=self.cmd_vel,
            cmd_w=self.cmd_w,
            orientation=data.qpos[3:7],
            crank_pos=crank_pos,
            crank_vel=crank_vel,
            info=self.info,
            gyro=gyro,
            accel=accel,
        )

        cmd = np.asarray(cmd)
        qpos_des = bruce.crank2pitch(np, cmd[:bruce.NDOF])
        qvel_des = bruce.crank2pitch(np, cmd[bruce.NDOF:])
        data.ctrl[:10] = bruce.pitch2bear(np, qpos_des)
        data.ctrl[10:20] = bruce.pitch2bear(np, qvel_des)

    # ----------------------------------------------------------- disturbance
    def trigger_disturbance(self, magnitude, angle):
        """Queue a horizontal disturbance force on the torso."""
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        self._disturbance_force = magnitude * direction
        self._disturbance_steps_left = DISTURBANCE_STEPS

    def _apply_disturbance(self, data):
        if self._disturbance_steps_left > 0:
            data.xfrc_applied[self.torso_id, :3] = self._disturbance_force
            self._disturbance_steps_left -= 1
        else:
            data.xfrc_applied[self.torso_id, :3] = 0.0

    # ------------------------------------------------------- mjviser callbacks
    def make_step_fn(self):
        def step_fn(model, data):
            self.data = data
            if self._step_count % self.steps_per_frame == 0:
                self._exchange(data)
            self._apply_disturbance(data)
            mujoco.mj_step(model, data)
            self._step_count += 1

        return step_fn

    def make_reset_fn(self):
        def reset_fn(model, data):
            # mjviser skips its default mj_resetData when reset_fn is set, so
            # clear stale state ourselves before applying the init pose.
            mujoco.mj_resetData(model, data)
            self.data = data
            self.set_initial_state()
            self._step_count = 0
            mujoco.mj_forward(model, data)
            # Restart the controller's internal gait state.
            self.reset_controller()

        return reset_fn

    # ---------------------------------------------------------------- viewer
    def _setup_command_gui(self, server):
        """Sliders to steer the robot (cmd_vel / cmd_w)."""
        vx_slider = server.gui.add_slider(
            "Forward vel vx (m/s)", min=-0.2, max=0.2, step=0.01, initial_value=0.0,
        )
        vy_slider = server.gui.add_slider(
            "Lateral vel vy (m/s)", min=-0.1, max=0.1, step=0.01, initial_value=0.0,
        )
        w_slider = server.gui.add_slider(
            "Yaw rate (rad/s)", min=-0.5, max=0.5, step=0.05, initial_value=0.0,
        )

        @vx_slider.on_update
        def _(_) -> None:
            self.cmd_vel[0] = vx_slider.value

        @vy_slider.on_update
        def _(_) -> None:
            self.cmd_vel[1] = vy_slider.value

        @w_slider.on_update
        def _(_) -> None:
            self.cmd_w[0] = w_slider.value

    def _setup_disturbance_gui(self, server):
        mag_slider = server.gui.add_slider(
            "Disturbance force (N)",
            min=0.0, max=200.0, step=5.0, initial_value=DISTURBANCE_FORCE,
        )
        dir_slider = server.gui.add_slider(
            "Disturbance direction (deg)",
            min=0.0, max=360.0, step=15.0, initial_value=0.0,
        )
        random_cb = server.gui.add_checkbox("Random sampling", initial_value=False)
        disturb_btn = server.gui.add_button("Apply disturbance", icon=viser.Icon.WIND)

        @random_cb.on_update
        def _(_) -> None:
            dir_slider.disabled = random_cb.value

        @disturb_btn.on_click
        def _(_) -> None:
            if random_cb.value:
                angle = np.random.uniform(0.0, 2.0 * np.pi)
            else:
                angle = np.radians(dir_slider.value)
            self.trigger_disturbance(mag_slider.value, angle)

    def run(self):
        self.set_initial_state()
        self.reset_controller()

        server = viser.ViserServer(port=VISER_PORT)
        server.gui.configure_theme(dark_mode=True)
        self._setup_command_gui(server)
        self._setup_disturbance_gui(server)

        print(f"NaviGait viewer ready. Open http://localhost:{VISER_PORT}")
        Viewer(
            self.model,
            self.data,
            step_fn=self.make_step_fn(),
            reset_fn=self.make_reset_fn(),
            server=server,
        ).run()


def main():
    # read_config() consumes sys.argv[1] as the YAML path (same convention as
    # hardware/navigait_controller.py).
    config = read_config()
    sim = NaviGaitViserSim(config)
    sim.run()


if __name__ == "__main__":
    main()
