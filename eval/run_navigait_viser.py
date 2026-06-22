"""Standalone BRUCE + NaviGait simulation with an mjviser web viewer.

Single-process replacement for the virtual_atalante setup: owns the MuJoCo model
+ mjviser viewer *and* calls the NaviGait controller (``ng.get_ctrl``) directly,
deleting the protobuf layer. Optional PS4 gamepad input drives the GUI sliders.

Run from the repo root in the ``viser`` conda env::

    python eval/run_navigait_viser.py config/bruce-navigait.yaml
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# pygame must load before anything that pulls in cv2: both ship their own
# libSDL2-2.0.0.dylib and whichever loads first wins the SDL Objective-C class
# registration on macOS. If cv2 wins, pygame's joystick subsystem opens but
# never receives HID samples. The dummy video driver keeps pygame.event.pump()
# from touching Cocoa AppKit from our worker thread (which would crash).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
try:
    import pygame
except ImportError:
    pygame = None

import threading
import time
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

CTRL_HZ = 50
VISER_PORT = 8080
DISTURBANCE_FORCE = 10.0
DISTURBANCE_DURATION_S = 0.1
# Disturbance arrow: drawn from the torso base along the applied force. Length
# scales with force magnitude (meters per Newton).
DISTURBANCE_ARROW_COLOR = (255, 70, 70)
DISTURBANCE_ARROW_M_PER_N = 0.02
MAX_DISTURBANCE = 40.0
DISTURBANCE_INCREMENT = 1.0

# Third-person follow camera. Positioned behind the robot's heading and
# re-aimed every render frame so the view chases the torso as it walks/turns.
# Distances are relative to the look-at point (raised above the torso base for
# nicer framing). Enabling this forces the camera each frame, so manual orbiting
# is suppressed while it is on.
FOLLOW_CAM_DISTANCE = 1.2      # meters behind the torso (horizontal)
FOLLOW_CAM_HEIGHT = 0.3        # meters above the look-at point
FOLLOW_CAM_LOOK_HEIGHT = 0.3   # raise look-at above the torso base
# Per-frame smoothing fraction toward the target pose (exponential low-pass).
# 1.0 = rigid/instant, lower = more lag. ~0.12 gives a gentle trailing feel at
# the ~60 Hz render rate.
FOLLOW_CAM_SMOOTHING = 0.02

# Slider ranges, shared by GUI and gamepad scaling.
VX_RANGE = (-0.17, 0.17)
VY_RANGE = (-0.08, 0.08)
W_RANGE = (-0.3, 0.3)

# PS4 axis mapping (SDL2): 0=LX, 1=LY (down=+1), 2=RX, 3=RY.
GAMEPAD_AXIS_VX = 1   # LY, inverted -> forward
GAMEPAD_AXIS_VY = 0   # LX, inverted -> left
GAMEPAD_AXIS_YAW = 2  # RX, inverted -> CCW
GAMEPAD_DEADZONE = 0.12
GAMEPAD_DEBUG = bool(int(os.environ.get("NAVIGAIT_GAMEPAD_DEBUG", "0")))

NUM_FREE_POS = geo.FREE3D_POS  # 7
NUM_FREE_VEL = geo.FREE3D_VEL  # 6


class NaviGaitViserSim:
    """Owns the MuJoCo model, the NaviGait controller, and the mjviser viewer."""

    def __init__(self, config):
        ng, _ = create_environment(config, for_training=False, idealistic=True)
        self.ng: Bruce = ng
        self.policy = jax.jit(load_policy(config))

        self.model = mujoco.MjModel.from_xml_path(bruce.OFFICIAL_XML.as_posix())
        self.data = mujoco.MjData(self.model)
        self.steps_per_frame = max(1, round((1.0 / CTRL_HZ) / self.model.opt.timestep))
        self._step_count = 0

        self.qpos_init_pitch = np.array(bruce.DEFAULT_FF + bruce.DEFAULT_JT, dtype=float)
        self.info = None

        self.cmd_vel = np.zeros(2)
        self.cmd_w = np.zeros(1)

        self.torso_id = bruce.TORSO_ID
        self._disturbance_force = np.zeros(3)
        self._disturbance_steps_left = 0

        self._server = None
        self._viewer = None
        self._arrow = None

        self._follow_cam_enabled = True
        self._follow_distance = FOLLOW_CAM_DISTANCE
        self._follow_height = FOLLOW_CAM_HEIGHT
        self._follow_smoothing = FOLLOW_CAM_SMOOTHING
        # Smoothed camera state; None until the first frame snaps to target.
        self._cam_position = None
        self._cam_look_at = None

        self._vx_slider = None
        self._vy_slider = None
        self._w_slider = None

    def set_initial_state(self):
        self.data.qpos = bruce.ext(np, bruce.pitch2full, self.qpos_init_pitch.copy(), NUM_FREE_POS)
        self.data.ctrl[:10] = bruce.pitch2bear(np, self.qpos_init_pitch[NUM_FREE_POS:].copy())

    def reset_controller(self):
        mujoco.mj_forward(self.model, self.data)
        start_pos = bruce.ext(np, bruce.full2pitch, self.data.qpos.copy(), NUM_FREE_POS)
        _, self.info = self.ng.reset_ctrl(
            initial_vdes=np.zeros(2),
            w_des_init=np.zeros(1),
            global_hzd_qpos=bruce.ext(np, bruce.pitch2crank, start_pos, geo.FREE3D_POS),
            gyro=self.data.sensordata[4:7].copy(),
            accel=self.data.sensordata[7:10].copy(),
            random_seed=95,
            policy=self.policy,
        )

    def _exchange(self, data):
        pitch_qpos = bruce.ext(np, bruce.full2pitch, data.qpos.copy(), NUM_FREE_POS)
        pitch_qvel = bruce.ext(np, bruce.full2pitch, data.qvel.copy(), NUM_FREE_VEL)
        crank_pos = bruce.pitch2crank(np, pitch_qpos[geo.FREE3D_POS:])
        # hardware/navigait_controller.py:71 reads crank_vel from qpos (a bug);
        # we use qvel.
        crank_vel = bruce.pitch2crank(np, pitch_qvel[geo.FREE3D_VEL:])

        cmd, self.info = self.ng.get_ctrl(
            time=data.time,
            cmd_vel=self.cmd_vel,
            cmd_w=self.cmd_w,
            orientation=data.qpos[3:7],
            crank_pos=crank_pos,
            crank_vel=crank_vel,
            info=self.info,
            gyro=data.sensordata[4:7].copy(),
            accel=data.sensordata[7:10].copy(),
        )
        cmd = np.asarray(cmd)
        qpos_des = bruce.crank2pitch(np, cmd[:bruce.NDOF])
        qvel_des = bruce.crank2pitch(np, cmd[bruce.NDOF:])
        data.ctrl[:10] = bruce.pitch2bear(np, qpos_des)
        data.ctrl[10:20] = bruce.pitch2bear(np, qvel_des)

    # --- gamepad ---------------------------------------------------------
    def _start_gamepad_thread(self):
        threading.Thread(target=self._gamepad_loop, name="gamepad", daemon=True).start()

    def _gamepad_loop(self):
        if pygame is None:
            print("[gamepad] pygame not installed; GUI sliders only.")
            return
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("[gamepad] no joystick detected; GUI sliders only.")
            return
        js = pygame.joystick.Joystick(0)
        js.init()
        print(f"[gamepad] using '{js.get_name()}' ({js.get_numaxes()} axes)")

        dt = 1.0 / CTRL_HZ
        tick = 0
        while True:
            pygame.event.pump()
            raw = [js.get_axis(i) for i in range(js.get_numaxes())]

            if GAMEPAD_DEBUG and tick % CTRL_HZ == 0:
                print(f"[gamepad] axes={[round(v, 3) for v in raw]}")
            tick += 1

            ax_vx = -_deadzone(raw[GAMEPAD_AXIS_VX])
            ax_vy = _deadzone(raw[GAMEPAD_AXIS_VY])
            ax_w = _deadzone(raw[GAMEPAD_AXIS_YAW])

            if ax_vx or ax_vy or ax_w:
                vx = _scale(ax_vx, *VX_RANGE)
                vy = _scale(ax_vy, *VY_RANGE)
                wz = _scale(ax_w, *W_RANGE)
                self.cmd_vel[0], self.cmd_vel[1], self.cmd_w[0] = vx, vy, wz
                if self._vx_slider is not None:
                    self._vx_slider.value = vx
                    self._vy_slider.value = vy
                    self._w_slider.value = wz

            time.sleep(dt)

    # --- disturbance -----------------------------------------------------
    def trigger_disturbance(self, magnitude, angle):
        self._disturbance_force = magnitude * np.array([np.cos(angle), np.sin(angle), 0.0])
        self._disturbance_steps_left = int(round(DISTURBANCE_DURATION_S / self.model.opt.timestep))

    def _apply_disturbance(self, data):
        if self._disturbance_steps_left > 0:
            data.xfrc_applied[self.torso_id, :3] = self._disturbance_force
            self._disturbance_steps_left -= 1
        else:
            data.xfrc_applied[self.torso_id, :3] = 0.0

    def _update_disturbance_arrow(self, data):
        """Show/refresh a force arrow rooted at the torso base while a
        disturbance is active; hide it otherwise."""
        if self._server is None:
            return
        mag = float(np.linalg.norm(self._disturbance_force))
        if self._disturbance_steps_left > 0 and mag > 1e-6:
            start = data.xpos[self.torso_id].copy()
            end = start + self._disturbance_force * DISTURBANCE_ARROW_M_PER_N
            # Parent under "/fixed_bodies": mjviser shifts that frame by its
            # scene_offset (= -tracked_body_pos when camera tracking is on) so
            # the centered robot stays at the origin. Adding the arrow there in
            # world coords keeps it pinned to the torso instead of drifting off
            # by the tracking offset. add_arrows with the same name replaces the
            # node in place, so this tracks the moving base + current force.
            self._arrow = self._server.scene.add_arrows(
                "/fixed_bodies/disturbance",
                points=np.array([[start, end]]),
                colors=DISTURBANCE_ARROW_COLOR,
                shaft_radius=0.012,
                head_radius=0.03,
                head_length=0.05,
            )
        elif self._arrow is not None:
            self._arrow.visible = False

    # --- follow camera ---------------------------------------------------
    def _update_follow_camera(self):
        """Aim every client camera from behind the robot's heading.

        mjviser's "Track camera" recenters the scene by shifting all geometry by
        ``scene._scene_offset`` (= -tracked_body_pos), so the displayed torso is
        at ``data.xpos[torso] + scene_offset``. We anchor to that displayed
        position (works whether or not tracking is on) and place the camera
        ``_follow_distance`` behind the torso's yaw heading and ``_follow_height``
        above the look-at point. Forward is the robot's +x (the cmd_vel forward
        axis); yaw uses the same convention as geo.extract_yaw."""
        if not self._follow_cam_enabled or self._server is None or self._viewer is None:
            return
        scene_offset = self._viewer.scene._scene_offset
        torso = self.data.xpos[self.torso_id] + scene_offset
        w, x, y, z = self.data.qpos[3:7]
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])

        look_at_target = torso + np.array([0.0, 0.0, FOLLOW_CAM_LOOK_HEIGHT])
        position_target = look_at_target - forward * self._follow_distance + np.array([0.0, 0.0, self._follow_height])

        # Exponential low-pass so the camera trails the robot instead of snapping
        # to it. Smoothing both vectors lags position and heading together.
        a = self._follow_smoothing
        if self._cam_position is None:
            self._cam_position = position_target
            self._cam_look_at = look_at_target
        else:
            self._cam_position += a * (position_target - self._cam_position)
            self._cam_look_at += a * (look_at_target - self._cam_look_at)

        for client in self._server.get_clients().values():
            client.camera.position = self._cam_position
            client.camera.look_at = self._cam_look_at

    # --- mjviser callbacks ----------------------------------------------
    def make_render_fn(self):
        def render_fn(scene):
            scene.update_from_mjdata(self.data)
            self._update_follow_camera()
        return render_fn

    def make_step_fn(self):
        def step_fn(model, data):
            self.data = data
            if self._step_count % self.steps_per_frame == 0:
                self._exchange(data)
                self._update_disturbance_arrow(data)
            self._apply_disturbance(data)
            mujoco.mj_step(model, data)
            self._step_count += 1
        return step_fn

    def make_reset_fn(self):
        def reset_fn(model, data):
            # mjviser skips its default mj_resetData when reset_fn is set.
            mujoco.mj_resetData(model, data)
            self.data = data
            self.set_initial_state()
            self._step_count = 0
            self._cam_position = None  # snap follow camera to the reset pose
            self._cam_look_at = None
            mujoco.mj_forward(model, data)
            self.reset_controller()
        return reset_fn

    # --- viewer ---------------------------------------------------------
    def _setup_command_gui(self, server):
        self._vx_slider = server.gui.add_slider(
            "Forward vel vx (m/s)", min=VX_RANGE[0], max=VX_RANGE[1], step=0.01, initial_value=0.0,
        )
        self._vy_slider = server.gui.add_slider(
            "Lateral vel vy (m/s)", min=VY_RANGE[0], max=VY_RANGE[1], step=0.01, initial_value=0.0,
        )
        self._w_slider = server.gui.add_slider(
            "Yaw rate (rad/s)", min=W_RANGE[0], max=W_RANGE[1], step=0.05, initial_value=0.0,
        )

        @self._vx_slider.on_update
        def _(_): self.cmd_vel[0] = self._vx_slider.value

        @self._vy_slider.on_update
        def _(_): self.cmd_vel[1] = self._vy_slider.value

        @self._w_slider.on_update
        def _(_): self.cmd_w[0] = self._w_slider.value

    def _setup_follow_cam_gui(self, server):
        with server.gui.add_folder("Camera"):
            # Two modes, toggled by one button:
            #   Follow - third-person camera locked behind the robot's heading
            #            (forced each render frame, so manual orbiting is off).
            #   Free   - default viser camera; we stop driving it so the user can
            #            orbit/pan/zoom freely.
            mode_btn = server.gui.add_button("Mode: Follow")
            dist_slider = server.gui.add_slider(
                "Distance (m)", min=0.5, max=6.0, step=0.1, initial_value=self._follow_distance,
            )
            height_slider = server.gui.add_slider(
                "Height (m)", min=0.0, max=4.0, step=0.1, initial_value=self._follow_height,
            )
            lag_slider = server.gui.add_slider(
                "Responsiveness", min=0.02, max=1.0, step=0.02, initial_value=self._follow_smoothing,
                hint="Lower = more lag (camera trails the robot). 1.0 = rigid.",
            )

            def _apply_mode():
                follow = self._follow_cam_enabled
                mode_btn.label = "Mode: Follow" if follow else "Mode: Free"
                dist_slider.disabled = not follow
                height_slider.disabled = not follow
                lag_slider.disabled = not follow

            @mode_btn.on_click
            def _(_):
                self._follow_cam_enabled = not self._follow_cam_enabled
                if self._follow_cam_enabled:
                    # Re-entering follow: snap to the robot, then ease as usual.
                    self._cam_position = None
                    self._cam_look_at = None
                _apply_mode()

            @dist_slider.on_update
            def _(_): self._follow_distance = dist_slider.value

            @height_slider.on_update
            def _(_): self._follow_height = height_slider.value

            @lag_slider.on_update
            def _(_): self._follow_smoothing = lag_slider.value

            _apply_mode()

    def _setup_disturbance_gui(self, server):
        mag_slider = server.gui.add_slider(
            "Disturbance force (N)", min=0.0, max=MAX_DISTURBANCE, step=DISTURBANCE_INCREMENT, initial_value=DISTURBANCE_FORCE,
        )
        dir_slider = server.gui.add_slider(
            "Disturbance direction (deg)", min=0.0, max=360.0, step=15.0, initial_value=0.0,
        )
        random_cb = server.gui.add_checkbox("Random sampling", initial_value=False)
        disturb_btn = server.gui.add_button("Apply disturbance", icon=viser.Icon.WIND)

        @random_cb.on_update
        def _(_): dir_slider.disabled = random_cb.value

        @disturb_btn.on_click
        def _(_):
            angle = np.random.uniform(0.0, 2.0 * np.pi) if random_cb.value else np.radians(dir_slider.value)
            self.trigger_disturbance(mag_slider.value, angle)

    def run(self):
        self.set_initial_state()
        self.reset_controller()

        server = viser.ViserServer(port=VISER_PORT)
        self._server = server
        server.gui.configure_theme(dark_mode=True)
        self._setup_command_gui(server)
        self._setup_follow_cam_gui(server)
        self._setup_disturbance_gui(server)
        self._start_gamepad_thread()

        print(f"NaviGait viewer ready. Open http://localhost:{VISER_PORT}")
        self._viewer = Viewer(
            self.model,
            self.data,
            step_fn=self.make_step_fn(),
            render_fn=self.make_render_fn(),
            reset_fn=self.make_reset_fn(),
            server=server,
        )
        self._viewer.run()


def _deadzone(v, dz=GAMEPAD_DEADZONE):
    if abs(v) < dz:
        return 0.0
    sign = 1.0 if v > 0 else -1.0
    return sign * (abs(v) - dz) / (1.0 - dz)


def _scale(axis, lo, hi):
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return float(np.clip(mid + axis * half, lo, hi))


def main():
    sim = NaviGaitViserSim(read_config())
    sim.run()


if __name__ == "__main__":
    main()
