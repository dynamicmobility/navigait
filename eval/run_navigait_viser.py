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

# --- pygame must be imported FIRST on macOS ----------------------------------
# Several downstream deps (opencv-python in particular) ship their own bundled
# copy of libSDL2-2.0.0.dylib. Whichever SDL2 dylib loads first wins the
# Objective-C class registration race for SDLApplication / IOHIDManager. If
# cv2's SDL2 wins, pygame's joystick subsystem opens but never receives HID
# events (all axes stay at 0.0 forever). Importing pygame here -- before
# numpy/jax/utils.geometry etc. transitively pull in cv2 -- ensures pygame's
# SDL2 registers the SDL classes and owns HID polling.
# Hints must be set before pygame loads SDL2.
# Force the cross-platform HIDAPI backend instead of the IOKit one. SDL's
# IOKit joystick driver on macOS regularly stalls (HID callbacks stop firing
# and get_axis() returns the last cached sample forever); HIDAPI polls the
# device directly and is robust to that. The PS4-specific hint enables SDL's
# DualShock 4 protocol handler so axes/buttons map correctly.
os.environ.setdefault("SDL_JOYSTICK_HIDAPI", "1")
os.environ.setdefault("SDL_JOYSTICK_HIDAPI_PS4", "1")
os.environ.setdefault("SDL_JOYSTICK_HIDAPI_PS4_RUMBLE", "0")
os.environ.setdefault("SDL_JOYSTICK_THREAD", "1")
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
try:
    import pygame  # noqa: F401  (just to load its SDL2 first)
except ImportError:
    pygame = None
# -----------------------------------------------------------------------------

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

# PS4 / generic gamepad axis mapping (pygame). Most PS4 pads under SDL2 use:
#   axis 0: left stick X, axis 1: left stick Y (down = +1),
#   axis 2: right stick X, axis 3: right stick Y.
GAMEPAD_AXIS_VX = 1   # left stick Y -> forward velocity (inverted: up = +vx)
GAMEPAD_AXIS_VY = 0   # left stick X -> lateral velocity (inverted: left = +vy)
GAMEPAD_AXIS_YAW = 2  # right stick X -> yaw rate (inverted: left = +yaw)
GAMEPAD_DEADZONE = 0.12
# Set NAVIGAIT_GAMEPAD_DEBUG=1 to log live axis values (helps remap a pad whose
# axes don't match the defaults above). Rate-limited to once per second.
GAMEPAD_DEBUG = bool(int(os.environ.get("NAVIGAIT_GAMEPAD_DEBUG", "0")))

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

        # Gamepad state -- populated lazily in run() if pygame + a joystick are
        # available. Slider handles are kept so the gamepad can drive them.
        # Polling runs in a dedicated daemon thread (see _gamepad_loop): on
        # macOS, SDL2 joystick events are tied to whichever thread initialized
        # the subsystem, and that thread must keep pumping events for the
        # IOKit HID manager to deliver new samples. Running pygame inside
        # mjviser's step_fn worked initially but silently stalled after any
        # hiccup; a dedicated polling thread avoids that.
        self._joystick = None
        self._gamepad_stop = False
        self._gamepad_lock = None
        self._vx_slider = None
        self._vy_slider = None
        self._w_slider = None
        # Slider ranges captured so we can scale axis values into them.
        self._vx_range = (-0.2, 0.2)
        self._vy_range = (-0.1, 0.1)
        self._w_range = (-0.5, 0.5)

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

    # -------------------------------------------------------------- gamepad
    def _start_gamepad_thread(self):
        """Spawn a daemon thread that owns pygame and polls the joystick.

        Pygame/SDL2 on macOS requires that whichever thread initialized the
        joystick subsystem is also the thread that pumps events; otherwise
        new samples eventually stop arriving. Keeping init + pump + read all
        on one dedicated thread sidesteps that.
        """
        import threading
        self._gamepad_lock = threading.Lock()
        t = threading.Thread(target=self._gamepad_loop, name="gamepad", daemon=True)
        t.start()

    def _gamepad_loop(self):
        # pygame was imported at module top (before cv2) so its SDL2 wins the
        # class-registration race. SDL_JOYSTICK_THREAD=1 was set there too.
        if pygame is None:
            print("[gamepad] pygame not installed; GUI sliders only.")
            return
        try:
            # Full pygame.init() so the event subsystem is alive: get_axis()
            # reads cached state that event.pump() refreshes from SDL's
            # joystick events.
            pygame.init()
            pygame.joystick.init()
        except Exception as e:
            print(f"[gamepad] pygame init failed ({e}); GUI sliders only.")
            return
        if pygame.joystick.get_count() == 0:
            print("[gamepad] no joystick detected; GUI sliders only.")
            return
        js = pygame.joystick.Joystick(0)
        js.init()
        self._joystick = js
        self._pygame = pygame
        print(f"[gamepad] using '{js.get_name()}' "
              f"({js.get_numaxes()} axes, {js.get_numbuttons()} buttons)")

        import time as _time
        dt = 1.0 / CTRL_HZ
        debug_tick = 0
        # Stall detection: if every axis stays bit-for-bit identical for this
        # many seconds, assume HID polling died and re-init the joystick.
        STALL_SECS = 2.0
        last_raw = None
        last_change_t = _time.monotonic()
        while not self._gamepad_stop:
            try:
                pygame.event.pump()
            except Exception:
                pass

            try:
                num_axes = js.get_numaxes()
                raw = tuple(js.get_axis(i) for i in range(num_axes))
            except pygame.error as e:
                # SDL invalidated the joystick handle (device idle-disconnect,
                # HIDAPI hiccup, etc.). Try to re-acquire it.
                print(f"[gamepad] joystick handle lost ({e}); reinitializing")
                try:
                    pygame.joystick.quit()
                    pygame.joystick.init()
                    if pygame.joystick.get_count() == 0:
                        _time.sleep(0.5)
                        continue
                    js = pygame.joystick.Joystick(0)
                    js.init()
                    self._joystick = js
                except Exception as e2:
                    print(f"[gamepad] reinit failed: {e2}")
                    _time.sleep(0.5)
                continue

            now = _time.monotonic()
            if raw != last_raw:
                last_raw = raw
                last_change_t = now
            elif now - last_change_t > STALL_SECS:
                print("[gamepad] axes stalled; reinitializing joystick subsystem")
                try:
                    js.quit()
                    pygame.joystick.quit()
                    pygame.joystick.init()
                    if pygame.joystick.get_count() == 0:
                        print("[gamepad] reinit found no joystick; will retry")
                        _time.sleep(0.5)
                        continue
                    js = pygame.joystick.Joystick(0)
                    js.init()
                    self._joystick = js
                except Exception as e:
                    print(f"[gamepad] reinit failed: {e}")
                last_change_t = now

            def axis(i):
                if i >= num_axes:
                    return 0.0
                return self._deadzone(raw[i])

            ax_vx = -axis(GAMEPAD_AXIS_VX)
            ax_vy = -axis(GAMEPAD_AXIS_VY)
            ax_w = -axis(GAMEPAD_AXIS_YAW)

            if GAMEPAD_DEBUG and debug_tick % CTRL_HZ == 0:
                print(f"[gamepad] axes={[round(v, 3) for v in raw]}")
            debug_tick += 1

            if ax_vx != 0.0 or ax_vy != 0.0 or ax_w != 0.0:
                vx = self._scale(ax_vx, *self._vx_range)
                vy = self._scale(ax_vy, *self._vy_range)
                wz = self._scale(ax_w, *self._w_range)
                with self._gamepad_lock:
                    self.cmd_vel[0] = vx
                    self.cmd_vel[1] = vy
                    self.cmd_w[0] = wz
                if self._vx_slider is not None:
                    try:
                        self._vx_slider.value = vx
                        self._vy_slider.value = vy
                        self._w_slider.value = wz
                    except Exception:
                        pass

            _time.sleep(dt)

    @staticmethod
    def _deadzone(v, dz=GAMEPAD_DEADZONE):
        if abs(v) < dz:
            return 0.0
        # Rescale so output starts at 0 just past the deadzone.
        sign = 1.0 if v > 0 else -1.0
        return sign * (abs(v) - dz) / (1.0 - dz)

    @staticmethod
    def _scale(axis, lo, hi):
        # axis in [-1, 1] -> [lo, hi] with 0 mapped to (lo+hi)/2.
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        return float(np.clip(mid + axis * half, lo, hi))

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
        self._vx_slider = vx_slider
        self._vy_slider = vy_slider
        self._w_slider = w_slider
        # Keep ranges in sync with the slider constructors above. (Looked up
        # here rather than via slider attrs because viser's slider handle does
        # not expose min/max in all versions.)
        self._vx_range = (-0.2, 0.2)
        self._vy_range = (-0.1, 0.1)
        self._w_range = (-0.5, 0.5)

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
        self._start_gamepad_thread()

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
