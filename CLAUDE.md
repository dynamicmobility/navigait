# CLAUDE.md

Guidance for working in the NaviGait repository.

## What this is

NaviGait combines a library of HZD-generated gaits with deep RL (Brax PPO) for
robust bipedal walking. It generates motion by (1) selecting a reference gait
from the gait library via a residual velocity on the commanded velocity, (2)
blending the new reference with the current one, and (3) adding joint-level
residuals for stability. Primary robot is the **BRUCE** humanoid; an Atalante
exoskeleton and G1 are also wired in.

Built on JAX + MuJoCo / MJX (`mujoco_playground`). Trained policies are Brax PPO
checkpoints.

## Conda environments

Ask the user which conda environment they want to use at the start of a session.

## Repo layout

- `learning/` — training + inference entry points.
  - `begin_run.py` — training entry (`python -m learning.begin_run <config.yaml>`).
  - `startup.py` — `read_config()`, `create_environment(config, for_training, idealistic)`.
  - `inference.py` — `load_policy(config)`, `rollout(...)`, `get_last_model()`.
  - `train.sh` — launches training in a tmux session.
- `envs/` — environments.
  - `generic/navigait.py` — the `NaviGait` base env (obs construction, gait blending, residual logic).
  - `bruce/navigait.py` — `Bruce` subclass; defines `reset_ctrl()` / `get_ctrl()` used for inference/hardware.
  - `bruce/interface_westwood.py` — **coordinate transforms** + BRUCE model constants (imported as `bruce`).
  - `bruce/model/` — MuJoCo XMLs (`flat_scene_westwood.xml` is the canonical one; `bruce_westwood.xml` is the robot).
  - `bruce/gaits/` — gait libraries (`gaitlib_path` in config points here).
- `eval/` — visualization / rollout scripts (run as `python -m eval.<name> <config.yaml>`).
- `hardware/` — `navigait_controller.py` runs NaviGait as a protobuf server for real/simulated BRUCE.
- `config/` — YAML run configs (e.g. `bruce-navigait.yaml`).
- `results/`, `icra-policies/` — saved checkpoints. `save_dir`/`name` in the config locate them.
- `externals/` — git submodules (`bruce_trajopt`, `roboprotobuf`).

## Config convention (important)

`read_config()` (`learning/startup.py:14`) reads `sys.argv[1]` as the YAML path
**and pops it**. So scripts take the config path as their single positional arg:

```bash
python -m eval.rollout_policy icra-policies/navigait/config.yaml
python -m learning.begin_run config/bruce-navigait.yaml
```

A checkpoint is located by `save_dir/name` in the config (e.g.
`results/navigait/may28/retraining/`); `load_policy` picks the latest iteration.

## Coordinate systems (the #1 gotcha)

BRUCE has 4-bar linkages, so there are four joint representations. Conversions
live in `envs/bruce/interface_westwood.py`:

- **full** — all 18 MuJoCo joints (includes passive linkage DOFs).
- **pitch** — 10 actuated joints in pitch space (5/leg). Used for state I/O.
- **crank** — 10 actuated joints in crank space. What `get_ctrl` consumes/emits.
- **bear** — 10 joints in bearing space. What MuJoCo `data.ctrl` actuators take.

Helpers: `full2pitch`, `pitch2full`, `pitch2crank`, `crank2pitch`, `pitch2bear`,
`crank2full`, etc. `ext(np, fn, q, num_free)` applies a transform to the joints
while passing the free-base DOFs (7 for qpos, 6 for qvel) through unchanged.

Typical inference chain: MuJoCo `qpos` → `full2pitch` → `pitch2crank` → `get_ctrl`
→ `crank2pitch` → `pitch2bear` → `data.ctrl`.

Sensor layout in `bruce_westwood.xml`: `sensordata[:4]` = foot touch,
`[4:7]` = `base_gyro`, `[7:10]` = `base_accelerometer`.

## Running NaviGait as a controller

`hardware/navigait_controller.py` is the canonical example of driving NaviGait
outside training: build env via `create_environment(config, for_training=False,
idealistic=True)`, JIT `load_policy(config)`, then `ng.reset_ctrl(...)` once and
`ng.get_ctrl(time, cmd_vel, cmd_w, orientation, crank_pos, crank_vel, info, gyro,
accel)` per 50 Hz tick. The controller never steps physics itself. `get_ctrl`
returns a length-20 vector: `[:10]` crank position targets, `[10:]` crank
velocity targets.

## Standalone web viewer

`eval/run_navigait_viser.py` (run in the `viser` env) is a single-process MuJoCo
+ mjviser sim that calls `get_ctrl` directly — no protobuf. See
`eval/HANDOFF_run_navigait_viser.md`.

### Gamepad input (PS4 / generic SDL2 pad)

The viewer optionally accepts a gamepad via pygame. Left stick → `cmd_vel`
(forward / lateral), right stick X → yaw rate; values are deadzoned and scaled
into the existing slider ranges, and the GUI sliders mirror the stick. If
pygame is missing or no pad is connected, the GUI alone still works.

Two macOS-specific quirks the script papers over (host dev box is a MacBook Pro;
Linux desktop is used occasionally):

- **SDL2 dylib load order.** `cv2` (pulled in transitively via `utils.plotting`)
  bundles its own `libSDL2-2.0.0.dylib`. If it loads before pygame, pygame's
  joystick subsystem opens but never receives HID events (all axes read 0
  forever). `pygame` is imported at the top of the script, *before*
  numpy/jax/`utils.geometry`, so pygame's SDL2 wins the Objective-C class
  registration race. Don't reorder those imports.
- **`SDL_VIDEODRIVER=dummy`.** Set before `import pygame`. The gamepad runs in
  a daemon thread that calls `pygame.event.pump()`; with a real video driver,
  pump enters Cocoa AppKit from a non-main thread and macOS aborts the process
  with `nextEventMatchingMask should only be called from the Main Thread`.

On Linux these env hints are harmless no-ops, so the same script runs
unmodified on the desktop.

**Wired vs Bluetooth on macOS.** Wired (USB-C) is the reliable path: low
latency, no idle suspend, well-tested HID report 0x01. The DS4 over Bluetooth
hits two macOS bugs — aggressive idle disconnect (SDL's joystick handle goes
stale) and flaky BT report (0x11 extended) handling in SDL's HIDAPI driver —
which manifest as axes that freeze at their last value and eventually
`pygame.error: Joystick not initialized`. If you must use BT, expect to
restart the script periodically. Earlier revisions of the script had stall
detection + handle re-acquire logic to paper over this; it was removed once
wired was confirmed to work. Re-add from git history if BT becomes the only
option.

Set `NAVIGAIT_GAMEPAD_DEBUG=1` to log live raw axis values once per second —
useful when remapping a non-PS4 pad (edit `GAMEPAD_AXIS_VX/VY/YAW`).

## Conventions / gotchas

- Run scripts as modules from the repo root (`python -m eval.x ...`) so `envs`/`learning`
  resolve. `eval/run_navigait_viser.py` also adds the repo root to `sys.path` so a
  direct `python eval/run_navigait_viser.py` works too.
- Training/eval default to GPU (`MUJOCO_GL=egl`); inference scripts often force CPU
  with `os.environ['JAX_PLATFORMS']='cpu'`.
- Commit/push only when asked. The default PR branch is `master`.
