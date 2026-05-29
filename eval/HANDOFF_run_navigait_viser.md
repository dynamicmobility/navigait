# Handoff: `eval/run_navigait_viser.py`

Standalone, single-process simulation of the **BRUCE** humanoid running
**NaviGait**, rendered in an **mjviser** web viewer. No protobuf.

## Why this exists

Previously, simulating BRUCE + NaviGait required **two processes in two conda
envs talking over a protobuf socket**:

- `~/Documents/virtual_atalante/atalante_twin/sim.py` — owned the MuJoCo model +
  mjviser viewer (ran in the `viser` env, protobuf *client* on port 9001).
- `hardware/navigait_controller.py` — loaded the NaviGait policy/gait library and
  computed commands (ran in the `navigait` env, protobuf *server*).

This script fuses both halves into one file: the MuJoCo sim + mjviser viewer call
the NaviGait controller (`ng.get_ctrl`) **directly in-process**, deleting the
socket layer entirely. Useful for quick visual evaluation and push-recovery tests
from a single command.

## How to run

From the repo root, in the `viser` conda env (it has mjviser + viser + the JAX
stack):

```bash
conda activate viser
python eval/run_navigait_viser.py config/bruce-navigait.yaml
# open the printed URL: http://localhost:8080
```

The config arg is consumed by `read_config()` (`sys.argv[1]`), same convention as
every other script here. The policy is loaded from `save_dir/name` in that config
(currently `results/navigait/may28/retraining/`).

### Viewer controls (viser GUI)

- **Forward vel vx / Lateral vel vy / Yaw rate** sliders — steer the robot
  (feed `cmd_vel` / `cmd_w` into `get_ctrl`). Default 0 = walk/step in place.
- **Disturbance force / direction / Random / Apply** — push the torso to test
  recovery. Ported from virtual_atalante.
- **Reset** (mjviser button) — restores the standing pose **and** restarts the
  controller's internal gait state.

## How it works

`NaviGaitViserSim` holds (a) the NaviGait controller env + JIT policy and (b) an
independent MuJoCo model/data for the viewer. Physics is stepped only on the
viewer's `MjData`; the controller never steps physics (same as the hardware
controller).

The mjviser `step_fn` runs the control **exchange** once every `steps_per_frame`
physics steps (= `round((1/50) / model.opt.timestep)`; with the westwood XML's
0.0005 s timestep that's 40 steps → 50 Hz control). The exchange replaces the old
protobuf round-trip:

```
data.qpos/qvel --full2pitch--> --pitch2crank--> crank_pos/crank_vel
gyro = sensordata[4:7], accel = sensordata[7:10]
cmd, info = ng.get_ctrl(time, cmd_vel, cmd_w, orientation=qpos[3:7],
                        crank_pos, crank_vel, info, gyro, accel)
cmd[:10] (pos), cmd[10:] (vel) --crank2pitch--> --pitch2bear--> data.ctrl[:20]
```

See `CLAUDE.md` → "Coordinate systems" for the full/pitch/crank/bear distinction;
all transforms come from `envs/bruce/interface_westwood.py` (imported as `bruce`).

## Key code references

- This script: `eval/run_navigait_viser.py`
- Controller call pattern (ground truth): `hardware/navigait_controller.py:35-102`
- Viewer/sim pattern (ground truth): `virtual_atalante/atalante_twin/sim.py`
- Controller API: `envs/bruce/navigait.py` — `reset_ctrl` (line 197), `get_ctrl` (line 362)

## Setup notes / dependencies

The `viser` env was topped up with packages pulled in transitively by
`utils/plotting.py`:

```bash
/home/njanwani/miniconda3/envs/viser/bin/pip install pandas matplotlib h5py opencv-python mediapy
```

The `Failed to import warp` / `mujoco_warp` lines on startup are harmless. Editor
(Pylance) warnings that `viser`/`mjviser` can't be resolved are just the IDE using
the `navigait` env, which doesn't have them — the script runs in the `viser` env.

## Decisions & known caveats

- **Initial pose**: canonical standing pose `bruce.DEFAULT_FF + bruce.DEFAULT_JT`
  (set in `__init__`). To start from virtual_atalante's walking pose, swap
  `qpos_init_pitch` for the `qpos_init` in its `config/bruce-config.yaml`.
- **`crank_vel` source**: this script derives `crank_vel` from `qvel` (correct).
  The deployed `hardware/navigait_controller.py:71` derives it from `qpos` with a
  velocity offset — almost certainly a bug. If you ever need bit-exact parity with
  the hardware controller, that's the single spot to change (in `_exchange`).
- **Foot contacts** are intentionally unused: `reset_ctrl`/`get_ctrl` don't take
  them, so the touch sensors are ignored here (unlike virtual_atalante, which
  packed them into the protobuf message).
- The viewer model and the controller's internal mjx model are **independent**;
  if you swap the XML, update both expectations accordingly.

## Verification done

- Imports resolve in the `viser` env.
- Headless smoke test: policy loads, controller runs 25 ctrl ticks, BRUCE stays
  upright (base_z ~0.43–0.45 m) under zero command.
- Viewer launches: viser listens on `:8080`, HTTP 200.

## Possible next steps

- Add a GUI toggle for an automatic `cmd_vel` sinusoid sweep (the hardware
  controller used `cmd_vel = (0.17*sin(0.3t), 0)`).
- Add live readouts (base velocity, gait phase, swing leg) as viser GUI text.
- Fold the missing `viser`-env deps into a dedicated `environment-viser.yml`.
