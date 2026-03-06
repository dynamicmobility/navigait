# Basic imports
from pathlib import Path
import numpy as np
from glob import glob
from tqdm import tqdm
import pickle

# Internal imports
import utils.plotting as plotting
from utils.state import MujocoState
import utils.geometry as geo
from learning.training import setup_training

# RL imports
import functools
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

# jax and MJX imports
from mujoco_playground._src.mjx_env import MjxEnv
import mujoco as mj
import jax


def get_last_model(config: dict) -> Path:
    """Returns the last model file in the model directory."""
    model_dir = Path(config['save_dir']) / config['name']
    dir_files = glob(str(model_dir / '*'))
    model_files = [Path(f).name for f in dir_files if '.' not in f]
    if not model_files:
        raise ValueError(f"No model files found in {model_dir}")
    
    return Path(model_dir / f"{max(model_files, key=int)}")

def get_all_models(config: dict) -> list[Path]:
    """Returns all model files in the model directory, sorted by iteration number."""
    model_dir = Path(config['save_dir']) / config['name']
    dir_files = glob(str(model_dir / '*'))
    model_files = [Path(f) for f in dir_files if '.' not in Path(f).name]
    if not model_files:
        raise ValueError(f"No model files found in {model_dir}")
    
    return sorted(model_files, key=lambda x: int(x.name))

def load_policy(config, deterministic=True):
    path = get_last_model(config)
    print(f'Loading model at {path.as_posix()}')
    policy = checkpoint.load_policy(
        path.resolve(),
        deterministic=deterministic
    )
    return policy

def load_specific_policy(path, deterministic=True):
    print(f'Loading model at {path.as_posix()}')
    policy = checkpoint.load_policy(
        path.resolve(),
        deterministic=deterministic
    )
    return policy

def get_make_inference_fn(
    config,
    network_factory= ppo_networks.make_ppo_networks,
    deterministic: bool = True,
):
    """Loads policy inference function from PPO checkpoint."""
    model_path = get_last_model(config)
    path = checkpoint.epath.Path(model_path)
    config = checkpoint.load_config(path)
    ppo_network = checkpoint._get_ppo_network(config, network_factory)
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    return lambda params: make_inference_fn(params, deterministic=deterministic)

def get_params(path):
    path = checkpoint.epath.Path(path)
    config = checkpoint.load_config(path)
    params = checkpoint.load(path)
    return params

def infer_frame_dim(
    mj_model, width, height
):
    if width is None:
        width = mj_model.vis.global_.offwidth
    if height is None:
        height = mj_model.vis.global_.offheight
    
    return width, height


def rollout(
    reset,
    step,
    inference_fn,
    env: MjxEnv,
    T=10.0,
    info_init_fn=lambda state: state.info,
    info_step_fn=lambda state: state.info,
    info_plot_key=None,
    width=None,
    height=None,
    gen_vid=True,
    show_progress=True,
    scene_option=plotting.get_mj_scene_option(contacts=False),
    camera='track',
    plot_vs=True
) -> tuple[list, plotting.RewardPlotter, plotting.MujocoPlotter, plotting.InfoPlotter]:
    width, height = infer_frame_dim(env.mj_model, width, height)
    
    # Set up the environment
    rng = jax.random.PRNGKey(np.random.randint(0, 100000))
    state: MujocoState = reset(rng)
    
    # Initialize the state
    initial_info = state.info | info_init_fn(state)
    state = state.replace(info=initial_info)

    # Setup reward plotting
    reward_plotter = plotting.RewardPlotter(state.metrics)
    data_plotter = plotting.MujocoPlotter()
    info_plotter = plotting.InfoPlotter(plotkey=info_plot_key)
    data_plotter.add_row(state.data)

    # Rollout and record data
    N = int(T / env.dt)
    traj = [state]
    vels = [state.info['vdes']]
    vel_target_key = 'vel_target' if 'vel_scale' in env.params else 'vdes'

    vel_targets = [state.info[vel_target_key]]
    for i in tqdm(range(N), disable=not show_progress):
        ctrl, _ = inference_fn(state.obs, rng)
        state = step(state, np.array(ctrl).copy())
        if('vel_target' not in state.info.keys()):
            state.info['vel_target'] = state.info['vdes']
        vels.append(state.info['vdes'])
        vel_targets.append(state.info[vel_target_key])
        state = state.replace(info=state.info | info_step_fn(state))
        data_plotter.add_row(state.data)
        reward_plotter.add_row(state.metrics, state.reward)
        info_plotter.add_row(state.data.time, state.info)
        traj.append(state)
        
        if state.done:
            break


    if gen_vid:
        print('Generating video...')
        frames = env.render(
            trajectory   = traj,
            camera       = camera,
            height       = height,
            width        = width,
            scene_option = scene_option,
        )
        if plot_vs:
            for i in range(len(frames)):
                curr_vdes = vels[i]
                vel_target = vel_targets[i]
                text = f'v = [{round(curr_vdes[0], 2)}, {round(curr_vdes[1], 2)}, {round(curr_vdes[2], 2)}]'
                frames[i] = plotting.add_text_to_frame(
                    pixels    = frames[i],
                    text      = text,
                    org       = (50, 100), 
                    size      = 1, 
                    thickness = 2,
                    color     = (255, 255, 255)
                )
                if vel_target_key != 'vdes':
                    text = f'v = [{round(vel_target[0], 2)}, {round(vel_target[1], 2)}]  m/s'
                    frames[i] = plotting.add_text_to_frame(
                        pixels    = frames[i],
                        text      = text,
                        org       = (50, 150), 
                        size      = 1, 
                        thickness = 2,
                        color     = (255, 255, 255)
                    )
    else:
        frames = None
    return frames, reward_plotter, data_plotter, info_plotter
        
def circle_vel(state, vxlim, vylim, freq):
    t = state.data.time
    info = state.info.copy()
    omega = 2 * np.pi * freq
    info['vdes'] = np.array([vxlim * np.sin(omega * t),
                             vylim * np.cos(omega * t),
                             0.0])
    return info

def vx_sine_vel(state, vxlim, vylim, freq):
    t = state.data.time
    info = state.info.copy()
    omega = 2 * np.pi * freq
    info['vdes'] = np.array([vxlim * np.sin(omega * t),
                             0.0])
    return info

def vy_sine_vel(state, vxlim, vylim, freq):
    t = state.data.time
    info = state.info.copy()
    info['vdes'] = np.array([0.0, vylim * np.sin(freq * 2 * np.pi * t)])
    return info