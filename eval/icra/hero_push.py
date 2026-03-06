import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
from utils.plotting import set_mpl_params
import cv2
set_mpl_params()
import matplotlib.pyplot as plt
from dynamo_figures import CompositeImage, CompositeMode
from utils.setupGPU import run_setup
from pathlib import Path
from learning.startup import read_config, create_environment, get_step_reset
from learning.inference import rollout, load_policy, circle_vel
from utils.plotting import save_video, save_metrics, save_trajectories, get_mj_scene_option
import jax
import numpy as np
from mujoco_playground import wrapper
import pandas as pd

def set_disturbance(state):
    state.info['push_override'] = True
    state.info['push_override_xy'] = [0.0, -4.5]
    return state.info

def main():
    # Set up the GPU environment
    # run_setup()

    # Read the config file from command line argument
    config = read_config()

    # Create the environment
    env, env_cfg = create_environment(config, idealistic=True, animate=False)
    env.params.initialization.vdes = [0.0, 0.0, 0.0]
    env.params.initialization.strategy = 'manual'
    env.params.start_stance = 'right'
    env.params.push.push_duration = 0.7
    env.params.push.nopush_duration = 4.6
    
    # Load the model    
    inference_fn = load_policy(config)
    # inference_fn = lambda obs, rng: (np.zeros(env.action_size, dtype=np.float32), None)
    jit_inference_fn = jax.jit(inference_fn)
    reset, step = get_step_reset(env, config['backend'])

    # Rollout the policy in the environment
    T = 10.0
    frames, reward_plotter, data_plotter, info_plotter = rollout(
        reset        = reset,
        step         = step, 
        inference_fn = jit_inference_fn,
        env          = env, 
        T            = T,
        height       = 1080,
        width        = 2560,
        camera       = 'front',
        scene_option = get_mj_scene_option(contacts=False, com=False, perts=True),
        info_init_fn = set_disturbance,
        plot_vs      = False,
        info_plot_key=['vdes_res']
        # info_step_fn = lambda state: circle_vel(state, vx_lim[1], vy_lim[1], 1 / T)
    )
    info_plotter.to_numpy()
    vr = info_plotter.data['vdes_res']
    df = pd.DataFrame(vr[:, :2], columns=['vdes_res_x', 'vdes_res_y'])
    df.to_csv('logs/hero_push_vr.csv', index=False)
    path = Path(f'eval/icra/videos/{config["env"]}_hero_push.mp4')
    save_video(
        frames,
        path=path,
        env_cfg=env_cfg,
    )
        

if __name__ == '__main__':
    main()