import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
from utils.setupGPU import run_setup
from pathlib import Path
from learning.startup import read_config, create_environment, get_step_reset
from learning.inference import rollout, load_policy, circle_vel
from utils.plotting import save_video, save_metrics, save_trajectories, get_mj_scene_option
import jax
import numpy as np
from mujoco_playground import wrapper

def set_disturbance(state):
    state.info['push_override'] = True
    state.info['push_override_xy'] = [30.0, 0.0]
    return state.info

def main():
    # Set up the GPU environment
    # run_setup()

    # Read the config file from command line argument
    config = read_config()

    # Create the environment
    env, env_cfg = create_environment(config, idealistic=True, animate=False)
    env.params.initialization.vdes = [0.0, -0.05, 0.0]
    env.params.initialization.strategy = 'manual'
    # env.params.start_stance = 'right'
    # env.params.noise_scale = 1.0
    # env.params.domain_randomization.obs_delay.enabled = True
    env.params.push.push_duration = 0.1
    env.params.push.nopush_duration = 5.1
    # env.params.push.enabled = False
    # env.params.initialization.random_jt_calibration.enabled = False
    
    # Load the model    
    inference_fn = load_policy(config)
    # inference_fn = lambda obs, rng: (np.zeros(env.action_size, dtype=np.float32), None)
    jit_inference_fn = jax.jit(inference_fn)
    reset, step = get_step_reset(env, config['backend'])

    vx_lim = config['env_config']['command']['lin_vel_x']
    vy_lim = config['env_config']['command']['lin_vel_y']
    # Rollout the policy in the environment
    T = 10.0
    frames, reward_plotter, data_plotter, info_plotter = rollout(
        reset        = reset,
        step         = step, 
        inference_fn = jit_inference_fn,
        env          = env, 
        T            = T,
        height       = 640,
        width        = 480,
        camera       = 'track',
        scene_option = get_mj_scene_option(contacts=False, com=False, perts=True),
        info_init_fn = set_disturbance
        # info_step_fn = lambda state: circle_vel(state, vx_lim[1], vy_lim[1], 1 / T)
    )

    # # Save metrics
    save_metrics(reward_plotter, path=Path(f'visualization/{config['env']}_metrics.png'))
    
    # # Save trajectories
    # save_trajectories(
    #     env.mj_model,
    #     env.nq,
    #     data_plotter,
    #     pos_path=Path(f'visualization/{config["env"]}_position_trajectories.png'),
    #     vel_path=Path(f'visualization/{config["env"]}_velocity_trajectories.png'),
    #     torque_path=Path(f'visualization/{config["env"]}_torque_trajectories.png'),
    #     sensor_path=Path(f'visualization/{config["env"]}_sensor_trajectories.png'),
    # )

    # # Save the video
    save_video(
        frames,
        path=Path(f'visualization/{config["env"]}_rollout.mp4'),
        env_cfg=env_cfg,
    )
        

if __name__ == '__main__':
    main()