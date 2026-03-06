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
from dynamo_figures import CompositeImage, CompositeMode


def main():
    # Set up the GPU environment
    # run_setup()

    # Read the config file from command line argument
    config = read_config()

    # Create the environment
    env, env_cfg = create_environment(config, idealistic=True, animate=False)
    env.params.initialization.vdes = [0.15, 0.0, 0.0]
    env.params.initialization.strategy = 'manual'
    env.params.start_stance = 'left'
    env.params.push.enabled = False
    
    # Load the model    
    inference_fn = load_policy(config)
    jit_inference_fn = jax.jit(inference_fn)
    reset, step = get_step_reset(env, config['backend'])

    # Rollout the policy in the environment
    T = 25.0
    kwargs = {
        'hero': {'height': 1080, 'width': 3000, 'camera': 'side'},
        'main': {'height': 3000, 'width': 1080, 'camera': 'track'}
    }
    frames, reward_plotter, data_plotter, info_plotter = rollout(
        reset        = reset,
        step         = step, 
        inference_fn = jit_inference_fn,
        env          = env, 
        T            = T,
        scene_option = get_mj_scene_option(contacts=False, com=False, perts=False),
        **kwargs['main']
    )

    # Save the video
    save_video(
        frames,
        path=Path(f'eval/icra/videos/{config["env"]}_rollout.mp4'),
        env_cfg=env_cfg,
    )
        

if __name__ == '__main__':
    main()