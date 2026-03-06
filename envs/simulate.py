import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
# os.environ['JAX_TRACEBACK_FILTERING']='off'

from utils.setupGPU import run_setup
run_setup() # Run the GPU setup

# Python imports
import argparse
import numpy as np
from pathlib import Path

# Local imports
from learning.startup import create_environment, get_step_reset, read_config
from learning.inference import rollout, circle_vel
from utils.plotting import save_video, save_metrics, save_trajectories
    
def main():
    # Load arguments
    config = read_config()
    env_name = config['env']
    
    # Loading the environment
    env, env_cfg = create_environment(config, idealistic=True, animate=True)
    env.params.initialization.vdes = [-0.0, 0.0, 0.0]
    env.params.initialization.strategy = 'manual'
    env.params.push.enabled = True
    
    # Get reset and step functions
    reset, step = get_step_reset(env, config['backend'])
    
    # Simulate the environment
    def inference_fn(obs, rng):
        """Dummy inference function for the environment."""
        ctrl = np.zeros(env.action_size)
        # ctrl = np.random.random(env.action_size) * 2 - 1  # Example control signal
        # ctrl = -np.ones(env.action_size) * 1.5  # Example control signal
        return ctrl, None

    T = 25.0
    frames, reward_plotter, data_plotter, info_plotter= rollout(
        reset,
        step,
        inference_fn,
        env,
        T=T,
        height=640,
        width=480,
        info_step_fn=lambda state: circle_vel(state, 0.2, 0.1, 1 / T),
    )

    # # Save metrics
    save_metrics(reward_plotter, path=Path(f'visualization/{env_name}_metrics.png'))

    # # Plot ctrl vs actual
    save_trajectories(
        mj_model = env.mj_model,
        nq=env.nq,
        plotter = data_plotter, 
        pos_path=Path(f'visualization/{env_name}_position_trajectories.png'),
        vel_path=Path(f'visualization/{env_name}_velocity_trajectories.png'),
        torque_path=Path(f'visualization/{env_name}_torque_trajectories.png')
    )
    # data_plotter.save_to_h5(Path(f'logs/{env_name}_trajectories.h5'))

    # Save video
    save_video(frames, env_cfg, path=Path(f'visualization/{env_name}_sim.mp4'))
    
if __name__ == '__main__':
    main()