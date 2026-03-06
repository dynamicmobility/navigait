import os
os.environ["MUJOCO_GL"] = "egl"
# os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"
# os.environ['JAX_PLATFORMS']='cpu'
# os.environ['JAX_CHECK_TRACER_LEAKS']='true'

# Internal imports
from utils.setupGPU import run_setup
from learning.startup import read_config, create_environment, get_step_reset, get_commit_hash
from learning.training import setup_training, train
from learning.inference import rollout

# Basic imports
from pathlib import Path
import yaml
import datetime
import numpy as np

# Graphics and plotting.
import utils.plotting as plotting

# jax and MJX imports
import jax

def main():
    run_setup()
    
    # Read the config file
    config = read_config()
    output_dir = Path(config['save_dir']) / config['name']
    os.makedirs(output_dir, exist_ok=False)
    
    # Create the environment
    print('Creating environment...')
    env, env_cfg = create_environment(config, for_training=True)
    ppo_params, network_params = setup_training(config['learning_params'])
    # print('PPO PARAMS')
    # print(ppo_params)
    # print('=========\n\n')
    # quit()
    # Save configuration
    config_save_path = Path(output_dir) / 'config.yaml'
    git_hash = get_commit_hash()
    config['git_hash'] = git_hash
    with open(config_save_path, 'w') as f:
        yaml.dump(dict(config), f)
    
    # Data for plotting
    print('Setting up data for plotting...')
    x_data, y_data, y_dataerr = [0], [0], [0]
    times = [datetime.datetime.now()]
    
    # Train
    print('Training...')
    eval_env, env_cfg = create_environment(config, for_training=False)
    make_inference_fn, params, metrics = train(
        config, output_dir, env, eval_env, ppo_params, network_params, times, x_data, y_data, y_dataerr
    )
    
    # jit compile stuff
    print('JIT compiling...')
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
    
    # Rollout
    print('Rolling out...')
    env, env_cfg = create_environment(config)
    reset, step = get_step_reset(env, backend='np')
    frames, reward_plotter, data_plotter, info_plotter = rollout(
        reset        = reset,
        step         = step,
        inference_fn = jit_inference_fn,
        env          = env,
        T            = 30.0
    )
    
    # Save video
    print('Saving video...')
    plotting.save_video(frames, env_cfg, path=Path(output_dir) / 'rollout.mp4')
    
    # Save plot
    print('Saving plot...')
    plotting.save_metrics(reward_plotter, path=Path(output_dir) / 'metrics.png')
    
    
if __name__ == "__main__":
    main()
