# Basic imports
from pathlib import Path
import time
import datetime
import pickle
import pandas as pd
from ml_collections import config_dict

# Graphics and plotting.
import matplotlib
# matplotlib.use("TkAgg")  # headless backend, no X server required
import matplotlib.pyplot as plt

# RL imports
import functools
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo.train import train as train_ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.sac.train import train as train_sac
from brax.training.agents.sac import networks as sac_networks
from brax.training.acme import running_statistics

# jax and MJX imports
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

# Other environments

def setup_training(config):
    """Sets up the right training variables for the given algorithm"""
    ppo_params = config_dict.ConfigDict(config['ppo_params'])
    network_params = config_dict.ConfigDict(config['network_params'])
    
    # learning_config = config['learning_params']
    # if 'ppo_params' in learning_config:
    #     for key in learning_config['ppo_params']:
    #         ppo_params[key] = learning_config['ppo_params'][key]
    # if 'network_params' in learning_config:
    #     for key in learning_config['network_params']:
    #         network_params[key] = eval(learning_config['network_params'][key])
    
    return ppo_params, network_params

def plot_progress(
    num_steps, metrics, times, x_data, y_data, y_dataerr, ppo_params, save_dir
):
    # clear_output(wait=True)
    print('=== TRAINING EPOCH ===')
    print('time', time.time())
    print('num_steps', num_steps)
    print('total steps', ppo_params["num_timesteps"])
    times.append(datetime.datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])
    pd.DataFrame(
        {
            'times': times,
            'x': x_data,
            'y': y_data,
            'yerr': y_dataerr
        }
    ).to_csv(
        save_dir / 'progress.csv',
        index=False
    )

    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ax.set_xlim([0, ppo_params["num_timesteps"] * 1.25])
    ax.set_xlabel("# environment steps")
    ax.set_ylabel("reward per episode")
    ax.set_title(f"y={y_data[-1]:.3f}")
    ax.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
    ax.scatter(x_data, y_data)
    for i, (t, value) in enumerate(zip(x_data, y_data)):
        ax.text(x_data[i], y_data[i], f'({t}, {value:.0f})', fontsize=8, ha='right', va='bottom', color='red')

    save_dir = save_dir / 'progress.png'
    plt.savefig(save_dir, dpi=300)
    
# def save_params(step, make_policy, params, output_dir):
#     print(f"Saving params at step {step} to {output_dir}_{step}")
#     fullpath = Path(output_dir) / f'{step}'
#     with open(fullpath, 'wb') as f:
#         pickle.dump(params, f)


    
def train(
    config_yaml, output_dir: Path, env, eval_env, ppo_params, network_params, times, x_data, y_data, y_dataerr
):
    train_algo = train_ppo
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **network_params
    )
    network_config = checkpoint.network_config(
        observation_size=eval_env.observation_size,
        action_size=eval_env.action_size,
        normalize_observations=ppo_params.normalize_observations,
        network_factory=network_factory,
    )
    
    train_fn = functools.partial(
        train_algo, **dict(ppo_params),
        network_factory=network_factory,
        progress_fn=lambda num_steps, metrics: plot_progress(
            num_steps, metrics, times, x_data, y_data, y_dataerr, ppo_params, output_dir
        ),
        policy_params_fn=lambda current_step, make_policy, params: checkpoint.save(
            path   = output_dir.resolve(),
            step   = current_step,
            params = params,
            config = network_config
        ),
    )
    
    make_inference_fn, trained_params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        # eval_env=eval_env,
    )
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    
    return make_inference_fn, trained_params, metrics