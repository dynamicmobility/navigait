
# Basic imports
import sys
import yaml
import numpy as np

# jax and MJX imports
from mujoco_playground._src.mjx_env import MjxEnv
import jax

# Other environments
from ml_collections import config_dict

def read_config():
    """Reads the YAML config file"""
    if len(sys.argv) != 2:
        print("Usage: python script.py <yaml_file>")
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    
    try:
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
            # print('\n=== Config ===')
            # print(yaml.dump(data, default_flow_style=False))
            # print('==============')
    except Exception as e:
        print(f"Error reading {yaml_file}: {e}")
        sys.exit(1)
    
    del sys.argv[1]
    return dict(data)

def get_atalante_env(config: dict, kwargs):
    if config['env'] == 'NaviGait':
        from envs.atalante.navigait import Exo
        env = Exo(
            gaitlib_path = config['gaitlib_path'],
            gait_type    = 'P2',
            **kwargs
        )
    elif config['env'] == 'NoRes':
        raise Exception('Broken')
        from envs.atalante.nores import NoRes
        env_cfg = NoRes.get_default_config()
        env = NoRes(
            gaitlib_path='control/gaits/2D_library_v6',
            config=env_cfg,
            config_overrides=None,
            num_vx_gaits=13,
            gait_type='P2',
            backend=backend,
            idealistic=idealistic
        )
    elif config['env'] == 'canonicalRL':
        raise Exception('Broken')
    else:
        raise Exception(f'Unknown environment {config["env"]} for Atalante')
    
    return env

def get_g1_env(config, kwargs):
    if config['env'] == 'canonicalRL':
        from envs.g1.canonicalRL import G1
        env = G1(**kwargs)
    elif config['env'] == 'NaviGait':
        from envs.g1.navigait import G1
        env = G1(
            gaitlib_path = 'control/gaits/G1_2D_Library_v3',
            gait_type    = 'P2',
            **kwargs
        )
    
    return env

def get_bruce_env(config, kwargs):
    if config['env'] == 'canonicalRL':
        from envs.bruce.canonicalRL import Bruce
        env = Bruce(**kwargs)
    elif config['env'] == 'NaviGait':
        from envs.bruce.navigait import Bruce
        env = Bruce(
            gaitlib_path = config['gaitlib_path'],
            gait_type    = 'P2',
            **kwargs
        )
    elif config['env'] == 'Imitation':
        from envs.bruce.imitation import BruceImitation
        env = BruceImitation(
            gaitlib_path = config['gaitlib_path'],
            gait_type    = 'P2',
            **kwargs
        )

    return env

def create_config_dict(config: dict) -> config_dict.ConfigDict:
    """Converts a dictionary to a ConfigDict."""
    config_dict_obj = config_dict.ConfigDict()
    for key, value in config.items():
        if isinstance(value, dict):
            config_dict_obj[key] = create_config_dict(value)
        else:
            config_dict_obj[key] = value
    return config_dict_obj

def create_environment(
    config: dict,
    for_training = False,
    idealistic = False,
    animate = False
) -> tuple[MjxEnv, config_dict.ConfigDict]:
    """Creates an MJX enironment for training"""
    env_params = create_config_dict(config['env_config'])
    backend = 'jnp' if for_training else config['backend']
    
    get_env_fns = {
        'Atalante': get_atalante_env,
        'G1':       get_g1_env,
        'BRUCE':    get_bruce_env
    }
    get_env_fn = get_env_fns[config['robot']]
    kwargs = {
        'idealistic': idealistic,
        'animate': animate,
        'backend': backend,
        'env_params': env_params
    }
    env = get_env_fn(
        config,
        kwargs
    )

    return env, env_params


def get_step_reset(env, backend):
    """Returns the reset and step functions based on the backend."""
    if backend == 'jnp':
        print('jitting')
        reset = jax.jit(env.reset)
        step = jax.jit(env.step)
    else:
        reset = env.reset
        step = env.step
    return reset, step

def get_commit_hash():
    import subprocess

    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('utf-8').strip()

        # Check for unadded / uncommitted changes
        status_output = subprocess.check_output(
            ['git', 'status', '--porcelain']
        ).decode('utf-8').strip()

        if status_output:
            input("⚠️ Warning: There are unadded or uncommitted changes in the repository. Press ENTER to continue...")

        return commit_hash

    except subprocess.CalledProcessError as e:
        print(f"Error getting commit hash: {e}")
        return None