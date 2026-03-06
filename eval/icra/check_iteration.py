import os
os.environ["MUJOCO_GL"] = "egl"
# os.environ['JAX_PLATFORMS']='cpu'
from utils.setupGPU import run_setup
from pathlib import Path
from learning.startup import read_config, create_environment, get_step_reset
from learning.inference import rollout, load_policy, get_all_models, get_make_inference_fn, get_params
from utils.plotting import set_mpl_params
import jax
import numpy as np
from mujoco_playground import wrapper
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.ticker as mticker
import pandas as pd


def main():
    run_setup()
    config = read_config()
    env, env_cfg = create_environment(config, for_training=True, idealistic=False)
    test = 'walk_forward'
    if test == 'alive':
        env.params.push.enabled = False
        env.params.initialization.vdes = [0.0, -0.0, 0.0]
        env.params.initialization.strategy = 'manual'
        env.params.command.enabled = False
    elif test == 'pushed':
        # env.params.push.enabled = False
        env.params.initialization.vdes = [0.0, -0.0, 0.0]
        env.params.initialization.strategy = 'manual'
        env.params.command.enabled = False
    elif test == 'walk_forward':
        env.params.push.enabled = False
        env.params.initialization.vdes = [0.15, -0.0, 0.0]
        env.params.initialization.strategy = 'manual'
        env.params.command.enabled = False
    else:
        raise Exception(f"Unknown test type: {test}")

    # Load the model    
    make_inference_fn = get_make_inference_fn(config)
    reset, step = env.reset, env.step

    N_ENVS = 128    
    
    T = 20.0
    N_STEPS = T / env.params.ctrl_dt
    rng = jax.random.PRNGKey(0)
    keys = jax.random.split(rng, num=N_ENVS)
    
    def rollout(key, params):
        inference_fn = make_inference_fn(params)
        def step_fn(carry, x):
            state = carry
            action, _ = inference_fn(state.obs, rng)
            new_state = step(state, action)
            new_state.info['done'] = new_state.info['done'] + new_state.done
            return new_state, None
        state = reset(key)
        state.info['done'] = 0
        return jax.lax.scan(step_fn, state, (), N_STEPS)
    
    batched_rollout = jax.jit(jax.vmap(rollout, in_axes=(0, None)))
    models = get_all_models(config)
    successes = [0]
    TEST = {
        'alive'             : lambda states: np.sum(states.info['done'] > 0),
        'pushed'            : lambda states: np.sum(states.info['done'] > 0),
        'walk_forward'      : lambda states: np.sum(states.data.qpos[:, 0] < 2.0)
    }
    for model in models:
        print(model)
        params = get_params(Path(model).resolve()) 
        states, _ = batched_rollout(keys, params)
        fails = TEST[test](states)
        successes.append(1 - fails / N_ENVS)
        print(successes[-1])
    
    with open(f'logs/{test}/{config['env']}_{config["name"]}.txt', 'w') as f:
        f.write(f'config: {Path(config['save_dir']) / config['name']}\n')
        f.write(f'test name = {test}')
        progress_df = pd.read_csv(Path(config['save_dir']) / config['name'] / 'progress.csv')
        new_df = pd.DataFrame({'times': progress_df['times'], 'x': progress_df['x'], 'success_rate': successes})
        new_df.to_csv(f, index=False)


            

if __name__ == '__main__':
    main()
