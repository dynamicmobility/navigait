import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
from utils.setupGPU import run_setup
from pathlib import Path
from learning.startup import read_config, create_environment, get_step_reset
from learning.inference import rollout, load_policy, vx_sine_vel, circle_vel
from utils.plotting import save_video, set_mpl_params
import jax
import numpy as np
from mujoco_playground import wrapper
import matplotlib.pyplot as plt
from eval.icra.imitation_accuracy import run_imitation_accuracy
import yaml

T_PUSH = 5.0

def set_disturbance(state):
    state.info['push_override'] = True
    state.info['push_override_xy'] = [20.0, 0.0]
    # state.info['push_override_xy'] = [5.0, 0.0]
    return state.info

def get_gait_error(config):
    # Create the environment
    env, env_cfg = create_environment(config, idealistic=True, animate=False)

    env.params.initialization.vdes = [0.05, 0.0, 0.0]
    env.params.push.enabled = True
    env.params.push.push_duration = 0.1
    env.params.push.nopush_duration = T_PUSH
    
    
    # Load the model    
    inference_fn = load_policy(config)
    jit_inference_fn = jax.jit(inference_fn)
    reset, step = get_step_reset(env, config['backend'])

    frames, data_plotter, gait_error = run_imitation_accuracy(
        env = env,
        reset = reset,
        step = step,
        inference_fn = jit_inference_fn,
        vdes = [0.0, 0.0, 0.0],
        T = 10.0,
        info_init_fn=set_disturbance
    )

    data_plotter.to_numpy()
    return data_plotter, gait_error


def main():
    with open('icra-policies/imitation/config.yaml', 'r') as file:
        imitation_config = yaml.safe_load(file) 

    with open('icra-policies/navigait/config.yaml', 'r') as file:
        navigait_config = yaml.safe_load(file) 

    set_mpl_params()
    fig, ax = plt.subplots()
    data_plotter, imitation_error = get_gait_error(imitation_config)
    data_plotter, navigait_error = get_gait_error(navigait_config)
    all_error = np.hstack((navigait_error, imitation_error))

    ax.plot(data_plotter.data['time'][1:], imitation_error, label='Imitation Policy Error')
    ax.plot(data_plotter.data['time'][1:], navigait_error, label=r'\textsc{NaviGait} Policy Error')
    ax.plot(np.linspace(T_PUSH, T_PUSH, num=50), np.linspace(np.min(all_error), np.max(all_error)), ls='--', c='black')
    # Add a text box
    # Add annotation with arrow
    # Lock current limits so they won't auto-adjust\
    FONTSIZE = 22
    ax.annotate(
        "robot pushed",
        xy=(T_PUSH, 0.3),
        xytext=(0.2, 0.8),  # relative position
        textcoords="axes fraction",  # interpret xytext as fraction of axes
        arrowprops=dict(facecolor='black', arrowstyle="->"),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
        fontsize=FONTSIZE
    )
    ax.set_xlabel('$t$', fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    ax.set_ylabel(r'Error ($||\cdot||_2$)', fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE)
    fig.set_size_inches((12, 3.5))
    fig.tight_layout()
    fig.savefig('paper_plots/imitation_push.pdf', dpi=1000)
    # Save the video


if __name__ == '__main__':
    main()