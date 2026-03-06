import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
from utils.setupGPU import run_setup
from pathlib import Path
from learning.startup import read_config, create_environment, get_step_reset
from learning.inference import rollout, load_policy, vx_sine_vel, circle_vel
from utils.plotting import save_video, set_mpl_params, get_mj_scene_option
import jax
import numpy as np
from mujoco_playground import wrapper
import matplotlib.pyplot as plt
import yaml

LABEL  = 16
LEGEND = 12
    
def plot_tracking(name, ax_vx, ax_vy, ax_vw):
    with open(f'icra-policies/{name}/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Create the environment
    env, env_cfg = create_environment(config, idealistic=True, animate=False)
    env.params.initialization.vdes = [0.0, 0.0, 0.0]
    env.params.push.enabled = False
    vx_lim = env.params.command.lin_vel_x
    vy_lim = env.params.command.lin_vel_y
    
    # Load the model    
    inference_fn = load_policy(config)
    jit_inference_fn = jax.jit(inference_fn)
    reset, step = get_step_reset(env, config['backend'])

    # Rollout the policy in the environment
    T = 60.0
    frames, reward_plotter, data_plotter, info_plotter = rollout(
        reset        = reset,
        step         = step, 
        inference_fn = jit_inference_fn,
        env          = env, 
        T            = T,
        height       = 720, #1440,
        width        = 1080, #2560,
        info_step_fn = lambda state: circle_vel(state, vx_lim[1], vy_lim[1], 1 / T),
        gen_vid      = False,
        camera       = 'top_side',
        scene_option = get_mj_scene_option(contacts=False, perts=False, com=False)
    )
    # save_video(frames, env_cfg, Path(f'eval/icra/videos/{name}-velocity-tracking.mp4'))

    set_mpl_params()
    data_plotter.to_numpy()
    info_plotter.to_numpy()
    LW = 2
    ax_vx.plot(data_plotter.data['time'], data_plotter.data['sensordata'][:, -3], lw=LW, label='Actual')
    ax_vx.plot(info_plotter.data['time'], info_plotter.data['vdes'][:, 0], lw=LW, label='Command')
    ax_vy.plot(data_plotter.data['time'], data_plotter.data['sensordata'][:, -2], lw=LW, label='Actual')
    ax_vy.plot(info_plotter.data['time'], info_plotter.data['vdes'][:, 1], lw=LW, label='Command')
    ax_vw.plot(data_plotter.data['time'], data_plotter.data['qvel'][:, 2], lw=LW, label='Actual')
    ax_vw.plot(info_plotter.data['time'], info_plotter.data['vdes'][:, 2], lw=LW, label='Command')
    # ax_vx.grid()
    # ax_vy.grid()
    
    # fig.suptitle(f'Velocity Tracking: {config["env"]} on {config["robot"]}')
    # ax_vx.set_ylabel(f'$v_x$ $(m/s)$')
    # ax_vx.set_xlabel(f'$t$ (sec)')
    # ax_vx.legend()
    # ax_vy.set_ylabel(f'$v_y$ $(m/s)$')
    # ax_vy.set_xlabel(f'$t$ (sec)')
    ax_vx.set_ylim(-0.26, 0.26)
    ax_vy.set_ylim(-0.55, 0.55)
    ax_vw.set_ylim(-0.5, 0.5)
    
    ax_vx.set_yticks([-0.2, 0, 0.2])
    ax_vy.set_yticks([-0.4, 0, 0.4])
    ax_vw.set_yticks([-0.4, 0, 0.4])
    
    ax_vx.tick_params(axis='x', which='both', labelsize=0) 
    ax_vw.set_xlabel(f'$t$ (sec)', fontsize=LABEL)
    ax_vw.tick_params(labelsize=LEGEND)
    ax_vy.tick_params(labelsize=LEGEND)
    ax_vx.tick_params(labelsize=LEGEND)
    # ax_vx.set_title(title, fontsize=24)

    return ax_vx, ax_vy, ax_vw


def main():
    set_mpl_params()
    fig, axs = plt.subplots(nrows=3, ncols=3)
    titles = [
        r'\textsc{NaviGait}',
        'Imitation RL',
        'Canonical RL'
    ]
    for col, env in enumerate(['navigait', 'imitation', 'canonical']):
        axs[:, col] = plot_tracking(env, axs[0, col], axs[1, col], axs[2, col])
    axs[0, 0].set_ylabel(r'$v_x$ $(m/s)$', fontsize=LABEL)
    axs[1, 0].set_ylabel(r'$v_y$ $(m/s)$', fontsize=LABEL)
    axs[2, 0].set_ylabel(r'$\omega_z$ $(rad/s)$', fontsize=LABEL)
    
    # Removing axes labels
    axs[0, 0].tick_params(axis='x', which='both', labelsize=0) 
    axs[0, 0].set_xlabel('')
    
    axs[1, 0].tick_params(axis='x', which='both', labelsize=0) 
    axs[1, 0].set_xlabel('')
    for row in [0, 1]:
        for col in [1,2]:
            axs[row, col].tick_params(axis='both', which='both', labelsize=0) 
            axs[row, col].tick_params(axis='both', which='both', labelsize=0) 
            axs[row, col].set_xlabel('')
            
    axs[2, 1].tick_params(axis='y', which='both', labelsize=0) 
    axs[2, 2].tick_params(axis='y', which='both', labelsize=0) 
            
    # for col in [1,2]:
    #     axs[0, col].tick_params(axis='both', which='both', labelsize=0) 
    #     axs[0, col].tick_params(axis='both', which='both', labelsize=0) 
    #     axs[1, col].tick_params(axis='y', which='both', labelsize=0) 

    # axs[0, 0].legend(fontsize=LEGEND)
    for ax in axs.flat:
        ax.margins(x=0.02, y=0.02)
    fig.set_size_inches(15, 4)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.03, hspace=0.18)
    fig.savefig('paper_plots/velocity_tracking.svg')


        

if __name__ == '__main__':
    main()