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
import utils.geometry as geo
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import yaml
FONTSIZE=22
SIDE_LENGTH = 12
VX_LIM = [-0.2, 0.2]
VY_LIM = [-0.1, 0.1]

def run_imitation_accuracy(env, reset, step, inference_fn, vdes, T=2.0, info_init_fn=lambda state: state.info):
    env.params.initialization.vdes = list(vdes)
    env.params.initialization.strategy = 'manual'
    # env.params.noise_scale = 1.0

    # Rollout the policy in the environment
    frames, reward_plotter, data_plotter, info_plotter = rollout(
        reset        = reset,
        step         = step, 
        inference_fn = inference_fn,
        env          = env, 
        T            = T,
        height       = 640,
        width        = 480,
        # info_step_fn = lambda state: circle_vel(state, vx_lim[1], vy_lim[1], 1 / T),
        info_plot_key= ['gait_history', 'base_history', 'qpos_history'],
        gen_vid=False,
        show_progress=False,
        info_init_fn=info_init_fn
    )
    info_plotter.to_numpy()
    gait_error = info_plotter.data['qpos_history'][:, 0, geo.FREE3D_POS:] - info_plotter.data['gait_history'][:, 1, :env.nq - geo.FREE3D_POS]
    gait_error = np.linalg.norm(gait_error, axis=1)
    return frames, data_plotter, gait_error


def grid_imitation_accuracy(name):
    # Read the config file from command line argument
    with open(f'icra-policies/{name}/config.yaml', 'r') as file:
        config = yaml.safe_load(file) 
    
    # Create the environment
    env, env_cfg = create_environment(config, idealistic=True, animate=False)
    env.params.push.enabled = False
    
    # Load the model    
    inference_fn = load_policy(config)
    jit_inference_fn = jax.jit(inference_fn)
    reset, step = get_step_reset(env, config['backend'])

    
    gait_errors = np.zeros((SIDE_LENGTH, SIDE_LENGTH))
    vxs = np.linspace(*VX_LIM, num=SIDE_LENGTH)
    vys = np.linspace(*VY_LIM, num=SIDE_LENGTH)
    for i, vx in tqdm(enumerate(vxs), total=SIDE_LENGTH):
        for j, vy in enumerate(vys):
            _, _, gait_error = run_imitation_accuracy(
                env          = env,
                reset        = reset,
                step         = step,
                inference_fn = jit_inference_fn,
                vdes         = np.array([vx, vy, 0.0]),
                T            = 10.0
            )
            gait_errors[i, j] = np.mean(gait_error)
    return gait_errors

def plot_imitation_accuracy(gait_errors, ax, err_max):
    red_green_cmap = LinearSegmentedColormap.from_list("red_green", ["green", "red"])
    im = ax.imshow(gait_errors, cmap=red_green_cmap, vmin=0.0, vmax=err_max, origin='lower')
    
    numticks = 3
    ax.set_yticks(np.linspace(0, SIDE_LENGTH - 1, num=numticks), np.linspace(*VX_LIM, num=numticks))  # custom column labels
    ax.set_xticks(np.linspace(0, SIDE_LENGTH - 1, num=numticks), np.linspace(*VY_LIM, num=numticks))  # custom row labels
    ax.set_xlabel("Lateral Velocity (m/s)", fontsize=FONTSIZE-2)
    ax.tick_params(labelsize=FONTSIZE-2)

    # # Show values on grid cells
    # for (i, j), val in np.ndenumerate(gait_errors):
    #     plt.text(j, i, f"{val:.2f}", ha='center', va='center', color="black")

    ax.set_aspect("equal")
    return ax, im

def main():
    set_mpl_params()
    fig, axs = plt.subplots(nrows=1, ncols=2) #, constrained_layout=True)
    axs = axs.flatten()
    navigait_errors = grid_imitation_accuracy('navigait')
    imitation_errors = grid_imitation_accuracy('imitation')
    
    err_max = np.max([navigait_errors, imitation_errors])
    print(err_max)
    ax_navigait, im = plot_imitation_accuracy(navigait_errors, axs[0], err_max)
    ax_imitation, im = plot_imitation_accuracy(imitation_errors, axs[1], err_max)
    # img = plt.imshow(np.array([[0,1]]))
    red_green_cmap = LinearSegmentedColormap.from_list("red_green", ["green", "red"])
    # fig.colorbar(img, label="Accuracy", shrink=0.5, cmap=red_green_cmap)
    # img.set_visible(False)
    # axs[2].axis('off')
    # from mpl_toolkits.axes_grid1 import make_axes_locatable

    # divider = make_axes_locatable(axs[1])
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cbar = fig.colorbar(im, cax=cax)
    cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cbar_ax, cmap=red_green_cmap)
    cbar.set_label('Imitation Error', size=FONTSIZE-2)
    cbar.ax.tick_params(labelsize=FONTSIZE-2)
    axs[1].tick_params(axis='y', which='both', labelsize=0)
    axs[0].set_ylabel("Forward Velocity (m/s)", fontsize=FONTSIZE-2)
    axs[0].set_title(r'\textsc{NaviGait} Error', fontsize=FONTSIZE)
    axs[1].set_title(r'Imitation RL Error', fontsize=FONTSIZE)
    fig.set_size_inches((10, 4))
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.savefig(f'paper_plots/imitation_accuracy.pdf', dpi=700)

if __name__ == '__main__':
    main()