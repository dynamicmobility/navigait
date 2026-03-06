import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
from utils.setupGPU import run_setup
from pathlib import Path
from learning.startup import read_config, create_environment, get_step_reset
from learning.inference import rollout, load_policy
from utils.plotting import set_mpl_params
import jax
import numpy as np
from mujoco_playground import wrapper
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time
import h5py
import matplotlib.ticker as mticker



def plot_fall_polar(angles, magnitudes, fall_probabilities, successes):
    """
    Plots a polar heatmap of fall probability for a biped under perturbations.

    Parameters:
    -----------
    angles : array-like
        Array of perturbation angles (radians).
    magnitudes : array-like
        Array of perturbation magnitudes.
    fall_probabilities : 2D array-like
        Matrix of shape (len(magnitudes), len(angles)) containing
        fraction of trials that resulted in a fall (0 to 1).
    """
    LBLSIZE=24
    TXTCOL='black'
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    
    # Plot using pcolormesh for smooth color interpolation
    c = ax.pcolormesh(angles, magnitudes, fall_probabilities, vmin=0.0, vmax=1.0, 
                      cmap='RdYlGn_r',
                    #   cmap='Blues_r', 
                      shading='gouraud'
        )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: f"{x:.0f} N")
    )
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.set_theta_zero_location("N")  # 0 rad points upwards
    ax.set_theta_direction(-1)       # Clockwise angles
    ax.set_rlabel_position(22)  # Change the angle of radial labels
    # ax.set_rlabel_position(135)  # Move radial labels to 135 degrees to avoid overlap

    # cbar = fig.colorbar(c, ax=ax)
    # cbar.set_label(label="Fall Probability (0=Stable, 1=Fall)", size=19, labelpad=15)
    # cbar.ax.tick_params(labelsize=18)
    ax.tick_params(labelsize=LBLSIZE)
    ax.tick_params(axis='y', colors=TXTCOL)
    C = 0.26
    ax.grid(linewidth=1.0, color=[C, C, C, 1], ls='--') #
    # ax.set_radial_lim_em(5)
    # ax.set_rlabel_position(45)  # Move radial labels to avoid overlap
    ax.tick_params(pad=15)  # Add padding to tick labels
    ax.set_yticklabels([f"\\textbf{{{label.get_text()}}}" for label in ax.get_yticklabels()])
    
    ax2 = fig.add_subplot(1, 1, 1, projection='polar', label="ax2")
    ax2.tick_params(labelsize=LBLSIZE)
    ax2.patch.set_alpha(0) # Makes the plotting area transparent
    ax2.spines['polar'].set_visible(False) # Hides the outer polar circle of the top axis
    ax2.grid(False)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax2.set_xticklabels([])  # Remove degree labels from ax2
    ax2.set_rlabel_position(240)  # Change the angle of radial labels
    ax2.set_yticks(range(len(successes) + 1))
    ax2.set_yticklabels([''] + [f"\\textbf{{{s*100:.1f}\\%}}" for s in successes])
    ax2.tick_params(axis='y', colors=TXTCOL)


    return fig, ax

def main():
    run_setup()
    set_mpl_params()
    config = read_config()
    env, env_cfg = create_environment(config, for_training=True)
    env.params.push.enabled = True
    
    # Load the model    
    inference_fn = load_policy(config) 
    reset, step = env.reset, env.step

    PUSH_MAG_MAX = 40
    PUSH_PER_ANGLE = 32
    N_PUSH_ANGLES = 32
    PUSH_DURATION = 0.1
    NOPUSH_DURATION = [3, 5] # uniformly sampled from here
    N_TRIALS = 8
    N_ENVS = N_PUSH_ANGLES * PUSH_PER_ANGLE * N_TRIALS
    
    push_angles = np.repeat(np.linspace(0, 2 * np.pi, num=N_PUSH_ANGLES), N_TRIALS)
    push_mags   = np.linspace(0, PUSH_MAG_MAX, num=PUSH_PER_ANGLE)
    push_angles, push_mags = np.meshgrid(push_angles, push_mags)
    pushes      = np.vstack((push_angles.ravel(), push_mags.ravel())).T
    
    
    T = 7.0
    N_STEPS = T / env.params.ctrl_dt
    rng = jax.random.PRNGKey(1)
    keys = jax.random.split(rng, num=N_ENVS)
    print(N_ENVS, keys.shape, pushes.shape)

    def step_fn(carry, x):
        state = carry
        action, _ = inference_fn(state.obs, rng)
        new_state = step(state, action)
        new_state.info['done'] = new_state.info['done'] + new_state.done
        return new_state, None
    
    def rollout(key, push):
        state = reset(key)
        
        state.info['push_override'] = True
        state.info['push_override_xy'] = jax.numpy.array(push)
        state.info['push_duration'] = PUSH_DURATION
        _, key = jax.random.split(key)
        state.info['nopush_duration'] = jax.random.uniform(
            key, minval=NOPUSH_DURATION[0], maxval=NOPUSH_DURATION[1]
        )
        state.info['done'] = 0.0
        
        return jax.lax.scan(step_fn, state, (), N_STEPS)
    
    start = time.time()
    # batched_rollout = jax.jit(jax.vmap(rollout))
    # states, _ = batched_rollout(keys, pushes)
    # Write to HDF5 file
    # with h5py.File(f"logs/{config['env']}_pushes.h5", "w") as f:
    #     f.create_dataset("pushes", data=pushes)
    #     f.create_dataset("dones", data=states.done)
    with h5py.File(f"logs/{config['env']}_pushes.h5", "r") as f:
        pushes = np.array(f["pushes"][:])
        dones = np.array(f["dones"][:])


    def check_success(inner, outer):
        print(inner, outer)
        push_mags = np.linalg.norm(pushes, axis=1)
        check_idxs = (inner < push_mags) & (push_mags < outer)
        return 1 - np.sum(dones[check_idxs]) / np.sum(check_idxs)
    
    # print(check_success(0, 10))
    # print(check_success(10, 20))
    # print(check_success(20, 30))
    # print(check_success(30, 40))
    # quit()
    
    dones_by_trial = dones.reshape((N_PUSH_ANGLES * PUSH_PER_ANGLE, -1))
    print('Total success', np.sum(dones) / N_ENVS)
    dones_by_trial = np.sum(dones_by_trial, axis=1) / N_TRIALS
    
    push_angles = np.linspace(0, 2 * np.pi, num=N_PUSH_ANGLES)
    push_mags   = np.linspace(0, PUSH_MAG_MAX, num=PUSH_PER_ANGLE)
    push_angles, push_mags = np.meshgrid(push_angles, push_mags)
    push_results = dones_by_trial.reshape(push_angles.shape)
    
    fig, ax = plot_fall_polar(
        push_angles, 
        push_mags, 
        push_results,
        [check_success(10 * i, 10 * (i + 1)) for i in range(4)]
    )

    names = {
        'NaviGait': r'\textsc{NaviGait}',
        'Imitation': 'Imitation RL',
        'canonicalRL': 'Canonical RL'
    }

    


    ax.set_title(f'{names[config["env"]]}', va='bottom', fontsize=30)
    # fig.set_size_inches((10, 6))
    fig.tight_layout()
    fig.savefig(f'paper_plots/{config["env"]}_disturbance.pdf', dpi=200)
    
    print(dones)
    print('Runtime =', time.time() - start)
            

if __name__ == '__main__':
    main()
