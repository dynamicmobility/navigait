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
import matplotlib.animation as animation
from eval.icra.imitation_accuracy import run_imitation_accuracy
import yaml

T_PUSH = 5.0

def set_disturbance(state):
    state.info['push_override'] = True
    state.info['push_override_xy'] = [-6.0, 0.0]
    return state.info

def get_gait_error(config):
    # Create the environment
    env, env_cfg = create_environment(config, idealistic=True, animate=False)

    env.params.initialization.vdes = [0.15, 0.0, 0.0]
    # env.params.push.enabled = True
    # env.params.push.push_duration = 0.1
    # env.params.push.nopush_duration = T_PUSH
    env.params.push.push_duration = 0.6
    env.params.push.nopush_duration = 4.0
    
    
    # Load the model    
    inference_fn = load_policy(config)
    jit_inference_fn = jax.jit(inference_fn)
    reset, step = get_step_reset(env, config['backend'])

    frames, data_plotter, gait_error = run_imitation_accuracy(
        env = env,
        reset = reset,
        step = step,
        inference_fn = jit_inference_fn,
        vdes = [0.15, 0.0, 0.0],
        T = 9.0,
        info_init_fn=set_disturbance,
        gen_vid=True
    )

    data_plotter.to_numpy()
    return frames, env_cfg, data_plotter, gait_error


def main():
    with open('icra-policies/imitation/config.yaml', 'r') as file:
        imitation_config = yaml.safe_load(file) 

    with open('icra-policies/navigait/config.yaml', 'r') as file:
        navigait_config = yaml.safe_load(file) 

    set_mpl_params()
    fig, ax = plt.subplots()
    ng_frames, ng_cfg, data_plotter, imitation_error = get_gait_error(imitation_config)
    im_frames, im_cfg, data_plotter, navigait_error = get_gait_error(navigait_config)
    all_error = np.hstack((navigait_error, imitation_error))

    save_video(ng_frames, ng_cfg, Path('eval/icra/videos/ng_accuray.mp4'))
    save_video(im_frames, im_cfg, Path('eval/icra/videos/im_accuray.mp4'))

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

    animate_error_plot(
        time=np.asarray(data_plotter.data['time'][1:]),
        imitation_error=np.asarray(imitation_error),
        navigait_error=np.asarray(navigait_error),
        out_path=Path('eval/icra/videos/imitation_push.mp4'),
    )


def animate_error_plot(time, imitation_error, navigait_error, out_path):
    FONTSIZE = 22
    dt = np.median(np.diff(time))
    fps = int(round(1.0 / dt)) if dt > 0 else 50

    all_error = np.concatenate([imitation_error, navigait_error])
    ymin, ymax = float(all_error.min()), float(all_error.max())
    ypad = 0.05 * (ymax - ymin) if ymax > ymin else 0.01

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(float(time[0]), float(time[-1]))
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_xlabel('$t$', fontsize=FONTSIZE)
    ax.set_ylabel(r'Error ($||\cdot||_2$)', fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)

    im_line, = ax.plot([], [], label='Imitation Policy Error')
    ng_line, = ax.plot([], [], label=r'\textsc{NaviGait} Policy Error')
    push_line, = ax.plot([], [], ls='--', c='black')
    push_annot = ax.annotate(
        "robot pushed",
        xy=(T_PUSH, 0.3),
        xytext=(0.2, 0.8),
        textcoords="axes fraction",
        arrowprops=dict(facecolor='black', arrowstyle="->"),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
        fontsize=FONTSIZE,
    )
    push_annot.set_visible(False)
    ax.legend(fontsize=FONTSIZE, loc='upper right')
    fig.tight_layout()

    n = len(time)

    def init():
        im_line.set_data([], [])
        ng_line.set_data([], [])
        push_line.set_data([], [])
        push_annot.set_visible(False)
        return im_line, ng_line, push_line, push_annot

    def update(i):
        im_line.set_data(time[: i + 1], imitation_error[: i + 1])
        ng_line.set_data(time[: i + 1], navigait_error[: i + 1])
        if time[i] >= T_PUSH:
            push_line.set_data(
                [T_PUSH, T_PUSH],
                [ymin - ypad, ymax + ypad],
            )
            push_annot.set_visible(True)
        else:
            push_line.set_data([], [])
            push_annot.set_visible(False)
        return im_line, ng_line, push_line, push_annot

    anim = animation.FuncAnimation(
        fig, update, frames=n, init_func=init,
        interval=1000 / fps, blit=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
    print(f'Writing animation to {out_path} ({n} frames @ {fps} fps)')
    anim.save(out_path, writer=writer, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()