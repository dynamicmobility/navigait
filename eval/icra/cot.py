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
from utils import geometry as geo
import numpy as np
import yaml

def cot(time, position, velocity, force):
    P = np.sum(np.abs(force*velocity), axis=1)
    E = np.trapezoid(P, time)
    dx = position[-1, 0] - position[0, 0]
    print("E: ", E, " dx: ", dx)
    m = 4.8
    g = 9.8
    return E/(m*g*dx)

def run_cot(config_name):
    # Read the config file from command line argument
    with open(config_name, 'r') as file:
        config = yaml.safe_load(file) 

    # Create the environment
    env, env_cfg = create_environment(config, idealistic=True, animate=False)
    env.params.initialization.vdes = [0.2, 0.0, 0.0]
    env.params.command.enabled = False
    env.params.initialization.strategy = 'manual'
    env.params.push.enabled = False
    if 'start_stance' in env.params:
        env.params.start_stance = 'right'
    vx_lim = env.params.command.lin_vel_x
    vy_lim = env.params.command.lin_vel_y
    
    # Load the model    
    inference_fn = load_policy(config)
    jit_inference_fn = jax.jit(inference_fn)
    reset, step = get_step_reset(env, config['backend'])

    # Rollout the policy in the environment
    T = 15.0
    frames, reward_plotter, data_plotter, info_plotter = rollout(
        reset        = reset,
        step         = step, 
        inference_fn = jit_inference_fn,
        env          = env, 
        T            = T,
        height       = 640,
        width        = 480,
        gen_vid      = False,
        show_progress= False
    )

    time = np.array(data_plotter.data['time'])
    position = np.array(data_plotter.data['qpos'])
    velocity = np.array(data_plotter.data['qvel'])[:, geo.FREE3D_VEL:]
    frc = np.array(data_plotter.data['qfrc_actuator'])[:, geo.FREE3D_VEL:]

    # plt.plot(time, position[:, 0:2])
    # plt.show()

    # set_mpl_params()
    # fig, axs = plt.subplots(nrows=1, ncols=3)
    # ax_vx, ax_vy, ax_cot = axs.flatten()
    # data_plotter.to_numpy()
    # info_plotter.to_numpy()
    # ax_vx.plot(data_plotter.data['time'], data_plotter.data['sensordata'][:, -3], label='Command')
    # ax_vx.plot(info_plotter.data['time'], info_plotter.data['vdes'][:, 0], label='Actual')
    # ax_vy.plot(data_plotter.data['time'], data_plotter.data['sensordata'][:, -2], label='Command')
    # ax_vy.plot(info_plotter.data['time'], info_plotter.data['vdes'][:, 1], label='Actual')
    
    # fig.suptitle(f'Velocity Tracking: {config["env"]} on {config["robot"]}')
    # ax_vx.set_ylabel(f'$v_x$ $(m/s)$')
    # ax_vx.set_xlabel(f'$t$ (sec)')
    # ax_vx.legend()
    # ax_vy.set_ylabel(f'$v_y$ $(m/s)$')
    # ax_vy.set_xlabel(f'$t$ (sec)')
    # fig.set_size_inches((10, 5))
    # fig.tight_layout()
    # fig.savefig(f'paper_plots/{config["env"]}_vel_tracking.png', dpi=400)

    # Save the video
    # save_video(
    #     frames,
    #     path=Path(f'visualization/{config["env"]}_cot.mp4'),
    #     env_cfg=env_cfg,
    # )
    return cot(time, position, velocity, frc)


def main():
    for env in ['canonical', 'navigait', 'imitation']:
        cot = run_cot(f'icra-policies/{env}/config.yaml')
        print(env, cot)
        

if __name__ == '__main__':
    main()