import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
from utils.setupGPU import run_setup
from pathlib import Path
from learning.startup import read_config, create_environment, get_step_reset
from learning.inference import load_policy
from utils.plotting import RewardPlotter, MujocoPlotter, get_mj_scene_option, save_video, save_trajectories, save_metrics
import jax
import numpy as np
from tqdm import tqdm
from utils.state import MujocoState
from utils.geometry import FREE3D_POS, FREE3D_VEL
from utils.plotting import add_text_to_frame, MujocoPlotter

from eval.curve_animation import animate_beziers
from utils import geometry as geo
from envs.bruce import interface4bar as bruce
# from envs.bruce import interfacedirect as bruce


def rollout_gaits(
    env,
    inference_fn,
    T=10.0,
    info_init_fn=lambda state: state.info,
    info_step_fn=lambda state: state.info,
    animate=False,
    width=None,
    height=None,
    change_v0_fn=None
):
    if width is None:
        width = env.mj_model.vis.global_.offwidth
    if height is None:
        height = env.mj_model.vis.global_.offheight
    # Set up the environment
    rng = jax.random.PRNGKey(1)
    state = env.reset(rng)
    _ = inference_fn(state.obs, rng)
    
    # Initialize the state
    new_info = info_init_fn(state)
    old_info = state.info
    initial_info = old_info | new_info
    state = state.replace(info=initial_info)
    initial_data = state.data
    initial_obs = state.obs

    # Setup reward plotting
    plotter = RewardPlotter(state.metrics)
    data_plotter = MujocoPlotter(plotkey=['sensordata'])
    data_plotter.add_row(state.data)

    # Rollout and record data
    N = int(T / env.dt)
    traj = []
    bezier_coeffs = []
    vdes = np.array([[0.0, 0.0],
                     [0.1, 0.0],
                     [0.1, 0.075],
                     [0.0, 0.075],
                     [-0.1, 0.075],
                     [-0.1, 0.0],
                     [-0.1, -0.075],
                     [0.0, -0.075],
                     [0.1, -0.075]])
    vdes_idx = 0
    vdes_chosen = []
    vel_targets = []
    for i in tqdm(range(N)):
        if animate: vdes_idx = int(i / N * len(vdes))
        state.info['vdes'] = vdes[vdes_idx]
        vdes_chosen.append(state.info['vdes'])
        ctrl, _ = inference_fn(state.obs, rng)
        ctrl = np.array(ctrl)
        state: MujocoState = env.step(state, ctrl)
        
        if animate:
            # set the base position
            state.data.qpos[:FREE3D_POS] = geo.apply_transform(
                np,
                state.info['base_history'][0][:FREE3D_POS],
                state.info['base2global']
            )
            state.data.qvel[:FREE3D_VEL] = state.info['base_history'][0][FREE3D_POS:]

            # Set the joints positions
            state.data.qpos[FREE3D_POS:] = bruce.crank2full(np, state.info['gait_history'][0][:env.nq - FREE3D_POS])
            state.data.qvel[FREE3D_VEL:] = bruce.crank2full(np, state.info['gait_history'][0][env.nv - FREE3D_VEL:])
            state.data.qacc = np.zeros(env.mj_model.nv)
        # bezier_coeffs.append((state.info['gaitlib'].curr_period,
        #                       state.info['gaitlib'].get_phase(state.data.time),
        #                       state.info['gaitlib'].curr_jt[0,:,:],
        #                       vdes_idx,
        #                       state.info['gait_des'][:env._config.ndof]))
        # print(bezier_coeffs[-1][0])
        vel_targets.append(state.info['vel_target'])
        if state.done or i / N * len(vdes) > vdes_idx + 1:
            print('resetting...')
            if not animate:
                print('incrementing')
                vdes_idx += 1
            
            if vdes_idx >= len(vdes):
                break
            env = change_v0_fn(env, vdes[vdes_idx])
            state = env.reset(rng)
            state.info['vdes'] = vdes[vdes_idx]
        
        traj.append(state)
        plotter.add_row(state.metrics, state.reward)
        data_plotter.add_row(state.data)

    # Render the rollout
    render_every = 1
    fps = 1.0 / env.dt / render_every
    print(f"fps: {fps}")
    traj = traj[::render_every]

    scene_option = get_mj_scene_option(contacts=False)

    print('Generating video...')
    frames = env.render(
        trajectory   = traj,
        camera       = 'track',
        height       = height,
        width        = width,
        scene_option = scene_option,
    )
    text_frames = []
    for i, frame in enumerate(frames):
        curr_vdes = vdes_chosen[i]
        target_vel = vel_targets[i]
        text = f'v = [{round(curr_vdes[0], 2)}, {round(curr_vdes[1], 2)}]  m/s'
        text2 = f'v2 = [{round(target_vel[0], 3)}, {round(target_vel[1], 3)}] m/s'
        text_frame = add_text_to_frame(
            pixels    = frame,
            text      = text, 
            org       = (50, 100), 
            size      = 1, 
            thickness = 2,
            color     = (255, 255, 255))
        text_frame = add_text_to_frame(
            pixels    = text_frame,
            text      = text2, 
            org       = (50, 160), 
            size      = 1, 
            thickness = 2,
            color     = (255, 255, 255))
        text_frames.append(text_frame)

    return frames, plotter, traj, bezier_coeffs, data_plotter

def main():
    # Set up the GPU environment
    run_setup()

    # Read the config file from command line argument
    config = read_config()
    if config['backend'] == 'jnp':
        raise Exception('jumpy not working for run gaits yet')

    # Create the environment
    env, env_cfg = create_environment(config, idealistic=False)
    
    # Open policy
    # fake_inference_fn = jax.jit(load_policy(config))
    
    # Load the model    
    fake_inference_fn = lambda obs, rng: (np.zeros(env.action_size, dtype=np.float32), None)

    # Get reset and step functions
    def change_v0_fn(env, vdes):
        env.params.initialization.strategy = 'manual'
        env.params.initialization.vdes = list(vdes)
        
        return env

    # Rollout the policy in the environment
    frames, reward_plotter, traj, bezier_coeffs, data_plotter = rollout_gaits(
        env          = env,
        inference_fn = fake_inference_fn,
        T            = 50.0,
        animate      = True,
        width        = 720,
        height       = 1080,
        change_v0_fn = change_v0_fn
    )
    
    # animate_beziers(
    #     beziers=bezier_coeffs,
    #     vid_length=20.0,
    #     path=Path(f'visualization/{config["env"]}_curve_animation.mp4')
    # )

    data_plotter.save_to_h5('logs/run_gaits.h5')

    # Save the video
    save_video(
        frames,
        path=Path(f'visualization/{config["env"]}_run_gaits.mp4'),
        env_cfg=env_cfg,
    )
    # plotter.save_to_h5('visualization/animation_4bar.h5')

    env_name = config['env']
    # Plot ctrl vs actual
    save_metrics(reward_plotter, path=Path(f'visualization/{env_name}_metrics.png'))
        

if __name__ == '__main__':
    main()