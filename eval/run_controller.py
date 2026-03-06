import os
# os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
import jax
from learning.startup import read_config, create_environment
from learning.inference import load_policy
from envs.atalante.navigait import Exo
from envs.bruce.navigait import Bruce
from envs.bruce import interface4bar as bruce
import mujoco as mj
import mujoco.viewer
import time
import os
from pathlib import Path
import numpy as np
from pynput import keyboard
from utils.plotting import MujocoPlotter, save_trajectories
import utils.geometry as geo
import matplotlib.pyplot as plt
import pygame


HZ = 50

def init_pygame():
    pygame.init()
    pygame.joystick.init()
    while not pygame.joystick.get_count():
        print("This program only works with at least one joystick plugged in. No joysticks were detected.")
        time.sleep(1.0)
    id = 0
    joy = pygame.joystick.Joystick(id)
    name = joy.get_name()
    print("Found Joystick: ", name)
    joy.init()
    return joy

def get_vel_command(joy):
    vx, vy, wz = 0,0,0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # for event in pygame.event.get():
    #     if event.type == pygame.JOYAXISMOTION:
    #         if(event.axis == 3):
    #             print('here')
    #             vx = -event.value
    #         elif(event.axis == 2):
    #             vy = -event.value
    #         elif(event.axis == 1):
    #             wz = -event.value
    vx = -0.2 * joy.get_axis(1)  # Left stick X
    vy = -0.1 * joy.get_axis(0)  # Left stick Y
    wz = 0.4 * joy.get_axis(3)  # Right stick X
            
    print(f'{vx=}, {vy=}, {wz=}')
    return np.array((vx, vy, wz))


if __name__ == '__main__':
    # open and jit compile model
    config = read_config()
    env, env_cfg = create_environment(config, idealistic=True)
    env: Bruce = env
    inference_fn = load_policy(config)
    inference_fn = jax.jit(inference_fn)
    # inference_fn = make_inference_fn(params, deterministic=True)
    model = mj.MjModel.from_xml_path(bruce.PD_XML.as_posix())
    data = mj.MjData(model)
    plotter = MujocoPlotter()

    qpos_init, qvel_init = env.get_gait_qpos_init(np.zeros(2), 95)
    # print(repr(bruce.crank2pitch(np, qpos_init[geo.FREE3D_POS:])))
    qpos_init = np.hstack([bruce.DEFAULT_FF, bruce.DEFAULT_JT])
    # jt_qpos_init = [-0.008242641896210911, 0.4692675710437233, 0.018224926402206444, -0.9471478786563116, 0.47781331523852566, 0.008242641896210911, 0.4692675710437233, -0.018224926402206444, -0.9471478786563116, 0.47781331523852566, -0.7, 1.3, 2.0, 0.7, -1.3, -2.0]
    # ff_qpos_init = bruce.DEFAULT_FF
    # ff_qpos_init[2] = 0.45
    # qpos_init = np.hstack([ff_qpos_init, jt_qpos_init])
    # qpos_init = np.array([0.0, 0.0, 0.4728, 0.7071068, 0, 0, 0.7071068,
    #     0.0, 0.0, 0.0, 0.0,
    #     0.0, 0.0, 0.0, 0.0,
    #     0.0, 0.0, 0.0, 0.0,
    #     0.0, 0.0, 0.0, 0.0])

    data.qpos = bruce.ext_crank2ext_full(np, qpos_init, geo.FREE3D_POS)
    # data.qvel = bruce.ext_crank2ext_full(np, qvel_init, geo.FREE3D_VEL)
    print(repr(bruce.ext_crank2ext_pitch(np, qpos_init, 7)))
    # quit()

    # data.qvel = bruce.hzd_pos_to_mj_qpos(np, qvel_init, geo.FREE3D_VEL)
    # data.qpos = qpos_init

    rng, noisy_gyro, noisy_accel, raw_contacts = env.get_sensor_values(
        rng=None,
        data=data,
        curr_level=0.0
    )
    obs, info = env.reset_ctrl(
        initial_vdes = np.zeros(2),
        global_hzd_qpos = qpos_init, # none if determined by the gait
        random_seed  = 95,
        gyro = noisy_gyro,
        accel = noisy_accel
    )
    for i in range(10):
        motor_targets, info = env.get_ctrl(
                time          = i / 10,
                ext_crank_pos = bruce.ext_full_2ext_crank(np, data.qpos, geo.FREE3D_POS),
                ext_crank_vel = bruce.ext_full_2ext_crank(np, data.qvel, geo.FREE3D_VEL),
                info          = info,
                gyro          = noisy_gyro,
                accel         = noisy_accel,
                policy        = inference_fn
            )
        
    obs, info = env.reset_ctrl(
        initial_vdes = np.zeros(2),
        global_hzd_qpos = qpos_init, # none if determined by the gait
        random_seed  = 95,
        gyro = noisy_gyro,
        accel = noisy_accel
    )
    # model = env.mj_model

    # Set up keyboard listener for velocity control
    joy = init_pygame()

    print('Model params')
    print('friction', model.geom_friction)
    print('dof_frictionloss', model.dof_frictionloss)
    print('dof_armature', model.dof_armature)
    print('damping', model.dof_damping)
    print('masses', model.body_mass)
    # print('gains', model.actuator_gainprm)
    GAIN_FACTOR = 1.0
    model.actuator_gainprm = GAIN_FACTOR * model.actuator_gainprm
    model.actuator_biasprm = GAIN_FACTOR * model.actuator_biasprm

    # Run the simulation with GUI
    step = 0
    timestep_noise = 0.0
    avg_inference_time = 0.0
    step_time = 0
    print_iters = 100
    sim_start = data.time

    T = np.inf #10.0
    times = []
    plotter.add_row(data)

    old_targets = data.ctrl
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer launched. Running simulation...")
        while viewer.is_running() and data.time - sim_start < T:
            step_start = time.time()


            if data.time == 0 or env.get_fall_termination(data, 0.3):
                obs, info = env.reset_ctrl(
                    initial_vdes = np.zeros(2),
                    global_hzd_qpos = qpos_init, # none if determined by the gait
                    random_seed  = 95,
                    gyro = noisy_gyro,
                    accel = noisy_accel
                )

            rng, noisy_gyro, noisy_accel, raw_contacts = env.get_sensor_values(
                rng=None,
                data=data,
                curr_level=0.0
            )

            inference_start = time.time()
            motor_targets, info = env.get_ctrl(
                time          = data.time,
                ext_crank_pos = bruce.ext_full_2ext_crank(np, data.qpos, geo.FREE3D_POS),
                ext_crank_vel = bruce.ext_full_2ext_crank(np, data.qvel, geo.FREE3D_VEL),
                info          = info,
                gyro          = noisy_gyro,
                accel         = noisy_accel,
                policy        = inference_fn
            )
            motor_targets = np.array(motor_targets)
            motor_targets_4bar = motor_targets.copy()
            this_inference_time = time.time() - inference_start
            avg_inference_time += this_inference_time

            substeps = round((env._config.ctrl_dt) / model.opt.timestep)
            delayed = 0 #np.random.randint(low=0, high=15)
            for i in range(delayed):
                data.ctrl[:] = old_targets.copy()
                mj.mj_step(model, data)
                if T != np.inf: plotter.add_row(data)

            old_targets = motor_targets_4bar.copy()
            
            for i in range(substeps - delayed):
                data.ctrl[:] = old_targets
                mj.mj_step(model, data)
                if T != np.inf: plotter.add_row(data)

            # print(bruce.get_accelerometer(env.mj_model, data))
            info['vdes'] = get_vel_command(joy)
            # info['vdes'] = np.array([0.0, 0.0, 2.5 * np.sin(0.3 * data.time)])
            # info['vdes'] = np.array([0.2 * np.sin(0.3 * data.time), 0.0, 0.0])
            # info['vdes'] = np.array([0.0, 0.1 * np.sin(0.3 * data.time), 0.0])
            # Adjust to real-time simulation
            viewer.sync()
            step += 1
            this_step_time = time.time() - step_start
            times.append(this_step_time)
            step_time += this_step_time
            if step % print_iters == 0:
                print(f'== Iter #{step} ===')
                print('Avg step time', step_time / print_iters)
                print('Avg inference time', avg_inference_time / print_iters)
                print()
                avg_inference_time = 0
                step_time = 0
            if env._config.ctrl_dt - this_step_time > 0:
                time.sleep(env._config.ctrl_dt - this_step_time)
                pass
    

    if T != np.inf:
        fig, ax = plt.subplots()
        iters = np.arange(len(times))
        ax.plot(iters, times, lw=1, c='b')
        ax.scatter(iters, times, s=10, c='r')
        plt.show()
        # plotter.save_to_h5("visualization/controller_NaviGait.h5")

        save_trajectories(
            env.mj_model,
            env.nq,
            plotter,
            pos_path=Path(f'visualization/controller_{config['env']}_position.png'),
            vel_path=Path(f'visualization/controller_{config['env']}_velocity.png'),
            torque_path=Path(f'visualization/controller_{config['env']}_torque.png'),
            sensor_path=Path(f'visualization/controller_{config['env']}_sensor.png')
        )
