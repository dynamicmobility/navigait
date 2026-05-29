import time
import os
import numpy as np
from externals.roboprotobuf.protoutil.control_server import ControlServer
from externals.roboprotobuf.protoutil.robotproto.bruce_pb2 import BruceControllerEvent, BruceEvent
from externals.roboprotobuf.protoutil.robotproto.pack_data import pack_numpy_array
from envs.bruce import interface_westwood as bruce
from envs.bruce.navigait import Bruce
from learning.startup import read_config, create_environment
from learning.inference import load_policy
import utils.geometry as geo
import jax

NDOF = 16
CTRL_RATE = 50
FREE3D_POS = 7
FREE3D_VEL = 6

def copy_proto_data(src: BruceEvent, dst):
    """Copy data from src to dst."""
    dst.qpos = np.array(src.proprioception.qpos.data).copy()
    dst.qvel = np.array(src.proprioception.qvel.data).copy()
    dst.time = np.copy(src.time)
    dst.sensordata[:] = np.hstack((np.array(src.proprioception.contacts.data), np.array(src.gyro.data), np.array(src.accel.data)))
    return dst

def pack_control(time: float, qpos_des, qvel_des) -> BruceControllerEvent:
    out = BruceControllerEvent()
    out.time = time
    pack_numpy_array(out.motorcommand.qpos_des, qpos_des)
    pack_numpy_array(out.motorcommand.qvel_des, qvel_des)
    return out


def main():
    config = read_config()
    ng, _ = create_environment(config, for_training=False, idealistic=True)
    ng: Bruce = ng

    # Start up server for accepting remote state data
    server = ControlServer(
        # host      = "192.168.0.102",
        host      = "127.0.0.1",
        port      = 9001,
        ctrl_msg  = BruceControllerEvent,
        state_msg = BruceEvent,
    )

    server.start_server()
    i = 0
    loop_accuracy_start = time.perf_counter()
    PRINT_FREQ = 100
    state_msg: BruceEvent = server.handle_request()
    start_pos = np.array(state_msg.proprioception.qpos.data).copy()
    inference_fn = load_policy(config)
    jit_inference_fn = jax.jit(inference_fn)
    obs, info = ng.reset_ctrl(
        initial_vdes    = np.zeros(2),
        w_des_init      = np.zeros(1),
        global_hzd_qpos = bruce.ext(np, bruce.pitch2crank, start_pos, geo.FREE3D_POS),
        gyro            = np.array(state_msg.gyro.data),
        accel           = np.array(state_msg.accel.data),
        random_seed     = 95,
        policy          = jit_inference_fn
    )

    while True:
        # Evaluate NaviGait
        crank_pos = bruce.pitch2crank(np, state_msg.proprioception.qpos.data[geo.FREE3D_POS:])
        crank_vel = bruce.pitch2crank(np, state_msg.proprioception.qpos.data[geo.FREE3D_VEL:])
        cmd, info = ng.get_ctrl(
            time=state_msg.time,
            orientation=state_msg.proprioception.qpos.data[3:7],
            crank_pos=crank_pos,
            crank_vel=crank_vel,
            info=info,
            gyro=np.array(state_msg.gyro.data),
            accel=np.array(state_msg.accel.data)
        )

        # Send commands to the robot
        output_msg = pack_control(
            time     = state_msg.time, 
            qpos_des = np.hstack([bruce.crank2pitch(np, cmd[:bruce.NDOF]), np.zeros(6)]),
            qvel_des = np.hstack([bruce.crank2pitch(np, cmd[bruce.NDOF:]), np.zeros(6)]),
        )
        server.send_control(output_msg)

        if i % PRINT_FREQ == 0:
            many_loops_time = time.perf_counter() - loop_accuracy_start
            print(f'LOOP ACCURACY = {many_loops_time - (1 / CTRL_RATE * PRINT_FREQ)} s')
            loop_accuracy_start = time.perf_counter()
        
        # Recieve messages back
        state_msg: BruceEvent = server.handle_request()


if __name__ == '__main__':
    main()