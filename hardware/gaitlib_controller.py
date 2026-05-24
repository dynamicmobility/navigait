import time
import os
import numpy as np
from externals.roboprotobuf.protoutil.control_server import ControlServer
from externals.roboprotobuf.protoutil.robotproto.bruce_pb2 import BruceControllerEvent, BruceEvent
from externals.roboprotobuf.protoutil.robotproto.pack_data import pack_numpy_array
from kinodynamo import CubicSpline
from control.gait import GaitLibrary, Leg, cpu_cond
from scipy.special import factorial
from pathlib import Path
from envs.bruce import interface_westwood as bruce


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
    
    gl = GaitLibrary.from_directory(
        path          = Path('envs/bruce/gaits/BRUCE_GL_4bar_noarms_v1').as_posix(),
        v0            = np.zeros(2),
        num_states    = 10,
        num_degree    = 7,
        gnp           = np,
        fact          = factorial,
        swing_leg     = Leg.LEFT,
        gait_type     = 'P2'
    )
    # cmd = gl.evaluate(0.0, gl.swing_leg)
    # print(repr(gl.ff_evaluate(0.0)))
    # print()
    # print(repr(bruce.crank2pitch(np, cmd[:10])))
    # print(repr(bruce.crank2pitch(np, cmd[10:])))
    # quit()
    
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
    start_pos = np.array(state_msg.proprioception.qpos.data)[FREE3D_POS:].copy()
    leg = Leg.LEFT

    while True:
        time_start = time.perf_counter()

        # Evaluate the gait library
        s = gl.get_phase(state_msg.time)
        if s >= 1.0:
            gl = gl.impact_reset(state_msg.time, cpu_cond)
            s = gl.get_phase(state_msg.time)
        cmd = gl.evaluate(s, gl.swing_leg)

        # Send commands to the robot
        output_msg = pack_control(
            time     = state_msg.time, 
            qpos_des = np.hstack([bruce.crank2pitch(np, cmd[:10]), np.zeros(6)]),
            qvel_des = np.hstack([bruce.crank2pitch(np, cmd[10:]), np.zeros(6)]),
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