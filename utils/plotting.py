import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import mujoco
import h5py
import cv2
from pathlib import Path
from utils import geometry as geo
import mediapy as media

def set_mpl_params():
    # Make fonts LaTeX-like (good for papers)
    mpl.rcParams.update({
        "text.usetex": True,  # Enable LaTeX
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",   # Computer Modern for math
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 18
    })

def get_subplot_grid(n):
    ncols = np.ceil(np.sqrt(n)).astype(int)
    nrows = np.ceil(n / ncols).astype(int)
    return nrows, ncols

def get_mj_scene_option(contacts=True, perts=True, com=True):
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = contacts
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = perts
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_COM] = com
    return scene_option
    
class InfoPlotter:
    
    DEFAULT_PLOTKEY = ['vdes']
    
    def __init__(
        self,
        plotkey=None
    ):
        self.plotkey = plotkey if plotkey is not None else self.DEFAULT_PLOTKEY
        self.data = {}
        for key in self.plotkey:
            self.data[key] = []
        
        self.data['time'] = []
        
    def add_row(self, time, info):
        for key in self.plotkey:
            self.data[key].append(info[key].copy())
            
        self.data['time'].append(time)
    
    def to_numpy(self):
        for key in self.data:
            self.data[key] = np.array(self.data[key])
                
    def save_to_h5(self, filename):
        with h5py.File(filename, 'w') as f:
            for key in self.data:
                f.create_dataset(key, data=np.array(self.data[key]))
        

class MujocoPlotter:

    DEFAULT_PLOTKEY = ['qpos', 'qvel', 'ctrl', 'sensordata', 'qfrc_actuator']

    def __init__(
        self,
        plotkey=None,
        record_time=True
    ):
        if plotkey is None:
            plotkey = MujocoPlotter.DEFAULT_PLOTKEY
        self.plotkey = plotkey
        self.record_time = record_time
        
        self.data = {}
        for key in self.plotkey:
            self.data[key] = []
        
        if record_time:
            self.data['time'] = []

    def add_row(self, data):
        if self.record_time:
            self.data['time'].append(getattr(data, 'time'))
        
        for key in self.plotkey:
            self.data[key].append(getattr(data, key).copy())

    def to_numpy(self):
        for key in self.plotkey:
            self.data[key] = np.array(self.data[key])

    def save_to_h5(self, filename):
        with h5py.File(filename, 'w') as f:
            for key in self.data:
                f.create_dataset(key, data=np.array(self.data[key]))

    @classmethod
    def time_idx(cls, time, data):
        if 'time' not in data:
            raise ValueError("Time data is required for indexing.")
        
        time_array = np.array(data['time'])
        idx = np.searchsorted(time_array, time)
        if idx == 0 or idx == len(time_array):
            raise ValueError("Time index out of bounds.")
        return idx - 1

def add_text_to_frame(pixels, text, org, size=1, thickness=2, color=(255, 255, 255)):
    # Text settings
    # org = (50, 100) 
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = size

    # Add text to the image
    cv2.putText(
        pixels, text, org, font, fontScale, color, thickness, cv2.LINE_AA
    )
    return pixels

def get_subplot_grid(n):
    ncols = np.ceil(np.sqrt(n)).astype(int)
    nrows = np.ceil(n / ncols).astype(int)
    return nrows, ncols

class RewardPlotter:

    def __init__(self, metrics):
        self.axkey = {}
        self.axdata = {}
        self.rewards = []
        for i, metric in enumerate(metrics):
            self.axkey[metric] = i
            self.axdata[metric] = []

    def add_row(self, metrics, reward):
        self.rewards.append(reward)
        for metric in metrics:
            self.axdata[metric].append(metrics[metric])

    def plot(self):
        ncols, nrows = get_subplot_grid(len(self.axdata) + 1)
        fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10))
        axs = axs.flatten()
        fig.suptitle('Metrics')

        for metric in self.axdata:
            axs[self.axkey[metric]].set_title(metric)
            axs[self.axkey[metric]].plot(self.axdata[metric])

        axs[len(self.axdata)].set_title('Total Reward')
        axs[len(self.axdata)].plot(self.rewards)
        return fig, axs
    

def save_metrics(plotter, path=Path('visualization/metrics.png')):
    print(f'Saving metrics to {path}...')
    fig, axs = plotter.plot()
    fig.tight_layout()
    ans = ensure_dir_exists(path)
    if ans:
        plt.savefig(path, dpi=200)
    plt.close(fig)

def save_trajectories(
    mj_model,
    nq,
    plotter: MujocoPlotter,
    pos_path: Path = None,
    vel_path: Path = None,
    torque_path: Path = None,
    sensor_path: Path = None
):
    njoints = mj_model.nq - geo.FREE3D_POS
    nmotors = int(mj_model.nu / 2)

    def get_actuator_to_jt_idx(mj_model):
        """
        Returns a list where each entry corresponds to an actuator,
        and the value is the index in qpos (0..nq-1) of the joint it controls.
        If the actuator is not directly connected to a joint, returns None.
        """
        result = []
        for i in range(mj_model.nu):
            jntid = mj_model.actuator_trnid[i, 0]
            if jntid == -1:
                result.append(None)  # actuator not attached to a joint
            else:
                qpos_adr = mj_model.jnt_qposadr[jntid]
                result.append(qpos_adr - geo.FREE3D_POS)
        return np.array(result)
    
    all_ctrl2joint = get_actuator_to_jt_idx(mj_model)

    def plot_arr(
        axs: list[plt.Axes],
        time: np.ndarray,
        ctrl: np.ndarray,
        act: np.ndarray,
        names: list[str],
        ctrl2joint: np.ndarray
    ):
        for idx in range(len(names)):
            name = names[idx]
            ctrl_idx = None
            if idx in ctrl2joint:
                ctrl_idx = np.arange(nmotors)[ctrl2joint == idx]

            if act is not None: axs[idx].plot(time, act[:, idx], c='b', label='actual')
            if ctrl is not None and ctrl_idx is not None: axs[idx].plot(time, ctrl[:, ctrl_idx], c='r', ls='--', label='cmd')
            axs[idx].set_title(name)

        return axs
    
    def get_joint_names(mj_model):
        """
        Returns a list of joint names for all joints in the model.
        """
        names = []
        for jntid in range(mj_model.njnt):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jntid)
            names.append(name)
        return names[1:]
    
    def get_actuator_names(mj_model):
        """
        Returns a list of actuator names for all actuators in the model.
        """
        names = []
        for actid in range(mj_model.nu):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actid)
            names.append(name)
        return names[1:]
    
    def get_sensor_names_and_dims(mj_model):
        """
        Returns a list of (sensor_name, sensor_dim) for all sensors in the model.
        """
        sensors = []
        for sensid in range(mj_model.nsensor):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensid)
            dim = mj_model.sensor_dim[sensid]
            labels = ['val']
            if dim == 3:
                labels = ['x', 'y', 'z']
            for label in labels:
                sensor_name = name + '_' + label
                sensors.append(sensor_name)
        
        return sensors

    

    plotter.to_numpy()
    nrows, ncols = get_subplot_grid(n=njoints)
    joint_names = get_joint_names(mj_model)
    
    DPI = 400
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    axs = axs.flatten()

    if pos_path is not None:
        axs = plot_arr(
            axs   = axs,
            time  = plotter.data['time'],
            ctrl  = plotter.data['ctrl'],
            act   = plotter.data['qpos'][:, geo.FREE3D_POS:],
            names = joint_names,
            ctrl2joint=all_ctrl2joint[:nmotors]
        )
        axs[0].legend()
        
        fig.set_size_inches((nrows * 4, ncols * 2))
        fig.tight_layout()
        fig.suptitle('Positions')
        fig.savefig(pos_path, dpi=DPI)
        # plt.show()
        # plt.close(fig)

        print(f'Saved position trajectories to {pos_path}')
        
    if vel_path is not None:
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
        axs = axs.flatten()

        axs = plot_arr(
            axs   = axs,
            time  = plotter.data['time'],
            ctrl  = plotter.data['ctrl'][:, nmotors:],
            act   = plotter.data['qvel'][:, geo.FREE3D_VEL:],
            names = joint_names,
            ctrl2joint=all_ctrl2joint[nmotors:]
        )
        axs[0].legend()

        fig.set_size_inches((nrows * 4, ncols * 2))
        fig.tight_layout()
        fig.savefig(vel_path, dpi=DPI)
        fig.suptitle('Velocities')
        # plt.show()
        # plt.close(fig)

        print(f'Saved velocity trajectories to {vel_path}')

    if torque_path is not None:
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
        axs = axs.flatten()
        actuator_names = get_actuator_names(mj_model)

        axs = plot_arr(
            axs   = axs,
            time  = plotter.data['time'],
            act  = plotter.data['qfrc_actuator'][:,geo.FREE3D_VEL:],
            ctrl   = None,
            names = joint_names[:njoints],
            ctrl2joint = []
        )
        axs[0].legend()
        fig.suptitle('Torques')

        fig.set_size_inches((nrows * 4, ncols * 2))
        fig.tight_layout()
        fig.savefig(torque_path, dpi=DPI)
        fig.suptitle('Torques')
        # plt.show()
        # plt.close(fig)

        print(f'Saved torque trajectories to {torque_path}')

    if sensor_path is not None:
        sensors = get_sensor_names_and_dims(mj_model)
        print(len(sensors))
        nrows, ncols = get_subplot_grid(len(sensors))
        fig, axs = plt.subplots(nrows, ncols)
        axs = axs.flatten()

        axs = plot_arr(
            axs = axs,
            time = plotter.data['time'],
            act = plotter.data['sensordata'],
            ctrl = None,
            names = sensors,
            ctrl2joint = []
        )

        fig.set_size_inches((nrows * 4, ncols * 2))
        fig.tight_layout()
        fig.suptitle('Sensors')
        fig.savefig(sensor_path, dpi=DPI)
        # plt.show()
        # plt.close(fig)

        print(f'Saved sensor trajectories to {sensor_path}')


def ensure_dir_exists(path):
    ans = 'y'
    if not path.parent.exists():
        ans = input(f"Directory {path.parent} does not exist. Create it? [y/n]: ")
        if ans.lower() in ['y', 'yes']:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            print("Skipping save.")
            return False
    return True

def save_video(frames, env_cfg, path=Path('visualization/policy_rollout.mp4')):
    print(f'Saving video to {path}')
    ans = ensure_dir_exists(path)
    if ans:
        media.write_video(path, frames, fps=round(1 / env_cfg.ctrl_dt))

def load_dict_from_hdf5(filename):
    out = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            out[key] = np.array(f[key][:])
    return out