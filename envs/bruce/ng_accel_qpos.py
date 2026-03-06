"""Joystick gait tracking for Atalante."""

from typing import Any, Dict, Optional, Union

import jax
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco import mjx

from envs.bruce import interface4bar as bruce
from envs.bruce.navigait import Bruce as NGBRUCE
from control.bezier import Leg

from utils.geometry import FREE3D_POS, FREE3D_VEL
from utils import geometry as geo

class Bruce(NGBRUCE):
    """
    A class for tracking joystick-controlled gait in a simulated environment.
    """

    def _get_obs(
        self,
        info: dict[str, Any],
        time,
        gyro,
        accel,
    ) -> jax.Array:
        
        references = self._np.hstack([
            info['gait_des'][:bruce.NDOF],
            info['base_des'][:FREE3D_POS],
            info['gaitlib'].swing_leg,
            info['gaitlib'].get_phase(time)
        ])

        output_feedback = self._np.hstack([
            info['act'],
            info['last_act'],
            info['last_last_act'],
            info['vdes_res'],
        ])

        command = info['vdes']

        history = self._np.hstack([
            info['noisy_qpos_history'][:, geo.FREE3D_POS:].flatten(),
            info['gait_history'].flatten(),
            info['base_history'].flatten(),
        ])

        privileged_history = self._np.hstack([
            info['qpos_history'][:, :7].flatten()
        ])

        disturbance = info['curr_pert']

        obs = self._np.hstack([
            accel,
            references,
            output_feedback,
            command,
            history
        ])

        privileged_obs = self._np.hstack([
            obs,
            gyro,
            accel, # REMOVE if time to retrain
            privileged_history,
            disturbance
        ])

        return {
            'state': obs,
            'privileged_state': privileged_obs
        }