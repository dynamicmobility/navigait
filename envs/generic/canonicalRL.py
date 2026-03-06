# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for Unitree G1."""

from typing import Any, Dict, Optional, Union

import jax
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from envs.generic.bipedal import BipedalBase
from utils import geometry as geo


class CanonicalRL(BipedalBase):
    """Track a joystick command."""

    def __init__(
        self,
        xml_path,
        env_params: config_dict.ConfigDict,
        curriculum_epochs: int = 19,
        backend = 'jnp'
    ):
        # Initialize the parent (bipedal) class
        super().__init__(
            xml_path         = xml_path,
            env_params       = env_params,
            backend          = backend
        )
        # Set the configuration
        self.params = env_params
        self._curriculum_epochs = curriculum_epochs
        self._post_init()


    def _post_init(self) -> None:
        # Note: First joint is freejoint.
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor
    
    def set_random_joint_state(
            self,
            rng: jax.Array,
            arr: jax.Array,
            minval: jax.Array,
            maxval: jax.Array
        ):
        rng, key = self._split(rng)
        val = self._uniform(key, arr.shape[0] - geo.FREE3D_POS, minval=minval, maxval=maxval)
        val = self._np.clip(val, *self._jt_lims)
        arr = self._set_val_fn(arr, val, min_idx=geo.FREE3D_POS, max_idx=None)
        return rng, arr
    
    def check_reset(
        self, state: mjx_env.State
    ) -> mjx_env.State:

        def env_reset(rng):
            s = self.reset(rng)
            info = state.info | s.info
            metrics = state.metrics | s.metrics
            return self._state_init_fn(
                data=s.data,
                obs=s.obs,
                reward=s.reward,
                done=s.done,
                metrics=metrics,
                info=info
            )
            
        new_state = self._cond(
            state.data.time < state.info['last_time'],
            lambda _: env_reset(state.info['rng']),
            lambda _: state,
            None
        )
        
        

        return new_state
    
    def reset(
        self,
        rng: jax.Array,
        qpos: jax.Array,
        qvel: jax.Array,
        cmd: jax.Array,
        torso_id: int,
        floor_id: int,
        ndof: int,
        njoint: int
    ) -> mjx_env.State:
        data = self._data_init_fn(
            time = 0.0,
            qpos = qpos,
            qvel = qvel,
            ctrl = cmd,
            xfrc_applied = self._np.zeros((self.mjx_model.nbody, 6)),
        )
        state = super().reset(rng, data, torso_id=torso_id, floor_id=floor_id, ndof=ndof, njoint=njoint)

        rng, key = self._split(rng)
        phase_dt = 2.0 * self._np.pi * self.dt * self._config.gait_freq
        phase = 0.0

        additional_info = {
            "rng":           rng,
            "last_time":     data.time - self.params.ctrl_dt,
            "step":          0,
            'delta_act':     self._np.zeros(ndof),
            "feet_air_time": self._np.zeros(2),
            "contact":       self._np.ones(2, dtype=bool),
            "last_contact":  self._np.zeros(2, dtype=bool),
            "first_contact": self._np.zeros(2, dtype=bool),
            "swing_peak":    self._np.zeros(2),
            "phase_dt":      phase_dt,
            "phase":         phase,
        }
        info = state.info | additional_info
        # self.randomize_velocity(state)

        metrics = {}
        obs = None
        reward, done = self._np.zeros(2)
        return self._state_init_fn(state.data, obs, reward, done, metrics, info)

    
    def pre_process_action(
        self,
        time: float,
        default_position: jax.Array,
        action: jax.Array,
        info: dict
    ):
        # Position
        des_pos = default_position + action * self._config.action_scale

        # Velocity
        if self.params.tracking == 'position':
            dt = time - info['last_time']
            jt_vdes = (des_pos - info['act_history'][0]) / dt
            info['delta_act'] = info['delta_act'] + dt / self.params.filter_tau * (jt_vdes - info['delta_act']) 
        else:
            info['deta_act'] = 0
        
        self.update_history(info['act_history'], action * self._config.action_scale)
        info['last_time'] = time
        return self._np.hstack([des_pos, info['delta_act']]), info
    
    def update_internal_state(
        self,
        info: dict[str, jax.Array],
        time: jax.Array,
        qpos: jax.Array,
        qvel: jax.Array,
        action: jax.Array,
        foot_pos: jax.Array,
        contact: jax.Array,
        gyro: jax.Array,
        accel: jax.Array,
        qpos_noise,
        qvel_noise,
        attitude_noise
    ):

        contact_filt = contact | info["last_contact"]
        first_contact = (info["feet_air_time"] > 0.0) * contact_filt
        info["feet_air_time"] += self.dt
        p_fz = foot_pos[..., -1]
        info["swing_peak"] = self._np.maximum(info["swing_peak"], p_fz)

        info['contact'] = contact
        info['first_contact'] = first_contact

        phase_tp1 = info["phase"] + info["phase_dt"]
        info["phase"] = self._np.fmod(phase_tp1 + self._np.pi, 2 * self._np.pi) - self._np.pi

        info["feet_air_time"] *= ~contact
        info["last_contact"] = contact
        info["swing_peak"] *= ~contact
        
        # info["jt_pos"] = jt_pos
        # info["jt_vel"] = jt_vel
        info['rng'], noisy_jt = self.noisy(
            rng        = info['rng'],
            value      = qpos[geo.FREE3D_POS:],
            lim        = qpos_noise,
            curr_level = self.get_curriculum_level(info)
        )
        info['rng'], noisy_omega = self.noisy(
            rng        = info['rng'],
            value      = self._np.zeros(3),
            lim        = attitude_noise,
            curr_level = self.get_curriculum_level(info)
        )
        noisy_quat_additional = geo.euler2quat(self._np, noisy_omega)
        noisy_quat = geo.quat_mul(self._np, noisy_quat_additional, qpos[3:geo.FREE3D_POS])
        noisy_qpos = self._np.hstack([
            qpos[:3],
            noisy_quat,
            noisy_jt
        ])
        info['qpos_history']  = self.update_history(info['qpos_history'], noisy_qpos)
        info['qvel_history']  = self.update_history(info['qvel_history'], qvel)
        info['act_history']   = self.update_history(info['act_history'], action)
        info['gyro_history']  = self.update_history(info['gyro_history'], gyro)
        info['accel_history'] = self.update_history(info['accel_history'], accel)
        

        return info

    # Begin shared rewards for canonical RL

    def reward_tracking_linvel(
        self, data: mjx.Data, body_vel: jax.Array, target_linvel: jax.Array, tracking_sigma: jax.Array
    ) -> jax.Array:
        """Reward term for linear velocity tracking. target_linvel should be
        of shape (2,) and describe the desired (vx, vy)"""
        error = self._np.square(self._np.linalg.norm(body_vel[:2] - target_linvel))
        return self._np.exp(-error / tracking_sigma)
            
    def reward_tracking_angvel(
        self, data: mjx.Data, target_angvel: jax.Array, tracking_sigma: jax.Array
    ) -> jax.Array:
        """Reward term for angular velocity tracking. target_angvel should be
        of shape (2,) and describe the desired (vx, vy)"""
        error = self._np.square(self._np.linalg.norm(data.qvel[3:6] - target_angvel))
        return self._np.exp(-error / tracking_sigma)
    
    def reward_upright(
        self, data: mjx.Data, upright_sigma
    ):
        """Rewards the base frame's z-axis for being aligned with the world's
        z-axis"""
        # Get the z vector of the robot's base from base quaternion
        quat = data.qpos[3:7]
        rot_mat = geo.rotmat(self._np, geo.quat2euler(self._np, quat))
        zvec = rot_mat @ self._np.array([0, 0, 1])
        
        # Compute the score
        score = self._np.dot(zvec, self._np.array([0, 0, 1]))
        error = self._np.square(1.0 - score)
        return self._np.exp(-error / upright_sigma)
    
    def reward_base_height(
        self, base_height: jax.Array, base_height_target: jax.Array, base_height_sigma: jax.Array
    ):
        """Reward for base height target"""
        error = self._np.square(base_height - base_height_target)
        return self._np.exp(-error / base_height_sigma)
    

    def get_rz(self, phase, foot_height):
        left = -self._np.cos(2.0 * phase) * foot_height / 2.0 + foot_height / 2.0
        right = 0.0
        left, right = self._np.where(phase > 0.0, self._np.array([right, left]), self._np.array([left, right]))
        return self._np.array([left, right])

    def reward_feet_stepping(
        self, 
        data: mjx.Data, 
        phase: jax.Array, 
        foot_height_des: jax.Array, 
        foot_pos_act: jax.Array,
        feet_stepping_sigma: jax.Array
    ) -> jax.Array:
        """Rewards stepping feet"""
        # Reward for tracking the desired foot height.
        right_foot_pos = foot_pos_act[0, :]
        left_foot_pos  = foot_pos_act[1, :]
        lf_z = left_foot_pos[2]
        rf_z = right_foot_pos[2]
        foot_z = self._np.array([rf_z, lf_z])
        rz = self.get_rz(phase, foot_height_des)
        error = self._np.sum(self._np.square(foot_z - rz.reshape(-1, 1)))
        return self._np.exp(-error / feet_stepping_sigma)
        # return error
    
    def cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
    ) -> jax.Array:
        del info  # Unused.
        body_vel = data.qvel[:2]
        reward = self._np.sum(self._np.linalg.norm(body_vel, axis=-1) * contact)
        return reward

    def cost_feet_clearance(
        self, data: mjx.Data, info: dict[str, Any], feet_pos: jax.Array, feet_vel: jax.Array, goal_height: jax.Array
    ) -> jax.Array:
        del info  # Unused.
        vel_xy = feet_vel[..., :2]
        vel_norm = self._np.sqrt(self._np.linalg.norm(vel_xy, axis=-1))
        foot_z = feet_pos[..., -1]
        delta = self._np.abs(foot_z - goal_height)
        return self._np.sum(delta * vel_norm)

    def cost_feet_height(
        self,
        swing_peak: jax.Array,
        first_contact: jax.Array,
        info: dict[str, Any],
        foot_height: jax.Array
    ) -> jax.Array:
        del info  # Unused.
        error = swing_peak / foot_height - 1.0
        return self._np.sum(self._np.square(error) * first_contact)

    def reward_feet_air_time(
        self,
        air_time: jax.Array,
        first_contact: jax.Array,
        commands: jax.Array,
        threshold_min: float = 0.2,
        threshold_max: float = 0.5,
    ) -> jax.Array:
        # cmd_norm = self._np.linalg.norm(commands)
        # air_time = (air_time - threshold_min) * first_contact
        # air_time = self._np.clip(air_time, max=threshold_max - threshold_min)
        # reward = self._np.sum(air_time)

        # reward *= cmd_norm > 0.02  # No reward for zero commands.
        mag2 = self._np.sum(air_time)**2
        return self._np.tanh(mag2 / 0.1)
    
    def reward_feet_flat(
        self, 
        foot_z: jax.Array,
        feet_flat_sigma: float
    ) -> jax.Array:
        """Rewards stepping feet"""
        # Reward for tracking the desired foot height.
        # foot_z = self._np.clip(foot_z, -1.0, 1.0)  # numerical safety
        # foot_angle = self._np.arccos(foot_z)
        err = self._np.sum(self._np.square(foot_z - (-1.0)))
        return self._np.exp(-err / feet_flat_sigma)

        # return error