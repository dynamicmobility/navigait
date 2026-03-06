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
"""Velocity Tracking Task for BRUCE."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import collision
from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from envs.generic.canonicalRL import CanonicalRL
from envs.bruce import interface4bar as bruce
# from envs.bruce import interfacedirect as bruce
from utils import geometry as geo



class Bruce(CanonicalRL):
    """Track a joystick command."""

    def __init__(
        self,
        env_params: config_dict.ConfigDict,
        curriculum_epochs: int = 19,
        backend = 'jnp',
        idealistic=False,
        animate=False
    ):
        super().__init__(
            xml_path=bruce.PD_XML,
            env_params=env_params,
            curriculum_epochs=curriculum_epochs,
            backend=backend
        )
        if idealistic:
            self.params.domain_randomization.enabled = False
            self.params.domain_randomization.obs_delay.enabled = False
            self.params.noise_scale = 0.0
            self.params.curriculum.enabled = False
            self.params.initialization.strategy = 'manual'
            self.params.initialization.add_random_yaw = False
            self.params.initialization.add_random_jt.enabled = False
            self.params.command.enabled = False

        self._post_init()
        self.nq = geo.FREE3D_POS + bruce.NDOF
        self.nv = geo.FREE3D_VEL + bruce.NDOF

    def _post_init(self) -> None:        
        self._site2id = {}
        for site in bruce.FEET_SITES:
            self._site2id[site] = self._mj_model.site(site).id
            
        # setup sensor indices from MJ model
        self._sensor2idx = {} # TODO: make for all sensors
        for sensor in bruce.FEET_SENSORS:
            sensor_id = self._mj_model.sensor(sensor).id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            self._sensor2idx[sensor] = sensor_adr
    

    def reset(self, rng: jax.Array) -> mjx_env.State:
        ext_crank_qpos = self._np.hstack((bruce.DEFAULT_FF, bruce.DEFAULT_JT))
        # rng, qpos = self.get_random_ff(rng, qpos)

        initialization = self.params.initialization
        if initialization.add_random_yaw:
            rng, ff_key = self._split(rng)
            ext_crank_qpos = self.add_random_ff(ff_key, ext_crank_qpos)
        
        if initialization.add_random_jt.enabled:
            rng, jt_key = self._split(rng)
            ext_full_qpos = self.add_random_joint_state(
                jt_key = jt_key, 
                qpos   = bruce.ext_crank2ext_full(self._np, ext_crank_qpos, geo.FREE3D_POS),
                minval = initialization.add_random_jt.minval,
                maxval = initialization.add_random_jt.maxval
            )
            ext_crank_qpos = bruce.ext_full_2ext_crank(self._np, ext_full_qpos, geo.FREE3D_POS)

        rng, z0_key = self._split(rng)
        z0 = self._uniform(z0_key, minval=initialization.z0[0], maxval=initialization.z0[1])
        ext_crank_qpos = self._set_val_fn(ext_crank_qpos, val=z0, min_idx=2, max_idx=3)

        ext_crank_qvel = self._np.zeros(geo.FREE3D_VEL + bruce.NDOF)
        ctrl = self._np.zeros(self.mjx_model.nu)
        parent_state = super().reset(rng,
            qpos     = bruce.ext_crank2ext_full(self._np, ext_crank_qpos, geo.FREE3D_POS),
            qvel     = bruce.ext_crank2ext_full(self._np, ext_crank_qvel, geo.FREE3D_VEL),
            cmd      = ctrl,
            torso_id = bruce.TORSO_ID,
            floor_id = bruce.GROUND_GEOM_ID,
            ndof     = bruce.NDOF,
            njoint   = bruce.NJOINT
        )
        
        rng, noisy_gyro, noisy_accel, ground_contacts = self.get_sensor_values(
            rng  = parent_state.info['rng'],
            data = parent_state.data,
            curr_level = 1
        )
        
        his_len        = self.params.history_length
        obs_delay      = self.params.domain_randomization.obs_delay
        qpos_history   = self.make_history(ext_crank_qpos, his_len + obs_delay.qpos[1] + 1)
        qvel_history   = self.make_history(ext_crank_qvel, his_len + obs_delay.qpos[1] + 1)
        gyro_history   = self.make_history(noisy_gyro, his_len + obs_delay.gyro[1] + 1)
        accel_history  = self.make_history(noisy_accel, his_len + obs_delay.accel[1] + 1)
        action_history = self.make_history(ctrl[:bruce.NDOF], his_len)

        additional_info = {
            'qpos_history':   qpos_history,
            'noisy_qpos_history':   qpos_history.copy(),
            'qvel_history':   qvel_history,
            'gyro_history':   gyro_history,
            'accel_history':  accel_history,
            'action_history': action_history,
        }
        updated_info = parent_state.info | additional_info

        reward, metrics = self.get_reward_and_metrics(
            data    = parent_state.data,
            info    = updated_info,
            metrics = parent_state.metrics,
            done    = False,
            action  = ctrl
        )

        obs = self._get_obs(
            info     = updated_info,
            contact  = ground_contacts,
        )

        reward, done = self._np.zeros(2)
        return parent_state.replace(
            obs=obs, reward=reward, done=done, metrics=metrics, info=updated_info
        )
        
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = self.check_reset(state)
        state, rand_model = self.handle_env_customization(state, bruce.TORSO_ID)
        
        default_crank_pos = self._np.array(bruce.DEFAULT_JT)
        motor_targets, updated_info = self.pre_process_action(
            time             = state.data.time,
            default_position = default_crank_pos,
            action           = action,
            info             = state.info
        )

        # Step the simulation
        data = self._step_fn(state.data, motor_targets, model=rand_model)
        rng, noisy_gyro, noisy_accel, ground_contacts = self.get_sensor_values(
            updated_info['rng'], 
            data,
            curr_level = self.get_curriculum_level(state.info)
        )
        
        # Randomize training variables
        if self.params.command.enabled:
            updated_info = self.randomize_velocity((data, updated_info.copy()))

        updated_info = self.update_internal_state(
            info            = updated_info,
            time            = data.time,
            qpos            = bruce.ext_full_2ext_crank(self._np, data.qpos, geo.FREE3D_POS),
            qvel            = bruce.ext_full_2ext_crank(self._np, data.qvel, geo.FREE3D_VEL),
            action          = motor_targets[:bruce.NDOF],
            foot_pos        = bruce.get_foot_pos(self._np, self._mj_model, data),
            contact         = ground_contacts,
            gyro            = noisy_gyro,
            accel           = noisy_accel,
            qpos_noise      = self.params.noise_scale * bruce.QPOS_NOISE,
            qvel_noise      = self.params.noise_scale * bruce.QVEL_NOISE,
            attitude_noise  = self.params.noise_scale * bruce.ATTITUDE_NOISE
        )

        obs = self._get_obs(
            info     = updated_info,
            contact  = ground_contacts,
        )

        done = self.get_fall_termination(
            data = data,
            base_height_min = self._config.termination.base_height
        )
        
        reward, metrics = self.get_reward_and_metrics(
            data, updated_info, state.metrics, done, action
        )

        done = done.astype(reward.dtype)
        new_state = self._state_init_fn( data, obs, reward, done, metrics, updated_info)
        return new_state
    
    @property
    def action_size(self):
        return bruce.NDOF

    def get_sensor_values(
        self,
        rng,
        data,
        curr_level
    ):
        rng, noisy_gyro = self.noisy(
            rng,
            value=bruce.get_gyro(self._mj_model, data),
            lim=self.params.noise_scale * bruce.GYRO_NOISE,
            curr_level = curr_level
        )
        rng, noisy_accel = self.noisy(
            rng,
            value=bruce.get_accelerometer(self._mj_model, data),
            lim=self.params.noise_scale * bruce.ACCEL_NOISE,
            curr_level = curr_level
        )

        raw_contacts = bruce.get_raw_contacts(
            self._np,
            self._mj_model,
            data,
            threshold=bruce.CONTACT_THRESHOLD,
        )

        contacts = bruce.get_ground_contact(self._np, raw_contacts)
        return rng, noisy_gyro, noisy_accel, contacts

    def _get_obs(
        self,
        info: dict[str, Any],
        contact,
    ) -> mjx_env.Observation:
        
        info['rng'], key = self._split(info['rng'])
        obs_delay = self.params.domain_randomization.obs_delay
        rand_delays = self._np.round(self._uniform(
            key,
            shape=(3,),
            minval=self._np.array([obs_delay.gyro[0], obs_delay.accel[0], obs_delay.qpos[0]]),
            maxval=self._np.array([obs_delay.gyro[1], obs_delay.accel[1], obs_delay.qpos[1]])
        )).astype(self._np.int32)

        gyro_delay, accel_delay, qpos_delay = self._np.where(
            self.params.domain_randomization.obs_delay.enabled,
            rand_delays,
            self._np.ones(3).astype(self._np.int32)
        )

        his_len = self.params.history_length
        proprioception = self._np.hstack([
            self._splice(info['gyro_history'], (gyro_delay, 0), (his_len, 3)).flatten(),
            self._splice(info['accel_history'], (accel_delay, 0), (his_len, 3)).flatten(),
        ])

        references = self._np.hstack([
            self._np.cos(info['phase']),
            self._np.sin(info['phase'])
        ])

        output_feedback = self._np.hstack([
            info['act_history'].flatten()
        ])

        command = info['vdes']

        history = self._np.hstack([
            self._splice(info['noisy_qpos_history'], (qpos_delay, 3), (his_len, self.nq - 3)).flatten(),
        ])

        privileged_history = self._np.hstack([
            info['qpos_history'][:his_len].flatten(),
        ])

        disturbance = info['curr_pert']

        obs = self._np.hstack([
            proprioception,
            references,
            output_feedback,
            command,
            history
        ])

        privileged_obs = self._np.hstack([
            obs,
            privileged_history,
            disturbance,
            contact
        ])

        return {
            "state": obs,
            "privileged_state": privileged_obs,
        }
        
    def reward_function(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        sigmas = self._config.reward.sigmas
        return {
            'linvel_tracking': self.reward_tracking_linvel(
                data           = data, 
                body_vel       = bruce.get_body_vel(self._np, self.mj_model, data),
                target_linvel  = info['vdes'][:2],
                tracking_sigma = sigmas.linvel_tracking
            ),
            'angvel_tracking': self.reward_euclidean_imitation(
                reference       = info['vdes'][2],
                qpos            = data.qvel[5],
                imitation_sigma = sigmas.angvel_tracking
            ),
            'upright': self.reward_upright(
                data          = data,
                upright_sigma = sigmas.upright,
            ),
            'base_height': self.reward_base_height(
                base_height        = data.qpos[2],
                base_height_target = self._config.base_height_target,
                base_height_sigma  = sigmas.base_height
            ),
            'minimize_energy': self.reward_vector_size(
                v            = data.qfrc_actuator[geo.FREE3D_VEL:],
                size_sigma   = sigmas.minimize_energy
            ),
            'alive': self.reward_alive(),
            'jt_imitation': self.reward_euclidean_imitation(
                qpos            = bruce.full2crank(self._np, data.qpos[geo.FREE3D_POS:]),
                reference       = self._np.array(bruce.DEFAULT_JT),
                imitation_sigma = sigmas.jt_imitation
            ),
            # Feet Rewards
            'feet_stepping': self.reward_feet_stepping(
                data                = data,
                phase               = info['phase'],
                foot_height_des     = self._config.foot_height,
                foot_pos_act        = bruce.get_foot_pos(self._np, self._mj_model, data),
                feet_stepping_sigma = sigmas.feet_stepping
            ),
            "feet_slip": self.cost_feet_slip(
                data, bruce.get_ground_contact(self._np, bruce.get_raw_contacts(self._np, self.mj_model, data, bruce.CONTACT_THRESHOLD)), info,
            ),
            "feet_clearance": self.cost_feet_clearance(
                data    = data,
                info    = info,
                feet_pos=bruce.get_foot_pos(self._np, self.mj_model, data),
                feet_vel=bruce.get_foot_vel(self._np, self.mj_model, data),
                goal_height= self._config.foot_height,
            ),
            "feet_height": self.cost_feet_height(
                swing_peak=info["swing_peak"],
                first_contact=info['first_contact'],
                info=info,
                foot_height=self._config.foot_height
            ),
            "feet_air_time": self.reward_feet_air_time(
                info["feet_air_time"], info['first_contact'], info["vdes"]
            ),
            'action_rate': self.reward_action_rate(
                act           = info['act_history'][0], 
                last_act      = info['act_history'][1], 
                last_last_act = info['act_history'][2]
            ),
            'termination': self._np.array(done).astype(int),
            'periodic_contact': self.periodic_contact(
                phase               = info['phase'],
                foot_contact        = bruce.get_ground_contact(self._np, bruce.get_raw_contacts(self._np, self.mj_model, data, threshold=bruce.CONTACT_THRESHOLD)),
                feet_stepping_sigma = sigmas.feet_stepping
            ),
            'feet_apart': self.feet_apart(
                foot_pos = bruce.get_foot_pos(self._np, self.mj_model, data)
            ),
            'feet_flat': self.reward_feet_flat(
                foot_z=self._np.array([
                    bruce.get_right_z_axis(self._np, self.mj_model, data)[2],
                    bruce.get_left_z_axis(self._np, self.mj_model, data)[2],
                    ]),
                feet_flat_sigma=sigmas.feet_flat
            )
        }
        
    def feet_apart(
        self,
        foot_pos
    ):
        f_xys = foot_pos[:2,:]
        dist = self._np.linalg.norm(f_xys[:,0] - f_xys[:,1])
        error = self._np.square(dist - 0.12)
        return self.exp_reward(error, 0.01)


    def periodic_contact(
        self, 
        phase: jax.Array, 
        foot_contact: jax.Array,
        feet_stepping_sigma: jax.Array
    ):
        """Rewards stepping feet"""
        # Reward for tracking the desired foot height.
        rz = self.get_rz(phase, 1.0)
        rz = self._np.array(rz == 0)
        correct = self._np.sum(foot_contact == rz)
        return correct #self._np.exp(-error / feet_stepping_sigma)