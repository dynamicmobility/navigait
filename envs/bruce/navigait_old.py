"""Joystick gait tracking for Atalante."""

from typing import Any, Dict, Optional, Union

import jax
from ml_collections import config_dict
from mujoco import mjx
import numpy as np
np.set_printoptions(linewidth=300)

from mujoco_playground._src import mjx_env
from mujoco import mjx

from envs.bruce import interface4bar as bruce
from envs.generic.navigait import NaviGait
from control.bezier import Leg

from utils import geometry as geo

class Bruce(NaviGait):
    """
    A class for tracking joystick-controlled gait in a simulated environment.
    """

    def __init__(
        self,
        gaitlib_path: str,
        env_params: config_dict.ConfigDict,
        gait_type='P2',
        backend='jnp'
    ):  
        # Initialize the parent class
        super().__init__(
            xml_path          = bruce.OLD_PD_XML,
            env_params        = env_params,
            num_states        = bruce.NDOF,
            backend           = backend,
            gait_type         = gait_type,
            gaitlib_path      = gaitlib_path,
        )
        self.nq = geo.FREE3D_POS + bruce.NDOF
        self.nv = geo.FREE3D_VEL + bruce.NDOF
        jt_lb_hzd = self._mj_model.jnt_range[1:, 0].T
        jt_ub_hzd = self._mj_model.jnt_range[1:, 1].T
        self._jt_lims = self._np.vstack((jt_lb_hzd[np.newaxis, :], jt_ub_hzd[np.newaxis, :]))
    
    def reset(
        self,
        rng: jax.Array,
        num_resets: int = 0
    ):
        
        rng, gaitlib, hzd_qpos, hzd_qvel = self.initialization_randomization(
            rng, 
            self._np.hstack([bruce.DEFAULT_FF, bruce.DEFAULT_POSE]),
            bruce.NDOF
        )
        
        rng, ff_key, jt_key = self._split(rng, 3)
        hzd_qpos = self._np.where(
            self.params.initialization.add_random_yaw,
            self.add_random_ff(ff_key, hzd_qpos),
            hzd_qpos
        )
        
        hzd_qpos = self._np.where(
            self.params.initialization.add_random_jt.enabled,
            self.add_random_joint_state(
                jt_key, 
                hzd_qpos,
                minval=self.params.initialization.add_random_jt.minval,
                maxval=self.params.initialization.add_random_jt.maxval
            ),
            hzd_qpos
        )
        
        hzd_motor_targets = self._np.hstack((hzd_qpos[geo.FREE3D_POS:], hzd_qvel[geo.FREE3D_VEL:]))
        motor_targets_linkage = self._np.hstack((
            hzd_motor_targets[0:bruce.NDOF],
            hzd_motor_targets[bruce.NDOF:]
        ))

        parent_state = super().reset(
            rng      = rng, 
            gaitlib  = gaitlib,
            qpos     = hzd_qpos,
            qvel     = hzd_qvel,
            ctrl     = motor_targets_linkage,
            ndof     = bruce.NDOF,
            torso_id = bruce.TORSO_ID,
            floor_id = bruce.GROUND_GEOM_ID,
            num_resets = num_resets,
        )
        parent_state.info['rng'], noisy_gyro, noisy_accel, raw_contacts = self.get_sensor_values(
            parent_state.info['rng'],
            parent_state.data,
            curr_level=self.get_curriculum_level(parent_state.info)
        )
       
        his_len = self.params.history_length
        hzd_relative_base = geo.apply_transform(self._np, hzd_qpos, parent_state.info['base_transform'])
        hzd_relative_qpos = self._np.hstack([hzd_relative_base, hzd_qpos[geo.FREE3D_POS:]])
        qpos_history = self._np.repeat(hzd_relative_qpos[self._np.newaxis, :], his_len, axis=0)
        noisy_qpos_history = qpos_history.copy()
        qvel_history = self._np.repeat(hzd_qvel[self._np.newaxis, :], his_len, axis=0)
        noisy_qvel_history = qvel_history.copy()
        
        additional_info = {
            'gyro_history'      : self._np.repeat(noisy_gyro[self._np.newaxis, :], his_len, axis=0),
            'accel_history'     : self._np.repeat(noisy_accel[self._np.newaxis, :], his_len, axis=0),
            'qpos_history'      : qpos_history,
            'noisy_qpos_history': noisy_qpos_history,
            'qvel_history'      : qvel_history,
            'noisy_qvel_history': noisy_qvel_history
        }
        additional_info['transform_init'] = self._np.copy(parent_state.info['base_transform'])
        updated_info = parent_state.info | additional_info
        
        obs = self._get_obs(
            time        = parent_state.data.time,
            info        = updated_info,
            ndof        = bruce.NDOF
        )
        return parent_state.replace(obs=obs, info=updated_info)

    def get_sensor_values(
        self,
        rng,
        data,
        curr_level
    ):
        rng, noisy_gyro = self.noisy(
            rng,
            value      = bruce.get_gyro(self._mj_model, data),
            lim        = self.params.noise_scale * bruce.GYRO_NOISE,
            curr_level = curr_level
        )
        rng, noisy_accel = self.noisy(
            rng,
            value      = bruce.get_accelerometer(self._mj_model, data),
            lim        = self.params.noise_scale * bruce.ACCEL_NOISE,
            curr_level = curr_level
        )

        raw_contacts = bruce.get_raw_contacts(
            self._np,
            self._mj_model,
            data,
            threshold=bruce.CONTACT_THRESHOLD
        )
        return rng, noisy_gyro, noisy_accel, raw_contacts

    def get_gait_qpos_init(
        self,
        vdes,
        random_seed
    ):
        rng = jax.random.PRNGKey(random_seed)
        gaitlib = self.set_gaitlib(
            rng  = rng,
            vdes = vdes
        )
        
        ff_state = gaitlib.ff_evaluate(0)[:geo.FREE3D_POS]
        jt_state = gaitlib(0)[:bruce.NDOF]
        ff_vel = gaitlib.ff_evaluate(0)[geo.FREE3D_POS:]
        jt_vel = gaitlib(0)[bruce.NDOF:]
        return self._np.hstack((ff_state, jt_state)), self._np.hstack((ff_vel, jt_vel))
    
    def reset_ctrl(
        self,
        initial_vdes: np.ndarray,
        initial_qpos: np.ndarray,
        random_seed: int
    ):
        rng = jax.random.PRNGKey(random_seed)
        gaitlib = self.set_gaitlib(
            rng  = rng,
            vdes = initial_vdes
        )
        
        ff_state = gaitlib.ff_evaluate(0)[:geo.FREE3D_POS]
        jt_state = gaitlib(0)[:bruce.NDOF]
        desired_qpos = self._np.hstack((ff_state, jt_state))
        initial_qvel = self._np.zeros(self.mjx_model.nv)
        
        initial_ctrl = gaitlib(0)

        if self._np.linalg.norm(desired_qpos - initial_qpos) < 0.1:
            # TODO: Change to warning if this is not always an error
            print('WARNING: gait initial qpos is different from actual')
        
        state = super().reset(
            rng        = rng, 
            gaitlib    = gaitlib,
            qpos       = initial_qpos,
            qvel       = initial_qvel,
            ctrl       = initial_ctrl,
            ndof       = bruce.NDOF,
            njoint     = bruce.NJOINT,
            torso_id   = bruce.TORSO_ID,
            floor_id   = bruce.GROUND_GEOM_ID,
            num_resets = 0
        )

        initial_obs = self._get_obs(
            time         = state.data.time,
            info         = state.info,
            ndof         = bruce.NDOF
        )

        initial_info = state.info

        return initial_obs, initial_info        

    def step(
        self, state: mjx_env.State, res_action: jax.Array
    ) -> mjx_env.State:
        # Check if the environment has been reset
        state = self.check_reset(state)
        state, rand_model = self.handle_env_customization(state, bruce.TORSO_ID)

        # Pre-process the action to get the motor targets
        res_jt = res_action[:bruce.NDOF]
        res_vel = res_action[bruce.NDOF:]
        motor_targets, info = self.pre_process_action(
            time       = state.data.time,
            qpos       = state.data.qpos,
            info       = state.info,
            res_joints = res_jt,
            res_vel    = res_vel,
            ndof       = bruce.NDOF
        )
        motor_targets_linkage = self._np.hstack((
            motor_targets[0:bruce.NDOF],
            motor_targets[bruce.NDOF:]
        ))
        # Step the simulation
        # data = self._step_fn(state.data, motor_targets_linkage, model=rand_model)
        data = self._step_fn(state.data, motor_targets_linkage)
        
        # UNCOMMENT FOR ANIMATION INSTEAD OF PHYSICS
        # base_qpos = geo.apply_transform(
        #     self._np,
        #     state.info['base_history'][0][:geo.FREE3D_POS],
        #     state.info['base_transform']
        # )
        # my_qpos = state.info['gait_history'][0][:bruce.NDOF]
        # data.qpos = self._np.hstack([base_qpos, bruce.hzd_jt_to_mj_jtpos(self._np, my_qpos)])
        # data.qvel = self._np.zeros(self._mj_model.nv)
        # data.qvel[:geo.FREE3D_VEL] = state.info['base_history'][0][geo.FREE3D_POS:]
        
        # UNCOMMENT TO HOIST ROBOT IN AIR
        # data.qpos[2] = 1.0
        # data.qvel[:geo.FREE3D_VEL] = 0.0
        # data.qacc[:geo.FREE3D_VEL] = 0.0

        # Compute and update contact information
        state.info['rng'], noisy_gyro, noisy_accel, raw_contacts = self.get_sensor_values(
            state.info['rng'],
            data,
            curr_level = self.get_curriculum_level(state.info)
        )

        # Update the internal state of the environment
        updated_info = self.update_internal_state(
            time                  = data.time,
            qpos                  = data.qpos,
            qvel                  = data.qvel,
            info                  = info,
            res_joints            = res_jt,
            right_ground_contact  = False,
            left_ground_contact   = False,
            gyro                  = noisy_gyro,
            accel                 = noisy_accel,
            ndof                  = bruce.NDOF,
            qpos_noise            = bruce.QPOS_NOISE,
            qvel_noise            = bruce.QVEL_NOISE
        )

        # Randomize training variables
        updated_info = self._cond(
            self.params.command.enabled,
            self.randomize_velocity,
            lambda data_info: data_info[1],
            (data, updated_info.copy())
        )
        # print(state.info['vdes'])

        # Compute the observation
        obs = self._get_obs(
            info         = updated_info,
            time         = data.time,
            ndof         = bruce.NDOF
        )
        
        # Check if done (fallen over)
        done = self.get_fall_termination(
            data            = data, 
            base_height_min = bruce.MIN_BASE_HEIGHT,  
        )
        
        # Compute rewards
        reward, metrics = self.get_reward_and_metrics(
            data, updated_info, state.metrics, state.done, res_action
        )
        
        # reward = self._np.where(
        #     done,
        #     reward - 1000,
        #     reward
        # )

        done = done.astype(self._np.float32)
        new_state = self._state_init_fn(data, obs, reward, done, metrics, updated_info)
        return new_state
    
    @property
    def action_size(self) -> int:
        return bruce.NDOF + 2
    
    @property
    def observation_size(self):
        rng = jax.random.PRNGKey(0)
        state = self.reset(rng)
        return len(state.obs['state'])
    
    def get_ctrl(
        self,
        time: float,
        qpos: np.ndarray,
        qvel: np.ndarray,
        info: dict[str | np.ndarray],
        gyro: np.ndarray,
        accel: np.ndarray,
        policy
    ) -> jax.Array:
        """Returns the control signal for the environment."""
        
        # Update the internal state of the environment        
        updated_info: dict = self.update_internal_state(
            time                 = time,
            qpos                 = qpos,
            qvel                 = qvel,
            info                 = info,
            res_joints           = info['act_history'][0, :bruce.NDOF],
            left_ground_contact  = False,
            right_ground_contact = False,
            gyro                 = gyro,
            accel                = accel,
            ndof                 = bruce.NDOF,
            qpos_noise           = 0.0,
            qvel_noise           = 0.0
        )

        # Compute the observation
        obs = self._get_obs(
            info         = updated_info,
            time         = time,
            ndof         = bruce.NDOF
        )

        # Run an inference and preprocess the outputs
        res, _      = policy(obs, None)
        res_joints  = self._np.array(res[:bruce.NDOF])
        res_vel     = self._np.array(res[bruce.NDOF:])
        # res_joints = self._np.zeros(bruce.NDOF)
        # res_vel = self._np.zeros(2)
        
        motor_targets, updated_info = self.pre_process_action(
            time       = time,
            qpos       = qpos,
            info       = updated_info,
            res_joints = res_joints,
            res_vel    = res_vel,
            ndof       = bruce.NDOF
        )

        return motor_targets, updated_info.copy()

    def reward_function(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        done: jax.Array,
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        sigmas = self.params.reward.sigmas
        base_des_old = info['base_history'][1]
        gait_des_old = info['gait_history'][1]
        global_des = geo.apply_transform(
            self._np, 
            base_des_old, 
            info['old_transform'])
        
        local_act = geo.apply_transform(
            self._np,
            data.qpos[:geo.FREE3D_POS],
            geo.inv_transform(self._np,
                              info['old_transform']
            )
        )
        global_init_qpos = geo.apply_transform(
            self._np,
            base_des_old,
            info['transform_init']
        )
        rewards = {
            'gait_tracking' : self.reward_euclidean_imitation(
                qpos            = data.qpos[geo.FREE3D_POS:],
                reference       = gait_des_old[:bruce.NDOF],
                imitation_sigma = sigmas.gait_tracking
            ),
            'base_xyz_tracking': self.reward_euclidean_imitation(
                qpos            = local_act[:3],
                reference       = base_des_old[:3],
                imitation_sigma = sigmas.base_xyz_tracking
            ),
            'base_quat_tracking': self.exp_reward(
                mag2  = geo.quat_dist(
                            _np = self._np,
                            q1  = data.qpos[3:geo.FREE3D_POS],
                            q2  = global_des[3:geo.FREE3D_POS]
                )**2,
                sigma = sigmas.base_quat_tracking
            ),
            'base_vel_tracking': self.reward_euclidean_imitation(
                qpos            = data.qvel[:2],
                reference       = base_des_old[geo.FREE3D_POS:geo.FREE3D_POS + 2],
                imitation_sigma = sigmas.base_vel_tracking
            ),
            'minimize_energy': self.reward_vector_size(
                v            = data.qfrc_actuator[geo.FREE3D_VEL:],
                size_sigma   = sigmas.minimize_energy
            ),
            'vel_residual_size' : self.reward_vector_size(
                v          = self._vel_scale * info['act_history'][0, bruce.NDOF:],
                size_sigma = sigmas.vel_residual_size
            ),
            'vel_residual_rate' : self.reward_residual_vel_rate(
                vdes_res            = self._vel_scale * info['act_history'][0, bruce.NDOF:],
                last_vdes_res       = self._vel_scale * info['act_history'][1, bruce.NDOF:],
                vdes_res_rate_sigma = sigmas.vel_residual_rate
            ),
            'res_jt_rate': self.reward_action_rate(
                act           = self._jt_scale * info['act_history'][0, :bruce.NDOF],
                last_act      = self._jt_scale * info['act_history'][1, :bruce.NDOF],
                last_last_act = self._jt_scale * info['act_history'][2, :bruce.NDOF],
            ),
            'foot_contact': self.reward_foot_contact(
                ground_contact  = bruce.get_ground_contact(self._np, 
                                                           bruce.get_raw_contacts(self._np,
                                                                                  self.mj_model,
                                                                                  data,
                                                                                  threshold=bruce.CONTACT_THRESHOLD)),
                swing_foot      = info['gaitlib'].swing_leg
            ),
                
            'alive': self.reward_alive(),
            'global_xy_tracking': self.reward_euclidean_imitation(
                qpos            = data.qpos[:3],
                reference       = global_init_qpos[:3],
                imitation_sigma = sigmas.global_xy_tracking
            ),
            'global_z_tracking': self.exp_reward(
                mag2            = geo.quat_dist(
                    _np = self._np,
                    q1  = data.qpos[3:geo.FREE3D_POS],
                    q2  = global_init_qpos[3:geo.FREE3D_POS]
                )**2,
                sigma = sigmas.global_z_tracking
            ),
        }
        return rewards
    
    
    def reward_foot_contact(
        self,
        ground_contact: jax.Array,
        swing_foot: jax.Array
    ) -> jax.Array:
        """Reward for foot contact."""
        feet_des = self._np.array([Leg.LEFT, Leg.RIGHT]) == swing_foot
        # print(f'ground contact: {ground_contact}, feet des: {feet_des}')
        contact_reward = self._np.sum(feet_des == ground_contact.astype(self._np.float32))
        
        return contact_reward / 2.0