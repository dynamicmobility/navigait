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
# from envs.bruce import interfacedirect as bruce
from envs.bruce.navigait import Bruce
from control.bezier import Leg

from utils import geometry as geo

class BruceImitation(Bruce):
    """
    A class for tracking joystick-controlled gait in a simulated environment.
    """

    def __init__(
        self,
        gaitlib_path: str,
        env_params: config_dict.ConfigDict,
        gait_type='P2',
        backend='jnp',
        idealistic=False,
        animate=False
    ):  
        # Initialize the parent class
        super().__init__(
            gaitlib_path, env_params, gait_type, backend, idealistic, animate
        )

    
    def pre_process_action(
        self, 
        time: jax.Array,
        info: dict[str | jax.Array],
        res_joints: jax.Array,
        res_vel: jax.Array,
        ndof: int,
    ) -> jax.Array:
        """Pre-processes the residual actions and re-evaluates the gait library
        to compute the desired joint targets.
        
        res_joints are actually just the joint commands here"""

        # Run the residual joint commands through lowpass filter
        dt = time - info['last_time']
        scaled_res_action = self._jt_scale * res_joints
        info['res_target'] = info['res_target'] + dt / self.params.filter_tau * (scaled_res_action - info['res_target'])


        # Calculate a running derivative (Euler velocity) of delta q
        scaled_last_action = info['last_res_target']
        info['delta_qdot'] = info['delta_qdot'] + dt / self.params.filter_tau * ((info['res_target'] - scaled_last_action) / dt - info['delta_qdot'])
        info['last_res_target'] = info['res_target'].copy()

        # Run the residual velocity command through a lowpass filter
        vel_cmd_limits = self._np.vstack((
            self.params.command.lin_vel_x,
            self.params.command.lin_vel_y,
            self.params.command.ang_vel_z
        )).T
        info['vel_target'] = self._np.clip(info['vdes'], *vel_cmd_limits)
        # print(round(self._np.linalg.norm(info['vel_target']), 2))
        # print(self._np.round(info['vdes_res'], 2))

        # Pull the gait corresponding to the desired velocity + residual
        # velocity from the gait library
        def set_new_gait(args: tuple):
            gaitlib, vdes_gaitlib, t = args
            return gaitlib.set_gait(vdes_gaitlib, t)

        def keep_gait(args: tuple):
            gaitlib, _, _ = args
            return gaitlib
        gaitlib = info['gaitlib'] 

        gaitlib = self._cond(
            gaitlib.get_step_phase(time) < 0.7,
            set_new_gait,
            keep_gait,
            (gaitlib, info['vel_target'][:2], time)
        )
        info['gaitlib'] = gaitlib
        
        jtdot_desired = info['delta_qdot']
        
        # Compute the final motor targets by adding the residual to desired
        q_targets = self._np.array(bruce.DEFAULT_JT) + info['res_target'] + info['jt_offset']
        # info['jtdot_desired'] = info['jtdot_desired'] + 0.6 * (change_in_jt - info['jtdot_desired'])

        if self.params.tracking == 'position':
            vel_enabled = self._np.zeros(1)
        elif self.params.tracking == 'position-velocity':
            vel_enabled = self._np.ones(1)
        else:
            raise Exception(f'Tracking mode {self.params.tracking} DNE')

        motor_targets = self._np.hstack([q_targets, vel_enabled * jtdot_desired])
        info['last_time'] = time
    
        info['act_history'] = self.update_history(
            info['act_history'],
            self._np.hstack([res_joints, res_vel])
        )

        return motor_targets, info

    
    @property
    def action_size(self) -> int:
        return bruce.NDOF

    def reward_function(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        done: jax.Array,
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        sigmas = self.params.reward.sigmas
        base_des = info['base_history'][1]
        gait_des = info['gait_history'][1]
        global_qpos_act = bruce.ext_full_2ext_crank(
            self._np,
            data.qpos,
            geo.FREE3D_POS
        )
        # rel_qpos_act = self.get_relative_qpos(
        #     qpos        = global_qpos_act,
        #     base2global = geo.inv_transform(self._np, info['old_base2global'])
        # )
        global_base = geo.apply_transform(
            _np = self._np,
            qpos = base_des,
            offset = info['old_base2global']
        )
        global_init_qpos = geo.apply_transform(
            self._np,
            base_des,
            info['transform_init']
        )
        rewards = {
            'gait_tracking' : self.reward_euclidean_imitation(
                qpos            = global_qpos_act[geo.FREE3D_POS:],
                reference       = gait_des[:bruce.NDOF],
                imitation_sigma = sigmas.gait_tracking
            ),
            'base_xyz_tracking': self.reward_euclidean_imitation(
                qpos            = global_qpos_act[:3],
                reference       = global_base[:3],
                imitation_sigma = sigmas.base_xyz_tracking
            ),
            'base_quat_tracking': self.exp_reward(
                mag2  = geo.quat_dist(
                            _np = self._np,
                            q1  = global_qpos_act[3:geo.FREE3D_POS],
                            q2  = global_base[3:geo.FREE3D_POS]
                )**2,
                sigma = sigmas.base_quat_tracking
            ),
            'base_vel_tracking': self.reward_euclidean_imitation(
                qpos            = data.qvel[:2],
                reference       = base_des[geo.FREE3D_POS:geo.FREE3D_POS + 2],
                imitation_sigma = sigmas.base_vel_tracking
            ),
            'minimize_energy': self.reward_vector_size(
                v            = data.qfrc_actuator[geo.FREE3D_VEL:],
                size_sigma   = sigmas.minimize_energy
            ),
            'res_jt_rate': self.reward_action_rate(
                act           = self._jt_scale * info['act_history'][0, :bruce.NDOF],
                last_act      = self._jt_scale * info['act_history'][1, :bruce.NDOF],
                last_last_act = self._jt_scale * info['act_history'][2, :bruce.NDOF],
            ),
            'foot_contact': self.reward_foot_contact(
                ground_contact  = bruce.get_ground_contact(
                                    self._np, 
                                    bruce.get_raw_contacts(
                                        self._np,
                                        self.mj_model,
                                        data,
                                        threshold=bruce.CONTACT_THRESHOLD
                                    )
                                ),
                swing_foot      = info['gaitlib'].swing_leg
            ),
                
            'alive': self.reward_alive(),
            'global_xy_tracking': self.reward_euclidean_imitation(
                qpos            = global_qpos_act[:3],
                reference       = global_init_qpos[:3],
                imitation_sigma = sigmas.global_xy_tracking
            ),
            'global_z_tracking': self.exp_reward(
                mag2            = geo.quat_dist(
                    _np = self._np,
                    q1  = global_qpos_act[3:geo.FREE3D_POS],
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