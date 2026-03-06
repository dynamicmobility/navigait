"""Bruce Interface"""

from pathlib import Path

# JAX and Mujoco imports
import jax
import jax.numpy as jnp
import numpy as np
import mujoco as mj
from mujoco_playground._src.dm_control_suite import common
from mujoco import mjx
from mujoco_playground._src import mjx_env

POSITION_XML = Path('envs/bruce/model/bruce_position.xml')
PD_XML = Path("envs/bruce/model/bruce_pd.xml")

_interface_model = mj.MjModel.from_xml_path(
    PD_XML.as_posix(), common.get_assets()
)

GROUND_GEOM = 'plane'
TORSO       = 'base_link'

GROUND_GEOM_ID = _interface_model.geom(GROUND_GEOM).id
TORSO_ID       = _interface_model.body(TORSO).id

NDOF = 16
NJOINT = 16

FEET_SENSORS = [
    'opto1',
    'opto2',
    'opto3',
    'opto4',
]

HEIGHT_FR = 'right_foot'
HEIGHT_FL = 'left_foot'
FOOT_VEL_R = 'right_foot_vel'
FOOT_VEL_L = 'left_foot_vel'
LF_ZAXIS = 'left_foot_z'
RF_ZAXIS = 'right_foot_z'
BASE_VEL = 'base_velocimeter'

FEET_SITES = [
    '0',
    '1',
    '2',
    '3',
    'midfoot_r',
    'midfoot_l',
]

# right foot sites
RF_SITES = [
    'opto1',
    'opto2',
]

# left foot sites 
LF_SITES = [
    'opto3',
    'opto4',
]

BASE_GYRO = 'base_gyro'
BASE_ACCELEROMETER = 'base_accelerometer'

DEFAULT_JT = [0.0] * 16
# DEFAULT_POSE = [
#     -0.008, 0.469, 0.018, -0.947, 0.478,
#      0.008, 0.469, -0.018, -0.947, 0.478,
#     -0.700, 1.300, 2.000, 0.700, -1.300, -2.000
# ]
DEFAULT_FF = [0.0, 0.0, 0.475, 1.0, 0.0, 0.0, 0.0]

ACCEL_NOISE = 0.05
GYRO_NOISE  = 0.04
QPOS_NOISE  = 0.05
QVEL_NOISE  = 0.1
CONTACT_THRESHOLD = 0.1
MIN_BASE_HEIGHT = 0.2

def mj_qpos_to_hzd(
    _np,
    qpos: jax.Array,
    num_free: jax.Array,
):
    """Convert qpos (without 4-bar) to motor configuration (returns angle for the 4-bar linkage on the ankle)"""
    jt_pos = qpos[num_free:]
    return _np.hstack((qpos[0:num_free], mj_jt_to_hzd(_np, jt_pos)))

def mj_jt_to_hzd(
    _np,
    jt_pos: jax.Array
):
    """Convert joint configuration (without 4-bar) to motor configuration (returns angle for the 4-bar linkage on the ankle)"""
    idxs = mj_jt_to_hzd_idx(_np)
    return jt_pos[idxs]

def mj_jt_to_hzd_idx(
    _np
):
    return _np.arange(NDOF)

def hzd_pos_to_mj_qpos(
    _np,
    qpos: jax.Array,
    num_free: jax.Array
):
    jt_pos = qpos[num_free:]
    return _np.hstack((qpos[:num_free], hzd_jt_to_mj_jtpos(_np, jt_pos)))

def hzd_jt_to_mj_jtpos(
    _np,
    jt_pos: jax.Array
):
    return jt_pos

def hzd_jt_to_mj_ctrl(
    _np,
    jt_pos: jax.Array
):
    return jt_pos

def get_raw_contacts(
    _np,
    mj_model: mj.MjModel,
    data: mjx.Data,
    threshold: float,
) -> jax.Array:
    """Return the contact state of the feet."""
    raw_contacts = _np.array([
        mjx_env.get_sensor_data(mj_model, data, sensor)[0]
        for sensor in FEET_SENSORS
    ]) > threshold
    return raw_contacts

def get_ground_contact(_np, raw_contacts: jax.Array) -> jax.Array:
    """Return the ground contact state of the feet."""
    right = _np.any(raw_contacts[:2])
    left = _np.any(raw_contacts[2:])
    return _np.array([right, left])

def get_gravity(_np, accel: jax.Array) -> jax.Array:
    """Return the gravity vector in the world frame."""
    return _np.array([accel @ _np.array([0, 0, -1])])

def get_accelerometer(mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the accelerometer readings in the local frame."""
    return mjx_env.get_sensor_data(mj_model, data, BASE_ACCELEROMETER)

def get_gyro(mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the gyroscope readings in the local frame."""
    return mjx_env.get_sensor_data(mj_model, data, BASE_GYRO)

def get_foot_pos( _np, mj_model: mj.MjModel, data: mjx.Data,) -> jax.Array:
    """Return the accelerometer readings in the local frame."""
    # return mjx_env.get_sensor_data(mj_model, data, )
    foot_height = _np.vstack((mjx_env.get_sensor_data(mj_model, data, HEIGHT_FR), mjx_env.get_sensor_data(mj_model, data, HEIGHT_FL)))
    return foot_height

def get_foot_vel( _np, mj_model: mj.MjModel, data: mjx.Data,) -> jax.Array:
    """Return the accelerometer readings in the local frame."""
    # return mjx_env.get_sensor_data(mj_model, data, )
    foot_vel = _np.vstack((mjx_env.get_sensor_data(mj_model, data, FOOT_VEL_R), mjx_env.get_sensor_data(mj_model, data, FOOT_VEL_L)))
    return foot_vel

def get_left_z_axis( _np, mj_model: mj.MjModel, data: mjx.Data,):
    axis = mjx_env.get_sensor_data(mj_model, data, LF_ZAXIS)
    return axis

def get_right_z_axis( _np, mj_model: mj.MjModel, data: mjx.Data,) ->jax.Array:
    axis = mjx_env.get_sensor_data(mj_model, data, RF_ZAXIS)
    return axis

def get_body_vel( _np, mj_model: mj.MjModel, data: mjx.Data,) ->jax.Array:
    vel = mjx_env.get_sensor_data(mj_model, data, BASE_VEL)
    return vel