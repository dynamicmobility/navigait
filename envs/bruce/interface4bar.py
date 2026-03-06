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
PD4BAR_XML = Path("envs/bruce/model/bruce_pd_4bar.xml")
PD_XML = PD4BAR_XML

_interface_model = mj.MjModel.from_xml_path(
    PD4BAR_XML.as_posix(), common.get_assets()
)

GROUND_GEOM = 'plane'
TORSO       = 'base_link'
RIGHT_FOOT  = 'ankle_pitch_link_r'
LEFT_FOOT  = 'ankle_pitch_link_l'

GROUND_GEOM_ID = _interface_model.geom(GROUND_GEOM).id
TORSO_ID       = _interface_model.body(TORSO).id
RIGHT_FOOT_ID  = _interface_model.body(RIGHT_FOOT).id
LEFT_FOOT_ID   = _interface_model.body(LEFT_FOOT).id

NDOF = 10
NJOINT = 14

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
LF_ZAXIS = 'left_foot_z'
RF_ZAXIS = 'right_foot_z'
BASE_ORIENTATION = 'base_orientation'
BASE_VEL = 'base_velocimeter'

# DEFAULT_JT = [0.0] * 10
DEFAULT_JT = [-0.008,  0.469,  0.018, -0.947, -0.469,  0.008,  0.469, -0.018, -0.947, -0.469]
DEFAULT_FF = [0.0, 0.0, 0.455, 1.0, 0.0, 0.0, 0.0]

ACCEL_NOISE       = 0.03
GYRO_NOISE        = 0.02
QPOS_NOISE        = 0.005
QVEL_NOISE        = 0.1
ATTITUDE_NOISE    = 0.02 # (in rad)
CONTACT_THRESHOLD = 0.1
MIN_BASE_HEIGHT   = 0.2

def ext_full_2ext_crank(
    _np,
    qpos: jax.Array,
    num_free: jax.Array,
):
    """Convert base + 4bar to base + crank"""
    jt_pos = qpos[num_free:]
    return _np.hstack((qpos[0:num_free], full2crank(_np, jt_pos)))

def full2crank(
    _np,
    jt_pos: jax.Array
):
    """Convert 4bar to crank"""
    idxs = _np.hstack((_np.arange(0, 4), _np.arange(6, 11), _np.array(13)))
    return jt_pos[idxs]


def ext_crank2ext_full(
    _np,
    qpos: jax.Array,
    num_free: jax.Array
):
    """Convert base + crank to base + 4bar"""
    jt_pos = qpos[num_free:]
    return _np.hstack((qpos[:num_free], crank2full(_np, jt_pos)))

def crank2full(
    _np,
    jt_pos: jax.Array
):
    """Convert crank to full 4bar by adding the passive joints"""
    return _np.hstack((jt_pos[0:4], -jt_pos[3] + jt_pos[4], jt_pos[3] - jt_pos[4], jt_pos[4], jt_pos[5:9], -jt_pos[8] + jt_pos[9], jt_pos[8] - jt_pos[9], jt_pos[9]))

def ext_pitch2ext_crank(
    _np,
    qpos: jax.Array,
    num_free: int
):
    """Convert extended pitch to extended crank"""
    q_crank = pitch2crank(_np, qpos[num_free:])
    return _np.hstack((qpos[:num_free], q_crank))


def ext_crank2ext_pitch(
    _np,
    qpos: jax.Array,
    num_free: int
):
    """Convert extended pitch to extended crank"""
    q_pitch = crank2pitch(_np, qpos[num_free:])
    return _np.hstack((qpos[:num_free], q_pitch))
        
def crank2pitch(
    _np,
    jt_pos: jax.Array
):
    """Convert crank to  pitch"""
    return _np.hstack((jt_pos[0:4], -jt_pos[3] + jt_pos[4], jt_pos[5:9], -jt_pos[8] + jt_pos[9]))

def pitch2crank(
    _np,
    jt_pos: jax.Array
):
    """Convert pitch to crank"""
    return _np.hstack((jt_pos[0:4], jt_pos[4] + jt_pos[3], jt_pos[5:9], jt_pos[9] + jt_pos[8]))


def ext_full2ext_pitch(
    _np,
    qpos: jax.Array,
    num_free: int
):
    return ext_crank2ext_pitch(
        _np,
        ext_full_2ext_crank(
            _np,
            qpos,
            num_free
        ),
        num_free
    )

def ext_pitch2ext_full(
    _np,
    qpos: jax.Array,
    num_free: int
):
    return ext_crank2ext_full(
        _np,
        ext_pitch2ext_crank(
            _np,
            qpos,
            num_free
        ),
        num_free
    )

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

def get_base_orientation(mj_model: mj.MjModel, data: mjx.Data) -> jax.Array:
    """Return the gyroscope readings in the local frame."""
    return mjx_env.get_sensor_data(mj_model, data, BASE_ORIENTATION)

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
    return mjx_env.get_sensor_data(mj_model, data, LF_ZAXIS)

def get_right_z_axis( _np, mj_model: mj.MjModel, data: mjx.Data,) ->jax.Array:
    return mjx_env.get_sensor_data(mj_model, data, RF_ZAXIS)

def get_body_vel( _np, mj_model: mj.MjModel, data: mjx.Data,) ->jax.Array:
    vel = mjx_env.get_sensor_data(mj_model, data, BASE_VEL)
    return vel
def get_right_z_axis( _np, mj_model: mj.MjModel, data: mjx.Data,) ->jax.Array:
    return mjx_env.get_sensor_data(mj_model, data, RF_ZAXIS)
