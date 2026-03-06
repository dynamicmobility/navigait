import os
# os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np
import mediapy as media
from tqdm import tqdm
from pathlib import Path
import glob
import cv2
from jax import numpy as jnp
from scipy.special import factorial
import jax
from control.gait import GaitLibrary, Leg
from utils.geometry import FREE3D_POS, FREE3D_VEL
from envs.atalante import interface as consts
import matplotlib.pyplot as plt
from PIL import Image

T = 5
VIS_HZ = 100
VIS_DT = 1 / VIS_HZ
GAIT_LIB_PATH = r'control/gaits/2D_library_v6'
SIM_SPEED = 1
NUM_STATES = 12
HEIGHT = 2560 
WIDTH = 1440
SLOWDOWN = 1


def setup_gaitlibrary(gaitlib_dir, model_xml_dir, num_states=12, blend=0.2):
    """Returns an environment and GaitTracker that corresponds to the provided 
    YAML file."""
    VARY_FUNC(0.0)
    controller = GaitLibrary.from_directory(
        path         = gaitlib_dir,
        v0           = np.array([vx, 0.0]),
        num_states   = num_states,
        num_degree   = 7,
        num_vx_gaits = 31,
        gnp          = np,
        fact         = factorial,
        blend        = blend,
        gait_type    = 'P1',
        swing_leg    = Leg.RIGHT
    )
    env = Exo(xml_path=model_xml_dir)
    data, obs, info = env.reset()
    
    # Set initial joint states
    data.qpos = np.zeros(19)
    data.qvel = np.zeros(18)
    data.qpos[FREE3D_POS:] = controller(0)[:num_states]
    data.qvel[FREE3D_VEL:] = controller(0)[num_states:]
    data.qpos[:FREE3D_POS] = controller.ff_evaluate(0)[:FREE3D_POS]
    data.qvel[:FREE3D_VEL] = controller.ff_evaluate(0)[FREE3D_POS:]
    
    data.ctrl = np.zeros(24)
    data, obs, info = env.step(data, info)
    
    return data, obs, info, env, controller

def add_text(pixels, text, org):
    # Text settings
    # org = (50, 100) 
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2

    # Add text to the image
    cv2.putText(
        pixels, text, org, font, fontScale, color, thickness, cv2.LINE_AA
    )
    return pixels


    
def forward_simulate(
    data,
    obs,
    info,
    env: Exo,
    controller: GaitLibrary,
    max_time=5,
    h=1080,
    w=1920,
    name=''
):
    global vx, tau
    qpos = []
    qvel = []
    time = []
    """Simulate the environment with the provided controller. Returns an array
    of RGB frames of the simulation"""
    # Create renderer
    max_steps = round(max_time / env.ctrl_dt)
    FRAME_SKIP = np.ceil(VIS_DT / env.ctrl_dt)
    frames = np.zeros((round(max_steps / FRAME_SKIP) , h, w, 3), dtype=np.uint8)
    renderer = mujoco.Renderer(env.model, h, w)
    iters = 0
    t0 = data.time
    vis_iter = 0
    offset_pos = np.zeros(FREE3D_POS)
    # ctrllr = jax.jit(controller.__call__)
    init_q = data.qpos.copy()
    init_qvel = data.qvel.copy()
    # Turn on scene flags
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = False
    scene_option.geomgroup[3] = True
    # scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    old_vdes = np.array([0.0, 0.0])
    VARY_FUNC(0.0)
    vdes = np.array([vx, 0.0])
    switch = 0
    for iters in tqdm(range(max_steps), disable=False):
        # check if the swing leg has made contact
        if not ANIMATE and info['swing_contact']:
            controller = controller.impact_reset(data.time)
            break

        VARY_FUNC(data.time)
        vdes = np.array([vx, 0.0])
        if controller.get_step_phase(data.time) < 0.7:
            switch += 1
            # print(f'{controller.get_phase(data.time)[0]:3f}\t{controller.curr_period[0]:3f}\t{data.time:3f}')
            controller = controller.set_gait(vdes.copy(), data.time)
        
        if ANIMATE and controller.get_step_phase(data.time) >= 1.0:
            # break
            """Time based reset for animation. Offset is commented out for now..."""
            # print('STEP', data.time)
            break
            if ANIMATE:
                offset_pos[0] += np.copy(controller.ff_evaluate(1)[:FREE3D_POS])[0]
                offset_pos[1] += np.copy(controller.ff_evaluate(1)[:FREE3D_POS])[1]
            controller = controller.impact_reset(data.time)
            if ANIMATE:
                offset_pos[0] -= np.copy(controller.ff_evaluate(0)[:FREE3D_POS:])[0]
                offset_pos[1] -= np.copy(controller.ff_evaluate(0)[:FREE3D_POS:])[1]
            
        # calculate control
        ctrl = controller(
            s=controller.get_phase(data.time),
        )
        data.ctrl = ctrl
        
        # Apply perturbations
        # perturb_time = 2.0
        # if data.time % perturb_time < env.ctrl_dt:
        #     body_id = env.model.body("torso").id
        #     max_force = [100, 100]
        #     mag_x = np.random.uniform(-1, 1)
        #     mag_y = np.random.uniform(-1, 1)
        #     perturbation_vec = [mag_x*max_force[0], mag_y*max_force[1], 0.0,   # force (x, y, z)
        #                         0.0, 0.0, 0.0] # torque (x, y, z)
        #     data.xfrc_applied[body_id] = np.array(perturbation_vec)   
        
        # step forward in the simulation
        data, obs, info = env.step(data, info)
        s = controller.get_phase(data.time)
        
        # # For animation....
        if ANIMATE:
            data.qpos[:FREE3D_POS] = controller.ff_evaluate(s)[:FREE3D_POS] + offset_pos
            data.qvel[:FREE3D_VEL] = controller.ff_evaluate(s)[FREE3D_POS:]
            data.qpos[FREE3D_POS:] = controller(s)[:NUM_STATES]
            data.qvel[FREE3D_VEL:] = controller(s)[NUM_STATES:]
            
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        time.append(data.time)
        
        if np.all(np.isclose(data.qpos[FREE3D_POS:], np.zeros(12), atol=1e-2)):
            print('what the helly')
        
        
        # qpos.append(data.qpos.copy())
        # qvel.append(vx)

        # render the scene
        if iters % FRAME_SKIP == 0:
            
            # Update renderer
            renderer.update_scene(data, camera='side', scene_option=scene_option)
            
            pixels = renderer.render()
            frames[vis_iter] = pixels
            vis_iter += 1
            
        # end simulation if the robot has fallen
        if data.qpos[2] < 0.3:
            # data, obs, info = env.reset()
            break
    frames = frames[:vis_iter]
    return frames, time, qpos, qvel

def save_tiles(frames, save_folder, num_tiles=2, separate=False):
    """Saves a list of list of provided frames to a video file
    at the provided path. Connects frames via add_type, which can be parallel or
    concatentation."""
    idxs = np.round(np.linspace(0, len(frames) - 1, num_tiles))
    frames = np.array(frames)
    tiles = frames[idxs.astype(int)]
    # tiles = np.array(tiles)
    if separate:
        for idx, tile in enumerate(tiles):
            im = Image.fromarray(tile)
            im.save(save_folder / f'{idx}.png')
    else:
        print(tiles.shape)
        tiles = np.concatenate([*tiles], axis=1)
        im = Image.fromarray(tiles)
        im.save(save_folder / 'tiles.png')

gait_frames = []
yaml_files = glob.glob(GAIT_LIB_PATH + '/*.yaml')
def sort_key(x):
    x = Path(x).stem
    nums = x.split('_')
    return int(nums[1]) * 11 + int(nums[2])

yaml_files.sort(key=sort_key)

data, obs, info, env, controller = setup_gaitlibrary(
    GAIT_LIB_PATH, consts.PD_EXO_XML, blend=0.2
)
vx = 0.3
print('Vx =', vx)
data, obs, info, env, controller = setup_gaitlibrary(
    GAIT_LIB_PATH, consts.PD_EXO_XML, blend=0.2
)
gait_frames, time, qpos, qvel = forward_simulate(
    data, obs, info, env, controller, h=HEIGHT, w=WIDTH, name=f'{vx}', max_time=T
)

# Save video
print('here')
tiles_path = Path('mj_envs/tiles')
save_tiles(gait_frames, tiles_path, num_tiles=4)
print(f"Tiles saved to {tiles_path}")
