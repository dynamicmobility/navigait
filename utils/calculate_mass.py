import mujoco
import numpy as np
from envs.bruce import interface4bar as bruce

# Load your model and data
model = mujoco.MjModel.from_xml_path(bruce.PD4BAR_XML.as_posix())
data = mujoco.MjData(model)

def print_body_names_and_masses(mj_model):
    """
    Prints the name and mass for each body in the model.
    """
    for body_id in range(mj_model.nbody):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        mass = mj_model.body_mass[body_id]
        print(f"Body: {name}, idx: {body_id}, Mass: {mass}")

print_body_names_and_masses(model)

# Sum all body masses
total_mass = np.sum(model.body_mass)

print(f"Total model mass: {total_mass:.4f} kg")
