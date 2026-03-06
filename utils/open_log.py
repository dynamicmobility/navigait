import h5py
import numpy as np
import matplotlib.pyplot as plt
from utils.plotting import get_subplot_grid

def load_dict_from_hdf5(filename):
    out = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            out[key] = np.array(f[key][:])
    return out

# print(data['time'])
# print()
# print(data['sensordata'])

def plot_sensordata(data, idxs, axs=None, fig=None):
    if axs is None or fig is None:
        nrows, ncols = get_subplot_grid(n=len(idxs))
        fig, axs = plt.subplots(nrows, ncols)
        axs = axs.flatten()

    for ax, idx in zip(axs, idxs):
        ax.plot(data['time'], data['sensordata'][:, idx])

    return fig, axs

old_model_data = load_dict_from_hdf5('logs/run_gaits.h5')
old_fig, old_axs = plot_sensordata(old_model_data, [-3, -2, -1])
old_fig.suptitle('Old model')
old_fig.tight_layout()

new_model_data = load_dict_from_hdf5('logs/animation_4bar.h5')
new_fig, new_axs = plot_sensordata(new_model_data, [-3, -2, -1], axs=old_axs, fig=old_fig)
# new_fig.suptitle('New 4-bar model')
new_fig.tight_layout()

plt.show()