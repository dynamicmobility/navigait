# from utils.plotting import set_mpl_params
# set_mpl_params()
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import numpy as np

# # Create figure
# fig, ax = plt.subplots()

# # Normalize from 0 to 1
# norm = mpl.colors.Normalize(vmin=0, vmax=1)

# # Green (0) -> Yellow -> Red (1)
# cmap = mpl.cm.get_cmap("RdYlGn_r")

# # Create scalar mappable
# sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
# sm.set_array([])

# # Create colorbar
# cbar = fig.colorbar(sm, ax=ax)

# # Set ticks
# ticks = np.linspace(0, 1, 6)
# cbar.set_ticks(ticks)
# cbar.set_ticklabels([f"{int(t * 100)}\\%" for t in ticks])

# # Label
# cbar.set_label(
#     "Fall Probability",
#     rotation=270,
#     labelpad=25,
# )# Add text at the bottom of the colorbar
# cbar.ax.text(0.55, -0.05, "Stable", ha='center', va='top', transform=cbar.ax.transAxes, fontsize=16)
# cbar.ax.text(0.55, 1.05, "Fall", ha='center', va='bottom', transform=cbar.ax.transAxes, fontsize=16)
# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# # Remove the empty axis
# fig.subplots_adjust(top=0.98, bottom=0.02)
# ax.remove()
# fig.set_size_inches((2, 6))
# plt.tight_layout()
# plt.savefig('paper_plots/colorbar.pdf')

from utils.plotting import set_mpl_params
set_mpl_params()
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


fig = plt.figure(figsize=(3.0,9))

# Explicit axis for the colorbar: [left, bottom, width, height]
cax = fig.add_axes([0.35, 0.1, 0.2, 0.8])

norm = mpl.colors.Normalize(vmin=0, vmax=1)
cmap = mpl.cm.RdYlGn_r
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm, cax=cax)
FS = 24
ticks = np.linspace(0,1,6)
cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{int(t*100)}\\%" for t in ticks], fontsize=FS)

cbar.set_label("Fall Probability", rotation=270, labelpad=30, fontsize=FS+8)

# top and bottom annotations
cax.set_title("Fall", pad=12, fontsize=FS)
fig.text(0.45, 0.07, "Stable", ha='center', va='center', fontsize=FS)

# plt.tight_layout()
plt.savefig('paper_plots/colorbar.pdf')