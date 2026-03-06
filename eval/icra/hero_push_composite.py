import cv2
from pathlib import Path
from dynamo_figures import CompositeImage, CompositeMode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = Path(f'eval/icra/videos/NaviGait_hero_push.mp4')
pic = CompositeImage(
    mode = CompositeMode.MIN_VALUE,
    video_path = path,
    start_t=1.25,
    end_t=5.1,
    skip_frame=20,
).merge_images()
cv2.imwrite(f'paper_plots/hero_push1.jpg', pic)

# pic = CompositeImage(
#     mode = CompositeMode.MIN_VALUE,
#     video_path = path,
#     alpha=0.2,
#     start_t=5.25,
#     end_t=6.4,
#     skip_frame=20,
# ).merge_images()
# cv2.imwrite(f'paper_plots/hero_push2.jpg', pic)
pic = CompositeImage(
    mode = CompositeMode.MIN_VALUE,
    video_path = path,
    start_t=5.3,
    end_t=6.0,
    skip_frame=10,
).merge_images()
cv2.imwrite(f'paper_plots/hero_push2.jpg', pic)

pic = CompositeImage(
    mode = CompositeMode.MIN_VALUE,
    video_path = path,
    start_t=6.0,
    end_t=99,
    skip_frame=20,
).merge_images()
cv2.imwrite(f'paper_plots/hero_push3.jpg', pic)

def make_vr_plot(vr, arrow_idxs, name):
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(vr[:, 1], vr[:, 0], c='blue', lw=2)
    def add_arrow(idx):
        ax.arrow(
            vr[idx, 1], 
            vr[idx, 0], 
            vr[idx+1, 1] - vr[idx, 1], 
            vr[idx+1, 0] - vr[idx, 0], 
            head_width=0.01, 
            head_length=0.01, 
            color='blue', 
            edgecolor='blue'
        )
    for idx in arrow_idxs:
        add_arrow(idx)
    
    ax.locator_params(nbins=4)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim((-0.175, 0.175))
    ax.set_ylim((-0.175, 0.175))
    ax.set_xlabel(r'$\nabla v_y$')
    ax.set_ylabel(r'$\nabla v_x$')
    fig.set_size_inches((3,3))
    fig.savefig(f'paper_plots/hero_push_{name}.svg')
    return fig, ax

df = pd.read_csv('logs/hero_push_vr.csv')
vr = df.values
make_vr_plot(vr[:250], [], 'part1')
make_vr_plot(vr[250:np.argmin(vr[:, 1]) + 1], [np.argmin(vr[:, 1]) - 250 - 1], 'part2')
make_vr_plot(vr[np.argmin(vr[:, 1]) - 2:], [2], 'part3')

