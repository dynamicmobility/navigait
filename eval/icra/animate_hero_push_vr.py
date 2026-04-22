import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

from utils.plotting import set_mpl_params


CSV_PATH = Path('logs/hero_push_vr.csv')
OUT_PATH = Path('eval/icra/videos/hero_push_vr.mp4')
FPS = 50
T = 10.0


def main():
    set_mpl_params()

    df = pd.read_csv(CSV_PATH)
    xs = df['vdes_res_x'].to_numpy()
    ys = df['vdes_res_y'].to_numpy()
    n = len(xs)

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    xpad = 0.05 * (xmax - xmin) if xmax > xmin else 0.01
    ypad = 0.05 * (ymax - ymin) if ymax > ymin else 0.01

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(ymin - ypad, ymax + ypad)
    ax.set_ylim(xmin - xpad, xmax + xpad)
    ax.set_box_aspect(1)
    ax.set_xlabel(r'$\Delta v_y$', fontsize=32)
    ax.set_ylabel(r'$\Delta v_x$', fontsize=32)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.axhline(0.0, color='0.7', linewidth=0.8, zorder=0)
    ax.axvline(0.0, color='0.7', linewidth=0.8, zorder=0)
    ax.grid(True, linestyle=':', alpha=0.5)

    fig.subplots_adjust(top=0.99, bottom=0.16, left=0.18, right=0.98)

    trail, = ax.plot([], [], color='C0', linewidth=1.2, alpha=0.7)
    current, = ax.plot([], [], 'o', color='red', markersize=6)

    def init():
        trail.set_data([], [])
        current.set_data([], [])
        return trail, current

    def update(i):
        trail.set_data(ys[: i + 1], xs[: i + 1])
        current.set_data([ys[i]], [xs[i]])
        return trail, current

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n,
        init_func=init,
        interval=1000 / FPS,
        blit=True,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=FPS, bitrate=2400)
    print(f'Writing animation to {OUT_PATH} ({n} frames @ {FPS} fps)')
    anim.save(OUT_PATH, writer=writer, dpi=150)
    plt.close(fig)
    print('Done.')


if __name__ == '__main__':
    main()
