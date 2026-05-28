"""Compose a "gait tile" image from a robot walking video.

Picks N evenly-spaced frames, crops each around the robot, and lays them
side by side into a single wide image.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_color(s):
    s = s.strip().lower()
    named = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
    }
    if s in named:
        return named[s]
    parts = [int(p) for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--bg must be a name or 'R,G,B', got {s!r}")
    return tuple(parts)


def grab_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if ok:
        return frame
    # Fallback: sequential read from start.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(idx + 1):
        ok, frame = cap.read()
        if not ok:
            return None
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path)
    p.add_argument("-n", "--num-frames", type=int, default=8)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--gap", type=int, default=0, help="Pixel gap between tiles.")
    p.add_argument("--bg", type=parse_color, default=(255, 255, 255),
                   help="Background color for gap: name or 'R,G,B'.")
    p.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                   default=None, help="Explicit crop box; skips interactive ROI.")
    args = p.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(args.video)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError("Could not determine frame count.")
    n = args.num_frames
    if n < 1 or n > total:
        raise ValueError(f"--num-frames must be in [1, {total}], got {n}")

    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for i in indices:
        f = grab_frame(cap, int(i))
        if f is None:
            raise RuntimeError(f"Failed to read frame {i}")
        frames.append(f)
    cap.release()

    if args.crop is None:
        win = "Select robot crop (drag, then Enter)"
        roi = cv2.selectROI(win, frames[0], showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(win)
        x, y, w, h = (int(v) for v in roi)
        if w == 0 or h == 0:
            raise RuntimeError("Empty crop selection.")
    else:
        x, y, w, h = args.crop

    print(f"Crop box: --crop {x} {y} {w} {h}")

    tiles = []
    for f in frames:
        H, W = f.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        crop = f[y0:y1, x0:x1]
        if crop.shape[0] != h or crop.shape[1] != w:
            pad = np.zeros((h, w, 3), dtype=f.dtype)
            pad[: crop.shape[0], : crop.shape[1]] = crop
            crop = pad
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tiles.append(Image.fromarray(rgb))

    gap = args.gap
    out_w = n * w + max(0, n - 1) * gap
    out = Image.new("RGB", (out_w, h), args.bg)
    for i, tile in enumerate(tiles):
        out.paste(tile, (i * (w + gap), 0))

    output = args.output or args.video.with_name(args.video.stem + "_tiles.png")
    out.save(output)
    print(f"Wrote {output} ({out.size[0]}x{out.size[1]})")


if __name__ == "__main__":
    main()
