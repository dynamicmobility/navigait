# manim-figures notes

Location: `presentation/manim-figures/`. Generated output (`media/`,
`__pycache__/`, `*.log`) is gitignored — only source `.py` files are tracked.


## Environment
- Use the `manim-env` conda env (Python 3.12). It was recreated fresh on 2026-04-20.
- `base` also auto-activates on this machine; shell out of it fully before activating `manim-env`:
  ```sh
  conda deactivate && conda deactivate && conda activate manim-env
  ```
  (two `deactivate`s — one leaves `manim-env`, the second leaves `base`.)
- Manim was installed via `pip install manim` (manim 0.20.1).

## Rendering
From this directory:
```sh
conda deactivate && conda deactivate && conda activate manim-env
manim -ql bezier7.py Bezier7Showcase     # low quality, fast
manim -qh bezier7.py Bezier7Showcase     # 1080p
```
Output lands in `media/videos/bezier7/<res>/Bezier7Showcase.mp4`.

### Quality flags
- `-ql` 480p15, `-qm` 720p30, `-qh` 1080p60, `-qp` 1440p60, `-qk` 2160p60.
- The script pins `config.pixel_width/height` to 1920×540, so the quality
  flag only controls frame rate in practice (folder name reflects the
  actual height — e.g., `-qh` lands in `540p60/`, not `1080p60/`).
- `-p` is a separate flag (`--preview`, open in player after render). Don't
  confuse it with `-qp`.

### Rendering higher-quality while keeping the 32:9 aspect
CLI flags beat script config, so the simplest way is `--resolution W,H`:
```sh
manim -qh --resolution 3840,1080 --fps 60 bezier7.py Bezier7Showcase
manim -qh --resolution 7680,2160 --fps 60 bezier7.py Bezier7Showcase  # 8K
```
Keep the ratio at 1920:540 (≈3.56:1) so the scene doesn't clip.

Alternatively, edit the `config.pixel_width/height` + `frame_width/height`
block at the top of `bezier7.py`:
```python
config.pixel_width = 3840
config.pixel_height = 1080
_FRAME_W = 20.0
config.frame_width = _FRAME_W
config.frame_height = _FRAME_W * (1080 / 3840)
```

### If output looks wrong
Manim caches partial renders. If a change doesn't appear, nuke the cache:
```sh
rm -rf media/videos/bezier7
```

## LaTeX gotcha
`Tex`/`MathTex` shell out to `latex` + `dvisvgm` and require the `standalone`
LaTeX class. On this machine `standalone.cls` is not installed and `tlmgr`
needs `sudo`, so I avoided LaTeX entirely by using `manim.Text` (Pango-based)
for all labels and captions. If you want real math typesetting later, install
standalone with:
```sh
sudo tlmgr install standalone everysel preview ucs
```

## Scene: `Bezier7Showcase`
- 7th-degree Bezier = 8 control points `P0..P7`.
- `bezier_point(pts, t)` evaluates `B(t) = Σ C(7,i) (1-t)^(7-i) t^i P_i`.
- Flow: draw axes → show title + formula → place 8 control points + control
  polygon → draw curve → stretch a subset of control points (curve recolors
  red and reshapes) → pull endpoints outward for a second stretch.
- Background is forced white via `config.background_color = "#FFFFFF"` at
  import time; foreground color `FG = #1a1a1a` is passed explicitly to every
  text/axis element so nothing disappears against white.

## Conventions I settled on
- Use `ParametricFunction(lambda t: bezier_point(pts, t), t_range=[0,1])` so
  the curve is always sampled from the current control points — no need to
  reimplement the De Casteljau algorithm.
- For "stretching," build a new `VGroup` of `Dot`s/`Line`s/`ParametricFunction`
  from the new control points and pass them to `Transform(old, new)` — this is
  simpler than animating each point with `MoveAlongPath` and Manim handles
  interpolation cleanly.
