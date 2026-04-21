# manim-figures notes

Location: `presentation/manim-figures/`. Generated output (`media/`,
`__pycache__/`, `*.log`) is gitignored — only source `.py` files are tracked.


## Environment
- Use the `manim-env` conda env (Python 3.12). Recreated 2026-04-20, and
  again 2026-04-21 on a second Ubuntu desktop.
- `base` also auto-activates on these machines; shell out of it fully before activating `manim-env`:
  ```sh
  conda deactivate && conda deactivate && conda activate manim-env
  ```
  (two `deactivate`s — one leaves `manim-env`, the second leaves `base`.)
- Manim installed via `pip install manim` (manim 0.20.1). `scipy` comes in
  as a transitive dep — used in the scene for `ConvexHull`.
- System deps (needed before `pip install manim` or `manimpango` build fails
  with "pangocairo >= 1.30.0 is required"):
  ```sh
  sudo apt install -y libcairo2-dev libpango1.0-dev ffmpeg pkg-config
  ```

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
- Represents a **2-phase periodic gait trajectory** for a walking robot joint:
  left curve = one phase (blue, `#1f77b4`), right curve = the other (red, `#d62728`).
  Each is a 7th-degree Bezier (8 control points).
- Constraints the scene preserves:
  - Shared middle control point (`left[-1] == right[0]`) → C0 at the phase transition.
  - `left[0].y == right[-1].y` and matching slope at that boundary → C0+C1
    periodic loop. With evenly-spaced x coords per curve, slope match reduces
    to `left[1].y - left[0].y == right[-1].y - right[-2].y` (kept at 0 here).
  - Stretches only vary interior y-values (and the shared middle y), so all
    boundary constraints are preserved automatically.
- Helpers `_phase_pts`, `_dots`, `_polygon`, `_hulls`, `_curves`, `_convex_hull`
  each rebuild fresh Mobjects from a `(8,3)` control-point array.
- Flow: fade in axes → fade in dots + control polygon → fade in convex hulls
  → `Create` both curves → morph to stretch 1 → morph to stretch 2.
- Background forced white via `config.background_color = "#FFFFFF"`; `FG = #1a1a1a`
  is passed explicitly to every text/axis element so nothing disappears.

## Animating convex hulls (important gotcha)
`Transform(old_hull, new_hull)` on a Manim `Polygon` interpolates the vertex
list element-by-element. The convex hull's vertex *ordering and count* can
differ between keyframes, so `Transform` produces a "rotating vertices"
artifact where the hull momentarily detaches from the points it should wrap.

Fix used in this scene: drive the morph with a `ValueTracker` + `always_redraw`
so every frame recomputes the hull (and all other mobjects) from freshly
interpolated control points.

```python
alpha = ValueTracker(1.0)
state = {"src_l": ..., "dst_l": ..., "src_r": ..., "dst_r": ...}
def cur():
    a = alpha.get_value()
    return (state["src_l"] * (1-a) + state["dst_l"] * a,
            state["src_r"] * (1-a) + state["dst_r"] * a)
dyn_hulls = always_redraw(lambda: _hulls(*cur()))
# ... same for dots, polygon, curves
self.play(alpha.animate.set_value(1.0), run_time=2.0)
```

The initial reveal still uses static mobjects (so `FadeIn`/`Create` behave
normally); they're `self.remove`'d and replaced with the `always_redraw`
versions before the first morph.

## Conventions I settled on
- Use `ParametricFunction(lambda t: bezier_point(pts, t), t_range=[0,1])` so
  the curve is always sampled from the current control points — no need to
  reimplement the De Casteljau algorithm.
- For "stretching," build a new `VGroup` of `Dot`s/`Line`s/`ParametricFunction`
  from the new control points and pass them to `Transform(old, new)` — this is
  simpler than animating each point with `MoveAlongPath` and Manim handles
  interpolation cleanly. **Exception:** `Polygon`s whose vertex set changes
  (e.g. convex hulls) — use the `ValueTracker` + `always_redraw` pattern above.
