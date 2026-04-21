"""Two 7th-degree Bezier polynomials representing a 2-phase periodic gait
trajectory for a walking robot joint (swing phase + stance phase).

Each phase is a 7th-degree Bezier with 8 control points P0..P7:
    B(t) = sum_{i=0}^{7} C(7,i) * (1-t)^(7-i) * t^i * P_i,   t in [0, 1]

Constraints enforced:
  - Left curve B1 ends where right curve B2 begins (shared middle control
    point) → C0 at the phase transition.
  - B1 starts at the same joint position as B2 ends, with the same velocity
    → periodic loop (C0 + C1 at the cycle boundary).

With evenly-spaced x control points per curve, velocity matching at the loop
boundary reduces to matching the y-offset between the first two / last two
control points (here both set to 0 for a simple zero-velocity boundary).
"""

from __future__ import annotations

from math import comb

import numpy as np
from scipy.spatial import ConvexHull
from manim import (
    BLUE_E,
    DOWN,
    RIGHT,
    Axes,
    Create,
    Dot,
    FadeIn,
    Line,
    ManimColor,
    ParametricFunction,
    Polygon,
    Scene,
    Text,
    ValueTracker,
    VGroup,
    always_redraw,
    config,
)

# Extra-wide, short frame (~32:9).
config.background_color = "#FFFFFF"
config.pixel_width = 1920
config.pixel_height = 540
_FRAME_W = 20.0
config.frame_width = _FRAME_W
config.frame_height = _FRAME_W * (540 / 1920)

FG = ManimColor("#1a1a1a")
CURVE_COLOR_LEFT = ManimColor("#1f77b4")
CURVE_COLOR_RIGHT = ManimColor("#d62728")
POLY_COLOR = ManimColor("#888888")
CTRL_COLOR = BLUE_E


def bezier_point(control_points: np.ndarray, t: float) -> np.ndarray:
    """Evaluate a Bezier curve of degree n = len(control_points) - 1 at t."""
    n = len(control_points) - 1
    point = np.zeros(3)
    for i, p in enumerate(control_points):
        point += comb(n, i) * (1 - t) ** (n - i) * t**i * p
    return point


def _phase_pts(x_start: float, x_end: float, ys: list[float]) -> np.ndarray:
    """Build 8 control points with x evenly spaced in [x_start, x_end]."""
    assert len(ys) == 8
    xs = np.linspace(x_start, x_end, 8)
    return np.stack([xs, np.array(ys), np.zeros(8)], axis=1)


def _make_curve(pts: np.ndarray, color: ManimColor) -> ParametricFunction:
    return ParametricFunction(
        lambda t: bezier_point(pts, t),
        t_range=[0, 1],
        color=color,
        stroke_width=5,
    )


def _convex_hull(pts: np.ndarray, color: ManimColor) -> Polygon:
    hull = ConvexHull(pts[:, :2])
    verts = [np.array([pts[i, 0], pts[i, 1], 0.0]) for i in hull.vertices]
    return Polygon(
        *verts,
        color=color,
        stroke_width=1.5,
        stroke_opacity=0.6,
        fill_color=color,
        fill_opacity=0.12,
    )


def _dots(left: np.ndarray, right: np.ndarray) -> VGroup:
    all_pts = np.vstack([left[:-1], right])  # skip duplicate middle
    return VGroup(*[Dot(p, color=CTRL_COLOR, radius=0.09) for p in all_pts])


def _polygon(left: np.ndarray, right: np.ndarray) -> VGroup:
    return VGroup(
        *[Line(left[i], left[i + 1], color=POLY_COLOR, stroke_width=2)
          for i in range(len(left) - 1)],
        *[Line(right[i], right[i + 1], color=POLY_COLOR, stroke_width=2)
          for i in range(len(right) - 1)],
    )


def _hulls(left: np.ndarray, right: np.ndarray) -> VGroup:
    return VGroup(
        _convex_hull(left, CURVE_COLOR_LEFT),
        _convex_hull(right, CURVE_COLOR_RIGHT),
    )


def _curves(left: np.ndarray, right: np.ndarray) -> VGroup:
    return VGroup(
        _make_curve(left, CURVE_COLOR_LEFT),
        _make_curve(right, CURVE_COLOR_RIGHT),
    )


class Bezier7Showcase(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-9, 9, 1],
            y_range=[-2.2, 2.2, 1],
            x_length=18,
            y_length=4.4,
            tips=False,
            axis_config={"color": FG, "stroke_opacity": 0.35},
        )
        x_label = Text("τ", color=FG, font="STIX Two Math", slant="ITALIC").scale(0.5)
        x_label.next_to(axes.x_axis.get_right(), DOWN, buff=0.15)
        y_label = Text("B(τ)", color=FG, font="STIX Two Math", slant="ITALIC").scale(0.5)
        y_label.next_to(axes.y_axis.get_top(), RIGHT, buff=0.15)
        self.play(FadeIn(axes), FadeIn(x_label), FadeIn(y_label))

        # Initial control points. Shared middle point: left[-1] == right[0].
        # Loop boundary has matching y (-1.0) and zero slope on both sides.
        # Shared middle y-value varies across stretches (left[-1].y == right[0].y
        # to preserve C0 continuity at the phase transition).
        left_initial = _phase_pts(-8.0, 0.0,
            [-1.0, -1.0, 1.0, 1.8, 1.5, 0.5, 0.1, 0.4])
        right_initial = _phase_pts(0.0, 8.0,
            [0.4, 0.0, -0.5, -1.0, -1.2, -1.2, -1.0, -1.0])

        left_s1 = _phase_pts(-8.0, 0.0,
            [-1.0, -1.0, -0.2, 2.1, 2.1, 1.8, 1.4, 1.3])
        right_s1 = _phase_pts(0.0, 8.0,
            [1.3, 0.2, -1.0, -1.8, -2.1, -1.9, -1.0, -1.0])

        left_s2 = _phase_pts(-8.0, 0.0,
            [-1.0, -1.0, 2.1, 1.3, -0.8, 0.5, -0.6, -0.9])
        right_s2 = _phase_pts(0.0, 8.0,
            [-0.9, 1.5, -0.3, -1.2, -2.1, -2.1, -1.0, -1.0])

        # Phase A: static reveal of initial configuration.
        dots_s = _dots(left_initial, right_initial)
        poly_s = _polygon(left_initial, right_initial)
        hulls_s = _hulls(left_initial, right_initial)
        curves_s = _curves(left_initial, right_initial)

        self.play(FadeIn(dots_s), FadeIn(poly_s), run_time=0.4)
        self.wait(0.2)
        self.play(FadeIn(hulls_s), run_time=0.5)
        self.wait(0.2)
        self.play(Create(curves_s), run_time=0.5)
        self.wait(0.3)

        # Phase B: swap to always_redraw versions for the morphing animations,
        # so the convex hull is recomputed from scratch every frame rather
        # than interpolating Polygon vertices (which causes the "rotation"
        # artifact when hull vertex ordering changes).
        self.remove(dots_s, poly_s, hulls_s, curves_s)

        state = {
            "src_l": left_initial.copy(), "dst_l": left_initial.copy(),
            "src_r": right_initial.copy(), "dst_r": right_initial.copy(),
        }
        alpha = ValueTracker(1.0)

        def cur():
            a = alpha.get_value()
            L = state["src_l"] * (1 - a) + state["dst_l"] * a
            R = state["src_r"] * (1 - a) + state["dst_r"] * a
            return L, R

        dyn_dots = always_redraw(lambda: _dots(*cur()))
        dyn_poly = always_redraw(lambda: _polygon(*cur()))
        dyn_hulls = always_redraw(lambda: _hulls(*cur()))
        dyn_curves = always_redraw(lambda: _curves(*cur()))
        self.add(dyn_hulls, dyn_poly, dyn_curves, dyn_dots)

        def morph_to(new_l: np.ndarray, new_r: np.ndarray, run_time: float) -> None:
            state["src_l"] = state["dst_l"].copy()
            state["src_r"] = state["dst_r"].copy()
            state["dst_l"] = new_l.copy()
            state["dst_r"] = new_r.copy()
            alpha.set_value(0.0)
            self.play(alpha.animate.set_value(1.0), run_time=run_time)

        morph_to(left_s1, right_s1, 2.0)
        self.wait(0.5)
        morph_to(left_s2, right_s2, 2.0)
        self.wait(0.8)
