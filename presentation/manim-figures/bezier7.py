"""7th-degree Bezier polynomial demo.

A 7th-degree Bezier curve is defined by 8 control points P0..P7:
    B(t) = sum_{i=0}^{7} C(7,i) * (1-t)^(7-i) * t^i * P_i,   t in [0, 1]

Scene walks through:
  1. Placing the 8 control points and connecting them with the control polygon.
  2. Drawing the resulting Bezier curve.
  3. "Stretching" the curve by animating several control points to new
     locations so the viewer can see how the curve reshapes itself.
"""

from __future__ import annotations

from math import comb

import numpy as np
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
    Scene,
    Text,
    Transform,
    VGroup,
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
CURVE_COLOR = ManimColor("#1f77b4")
POLY_COLOR = ManimColor("#888888")
CTRL_COLOR = BLUE_E


def bezier_point(control_points: np.ndarray, t: float) -> np.ndarray:
    """Evaluate a Bezier curve of degree n = len(control_points) - 1 at t."""
    n = len(control_points) - 1
    point = np.zeros(3)
    for i, p in enumerate(control_points):
        point += comb(n, i) * (1 - t) ** (n - i) * t**i * p
    return point


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

        # Initial control points (8 total for degree 7).
        initial_pts = np.array(
            [
                [-8.0, -1.4, 0.0],
                [-5.7,  1.4, 0.0],
                [-3.4, -1.2, 0.0],
                [-1.1,  1.7, 0.0],
                [ 1.1, -1.6, 0.0],
                [ 3.4,  1.2, 0.0],
                [ 5.7, -0.8, 0.0],
                [ 8.0,  1.6, 0.0],
            ]
        )

        dots = VGroup(
            *[Dot(pt, color=CTRL_COLOR, radius=0.09) for pt in initial_pts]
        )
        polygon = VGroup(
            *[
                Line(initial_pts[i], initial_pts[i + 1], color=POLY_COLOR, stroke_width=2)
                for i in range(len(initial_pts) - 1)
            ]
        )

        self.play(FadeIn(dots), FadeIn(polygon), run_time=0.4)
        self.wait(0.3)

        def make_curve(pts: np.ndarray, color: ManimColor) -> ParametricFunction:
            return ParametricFunction(
                lambda t: bezier_point(pts, t),
                t_range=[0, 1],
                color=color,
                stroke_width=5,
            )

        curve = make_curve(initial_pts, CURVE_COLOR)
        self.play(Create(curve), run_time=0.5)
        self.wait(0.3)

        # Stretch: move a few control points and redraw the curve + polygon.
        stretched_pts = initial_pts.copy()
        stretched_pts[1] = [-5.7,  2.2, 0.0]
        stretched_pts[3] = [-1.1,  2.4, 0.0]
        stretched_pts[4] = [ 1.1, -2.2, 0.0]
        stretched_pts[6] = [ 6.4, -1.8, 0.0]
        stretched_pts[7] = [ 8.6,  2.1, 0.0]

        new_dots = VGroup(
            *[Dot(pt, color=CTRL_COLOR, radius=0.09) for pt in stretched_pts]
        )
        new_polygon = VGroup(
            *[
                Line(stretched_pts[i], stretched_pts[i + 1], color=POLY_COLOR, stroke_width=2)
                for i in range(len(stretched_pts) - 1)
            ]
        )
        new_curve = make_curve(stretched_pts, CURVE_COLOR)

        self.play(
            Transform(dots, new_dots),
            Transform(polygon, new_polygon),
            Transform(curve, new_curve),
            run_time=2,
        )
        self.wait(0.5)

        # Second stretch: vertical shifts on a few interior control points only.
        stretched2 = stretched_pts.copy()
        stretched2[2] = [-3.4,  1.8, 0.0]
        stretched2[3] = [-1.1, -1.8, 0.0]
        stretched2[5] = [ 3.4, -1.6, 0.0]

        new_dots2 = VGroup(
            *[Dot(pt, color=CTRL_COLOR, radius=0.09) for pt in stretched2]
        )
        new_polygon2 = VGroup(
            *[
                Line(stretched2[i], stretched2[i + 1], color=POLY_COLOR, stroke_width=2)
                for i in range(len(stretched2) - 1)
            ]
        )
        new_curve2 = make_curve(stretched2, CURVE_COLOR)

        self.play(
            Transform(dots, new_dots2),
            Transform(polygon, new_polygon2),
            Transform(curve, new_curve2),
            run_time=2,
        )
        self.wait(0.8)
