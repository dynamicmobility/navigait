"""Manim animation showing the Bezier transition used in set_gait
(control/gait.py).

Two 7th-degree Bezier curves B1 (bottom) and B2 (top) are shown on a shared
τ axis. At τ_split both curves are split via de Casteljau (matching
control/bezier.py: split). A new transition Bezier is then constructed on
[τ_split, 1] whose first `deg=3` control points are inherited from B1's
right-half split and last `deg=3` are inherited from B2's right-half split
(with the middle points averaged) — mirroring P1Bezier.interpolate. By
construction it matches position and the first two derivatives at both
boundaries: B1(τ_split) and B2(1).
"""

from __future__ import annotations

from math import comb

import numpy as np
from manim import (
    BLUE_E,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Axes,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    ManimColor,
    ParametricFunction,
    Scene,
    Text,
    Transform,
    VGroup,
    config,
)

config.background_color = "#FFFFFF"
config.pixel_width = 1920
config.pixel_height = 1080
_FRAME_W = 14.0
config.frame_width = _FRAME_W
config.frame_height = _FRAME_W * (1080 / 1920)

FG = ManimColor("#1a1a1a")
B1_COLOR = ManimColor("#1f77b4")
B2_COLOR = ManimColor("#d62728")
TRANS_COLOR = ManimColor("#9467bd")
POLY_COLOR = ManimColor("#888888")
CTRL_COLOR = BLUE_E


# ---------- bezier math ---------- #

def bezier_eval(ys: np.ndarray, u: float) -> float:
    """Scalar Bezier evaluation at u ∈ [0,1]."""
    n = len(ys) - 1
    return float(sum(comb(n, i) * (1 - u) ** (n - i) * u**i * ys[i]
                     for i in range(n + 1)))


def de_casteljau_split(ys: np.ndarray, z: float):
    """Split a scalar Bezier (given by control y-values) at z ∈ [0,1].

    Returns (left_ys, right_ys), each of the same length as ys. The right
    half reparameterized on u ∈ [0,1] represents the original curve over
    [z, 1]; likewise left over [0, z].
    """
    n = len(ys) - 1
    levels = [ys.astype(float).copy()]
    for _ in range(n):
        prev = levels[-1]
        levels.append((1 - z) * prev[:-1] + z * prev[1:])
    left = np.array([levels[k][0] for k in range(n + 1)])
    right = np.array([levels[n - k][k] for k in range(n + 1)])
    return left, right


def interpolate_transition(right_B1: np.ndarray,
                           right_B2: np.ndarray,
                           deg: int = 3) -> np.ndarray:
    """Mirror of P1Bezier.interpolate: keep first `deg` of start, last `deg`
    of end, average the middle."""
    mid = 0.5 * (right_B1[deg:-deg] + right_B2[deg:-deg])
    return np.concatenate([right_B1[:deg], mid, right_B2[-deg:]])


# ---------- manim helpers ---------- #

def _curve_mobject(axes: Axes, ys: np.ndarray,
                   t_start: float, t_end: float,
                   color: ManimColor, stroke_width: float = 5.0,
                   dashed: bool = False) -> ParametricFunction:
    def fn(s):
        # s ∈ [t_start, t_end] → u ∈ [0,1]
        u = (s - t_start) / (t_end - t_start)
        return axes.c2p(s, bezier_eval(ys, u))

    return ParametricFunction(
        fn, t_range=[t_start, t_end], color=color,
        stroke_width=stroke_width,
    )


def _control_points_mobject(axes: Axes, ys: np.ndarray,
                            t_start: float, t_end: float,
                            color: ManimColor) -> tuple[VGroup, VGroup]:
    """Returns (dots, polygon-lines)."""
    n = len(ys) - 1
    xs = np.linspace(t_start, t_end, n + 1)
    pts = [axes.c2p(x, y) for x, y in zip(xs, ys)]
    dots = VGroup(*[Dot(p, color=color, radius=0.07) for p in pts])
    lines = VGroup(*[
        Line(pts[i], pts[i + 1], color=POLY_COLOR, stroke_width=2)
        for i in range(n)
    ])
    return dots, lines


_SUPERS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUBS = "₀₁₂₃₄₅₆₇₈₉"


def _sup(i: int) -> str:
    return "".join(_SUPERS[int(d)] for d in str(i))


def _sub(i: int) -> str:
    return "".join(_SUBS[int(d)] for d in str(i))


def _cp_labels(axes: Axes, ys: np.ndarray,
               t_start: float, t_end: float,
               subscript, hat: bool,
               color: ManimColor, direction) -> VGroup:
    """Label each control point with α_{subscript}^{i} (or α̂ if hat=True).

    `subscript` is either an int (rendered via unicode subscript digits) or a
    pre-formatted string (used verbatim, e.g. "₁→₂")."""
    base = "α̂" if hat else "α"
    sub_str = _sub(subscript) if isinstance(subscript, int) else subscript
    n = len(ys) - 1
    xs = np.linspace(t_start, t_end, n + 1)
    labels = VGroup()
    for i, (x, y) in enumerate(zip(xs, ys)):
        txt = f"{base}{sub_str}{_sup(i + 1)}"
        t = Text(txt, color=color).scale(0.3)
        t.next_to(axes.c2p(x, y), direction, buff=0.08)
        labels.add(t)
    return labels


# ---------- scene ---------- #

class BezierTransition(Scene):
    def construct(self):
        # Axes
        axes = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-2.2, 2.2, 1],
            x_length=10,
            y_length=5,
            tips=False,
            axis_config={"color": FG, "stroke_opacity": 0.35},
        )
        x_label = Text("τ", color=FG, slant="ITALIC").scale(0.5)
        x_label.next_to(axes.x_axis.get_right(), DOWN, buff=0.15)
        y_label = Text("B(τ)", color=FG, slant="ITALIC").scale(0.5)
        y_label.next_to(axes.y_axis.get_top(), RIGHT, buff=0.15)
        self.play(FadeIn(axes), FadeIn(x_label), FadeIn(y_label))

        # Two 7th-degree Bezier curves — control y-values only.
        # B1 sits lower, B2 sits higher; same τ range [0, 1].
        b1_ys = np.array([-1.5, -1.3, -0.4, -0.2, -0.6, -1.1, -1.4, -1.3])
        b2_ys = np.array([0.8, 1.6, 1.9, 1.3, 0.9, 1.3, 1.7, 1.5])

        tau_split = 0.45

        # --- Phase A: draw B1 and B2 in full --- #
        b1_dots, b1_poly = _control_points_mobject(axes, b1_ys, 0.0, 1.0, B1_COLOR)
        b2_dots, b2_poly = _control_points_mobject(axes, b2_ys, 0.0, 1.0, B2_COLOR)
        b1_curve = _curve_mobject(axes, b1_ys, 0.0, 1.0, B1_COLOR)
        b2_curve = _curve_mobject(axes, b2_ys, 0.0, 1.0, B2_COLOR)

        b1_label = Text("B₁", color=B1_COLOR).scale(0.55)
        b1_label.next_to(axes.c2p(1.0, bezier_eval(b1_ys, 1.0)), RIGHT, buff=0.2)
        b2_label = Text("B₂", color=B2_COLOR).scale(0.55)
        b2_label.next_to(axes.c2p(1.0, bezier_eval(b2_ys, 1.0)), RIGHT, buff=0.2)

        self.play(FadeIn(b1_poly), FadeIn(b1_dots),
                  FadeIn(b2_poly), FadeIn(b2_dots), run_time=0.6)
        self.play(Create(b1_curve), Create(b2_curve), run_time=1.0)
        self.play(FadeIn(b1_label), FadeIn(b2_label), run_time=0.3)

        b1_cp_labels = _cp_labels(axes, b1_ys, 0.0, 1.0,
                                  subscript=1, hat=False,
                                  color=B1_COLOR, direction=DOWN)
        b2_cp_labels = _cp_labels(axes, b2_ys, 0.0, 1.0,
                                  subscript=2, hat=False,
                                  color=B2_COLOR, direction=UP)
        self.play(FadeIn(b1_cp_labels), FadeIn(b2_cp_labels), run_time=0.6)
        self.wait(0.4)

        # --- Phase B: show τ_split --- #
        split_line = DashedLine(
            axes.c2p(tau_split, -2.2), axes.c2p(tau_split, 2.2),
            color=FG, stroke_opacity=0.6, dash_length=0.12,
        )
        split_label = Text("τ_split", color=FG).scale(0.45)
        split_label.next_to(axes.c2p(tau_split, -2.2), DOWN, buff=0.15)
        self.play(Create(split_line), FadeIn(split_label), run_time=0.6)
        self.wait(0.3)

        # --- Phase C: split both curves at τ_split --- #
        b1_left_ys, b1_right_ys = de_casteljau_split(b1_ys, tau_split)
        b2_left_ys, b2_right_ys = de_casteljau_split(b2_ys, tau_split)

        # Build split-half mobjects
        b1_left_dots, b1_left_poly = _control_points_mobject(
            axes, b1_left_ys, 0.0, tau_split, B1_COLOR)
        b1_right_dots, b1_right_poly = _control_points_mobject(
            axes, b1_right_ys, tau_split, 1.0, B1_COLOR)
        b2_left_dots, b2_left_poly = _control_points_mobject(
            axes, b2_left_ys, 0.0, tau_split, B2_COLOR)
        b2_right_dots, b2_right_poly = _control_points_mobject(
            axes, b2_right_ys, tau_split, 1.0, B2_COLOR)

        b1_left_curve = _curve_mobject(axes, b1_left_ys, 0.0, tau_split, B1_COLOR)
        b1_right_curve = _curve_mobject(axes, b1_right_ys, tau_split, 1.0, B1_COLOR)
        b2_left_curve = _curve_mobject(axes, b2_left_ys, 0.0, tau_split, B2_COLOR)
        b2_right_curve = _curve_mobject(axes, b2_right_ys, tau_split, 1.0, B2_COLOR)

        # Replace original curves/CPs with split versions.
        self.play(
            FadeOut(b1_curve), FadeOut(b2_curve),
            FadeOut(b1_poly), FadeOut(b2_poly),
            FadeOut(b1_dots), FadeOut(b2_dots),
            FadeOut(b1_cp_labels), FadeOut(b2_cp_labels),
            FadeIn(b1_left_curve), FadeIn(b1_right_curve),
            FadeIn(b2_left_curve), FadeIn(b2_right_curve),
            FadeIn(b1_left_poly), FadeIn(b1_right_poly),
            FadeIn(b2_left_poly), FadeIn(b2_right_poly),
            FadeIn(b1_left_dots), FadeIn(b1_right_dots),
            FadeIn(b2_left_dots), FadeIn(b2_right_dots),
            run_time=1.0,
        )
        self.wait(0.6)

        # Labels for the pre-split halves: α̂. Appear AFTER the split has
        # visibly settled.
        b1_right_labels = _cp_labels(axes, b1_right_ys, tau_split, 1.0,
                                     subscript=1, hat=True,
                                     color=B1_COLOR, direction=DOWN)
        b2_right_labels = _cp_labels(axes, b2_right_ys, tau_split, 1.0,
                                     subscript=2, hat=True,
                                     color=B2_COLOR, direction=UP)
        self.play(FadeIn(b1_right_labels), run_time=0.5)
        self.play(FadeIn(b2_right_labels), run_time=0.5)
        self.wait(0.4)

        # --- Phase D: build transition Bezier --- #
        trans_ys = interpolate_transition(b1_right_ys, b2_right_ys, deg=3)
        trans_dots, trans_poly = _control_points_mobject(
            axes, trans_ys, tau_split, 1.0, TRANS_COLOR)
        trans_curve = _curve_mobject(axes, trans_ys, tau_split, 1.0, TRANS_COLOR)

        trans_label = Text("transition", color=TRANS_COLOR).scale(0.45)
        trans_label.next_to(axes.c2p(tau_split, 2.2), UP, buff=0.1)

        self.play(FadeIn(trans_poly), FadeIn(trans_dots),
                  FadeIn(trans_label), run_time=0.8)
        self.wait(0.2)
        self.play(Create(trans_curve), run_time=1.2)
        self.wait(0.6)

        # --- Phase E: drop the scaffolding polygons/dots and α̂ labels
        # instantly (so they don't visibly linger during any fade), then dim
        # the remaining scaffolding curves so they're still faintly visible.
        #
        # IMPORTANT: use set_stroke(opacity=...) on the curves rather than
        # set_opacity(...). ParametricFunction is a VMobject with a fill path
        # (implicitly closing to the origin). set_opacity() drops fill_opacity
        # to 0.2 too, which renders as a translucent "shaded region" under the
        # curve — not what we want. --- #
        self.remove(
            b1_right_poly, b1_right_dots,
            b2_left_poly, b2_left_dots,
            b2_right_poly, b2_right_dots,
            b1_right_labels, b2_right_labels,
        )
        dim_curves = VGroup(b1_right_curve, b2_left_curve, b2_right_curve)
        self.play(
            dim_curves.animate.set_stroke(opacity=0.2),
            b2_label.animate.set_opacity(0.2),
            run_time=1.0,
        )
        self.wait(0.3)

        trans_labels = _cp_labels(
            axes, trans_ys, tau_split, 1.0,
            subscript="₁→₂", hat=False,
            color=TRANS_COLOR, direction=UP,
        )
        self.play(FadeIn(trans_labels), run_time=0.6)
        self.wait(1.0)
