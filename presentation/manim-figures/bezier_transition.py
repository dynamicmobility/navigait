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
    ValueTracker,
    VGroup,
    always_redraw,
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


# ---------- shared world ---------- #
#
# All static mobjects used across the phases are built once by
# `_BezierWorld.setup_world` and stashed on self. Each phase Scene picks
# which mobjects to `self.add` (as the static "starting state" inherited
# from the previous phase) and which to animate into existence. The end
# state of phase N visually equals the start state of phase N+1, so the
# per-phase MP4s stitch seamlessly when advanced as reveal.js fragments.


B1_YS = np.array([-1.5, -1.3, -0.4, -0.2, -0.6, -1.1, -1.4, -1.3])
B2_YS = np.array([0.8, 1.6, 1.9, 1.3, 0.9, 1.3, 1.7, 1.5])
TAU_SPLIT = 0.45


class _BezierWorld:
    """Mixin for Scene subclasses. Builds every mobject used by any phase."""

    def setup_world(self):
        b1_ys, b2_ys, tau_split = B1_YS, B2_YS, TAU_SPLIT
        self.b1_ys, self.b2_ys, self.tau_split = b1_ys, b2_ys, tau_split

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

        b1_dots, b1_poly = _control_points_mobject(axes, b1_ys, 0.0, 1.0, B1_COLOR)
        b2_dots, b2_poly = _control_points_mobject(axes, b2_ys, 0.0, 1.0, B2_COLOR)
        b1_curve = _curve_mobject(axes, b1_ys, 0.0, 1.0, B1_COLOR)
        b2_curve = _curve_mobject(axes, b2_ys, 0.0, 1.0, B2_COLOR)

        b1_label = Text("B₁", color=B1_COLOR).scale(0.55)
        b1_label.next_to(axes.c2p(1.0, bezier_eval(b1_ys, 1.0)), RIGHT, buff=0.2)
        b2_label = Text("B₂", color=B2_COLOR).scale(0.55)
        b2_label.next_to(axes.c2p(1.0, bezier_eval(b2_ys, 1.0)), RIGHT, buff=0.2)

        b1_cp_labels = _cp_labels(axes, b1_ys, 0.0, 1.0,
                                  subscript=1, hat=False,
                                  color=B1_COLOR, direction=DOWN)
        b2_cp_labels = _cp_labels(axes, b2_ys, 0.0, 1.0,
                                  subscript=2, hat=False,
                                  color=B2_COLOR, direction=UP)

        split_line = DashedLine(
            axes.c2p(tau_split, -2.2), axes.c2p(tau_split, 2.2),
            color=FG, stroke_opacity=0.6, dash_length=0.12,
        )
        split_label = Text("τ_split", color=FG).scale(0.45)
        split_label.next_to(axes.c2p(tau_split, -2.2), DOWN, buff=0.15)

        b1_left_ys, b1_right_ys = de_casteljau_split(b1_ys, tau_split)
        b2_left_ys, b2_right_ys = de_casteljau_split(b2_ys, tau_split)
        self.b1_left_ys, self.b1_right_ys = b1_left_ys, b1_right_ys
        self.b2_left_ys, self.b2_right_ys = b2_left_ys, b2_right_ys

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

        b1_right_labels = _cp_labels(axes, b1_right_ys, tau_split, 1.0,
                                     subscript=1, hat=True,
                                     color=B1_COLOR, direction=DOWN)
        b2_right_labels = _cp_labels(axes, b2_right_ys, tau_split, 1.0,
                                     subscript=2, hat=True,
                                     color=B2_COLOR, direction=UP)

        trans_ys = interpolate_transition(b1_right_ys, b2_right_ys, deg=3)
        self.trans_ys = trans_ys
        trans_dots, trans_poly = _control_points_mobject(
            axes, trans_ys, tau_split, 1.0, TRANS_COLOR)
        trans_curve = _curve_mobject(axes, trans_ys, tau_split, 1.0, TRANS_COLOR)
        trans_label = Text("transition", color=TRANS_COLOR).scale(0.45)
        trans_label.next_to(axes.c2p(tau_split, 2.2), UP, buff=0.1)
        trans_labels = _cp_labels(
            axes, trans_ys, tau_split, 1.0,
            subscript="₁→₂", hat=False,
            color=TRANS_COLOR, direction=UP,
        )

        # Stash everything on self.
        for k, v in dict(
            axes=axes, x_label=x_label, y_label=y_label,
            b1_dots=b1_dots, b1_poly=b1_poly, b1_curve=b1_curve,
            b2_dots=b2_dots, b2_poly=b2_poly, b2_curve=b2_curve,
            b1_label=b1_label, b2_label=b2_label,
            b1_cp_labels=b1_cp_labels, b2_cp_labels=b2_cp_labels,
            split_line=split_line, split_label=split_label,
            b1_left_dots=b1_left_dots, b1_left_poly=b1_left_poly,
            b1_left_curve=b1_left_curve,
            b1_right_dots=b1_right_dots, b1_right_poly=b1_right_poly,
            b1_right_curve=b1_right_curve,
            b2_left_dots=b2_left_dots, b2_left_poly=b2_left_poly,
            b2_left_curve=b2_left_curve,
            b2_right_dots=b2_right_dots, b2_right_poly=b2_right_poly,
            b2_right_curve=b2_right_curve,
            b1_right_labels=b1_right_labels,
            b2_right_labels=b2_right_labels,
            trans_dots=trans_dots, trans_poly=trans_poly,
            trans_curve=trans_curve, trans_label=trans_label,
            trans_labels=trans_labels,
        ).items():
            setattr(self, k, v)

    # Each `add_end_of_*` installs (without animation) the visual state at
    # the end of the named phase, so the next phase's video begins exactly
    # where the previous one finished.

    def add_axes(self):
        self.add(self.axes, self.x_label, self.y_label)

    def add_end_of_a(self):
        self.add_axes()
        self.add(self.b1_poly, self.b1_dots, self.b1_curve,
                 self.b2_poly, self.b2_dots, self.b2_curve,
                 self.b1_label, self.b2_label,
                 self.b1_cp_labels, self.b2_cp_labels)

    def add_end_of_b(self):
        self.add_end_of_a()
        self.add(self.split_line, self.split_label)

    def add_end_of_c(self):
        self.add_axes()
        self.add(self.split_line, self.split_label,
                 self.b1_label, self.b2_label,
                 self.b1_left_curve, self.b1_left_poly, self.b1_left_dots,
                 self.b1_right_curve, self.b1_right_poly, self.b1_right_dots,
                 self.b2_left_curve, self.b2_left_poly, self.b2_left_dots,
                 self.b2_right_curve, self.b2_right_poly, self.b2_right_dots,
                 self.b1_right_labels, self.b2_right_labels)

    def add_end_of_d(self):
        self.add_end_of_c()
        self.add(self.trans_poly, self.trans_dots, self.trans_curve,
                 self.trans_label)

    def add_end_of_e(self):
        # End of E: scaffolding polys/dots and α̂ labels are gone; the
        # non-trajectory half-curves and B₂ label are dimmed.
        self.b1_right_curve.set_stroke(opacity=0.2)
        self.b2_left_curve.set_stroke(opacity=0.2)
        self.b2_right_curve.set_stroke(opacity=0.2)
        self.b2_label.set_opacity(0.2)
        self.add_axes()
        self.add(self.split_line, self.split_label,
                 self.b1_label, self.b2_label,
                 self.b1_left_curve, self.b1_left_poly, self.b1_left_dots,
                 self.b1_right_curve,
                 self.b2_left_curve, self.b2_right_curve,
                 self.trans_poly, self.trans_dots, self.trans_curve,
                 self.trans_label, self.trans_labels)


# ---------- monolithic scene (renders the whole thing in one MP4) ---------- #

class BezierTransition(_BezierWorld, Scene):
    def construct(self):
        self.setup_world()
        self.add_axes()

        _play_phase_a(self)
        _play_phase_b(self)
        _play_phase_c(self)
        _play_phase_d(self)
        _play_phase_e(self)
        _play_phase_f(self)


# ---------- per-phase animation helpers ---------- #
#
# These are module-level so both the monolithic `BezierTransition` scene
# and the per-phase scenes below run the exact same animations.


def _play_phase_a(s: "_BezierWorld"):
    s.play(FadeIn(s.b1_poly), FadeIn(s.b1_dots),
           FadeIn(s.b2_poly), FadeIn(s.b2_dots), run_time=0.6)
    s.play(Create(s.b1_curve), Create(s.b2_curve), run_time=1.0)
    s.play(FadeIn(s.b1_label), FadeIn(s.b2_label), run_time=0.3)
    s.play(FadeIn(s.b1_cp_labels), FadeIn(s.b2_cp_labels), run_time=0.6)
    s.wait(0.4)


def _play_phase_b(s: "_BezierWorld"):
    s.play(Create(s.split_line), FadeIn(s.split_label), run_time=0.6)
    s.wait(0.3)


def _play_phase_c(s: "_BezierWorld"):
    s.play(
        FadeOut(s.b1_curve), FadeOut(s.b2_curve),
        FadeOut(s.b1_poly), FadeOut(s.b2_poly),
        FadeOut(s.b1_dots), FadeOut(s.b2_dots),
        FadeOut(s.b1_cp_labels), FadeOut(s.b2_cp_labels),
        FadeIn(s.b1_left_curve), FadeIn(s.b1_right_curve),
        FadeIn(s.b2_left_curve), FadeIn(s.b2_right_curve),
        FadeIn(s.b1_left_poly), FadeIn(s.b1_right_poly),
        FadeIn(s.b2_left_poly), FadeIn(s.b2_right_poly),
        FadeIn(s.b1_left_dots), FadeIn(s.b1_right_dots),
        FadeIn(s.b2_left_dots), FadeIn(s.b2_right_dots),
        run_time=1.0,
    )
    s.wait(0.6)
    s.play(FadeIn(s.b1_right_labels), run_time=0.5)
    s.play(FadeIn(s.b2_right_labels), run_time=0.5)
    s.wait(0.4)


def _play_phase_d(s: "_BezierWorld"):
    s.play(FadeIn(s.trans_poly), FadeIn(s.trans_dots),
           FadeIn(s.trans_label), run_time=0.8)
    s.wait(0.2)
    s.play(Create(s.trans_curve), run_time=1.2)
    s.wait(0.6)


def _play_phase_e(s: "_BezierWorld"):
    # Drop scaffolding polys/dots and α̂ labels instantly.
    #
    # IMPORTANT: use set_stroke(opacity=...) on ParametricFunction, not
    # set_opacity(...) — the latter also drops fill_opacity and renders a
    # translucent "shaded region" under the curve.
    s.remove(
        s.b1_right_poly, s.b1_right_dots,
        s.b2_left_poly, s.b2_left_dots,
        s.b2_right_poly, s.b2_right_dots,
        s.b1_right_labels, s.b2_right_labels,
    )
    dim_curves = VGroup(s.b1_right_curve, s.b2_left_curve, s.b2_right_curve)
    s.play(
        dim_curves.animate.set_stroke(opacity=0.2),
        s.b2_label.animate.set_opacity(0.2),
        run_time=1.0,
    )
    s.wait(0.3)
    s.play(FadeIn(s.trans_labels), run_time=0.6)
    s.wait(1.0)


def _play_phase_f(s: "_BezierWorld"):
    axes = s.axes
    b1_ys, b2_ys, tau_split = s.b1_ys, s.b2_ys, s.tau_split
    tracker = ValueTracker(tau_split)

    def _dyn_split_line() -> DashedLine:
        u = tracker.get_value()
        return DashedLine(
            axes.c2p(u, -2.2), axes.c2p(u, 2.2),
            color=FG, stroke_opacity=0.6, dash_length=0.12,
        )

    def _dyn_split_label() -> Text:
        u = tracker.get_value()
        t = Text("τ_split", color=FG).scale(0.45)
        t.next_to(axes.c2p(u, -2.2), DOWN, buff=0.15)
        return t

    def _dyn_b1_left() -> ParametricFunction:
        u = tracker.get_value()
        left_ys, _ = de_casteljau_split(b1_ys, u)
        return _curve_mobject(axes, left_ys, 0.0, u, B1_COLOR)

    def _dyn_dim_curve(full_ys: np.ndarray, color: ManimColor,
                       side: str) -> ParametricFunction:
        u = tracker.get_value()
        left_ys, right_ys = de_casteljau_split(full_ys, u)
        if side == "left":
            c = _curve_mobject(axes, left_ys, 0.0, u, color)
        else:
            c = _curve_mobject(axes, right_ys, u, 1.0, color)
        c.set_stroke(opacity=0.2)
        return c

    def _trans_ys_now() -> np.ndarray:
        u = tracker.get_value()
        _, b1_r = de_casteljau_split(b1_ys, u)
        _, b2_r = de_casteljau_split(b2_ys, u)
        return interpolate_transition(b1_r, b2_r, deg=3)

    def _dyn_trans_curve() -> ParametricFunction:
        u = tracker.get_value()
        return _curve_mobject(axes, _trans_ys_now(), u, 1.0, TRANS_COLOR)

    def _dyn_trans_dots() -> VGroup:
        u = tracker.get_value()
        dots, _poly = _control_points_mobject(
            axes, _trans_ys_now(), u, 1.0, TRANS_COLOR)
        return dots

    def _dyn_trans_poly() -> VGroup:
        u = tracker.get_value()
        _dots, poly = _control_points_mobject(
            axes, _trans_ys_now(), u, 1.0, TRANS_COLOR)
        return poly

    def _dyn_trans_label() -> Text:
        u = tracker.get_value()
        t = Text("transition", color=TRANS_COLOR).scale(0.45)
        t.next_to(axes.c2p(u, 2.2), UP, buff=0.1)
        return t

    def _dyn_trans_labels() -> VGroup:
        u = tracker.get_value()
        return _cp_labels(
            axes, _trans_ys_now(), u, 1.0,
            subscript="₁→₂", hat=False,
            color=TRANS_COLOR, direction=UP,
        )

    s.remove(
        s.split_line, s.split_label,
        s.b1_left_curve,
        s.b1_right_curve, s.b2_left_curve, s.b2_right_curve,
        s.trans_curve, s.trans_poly, s.trans_dots,
        s.trans_label, s.trans_labels,
    )

    dyn_split_line = always_redraw(_dyn_split_line)
    dyn_split_label = always_redraw(_dyn_split_label)
    dyn_b1_left = always_redraw(_dyn_b1_left)
    dyn_b1_right = always_redraw(
        lambda: _dyn_dim_curve(b1_ys, B1_COLOR, "right"))
    dyn_b2_left = always_redraw(
        lambda: _dyn_dim_curve(b2_ys, B2_COLOR, "left"))
    dyn_b2_right = always_redraw(
        lambda: _dyn_dim_curve(b2_ys, B2_COLOR, "right"))
    dyn_trans_curve = always_redraw(_dyn_trans_curve)
    dyn_trans_poly = always_redraw(_dyn_trans_poly)
    dyn_trans_dots = always_redraw(_dyn_trans_dots)
    dyn_trans_label = always_redraw(_dyn_trans_label)
    dyn_trans_labels = always_redraw(_dyn_trans_labels)

    s.add(
        dyn_split_line, dyn_split_label,
        dyn_b1_right, dyn_b2_left, dyn_b2_right,
        dyn_b1_left,
        dyn_trans_poly, dyn_trans_dots, dyn_trans_curve,
        dyn_trans_label, dyn_trans_labels,
    )

    # Clamp away from degenerate endpoints so de Casteljau stays
    # well-conditioned and labels don't overlap.
    s.play(tracker.animate.set_value(0.20), run_time=2.5)
    s.wait(0.4)
    s.play(tracker.animate.set_value(0.75), run_time=3.0)
    s.wait(0.4)
    s.play(tracker.animate.set_value(0.45), run_time=2.0)
    s.wait(1.0)


# ---------- per-phase scenes (each renders a separate MP4) ---------- #
#
# Used as reveal.js fragments in `presentation/slides/gaitlib.qmd`.
# Each scene's first frame matches the previous scene's last frame.


class BezierPhaseA(_BezierWorld, Scene):
    def construct(self):
        self.setup_world()
        self.add_axes()
        _play_phase_a(self)


class BezierPhaseB(_BezierWorld, Scene):
    def construct(self):
        self.setup_world()
        self.add_end_of_a()
        _play_phase_b(self)


class BezierPhaseC(_BezierWorld, Scene):
    def construct(self):
        self.setup_world()
        self.add_end_of_b()
        _play_phase_c(self)


class BezierPhaseD(_BezierWorld, Scene):
    def construct(self):
        self.setup_world()
        self.add_end_of_c()
        _play_phase_d(self)


class BezierPhaseE(_BezierWorld, Scene):
    def construct(self):
        self.setup_world()
        self.add_end_of_d()
        _play_phase_e(self)


class BezierPhaseF(_BezierWorld, Scene):
    def construct(self):
        self.setup_world()
        self.add_end_of_e()
        _play_phase_f(self)
