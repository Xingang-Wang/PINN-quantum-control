"""
Generate a vector version of Fig. 1 (method overview).

The original manuscript only had a PNG copy.  This script recreates the same
conceptual layout as vector graphics and writes PDF/SVG/PNG copies to
writing/manuscript/figures/.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pinn_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/pinn_xdg_cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#1F4ED8"
GREEN = "#167A28"
RED = "#D62728"
GRAY = "#555555"
LIGHT = "#EDEDED"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8.6,
            "axes.linewidth": 0.7,
            "savefig.bbox": "standard",
            "savefig.pad_inches": 0.0,
        }
    )


def box(ax, xy, wh, edge, face="white", lw=1.1, radius=0.01):
    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color="black", lw=1.2, scale=12):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def panel_label(ax, x, y, label):
    ax.text(x, y, label, fontsize=9.8, fontweight="bold", ha="left", va="top")


def draw_step_pulse(ax, x0, y0, w, h, color=BLUE, smooth=False):
    ax.plot([x0, x0], [y0, y0 + h], color="black", lw=0.8)
    ax.plot([x0, x0 + w], [y0, y0], color="black", lw=0.8)
    if not smooth:
        ax.text(x0 - 0.01, y0 + h * 0.58, r"$\Omega(t)$", ha="right", va="center", fontsize=8)
    ax.text(x0 - 0.002, y0 - 0.02, r"$0$", ha="center", va="top", fontsize=7.5)
    if smooth:
        ax.text(x0 + w, y0 - 0.02, r"$T$", ha="center", va="top", fontsize=7.5)
    if smooth:
        t = np.linspace(0, 1, 250)
        om = 0.26 + 0.22 * np.sin(2 * np.pi * t) * np.exp(-0.9 * t) + 0.10 * np.sin(5 * np.pi * t)
        de = 0.13 + 0.16 * np.sin(2.5 * np.pi * t + 0.25) * np.exp(-0.7 * t)
        ax.plot(x0 + w * t, y0 + h * (0.25 + om), color=BLUE, lw=1.4)
        ax.plot(x0 + w * t, y0 + h * (0.12 + de), color=RED, lw=1.4)
        ax.text(x0 + 0.62 * w, y0 + 0.76 * h, r"$\Omega(t)$", color=BLUE, fontsize=7.5)
        ax.text(x0 + 0.72 * w, y0 + 0.43 * h, r"$\Delta(t)$", color=RED, fontsize=7.5)
    else:
        vals = np.array([0.42, 0.58, 0.46, 0.30, 0.34, 0.23, 0.36, 0.40])
        xs = np.linspace(x0, x0 + w, len(vals) + 1)
        for xx in xs[1:-1]:
            ax.plot([xx, xx], [y0, y0 + h * 0.75], color="#9A9A9A", lw=0.6, ls="--")
        ax.step(xs, np.r_[y0 + h * vals, y0 + h * vals[-1]], where="post", color=color, lw=1.3)
        ax.text(x0 + 0.15 * w, y0 - 0.02, r"$\Delta t$", ha="center", va="top", fontsize=7.5)
        ax.text(x0 + 0.50 * w, y0 - 0.02, r"$\cdots$", ha="center", va="top", fontsize=8)
        ax.text(x0 + w, y0 - 0.02, r"$T=N\Delta t$", ha="right", va="top", fontsize=7.5)


def draw_amplitude_bins(ax, x0, y0, w, h):
    labels = [r"$c_1$", r"$c_2$", r"$c_3$", r"$\cdots$", r"$c_N$"]
    n = len(labels)
    for i, lab in enumerate(labels):
        ax.add_patch(plt.Rectangle((x0 + i * w / n, y0), w / n, h, fill=False, ec=BLUE, lw=0.9))
        ax.text(x0 + (i + 0.5) * w / n, y0 + h / 2, lab, ha="center", va="center", fontsize=8.5)


def draw_bloch(ax, cx, cy, r, with_targets=True, label_states=False):
    """Draw four probe trajectories for a representative correct Z(pi) gate."""

    def project(vec):
        x, y, z = vec
        return cx + r * (0.82 * x - 0.30 * y), cy + r * (z + 0.18 * y)

    def rz(phi, vec):
        x, y, z = vec
        c, s = np.cos(phi), np.sin(phi)
        return np.array([c * x - s * y, s * x + c * y, z])

    ax.add_patch(Circle((cx, cy), r, ec="#AFAFAF", fc="#F7F7F7", lw=0.8))
    th = np.linspace(0, 2 * np.pi, 240)
    equator = np.array([project([np.cos(a), np.sin(a), 0.0]) for a in th])
    meridian_xz = np.array([project([np.cos(a), 0.0, np.sin(a)]) for a in th])
    meridian_yz = np.array([project([0.0, np.cos(a), np.sin(a)]) for a in th])
    ax.plot(equator[:, 0], equator[:, 1], color="#D0D0D0", lw=0.55)
    ax.plot(meridian_xz[:, 0], meridian_xz[:, 1], color="#D8D8D8", lw=0.45)
    ax.plot(meridian_yz[:, 0], meridian_yz[:, 1], color="#D8D8D8", lw=0.45)
    for axis_vec, lab, ha, va, dx, dy in [
        ([1, 0, 0], r"$x$", "left", "center", 0.030, -0.015),
        ([0, 1, 0], r"$y$", "right", "bottom", -0.150, 0.200),
        ([0, 0, 1], r"$z$", "center", "bottom", 0.000, 0.020),
    ]:
        px, py = project(np.array(axis_vec) * 1.10)
        ax.text(px + dx * r, py + dy * r, lab, ha=ha, va=va, fontsize=6.4, color=GRAY)

    probes = [
        (np.array([0.0, 0.0, 1.0]), BLUE),
        (np.array([0.0, 0.0, -1.0]), "#7B2CBF"),
        (np.array([1.0, 0.0, 0.0]), RED),
        (np.array([0.0, 1.0, 0.0]), GREEN),
    ]
    target_shift = np.array([0.020 * r, 0.020 * r])
    for vec, col in probes:
        path = np.array([project(rz(phi, vec)) for phi in np.linspace(0, np.pi, 120)])
        stationary = np.linalg.norm(path[-1] - path[0]) <= 1e-5
        if not stationary:
            ax.plot(path[:, 0], path[:, 1], color=col, lw=1.05, alpha=0.94)
        sx, sy = path[0]
        ex, ey = path[-1]
        if stationary:
            sx = sx - 0.025 * r
            sy = sy + 0.010 * r
            ax.plot([sx, ex], [sy, ey], color=col, lw=0.55, alpha=0.75)
        ax.plot(sx, sy, "o", ms=3.7, color=col, mec="black", mew=0.35, zorder=5)
        ax.plot(ex, ey, "*", ms=6.2, color=col, mec="black", mew=0.35, zorder=6)
        if with_targets:
            ax.plot(
                ex + target_shift[0],
                ey + target_shift[1],
                marker="*",
                ms=6.8,
                mfc="white",
                mec=col,
                mew=0.8,
                zorder=7,
            )

    if label_states:
        labels = [
            ([0, 0, 1], r"$+z$", BLUE, 0.00, 0.16, "center", "bottom"),
            ([0, 0, -1], r"$-z$", "#7B2CBF", 0.00, -0.15, "center", "top"),
            ([1, 0, 0], r"$+x$", RED, 0.13, -0.02, "left", "center"),
            ([-1, 0, 0], r"$-x$", RED, -0.12, 0.02, "right", "center"),
            ([0, 1, 0], r"$+y$", GREEN, -0.10, 0.10, "right", "bottom"),
            ([0, -1, 0], r"$-y$", GREEN, 0.11, -0.10, "left", "top"),
        ]
        for vec, lab, col, dx, dy, ha, va in labels:
            px, py = project(vec)
            ax.text(px + dx * r, py + dy * r, lab, color=col, fontsize=5.5, ha=ha, va=va)


def draw_loss_box(ax, x, y, w, h, continuous=False):
    box(ax, (x, y), (w, h), RED, lw=1.0)
    if continuous:
        eq = r"$L_{\rm total}=\alpha_{\rm dyn}L_{\rm dyn}+\beta_{\rm gate}L_{\rm gate}$" + "\n" + r"$+\lambda_T L_T + L_{\rm reg}$"
        body = (
            r"$L_{\rm dyn}$: Bloch residual at $t_i$" + "\n"
            r"$L_{\rm gate}$: terminal gate loss" + "\n"
            r"$L_T=T$" + "\n"
            r"$L_{\rm reg}$: amplitude, smoothness, boundary"
        )
    else:
        eq = r"$L_{\rm disc}=\beta_{\rm gate}L_{\rm gate}+L_{\rm reg}$"
        body = (
            r"$L_{\rm gate}$: terminal gate loss" + "\n"
            r"$L_{\rm reg}$: pulse regularization"
        )
    ax.text(x + w / 2, y + h * 0.72, eq, color=RED, ha="center", va="center", fontsize=8.2)
    ax.text(x + 0.05 * w, y + h * 0.34, body, ha="left", va="center", fontsize=7.2)


def draw_network(ax, x, y, w, h):
    box(ax, (x, y), (w, h), GREEN, face="#F5FBF5", lw=1.0)
    ax.text(x + w / 2, y + h * 0.90, r"$f_\theta$", ha="center", va="center", fontsize=9)
    layers = [3, 4, 3]
    xs = np.linspace(x + 0.18 * w, x + 0.82 * w, len(layers))
    node_pos = []
    for xi, n in zip(xs, layers):
        ys = np.linspace(y + 0.22 * h, y + 0.72 * h, n)
        node_pos.append([(xi, yi) for yi in ys])
        for yi in ys:
            ax.add_patch(Circle((xi, yi), 0.006, fc="#DCEEDD", ec=GREEN, lw=0.5))
    for left, right in zip(node_pos[:-1], node_pos[1:]):
        for a in left:
            for b in right:
                ax.plot([a[0], b[0]], [a[1], b[1]], color="#888888", lw=0.35, alpha=0.65)


def draw_time_map(ax, x, y, w, h):
    box(ax, (x, y), (w, h), GREEN, face="white", lw=1.0)
    ax.text(x + 0.055 * w, y + 0.72 * h, r"$T=e^\tau$", fontsize=8.6, ha="left", va="center")
    ax.text(x + 0.055 * w, y + 0.43 * h, r"$t_i=s_iT$", fontsize=8.6, ha="left", va="center")
    sx0, sx1 = x + 0.62 * w, x + 0.93 * w
    sy, ty = y + 0.64 * h, y + 0.28 * h
    pts = np.linspace(sx0, sx1, 6)
    ax.plot([sx0, sx1], [sy, sy], color="black", lw=0.8)
    ax.plot([sx0, sx1], [ty, ty], color=GREEN, lw=1.0)
    for px in pts:
        ax.plot(px, sy, "o", color="black", ms=3)
        ax.plot(px, ty, "o", color=GREEN, ms=3)
        ax.plot([px, px], [sy, ty], color=GREEN, lw=0.55, ls="--")
    ax.text(sx0 - 0.012, sy, r"$s_i$", ha="right", va="center", fontsize=7.2)
    ax.text(sx0, sy - 0.036 * h, r"$0$", ha="center", va="top", fontsize=7.0)
    ax.text(sx1, sy - 0.036 * h, r"$1$", ha="center", va="top", fontsize=7.0)
    ax.text(sx0 - 0.012, ty, r"$t_i$", ha="right", va="center", fontsize=7.2)
    ax.text(sx0, ty - 0.036 * h, r"$0$", ha="center", va="top", fontsize=7.0)
    ax.text(sx1, ty - 0.036 * h, r"$T$", ha="center", va="top", fontsize=7.0)


def make_figure() -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.65))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.015, 0.555), (0.970, 0.352), BLUE, lw=1.0)
    box(ax, (0.015, 0.045), (0.970, 0.468), GREEN, lw=1.0)
    ax.text(
        0.500,
        0.890,
        "Traditional discrete optimization",
        ha="center",
        va="center",
        fontsize=9.4,
        fontweight="bold",
        color=BLUE,
    )
    ax.text(
        0.500,
        0.497,
        "Continuous physics-informed representation",
        ha="center",
        va="center",
        fontsize=9.4,
        fontweight="bold",
        color=GREEN,
    )
    panel_label(ax, 0.030, 0.862, "(a)")
    ax.text(0.065, 0.850, r"Pulse amplitudes $\{c_1,c_2,\ldots,c_N\}$", ha="left", va="center", fontsize=7.6)
    draw_amplitude_bins(ax, 0.055, 0.790, 0.180, 0.036)
    draw_step_pulse(ax, 0.072, 0.645, 0.165, 0.122)
    ax.text(0.282, 0.710, "$T$ fixed\nscan", ha="center", va="center", fontsize=6.9)
    arrow(ax, (0.314, 0.720), (0.365, 0.720), scale=11)

    panel_label(ax, 0.335, 0.862, "(b)")
    ax.text(0.440, 0.848, "Propagator", ha="center", va="center", fontsize=8.0)
    draw_bloch(ax, 0.440, 0.715, 0.074, label_states=False)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="black", markersize=3.2, label="Initial"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="black", markersize=4.8, label="Evolved"),
        Line2D([0], [0], marker="*", color="black", markerfacecolor="white", markersize=4.8, label="Target"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.332, 0.562),
              fontsize=5.3, frameon=True, edgecolor="black", handlelength=0.86,
              handletextpad=0.55, borderpad=0.24, labelspacing=0.24,
              borderaxespad=0.0)
    arrow(ax, (0.523, 0.720), (0.603, 0.720))
    box(ax, (0.615, 0.668), (0.270, 0.106), RED, lw=1.0)
    ax.text(0.750, 0.735, r"$L_{\rm disc}=\beta_{\rm gate}L_{\rm gate}+L_{\rm reg}$",
            color=RED, ha="center", va="center", fontsize=8.3)
    ax.text(0.750, 0.699, r"terminal probes + pulse regularization",
            ha="center", va="center", fontsize=7.3)

    panel_label(ax, 0.030, 0.465, "(c)")
    ax.text(0.068, 0.422, r"labels $s_i$", ha="left", va="center", fontsize=7.4)
    ax.plot([0.030, 0.150], [0.396, 0.396], color="black", lw=0.8)
    for px in np.linspace(0.036, 0.144, 6):
        ax.plot(px, 0.396, "o", color="black", ms=3)
    ax.text(0.030, 0.370, r"$0$", fontsize=7.2)
    ax.text(0.144, 0.370, r"$1$", fontsize=7.2)
    ax.text(0.088, 0.372, r"$\cdots$", fontsize=8)
    arrow(ax, (0.156, 0.396), (0.197, 0.396))
    box(ax, (0.060, 0.302), (0.040, 0.044), GREEN, face="#F6FFF6", lw=0.9)
    ax.text(0.080, 0.324, r"$\tau$", ha="center", va="center", fontsize=10.5)
    ax.text(0.080, 0.286, "learnable time", ha="center", va="center", fontsize=6.5)
    arrow(ax, (0.104, 0.324), (0.197, 0.324))
    draw_network(ax, 0.197, 0.302, 0.070, 0.126)
    draw_time_map(ax, 0.030, 0.132, 0.238, 0.136)
    ax.text(0.055, 0.098, r"$\tau$ rescales time; it is not a network output",
            ha="left", va="center", fontsize=7.0)

    panel_label(ax, 0.310, 0.465, "(d)")
    ax.text(0.512, 0.420, "same parameters generate\nfields and trajectories",
            ha="center", va="center", fontsize=7.5)
    arrow(ax, (0.462, 0.385), (0.382, 0.304))
    arrow(ax, (0.562, 0.385), (0.600, 0.304))
    box(ax, (0.302, 0.132), (0.155, 0.170), GREEN, face="white", lw=0.9)
    ax.text(0.379, 0.287, "Control fields", ha="center", va="center", fontsize=7.3)
    draw_step_pulse(ax, 0.323, 0.168, 0.108, 0.094, smooth=True)
    box(ax, (0.505, 0.132), (0.172, 0.170), GREEN, face="white", lw=0.9)
    ax.text(0.591, 0.287, "4 probe trajectories", ha="center", va="center", fontsize=7.3)
    draw_bloch(ax, 0.591, 0.210, 0.057)
    arrow(ax, (0.682, 0.242), (0.720, 0.242))

    box(ax, (0.728, 0.166), (0.230, 0.176), RED, lw=1.0)
    ax.text(0.843, 0.302, r"$L_{\rm total}=\alpha_{\rm dyn}L_{\rm dyn}+\beta_{\rm gate}L_{\rm gate}$",
            color=RED, ha="center", va="center", fontsize=6.8)
    ax.text(0.843, 0.277, r"$+\lambda_TL_T+L_{\rm reg}$", color=RED, ha="center", va="center", fontsize=6.8)
    ax.text(0.744, 0.218, r"$L_{\rm dyn}$: Bloch residual" + "\n" + r"$L_{\rm gate}$: terminal gate loss" + "\n" + r"$L_T=T$",
            ha="left", va="center", fontsize=6.6)
    ax.text(0.843, 0.113, "physics enters directly\nduring training", color=RED,
            ha="center", va="center", fontsize=6.8)

    fig.savefig(OUT / "fig_method_overview.pdf", bbox_inches=None)
    fig.savefig(OUT / "fig_method_overview.svg", bbox_inches=None)
    fig.savefig(OUT / "fig_method_overview_vector.png", dpi=600, bbox_inches=None)
    plt.close(fig)
    print(f"saved {OUT / 'fig_method_overview.pdf'}")
    print(f"saved {OUT / 'fig_method_overview.svg'}")


if __name__ == "__main__":
    make_figure()
