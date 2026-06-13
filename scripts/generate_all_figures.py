"""
Regenerate APS-style Results figures for intro_methods.tex.

The figures are sized at final manuscript scale: most are single-column
figures (3.35 in wide), with fixed output canvases, readable 7--9 pt labels,
vector PDF output, and matching panel order for the Results narrative.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pinn_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/pinn_xdg_cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COL_W = 3.35
DBL_W = 7.0
H_SMALL = 2.70
H_MED = 3.25
H_TALL = 4.55
H_EXTRA = 6.55
FS_LEGEND = 7.2
FS_NOTE = 7.3
AX_LEFT = 0.22
AX_RIGHT = 0.90
BOTTOM_SMALL = 0.145
BOTTOM_TALL = 0.075
BOTTOM_EXTRA = 0.055

C_BLUE = "#0072B2"
C_RED = "#D55E00"
C_GREEN = "#009E73"
C_ORANGE = "#E69F00"
C_PURPLE = "#CC79A7"
C_SKY = "#56B4E9"
C_GRAY = "#4D4D4D"
C_LIGHT = "#BDBDBD"


def set_aps_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 9.0,
            "axes.labelsize": 9.2,
            "axes.linewidth": 0.65,
            "axes.unicode_minus": False,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": FS_LEGEND,
            "legend.frameon": False,
            "lines.linewidth": 1.25,
            "savefig.bbox": "standard",
            "savefig.pad_inches": 0.0,
        }
    )


def panel_label(ax, label: str, x: float = 0.02, y: float = 0.97) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.6,
        fontweight="bold",
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.6),
    )


def finish(fig, name: str, png: bool = True, crop_pdf: bool = False, **save_kwargs) -> None:
    kwargs = {"bbox_inches": None}
    kwargs.update(save_kwargs)
    pdf_path = OUT / f"{name}.pdf"
    fig.savefig(pdf_path, **kwargs)
    if crop_pdf:
        tmp_pdf = OUT / f"{name}.crop.pdf"
        try:
            subprocess.run(
                ["pdfcrop", str(pdf_path), str(tmp_pdf)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            tmp_pdf.replace(pdf_path)
        finally:
            if tmp_pdf.exists():
                tmp_pdf.unlink()
    if png:
        fig.savefig(OUT / f"{name}.png", dpi=600, **kwargs)
    plt.close(fig)
    print(f"saved {OUT / (name + '.pdf')}")


def align_frame(fig, *, bottom: float, top: float, hspace: float | None = None, wspace: float | None = None) -> None:
    """Use a common axes-frame left/right boundary across single-column figures."""
    kwargs = {"left": AX_LEFT, "right": AX_RIGHT, "bottom": bottom, "top": top}
    if hspace is not None:
        kwargs["hspace"] = hspace
    if wspace is not None:
        kwargs["wspace"] = wspace
    fig.subplots_adjust(**kwargs)


def load_json(path: str):
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw_rotation_axes() -> None:
    fig, ax = plt.subplots(figsize=(COL_W, H_SMALL), constrained_layout=False)
    align_frame(fig, bottom=0.08, top=0.96)

    th = np.linspace(0, 2 * np.pi, 500)
    ax.plot(np.cos(th), np.sin(th), color=C_LIGHT, lw=0.65)
    ax.axhline(0, color=C_GRAY, lw=0.55, alpha=0.35)
    ax.axvline(0, color=C_GRAY, lw=0.55, alpha=0.35)
    ax.annotate("", xy=(1.22, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=0.9))
    ax.annotate("", xy=(0, 1.22), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=0.9))
    ax.text(1.28, -0.03, r"$x$", color=C_BLUE, ha="left", va="top", fontsize=9)
    ax.text(-0.04, 1.27, r"$z$", color=C_RED, ha="right", va="bottom", fontsize=9)

    alphas = np.array([0, 15, 30, 45, 60, 75, 90])
    colors = [C_BLUE, C_SKY, C_GREEN, C_ORANGE, C_PURPLE, "#7B3294", C_RED]
    for a, color in zip(alphas, colors):
        rad = np.deg2rad(a)
        x, z = np.cos(rad), np.sin(rad)
        ax.annotate("", xy=(x, z), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, alpha=0.92))
        ax.plot(x, z, "o", ms=3.6, color=color, mec="black", mew=0.35)

    arc = np.linspace(0, np.deg2rad(45), 80)
    ax.plot(0.28 * np.cos(arc), 0.28 * np.sin(arc), color=C_GRAY, lw=0.85)
    ax.text(0.30, 0.12, r"$\alpha$", fontsize=8.4, color=C_GRAY)
    ax.text(0.72, 0.77, r"$\hat n(\alpha)=(\cos\alpha,0,\sin\alpha)$",
            ha="center", va="center", fontsize=8.0, color=C_GRAY)
    ax.text(1.02, -0.12, r"$X(\pi)$", ha="center", va="top", color=C_BLUE, fontsize=8.0)
    ax.text(-0.10, 1.03, r"$Z(\pi)$", ha="right", va="center", color=C_RED, fontsize=8.0)

    ax.set_xlim(-0.25, 1.36)
    ax.set_ylim(-0.22, 1.35)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    finish(fig, "fig_rotation_axes")


def draw_rotation_minimal() -> None:
    d = np.load(ROOT / "ppt/data/pinn_rotation_5gates_minimal.npz")
    alphas = np.array([0, 15, 30, 45, 60, 75, 90], dtype=float)
    T_pinn = np.array([0.392, 0.376, 0.335, 0.280, 0.336, 0.377, 0.392])
    alpha_s = np.linspace(0, 90, 400)
    T_direct = np.pi * np.maximum(np.cos(np.deg2rad(alpha_s)), np.sin(np.deg2rad(alpha_s))) / 8.0

    gates = [
        ("X_pi", r"$X(\pi)$"),
        ("R45", r"$R(45^\circ,\pi)$"),
        ("Z_pi", r"$Z(\pi)$"),
    ]
    fig = plt.figure(figsize=(COL_W, H_TALL), constrained_layout=False)
    align_frame(fig, bottom=BOTTOM_TALL, top=0.97, hspace=0.20)
    gs = fig.add_gridspec(
        4,
        1,
        height_ratios=[0.78, 0.78, 0.78, 1.25],
    )

    xmax = 0.42
    for row, (gate, label) in enumerate(gates):
        pax = fig.add_subplot(gs[row, 0])
        t = d[f"{gate}_t"]
        omega = d[f"{gate}_Omega"]
        delta = d[f"{gate}_Delta"]
        T = float(d[f"{gate}_T"])
        F = float(d[f"{gate}_F"])
        pax.plot(t, omega, color=C_BLUE, lw=1.35, label=r"$\Omega(t)$")
        pax.plot(t, delta, color=C_RED, lw=1.25, label=r"$\Delta(t)$")
        pax.axhline(0, color=C_LIGHT, lw=0.45)
        pax.set_xlim(-0.005, xmax)
        pax.set_ylim(-9.1, 9.1)
        pax.set_yticks([-8, 0, 8])
        pax.set_ylabel("")
        pax.text(
            0.030,
            0.82,
            label,
            transform=pax.transAxes,
            ha="left",
            va="top",
            fontsize=8.7,
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.0),
        )
        pax.text(
            0.985,
            0.86,
            rf"$T^*={T:.3f}$" + "\n" + rf"$F={F:.4f}$",
            transform=pax.transAxes,
            ha="right",
            va="top",
            fontsize=FS_NOTE,
            color=C_GRAY,
        )
        if row == 0:
            panel_label(pax, "(a)", y=0.93)
            pax.legend(loc="lower right", ncol=2, handlelength=1.4, columnspacing=0.9)
        if row == 1:
            pax.set_ylabel(r"$\Omega(t),\Delta(t)$")
        if row < len(gates) - 1:
            pax.set_xlabel("")
            pax.tick_params(labelbottom=False)
        else:
            pax.set_xlabel(r"$t$")

    ax = fig.add_subplot(gs[3, 0])
    ax.plot(alpha_s, T_direct, color=C_GRAY, lw=1.25, label=r"$T_{\rm direct}$")
    ax.fill_between(alpha_s, 0.96 * T_direct, 1.04 * T_direct, color=C_GRAY, alpha=0.12, lw=0)
    ax.plot(alphas, T_pinn, "o", color=C_BLUE, ms=4.4, mec="black", mew=0.45, label="PINN")
    ax.plot(45, np.pi / (8 * np.sqrt(2)), "s", ms=4.6, mfc="white", mec=C_BLUE, mew=0.9)
    ax.annotate(
        r"$T_{\min}$",
        xy=(45, np.pi / (8 * np.sqrt(2))),
        xytext=(53, 0.266),
        ha="left",
        va="center",
        fontsize=8.0,
        arrowprops=dict(arrowstyle="->", lw=0.6, color=C_GRAY),
    )
    ax.set_xlim(-3, 93)
    ax.set_ylim(0.24, 0.43)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_ylabel(r"$T^*$")
    ax.set_xlabel(r"$\alpha$ (deg.)")
    ax.legend(loc="upper center", ncol=2, handlelength=1.6, columnspacing=1.2)
    panel_label(ax, "(b)")
    finish(fig, "fig_rotation_minimal")


def draw_pulse_comparison() -> None:
    pinn = np.load(ROOT / "ppt/data/pinn_3levels_3gates.npz")
    grape_dir = ROOT / "quantum-gate/code/GRAPE/outputs_grape_lbfgsb_smooth"
    gates = [
        ("X", "X_pi", r"$X(\pi)$", 0.4123),
        ("Z", "Z_pi", r"$Z(\pi)$", 0.4123),
        ("Y", "Y_pi", r"$Y(\pi)$", 0.6500),
    ]

    fig = plt.figure(figsize=(COL_W, H_TALL), constrained_layout=False)
    align_frame(fig, bottom=BOTTOM_TALL, top=0.91, hspace=0.16, wspace=0.12)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.0])
    for row, (pinn_key, grape_key, label, grape_T) in enumerate(gates):
        ax_l = fig.add_subplot(gs[row, 0])
        ax_r = fig.add_subplot(gs[row, 1], sharey=ax_l)

        k = f"{pinn_key}_noenv"
        t = pinn[f"{k}_t"]
        omega = pinn[f"{k}_Omega"]
        delta = pinn[f"{k}_Delta"]
        T = float(pinn[f"{k}_T"])
        ax_l.plot(t, omega, color=C_BLUE, lw=1.15, label=r"$\Omega(t)$")
        ax_l.plot(t, delta, color=C_RED, lw=1.05, label=r"$\Delta(t)$")

        omega_g = np.load(grape_dir / f"best_{grape_key}_no_envelope_omega.npy")
        delta_g = np.load(grape_dir / f"best_{grape_key}_no_envelope_delta.npy")
        edges = np.linspace(0, grape_T, len(omega_g) + 1)
        ax_r.stairs(omega_g, edges, color=C_BLUE, lw=1.10)
        ax_r.stairs(delta_g, edges, color=C_RED, lw=1.00)

        for ax in (ax_l, ax_r):
            ax.axhline(0, color=C_LIGHT, lw=0.45)
            ax.set_ylim(-9.4, 9.4)
            ax.set_yticks([-8, 0, 8])
            ax.tick_params(labelsize=7.8)
        ax_l.set_ylabel(r"$\Omega(t),\Delta(t)$", labelpad=1.0)
        ax_l.set_xlim(-0.015, 0.95)
        ax_r.set_xlim(-0.015, 0.70)
        ax_l.text(
            0.035,
            0.80,
            label,
            transform=ax_l.transAxes,
            ha="left",
            va="top",
            fontsize=8.7,
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.8),
        )
        ax_l.text(0.98, 0.88, rf"$T^*={T:.3f}$", transform=ax_l.transAxes,
                  ha="right", va="top", fontsize=FS_NOTE, color=C_GRAY,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=0.7))
        ax_r.text(0.98, 0.88, rf"$T^*={grape_T:.3f}$", transform=ax_r.transAxes,
                  ha="right", va="top", fontsize=FS_NOTE, color=C_GRAY,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=0.7))
        if row == 0:
            ax_l.set_title("continuous", fontsize=8.2, pad=2)
            ax_r.set_title("GRAPE", fontsize=8.2, pad=2)
            ax_l.legend(
                loc="lower center",
                bbox_to_anchor=(0.54, 0.06),
                ncol=2,
                fontsize=FS_LEGEND,
                handlelength=1.2,
                columnspacing=0.6,
                frameon=True,
                framealpha=0.78,
                facecolor="white",
                edgecolor="none",
                borderpad=0.2,
                borderaxespad=0.0,
            )
            panel_label(ax_l, "(a)", y=0.96)
            panel_label(ax_r, "(b)", y=0.96)
        if row < len(gates) - 1:
            ax_l.tick_params(labelbottom=False)
            ax_r.tick_params(labelbottom=False)
        else:
            ax_l.set_xlabel(r"$t$")
            ax_r.set_xlabel(r"$t$")
        ax_r.tick_params(labelleft=False)

    finish(fig, "fig_pulse_comparison")


def draw_rotation_structure() -> None:
    noenv = load_json("ppt/data/pinn_general_gates_noenv.json")
    minimal = load_json("ppt/data/pinn_minimal_3gates.json")

    in_plane_noenv = sorted(
        [x for x in noenv if x.get("phase") == 1 and x.get("ny_zero")],
        key=lambda x: x["alpha_deg"],
    )
    alphas = np.array([x["alpha_deg"] for x in in_plane_noenv], dtype=float)
    ratios = np.array([x["ratio_OD"] for x in in_plane_noenv], dtype=float)

    fig, ax = plt.subplots(figsize=(COL_W, H_SMALL), constrained_layout=False)
    align_frame(fig, bottom=BOTTOM_SMALL, top=0.93)
    a_s = np.linspace(4, 86, 320)
    ax.plot(a_s, 1 / np.tan(np.deg2rad(a_s)), color=C_GRAY, lw=1.2, label=r"$\cot\alpha$")
    mask = (alphas > 0) & (alphas < 90)
    ax.plot(alphas[mask], ratios[mask], "o", color=C_GREEN, ms=4.2, mec="black", mew=0.45,
            label="PINN")
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 4.2)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_ylabel(r"$|\Omega|_{\max}/|\Delta|_{\max}$")
    ax.set_xlabel(r"$\alpha$ (deg.)")
    ax.legend(loc="upper right", handlelength=1.5)
    finish(fig, "fig_allocation_law")

    fig, ax = plt.subplots(figsize=(COL_W, H_SMALL), constrained_layout=False)
    align_frame(fig, bottom=BOTTOM_SMALL, top=0.93)
    gates = [r"$X$", r"$Y$", r"$Z$"]
    levels = ["min.", "reg.", "sin"]
    T_data = np.array(
        [
            [0.392, 0.565, 0.614],
            [0.570, 0.823, 0.927],
            [0.392, 0.565, 0.613],
        ]
    )
    F_data = np.array(
        [
            [0.9935, 0.9907, 0.9898],
            [0.9905, 0.9863, 0.9844],
            [0.9935, 0.9906, 0.9898],
        ]
    )
    x = np.arange(len(levels))
    width = 0.22
    colors = [C_BLUE, C_PURPLE, C_RED]
    for j, gate in enumerate(gates):
        ax.bar(x + (j - 1) * width, T_data[j], width=width, color=colors[j],
               edgecolor="white", linewidth=0.4, label=gate)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel(r"$T^*$")
    ax.set_ylim(0, 1.02)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.13, 1.0),
        ncol=3,
        handlelength=1.1,
        columnspacing=0.65,
        borderaxespad=0.0,
    )
    ax2 = ax.twinx()
    infid = (1 - F_data) * 1e3
    for j, gate in enumerate(gates):
        ax2.plot(x + (j - 1) * width, infid[j], "o", ms=3.2, mfc="white",
                 mec=colors[j], mew=0.8, linestyle="none")
    ax2.set_ylabel(r"$1-F_{\rm avg}$ ($10^{-3}$)")
    ax2.set_ylim(3.0, 16.8)
    finish(fig, "fig_pulse_shape_cost")

    fig, ax = plt.subplots(figsize=(COL_W, H_SMALL), constrained_layout=False)
    align_frame(fig, bottom=BOTTOM_SMALL, top=0.93)
    in_plane = [x for x in minimal if abs(x["n"][1]) < 1e-10]
    off_plane = [x for x in minimal if abs(x["n"][1]) >= 1e-10]
    ip_T = np.array([x["T_opt"] for x in in_plane])
    t_min = float(np.min(ip_T))
    ax.axvspan(-0.015, 0.065, color=C_SKY, alpha=0.16, lw=0)
    ax.axhline(t_min, color=C_GRAY, lw=0.75, ls="--", alpha=0.62)
    ax.text(0.78, t_min + 0.008, rf"$T_{{\min}}={t_min:.3f}$",
            ha="left", va="bottom", fontsize=FS_NOTE, color=C_GRAY)
    x_ip = np.linspace(0.000, 0.034, len(in_plane))
    ax.plot(x_ip, ip_T, "o",
            color=C_BLUE, ms=3.7, mec="black", mew=0.4, alpha=0.95,
            label="in plane")
    ax.text(0.041, 0.356, r"$n_y=0$" + "\n7 gates",
            ha="left", va="center", fontsize=FS_NOTE, color=C_BLUE)
    for xpos in [1 / np.sqrt(3), 1 / np.sqrt(2), 1.0]:
        ax.axvline(xpos, color=C_LIGHT, lw=0.55, ls=":", zorder=0)
    pi = "\u03c0"
    offsets = {
        f"R(xyz,{pi})": (0.018, -0.014, "left", "top"),
        f"R(xy,{pi})": (0.018, 0.012, "left", "bottom"),
        f"R(yz,{pi})": (0.018, -0.004, "left", "center"),
        f"Y({pi})": (-0.155, 0.010, "right", "bottom"),
    }
    for j, item in enumerate(off_plane):
        ny = abs(item["n"][1])
        T = item["T_opt"]
        ax.plot(ny, T, "o", color=C_RED, ms=4.6, mec="black", mew=0.45,
                label="out of plane" if j == 0 else None)
        dx, dy, ha, va = offsets.get(item["label"], (0.014, 0.008, "left", "bottom"))
        ax.text(ny + dx, T + dy, item["label"].replace(pi, r"$\pi$"),
                ha=ha, va=va, fontsize=FS_NOTE, color=C_RED, fontweight="bold")
    ax.set_xlim(-0.06, 1.08)
    ax.set_ylim(0.245, 0.605)
    ax.set_xticks([0, 1 / np.sqrt(3), 1 / np.sqrt(2), 1.0])
    ax.set_xticklabels([r"$0$", r"$1/\sqrt{3}$", r"$1/\sqrt{2}$", r"$1$"])
    ax.set_xlabel(r"$|n_y|$ (out-of-plane component)")
    ax.set_ylabel(r"$T^*$")
    ax.legend(loc="center", bbox_to_anchor=(0.54, 0.82), ncol=2,
              fontsize=FS_LEGEND, handlelength=1.1, columnspacing=0.8,
              borderaxespad=0.0)
    finish(fig, "fig_ny_time")


def draw_bloch_wire(ax):
    u = np.linspace(0, 2 * np.pi, 44)
    v = np.linspace(0, np.pi, 22)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="#CFCFCF", alpha=0.17, linewidth=0.25)
    ax.plot([-1.12, 1.12], [0, 0], [0, 0], color=C_LIGHT, lw=0.55)
    ax.plot([0, 0], [-1.12, 1.12], [0, 0], color=C_LIGHT, lw=0.55)
    ax.plot([0, 0], [0, 0], [-1.12, 1.12], color=C_LIGHT, lw=0.55)
    ax.text(1.22, 0, 0, r"$x$", fontsize=FS_NOTE, color=C_GRAY)
    ax.text(0, 1.22, 0, r"$y$", fontsize=FS_NOTE, color=C_GRAY)
    ax.text(0, 0, 1.22, r"$z$", fontsize=FS_NOTE, color=C_GRAY)
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.set_zlim(-1.18, 1.18)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1, 1, 1])
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor("white")
    ax.grid(False)


def draw_geometric_paths() -> None:
    data_dir = ROOT / "geometric-gate/results/outputs_turn_weighted_learnableT_env_beta0"
    ts = np.load(data_dir / "timeseries_turn_weighted.npz")
    t_grid = ts["t"]
    omega = ts["Omega"]
    delta = ts["Delta"]
    T = float(t_grid[-1])

    def interp_ctrl(t_val):
        return (
            float(np.interp(t_val, t_grid, omega)),
            float(np.interp(t_val, t_grid, delta)),
        )

    def bloch_rhs(r, Om, De):
        x, y, z = r
        return np.array([-De * y, De * x - Om * z, Om * y], dtype=float)

    def rk4_full(r0, n_steps=1800):
        h = T / n_steps
        traj = [np.array(r0, dtype=float)]
        r = traj[0].copy()
        tt = 0.0
        for _ in range(n_steps):
            Om1, De1 = interp_ctrl(tt)
            k1 = bloch_rhs(r, Om1, De1)
            Om2, De2 = interp_ctrl(tt + 0.5 * h)
            k2 = bloch_rhs(r + 0.5 * h * k1, Om2, De2)
            k3 = bloch_rhs(r + 0.5 * h * k2, Om2, De2)
            Om4, De4 = interp_ctrl(tt + h)
            k4 = bloch_rhs(r + h * k3, Om4, De4)
            r = r + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            tt += h
            traj.append(r.copy())
        return np.asarray(traj)

    traj_ref = ts["ref_plus"]
    traj_ref_neg = ts["ref_minus"]
    traj_probe_x = rk4_full([1.0, 0.0, 0.0])
    traj_probe_y = rk4_full([0.0, 1.0, 0.0])

    u = np.linspace(0, 2 * np.pi, 44)
    v = np.linspace(0, np.pi, 22)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    def draw_sphere(ax):
        ax.plot_wireframe(xs, ys, zs, color="#CFCFCF", alpha=0.18, linewidth=0.25)
        ax.plot([-1.12, 1.12], [0, 0], [0, 0], color=C_LIGHT, lw=0.55)
        ax.plot([0, 0], [-1.12, 1.12], [0, 0], color=C_LIGHT, lw=0.55)
        ax.plot([0, 0], [0, 0], [-1.12, 1.12], color=C_LIGHT, lw=0.55)
        ax.text(1.18, 0, 0, r"$x$", fontsize=FS_NOTE, color=C_GRAY)
        ax.text(0, 1.18, 0, r"$y$", fontsize=FS_NOTE, color=C_GRAY)
        ax.text(0, 0, 1.18, r"$z$", fontsize=FS_NOTE, color=C_GRAY)
        ax.set_xlim(-1.22, 1.22)
        ax.set_ylim(-1.22, 1.22)
        ax.set_zlim(-1.22, 1.22)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=22, azim=-60)
        ax.set_proj_type("ortho")
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill = False
            axis.pane.set_edgecolor("white")
        ax.grid(False)
        ax.set_axis_off()

    fig = plt.figure(figsize=(COL_W, 2.05), constrained_layout=False)
    ax_left = fig.add_subplot(121, projection="3d")
    ax_right = fig.add_subplot(122, projection="3d")
    draw_sphere(ax_left)
    draw_sphere(ax_right)

    c_ref = "#E65100"
    c_ref_neg = "#9C27B0"
    c_px = "#D32F2F"
    c_py = "#1565C0"
    c_arrow = C_GREEN

    ax_left.plot(traj_ref[:, 0], traj_ref[:, 1], traj_ref[:, 2],
                 color=c_ref, linewidth=1.6, alpha=0.95)
    ax_left.plot(traj_ref_neg[:, 0], traj_ref_neg[:, 1], traj_ref_neg[:, 2],
                 color=c_ref_neg, linewidth=1.1, alpha=0.78, ls="--")
    ax_left.scatter(*traj_ref[0], color=c_ref, s=18, marker="o",
                    edgecolors="black", linewidths=0.35)
    ax_left.scatter(*traj_ref[-1], color=c_ref, s=26, marker="*",
                    edgecolors="black", linewidths=0.35)
    ax_left.scatter(*traj_ref_neg[0], color=c_ref_neg, s=18, marker="o",
                    edgecolors="black", linewidths=0.35)

    ax_right.plot(traj_probe_x[:, 0], traj_probe_x[:, 1], traj_probe_x[:, 2],
                  color=c_px, linewidth=1.55, alpha=0.95)
    ax_right.plot(traj_probe_y[:, 0], traj_probe_y[:, 1], traj_probe_y[:, 2],
                  color=c_py, linewidth=1.35, alpha=0.85)
    for traj, color, size in [(traj_probe_x, c_px, 22), (traj_probe_y, c_py, 18)]:
        ax_right.scatter(*traj[0], color=color, s=size, marker="o",
                         edgecolors="black", linewidths=0.35)
        ax_right.scatter(*traj[-1], color=color, s=size + 4, marker="*",
                         edgecolors="black", linewidths=0.35)

    def add_control_arrows(ax, traj):
        idx = np.linspace(0, len(traj) - 1, 9).astype(int)[1:]
        for k in idx:
            tt = T * k / (len(traj) - 1)
            Om, De = interp_ctrl(tt)
            h_vec = np.array([Om, 0.0, De], dtype=float)
            h_norm = np.linalg.norm(h_vec)
            if h_norm < 0.25:
                continue
            direction = h_vec / h_norm * 0.26
            pt = traj[k]
            ax.quiver(
                pt[0], pt[1], pt[2],
                direction[0], direction[1], direction[2],
                color=c_arrow, alpha=0.70, linewidth=0.65,
                arrow_length_ratio=0.32,
            )

    add_control_arrows(ax_left, traj_ref)
    add_control_arrows(ax_right, traj_probe_x)

    ax_left.set_title("Reference loops", fontsize=8.2, pad=1)
    ax_right.set_title(r"Probe states under $Z$ gate" + "\n"
                       r"$|{+}x\rangle\!\to\!|{-}x\rangle$, "
                       r"$|{+}y\rangle\!\to\!|{-}y\rangle$",
                       fontsize=7.8, pad=1)

    left_handles = [
        Line2D([0], [0], color=c_ref, lw=1.6, label=r"Ref $|0\rangle$"),
        Line2D([0], [0], color=c_ref_neg, lw=1.1, ls="--", label=r"Ref $|1\rangle$"),
        Line2D([0], [0], color=c_arrow, lw=1.2, label=r"$\mathbf{h}(t)$"),
    ]
    right_handles = [
        Line2D([0], [0], color=c_px, lw=1.6, label=r"$|{+}x\rangle\to|{-}x\rangle$"),
        Line2D([0], [0], color=c_py, lw=1.35, label=r"$|{+}y\rangle\to|{-}y\rangle$"),
        Line2D([0], [0], color=c_arrow, lw=1.2, label=r"$\mathbf{h}(t)$"),
    ]
    ax_left.legend(handles=left_handles, loc="upper left", bbox_to_anchor=(-0.03, 0.96),
                   fontsize=6.8, handlelength=1.2, borderpad=0.20,
                   labelspacing=0.22, framealpha=0.86)
    ax_right.legend(handles=right_handles, loc="upper left", bbox_to_anchor=(-0.03, 0.96),
                    fontsize=6.8, handlelength=1.2, borderpad=0.20,
                    labelspacing=0.22, framealpha=0.86)
    # This Bloch-sphere panel has no visible Cartesian axes frame; let the
    # spherical content fill the single-column canvas instead of applying the
    # common plot-frame margins used for ordinary axes.
    fig.subplots_adjust(left=0.00, right=1.00, top=0.90, bottom=0.00, wspace=-0.02)
    finish(fig, "fig_geometric_paths", crop_pdf=True)


def load_beta_data():
    results = ROOT / "geometric-gate/results"
    betas = [0, 1, 3, 10, 30, 50]
    data = {}
    for beta in betas:
        d = results / f"outputs_turn_weighted_learnableT_env_beta{beta}"
        ts = np.load(d / "timeseries_turn_weighted.npz")
        with open(d / "metrics.json", "r", encoding="utf-8") as f:
            m = json.load(f)
        t = ts["t"]
        E = 0.5 * (np.abs(ts["E_plus"]) + np.abs(ts["E_minus"]))
        # Load RK4-validated E(t) if available
        E_rk4_path = d / "t_rk4.txt"
        if E_rk4_path.exists():
            t_rk4 = np.loadtxt(E_rk4_path)
            E_p_rk4 = np.loadtxt(d / "ref_plus_energy_rk4.txt")
            E_m_rk4 = np.loadtxt(d / "ref_minus_energy_rk4.txt")
            E_rk4 = 0.5 * (np.abs(E_p_rk4) + np.abs(E_m_rk4))
        else:
            t_rk4 = t
            E_rk4 = E
        kappa = compute_kappa(ts["Omega"], ts["Delta"], t)
        corr = np.corrcoef(E, kappa)[0, 1] if np.std(kappa) > 1e-12 else 0.0
        data[beta] = {"ts": ts, "metrics": m, "E": E, "E_rk4": E_rk4, "t_rk4": t_rk4,
                      "kappa": kappa, "corr": corr}
    return betas, data


def compute_kappa(Omega, Delta, t):
    dO = np.gradient(Omega, t)
    dD = np.gradient(Delta, t)
    return np.abs((Omega * dD - Delta * dO) / (Omega**2 + Delta**2 + 1e-9))


def draw_turning_bottleneck() -> None:
    _, data = load_beta_data()
    base = data[0]
    ts = base["ts"]
    t = ts["t"]
    E = base["E"]
    kappa = base["kappa"]
    kappa_norm = kappa / max(np.nanmax(kappa), 1e-12)
    E_norm = E / max(np.nanmax(E), 1e-12)

    fig, axes = plt.subplots(3, 1, figsize=(COL_W, H_TALL), sharex=True, constrained_layout=False)
    align_frame(fig, bottom=BOTTOM_TALL, top=0.96, hspace=0.12)
    axes[0].plot(t, ts["Omega"], color=C_BLUE, label=r"$\Omega(t)$")
    axes[0].plot(t, ts["Delta"], color=C_RED, label=r"$\Delta(t)$")
    axes[0].axhline(0, color=C_LIGHT, lw=0.45)
    axes[0].set_ylabel("control")
    axes[0].legend(loc="upper right", ncol=2, handlelength=1.3, columnspacing=0.9)
    panel_label(axes[0], "(a)")

    axes[1].plot(t, kappa, color=C_PURPLE)
    axes[1].set_ylabel(r"$\kappa(t)$")
    panel_label(axes[1], "(b)")

    axes[2].plot(t, E * 1e3, color=C_GRAY, label=r"$|E(t)|$")
    axes[2].fill_between(t, 0, E * 1e3, color=C_LIGHT, alpha=0.32, linewidth=0)
    axes[2].plot(t, E_norm * np.max(E * 1e3), color=C_GRAY, lw=0.0)
    axes[2].set_ylabel(r"$|E|$ ($10^{-3}$)")
    axes[2].set_xlabel(r"$t$")
    panel_label(axes[2], "(c)")

    peaks = kappa_norm > np.quantile(kappa_norm, 0.92)
    for ax in axes:
        ax.fill_between(t, 0, 1, where=peaks, transform=ax.get_xaxis_transform(),
                        color=C_PURPLE, alpha=0.08, linewidth=0)
    finish(fig, "fig_turning_bottleneck")


def draw_turn_weighted_loss() -> None:
    betas, data = load_beta_data()
    compare_red = "#D62728"
    x = np.arange(len(betas), dtype=float)
    mean_E = np.array([
        data[b]["metrics"].get("mean_abs_E_rk4",
            0.5 * (data[b]["metrics"]["mean_abs_E_plus"] + data[b]["metrics"]["mean_abs_E_minus"]))
        for b in betas
    ]) * 1e3
    abs_corr = np.array([abs(data[b]["corr"]) for b in betas])

    fig = plt.figure(figsize=(COL_W, H_EXTRA), constrained_layout=False)
    align_frame(fig, bottom=BOTTOM_EXTRA, top=0.96, hspace=0.16)
    gs = fig.add_gridspec(5, 1, height_ratios=[1.05, 0.78, 0.78, 0.95, 0.95])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b_pinn = fig.add_subplot(gs[1, 0])
    ax_b_grape = fig.add_subplot(gs[2, 0])
    ax_c = fig.add_subplot(gs[3, 0])
    ax_d = fig.add_subplot(gs[4, 0])

    ax_a.plot(x, mean_E, "o-", color=C_BLUE, ms=4.0, mec="black", mew=0.35,
              label=r"mean $|E|$")
    ax_a.plot([betas.index(30)], [mean_E[betas.index(30)]], "s", ms=4.8,
              mfc="white", mec=C_BLUE, mew=1.0)
    ax_a.set_ylabel(r"mean $|E|$ ($10^{-3}$)")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([str(b) for b in betas])
    ax_a.set_xlabel(r"PINN turn weight $\beta$")
    ax_a2 = ax_a.twinx()
    ax_a2.plot(x, abs_corr, "o-", color=compare_red, ms=3.7, mec="black", mew=0.35,
               label=r"$|r(E,\kappa)|$")
    ax_a2.set_ylabel(r"$|r(E,\kappa)|$")
    ax_a2.set_ylim(0, max(0.75, abs_corr.max() * 1.12))
    handles = ax_a.get_lines()[:1] + ax_a2.get_lines()
    labels = [h.get_label() for h in handles]
    ax_a.legend(handles, labels, loc="upper center", ncol=2, handlelength=1.4,
                columnspacing=0.8, bbox_to_anchor=(0.55, 1.02), borderaxespad=0.0)
    panel_label(ax_a, "(a)", y=0.88)

    def load_grape_turn(beta: int, T_val: float = 1.5):
        """Load GRAPE results. Try T=1.5 first, fall back to T=1.0."""
        d15 = ROOT / f"geometric-gate/results/outputs_grape_turn_beta{beta}_T{T_val}"
        d10 = ROOT / f"geometric-gate/results/outputs_grape_turn_beta{beta}"
        d = d15 if (d15 / "metrics.json").exists() else d10
        omega = np.loadtxt(d / "best_Omega.txt")
        delta = np.loadtxt(d / "best_Delta.txt")
        with open(d / "metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
        T = float(metrics.get("T", 1.0))
        t_ctrl = np.linspace(0, T, len(omega))
        # Use RK4 E(t) if available
        rk4_path = d / "ref_plus_energy_rk4.txt"
        if rk4_path.exists():
            t_err = np.loadtxt(d / "t_rk4.txt")
            e_plus = np.loadtxt(d / "ref_plus_energy_rk4.txt")
            e_minus = np.loadtxt(d / "ref_minus_energy_rk4.txt")
        else:
            e_plus = np.loadtxt(d / "ref_plus_energy.txt")
            e_minus = np.loadtxt(d / "ref_minus_energy.txt")
            t_err = np.linspace(0, T, len(e_plus))
        err = 0.5 * (np.abs(e_plus) + np.abs(e_minus))
        return t_ctrl, omega, delta, t_err, err, metrics, T

    g0 = load_grape_turn(0)
    g3 = load_grape_turn(3)

    def plot_bin_step(ax, t_ctrl, values, *, color, ls="-", label=None):
        T_ctrl = float(t_ctrl[-1]) if len(t_ctrl) else 1.0
        edges = np.linspace(0.0, T_ctrl, len(values) + 1)
        ax.step(edges, np.r_[values, values[-1]], where="post",
                color=color, lw=1.0, ls=ls, label=label)

    for beta, color, label in [
        (0, C_BLUE, r"$\beta=0$"),
        (30, compare_red, r"$\beta=30$"),
    ]:
        ax_b_pinn.plot(data[beta]["t_rk4"], data[beta]["E_rk4"] * 1e3, color=color, lw=1.15,
                       label=label)
    ax_b_grape.plot(g0[3], g0[4] * 1e3, color=C_BLUE, lw=1.05,
                    label=r"$\beta=0$")
    ax_b_grape.plot(g3[3], g3[4] * 1e3, color=compare_red, lw=1.05,
                    label=r"$\beta=3$")
    kappa0 = data[0]["kappa"]
    kappa0 = kappa0 / max(np.nanmax(kappa0), 1e-12)
    high_turn = kappa0 > np.quantile(kappa0, 0.92)
    t0_train = data[0]["ts"]["t"]
    ax_b_pinn.fill_between(
        t0_train,
        0,
        1,
        where=high_turn,
        transform=ax_b_pinn.get_xaxis_transform(),
        color=C_PURPLE,
        alpha=0.07,
        linewidth=0,
    )
    T_grape = max(g0[6], g3[6])  # GRAPE duration
    for ax in [ax_b_pinn, ax_b_grape]:
        ax.legend(loc="upper right", ncol=2, fontsize=FS_LEGEND, handlelength=1.35,
                  columnspacing=0.75)
    ax_b_pinn.set_xlim(-0.03, 1.58)
    ax_b_grape.set_xlim(-0.02, T_grape * 1.05)
    ax_b_pinn.set_ylim(-0.8, 23.5)
    ax_b_pinn.set_yticks([0, 10, 20])
    ax_b_pinn.set_ylabel("PINN\nerror\n" + r"($10^{-3}$)")
    ax_b_pinn.set_xticks([0.0, 0.5, 1.0, 1.5])
    ax_b_pinn.set_xlabel(r"$t$")
    ax_b_grape.set_ylim(-2.0, 72.0)
    ax_b_grape.set_yticks([0, 30, 60])
    ax_b_grape.set_ylabel("GRAPE\nerror\n" + r"($10^{-3}$)")
    ax_b_grape.set_xlabel(r"$t$")
    panel_label(ax_b_pinn, "(b)")
    panel_label(ax_b_grape, "(c)")

    for beta, color, alpha in [(0, C_BLUE, 0.95), (30, compare_red, 1.0)]:
        ts = data[beta]["ts"]
        t = ts["t"]
        ax_c.plot(t, ts["Omega"], color=color, ls="-", alpha=alpha,
                  label=rf"$\Omega$, $\beta={beta}$")
        ax_c.plot(t, ts["Delta"], color=color, ls="--", alpha=alpha,
                  label=rf"$\Delta$, $\beta={beta}$")
    ax_c.axhline(0, color=C_LIGHT, lw=0.45)
    ax_c.set_ylabel("PINN\ncontrol")
    ax_c.set_xlabel(r"$t$")
    ax_c.set_xlim(-0.03, 1.58)
    ax_c.legend(loc="upper right", ncol=2, fontsize=FS_LEGEND, handlelength=1.4,
                columnspacing=0.6)
    panel_label(ax_c, "(d)")

    plot_bin_step(ax_d, g0[0], g0[1], color=C_BLUE, label=r"$\Omega$, $\beta=0$")
    plot_bin_step(ax_d, g0[0], g0[2], color=C_BLUE, ls="--", label=r"$\Delta$, $\beta=0$")
    plot_bin_step(ax_d, g3[0], g3[1], color=compare_red, label=r"$\Omega$, $\beta=3$")
    plot_bin_step(ax_d, g3[0], g3[2], color=compare_red, ls="--", label=r"$\Delta$, $\beta=3$")
    ax_d.axhline(0, color=C_LIGHT, lw=0.45)
    ax_d.set_ylabel("GRAPE\ncontrol")
    ax_d.set_xlabel(r"$t$")
    ax_d.set_xlim(-0.02, T_grape * 1.05)
    ax_d.set_ylim(-6.9, 10.4)
    ax_d.legend(loc="upper left", bbox_to_anchor=(0.16, 1.00), ncol=2,
                fontsize=FS_LEGEND, handlelength=1.4, columnspacing=0.6)
    panel_label(ax_d, "(e)")
    finish(fig, "fig_turn_weighted_loss")


def main() -> None:
    set_aps_style()
    draw_rotation_axes()
    draw_rotation_minimal()
    draw_pulse_comparison()
    draw_rotation_structure()
    draw_geometric_paths()
    draw_turning_bottleneck()
    draw_turn_weighted_loss()


if __name__ == "__main__":
    main()
