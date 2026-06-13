"""
SuppMovie1: Direct vs indirect rotation gates following PPT make_animation.py style.
Each gate: Left=Bloch sphere, Right=Omega(t) top + Delta(t) bottom.
Side-by-side layout for X(pi) and Y(pi).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
    "font.size": 13,
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
})

def rotation_bloch_matrix(n, theta):
    nx, ny, nz = n
    c, s = np.cos(theta), np.sin(theta)
    K = 1 - c
    return np.array([
        [c+nx*nx*K,      nx*ny*K-nz*s,  nx*nz*K+ny*s],
        [ny*nx*K+nz*s,   c+ny*ny*K,      ny*nz*K-nx*s],
        [nz*nx*K-ny*s,   nz*ny*K+nx*s,   c+nz*nz*K]])

GATES = {
    r'$X(\pi)$':     {'n': [1,0,0], 'theta': np.pi},
    r'$R(45°,\pi)$': {'n': [np.cos(np.pi/4), 0, np.sin(np.pi/4)], 'theta': np.pi},
    r'$Z(\pi)$':     {'n': [0,0,1], 'theta': np.pi},
    r'$Y(\pi)$':     {'n': [0,1,0], 'theta': np.pi},
}

def rk4_traj(r0, t_ctrl, Omega, Delta, n_steps=400):
    T = t_ctrl[-1] - t_ctrl[0]; h = T / n_steps
    r = r0.copy(); tv = t_ctrl[0]; traj = [r.copy()]
    for _ in range(n_steps):
        def rhs(rv, ti):
            Om = float(np.interp(ti, t_ctrl, Omega))
            De = float(np.interp(ti, t_ctrl, Delta))
            return np.array([-De*rv[1], De*rv[0]-Om*rv[2], Om*rv[1]])
        k1 = rhs(r, tv); k2 = rhs(r+.5*h*k1, tv+.5*h)
        k3 = rhs(r+.5*h*k2, tv+.5*h); k4 = rhs(r+h*k3, tv+h)
        r += (h/6)*(k1+2*k2+2*k3+k4); tv += h; traj.append(r.copy())
    return np.array(traj)

def setup_bloch(ax):
    u = np.linspace(0, 2*np.pi, 50); v = np.linspace(0, np.pi, 25)
    xs = np.outer(np.cos(u), np.sin(v)); ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="#d0e4f5", alpha=0.07, edgecolor="none", shade=False)
    L = 1.35
    for d in [np.array([L,0,0]), np.array([0,L,0]), np.array([0,0,L])]:
        ax.plot([0,d[0]],[0,d[1]],[0,d[2]], "k-", lw=0.6, alpha=0.3)
        ax.plot([0,-d[0]],[0,-d[1]],[0,-d[2]], "k-", lw=0.6, alpha=0.3)
    th = np.linspace(0, 2*np.pi, 80)
    ax.plot(np.cos(th), np.sin(th), 0, "k-", lw=0.3, alpha=0.15)
    ax.plot(np.cos(th), 0, np.sin(th), "k-", lw=0.3, alpha=0.15)
    ax.plot(0, np.cos(th), np.sin(th), "k-", lw=0.3, alpha=0.15)
    ax.text(1.45, 0, 0, r"$x$", fontsize=14, ha="center")
    ax.text(0, 1.45, 0, r"$y$", fontsize=14, ha="center")
    ax.text(0, 0, 1.45, r"$z$", fontsize=14, ha="center")
    ax.set_xlim([-1.5, 1.5]); ax.set_ylim([-1.5, 1.5]); ax.set_zlim([-1.5, 1.5])
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.tick_params(length=0)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor("w")
    ax.view_init(elev=22, azim=35)

# ── Load data ──
cache = np.load(
    '/Users/strong/Desktop/research-workspace/projects/PINN/quantum-gate/results/fig2_pulse_data.npz',
    allow_pickle=True)

gate_order = [r'$X(\pi)$', r'$Y(\pi)$']
gate_keys  = ['X(π)', 'Y(π)']

N_STEPS = 400
probes_cfg = [
    (r"$|0\rangle$", np.array([0.,0.,1.]), "#2166ac", "o"),
    (r"$|+\rangle$", np.array([1.,0.,0.]), "#1b7837", "^"),
]
colors_om, colors_de = "#2166ac", "#b2182b"

# Pre-compute
all_data = {}
for gname, gkey in zip(gate_order, gate_keys):
    Om = cache[f'{gkey}_Omega']; De = cache[f'{gkey}_Delta']
    t  = cache[f'{gkey}_t']; T_val = float(cache[f'{gkey}_T'])
    R  = rotation_bloch_matrix(GATES[gname]['n'], GATES[gname]['theta'])
    trajs = [rk4_traj(p[1], t, Om, De, N_STEPS) for p in probes_cfg]
    targets = [R @ p[1] for p in probes_cfg]
    t_fine = np.linspace(t[0], t[-1], N_STEPS+1)
    all_data[gname] = {'Om': Om, 'De': De, 't': t, 'T': T_val,
                       'trajs': trajs, 'targets': targets, 't_fine': t_fine}

# ── Figure: 1 row x 2 cols, each cell = Bloch + pulses ──
fig = plt.figure(figsize=(16, 8), facecolor="white")

cell_els = []  # per-gate animated elements

for idx, (gname, gkey) in enumerate(zip(gate_order, gate_keys)):
    row, col = 0, idx
    d = all_data[gname]

    # Cell positions
    x0 = col * 0.5
    y0 = 0.0

    # Bloch sphere: left 30% of cell
    ax_b = fig.add_axes([x0 + 0.005, y0 + 0.07, 0.30, 0.84], projection="3d")
    setup_bloch(ax_b)

    # Gate label
    ax_b.text2D(0.5, 0.97, gname, transform=ax_b.transAxes, fontsize=15,
                ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=colors_om, alpha=0.9))

    # Start dots + target open circles
    for j, (plbl, pinit, pcol, pmarker) in enumerate(probes_cfg):
        ax_b.scatter(*pinit, c=pcol, s=58, marker=pmarker,
                     edgecolors="black", linewidths=0.5, zorder=10, depthshade=False)
        tgt = d['targets'][j]
        ax_b.scatter(*tgt, facecolors="none", s=130, marker="o",
                     edgecolors=pcol, linewidths=1.5, zorder=9, depthshade=False, alpha=0.45)

    bloch_lines, bloch_pts = [], []
    for j, (plbl, pinit, pcol, pmarker) in enumerate(probes_cfg):
        line, = ax_b.plot([], [], [], color=pcol, lw=2.0, alpha=0.88)
        pt, = ax_b.plot([], [], [], "o", color=pcol, markersize=6,
                        markeredgecolor="black", markeredgewidth=0.4, zorder=10)
        bloch_lines.append(line); bloch_pts.append(pt)

    ctrl_quiver = [ax_b.quiver(0, 0, 0, 0, 0, 0, color="#e74c3c",
                               linewidth=2.5, arrow_length_ratio=0.15,
                               alpha=0.9, zorder=20)]
    ctrl_dot, = ax_b.plot([], [], [], "o", color="#e74c3c", markersize=4.5, zorder=20)

    handles = [Line2D([0],[0], color=p[2], marker=p[3], ls="-", markersize=4,
               label=p[0], markeredgecolor="black", markeredgewidth=0.3, lw=1.2)
               for p in probes_cfg]
    handles.append(Line2D([0], [0], color="#e74c3c", lw=2.0, label=r"$h(t)$"))
    ax_b.legend(handles=handles, loc="lower left", fontsize=9, ncol=1,
                bbox_to_anchor=(0.00, -0.01), frameon=True, framealpha=0.82)

    time_text = ax_b.text2D(0.02, 0.03, "", transform=ax_b.transAxes, fontsize=11,
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                      edgecolor="gray", alpha=0.8))

    # Ω panel (top right of cell)
    ax_om = fig.add_axes([x0 + 0.325, y0 + 0.60, 0.16, 0.30])
    ax_om.set_title("Control fields", fontsize=13, pad=4)
    ax_om.set_ylabel(r"$\Omega(t)$", fontsize=13)
    ax_om.set_xlim([d['t'][0]-0.02, d['t'][-1]*1.05])
    ax_om.set_ylim([d['Om'].min()-1, d['Om'].max()+1])
    ax_om.axhline(0, color="gray", lw=0.3, ls="--")
    ax_om.plot(d['t'], d['Om'], color=colors_om, lw=0.6, alpha=0.15)
    ax_om.tick_params(direction="in", top=True, right=True, labelsize=10)
    ax_om.set_xticklabels([])
    pulse_om, = ax_om.plot([], [], color=colors_om, lw=2.0)
    marker_om, = ax_om.plot([], [], "o", color=colors_de, markersize=6,
                            markeredgecolor="black", markeredgewidth=0.5, zorder=10)

    # Δ panel (bottom right of cell)
    ax_de = fig.add_axes([x0 + 0.325, y0 + 0.16, 0.16, 0.30])
    ax_de.set_xlabel(r"$t$", fontsize=13)
    ax_de.set_ylabel(r"$\Delta(t)$", fontsize=13)
    ax_de.set_xlim([d['t'][0]-0.02, d['t'][-1]*1.05])
    ax_de.set_ylim([d['De'].min()-1, d['De'].max()+1])
    ax_de.axhline(0, color="gray", lw=0.3, ls="--")
    ax_de.plot(d['t'], d['De'], color=colors_de, lw=0.6, alpha=0.15)
    ax_de.tick_params(direction="in", top=True, right=True, labelsize=10)
    pulse_de, = ax_de.plot([], [], color=colors_de, lw=2.0)
    marker_de, = ax_de.plot([], [], "o", color=colors_om, markersize=6,
                            markeredgecolor="black", markeredgewidth=0.5, zorder=10)

    fill_om = [None]; fill_de = [None]

    cell_els.append({
        'bloch_lines': bloch_lines, 'bloch_pts': bloch_pts,
        'pulse_om': pulse_om, 'pulse_de': pulse_de,
        'marker_om': marker_om, 'marker_de': marker_de,
        'ax_b': ax_b, 'ax_om': ax_om, 'ax_de': ax_de,
        'time_text': time_text,
        'ctrl_quiver': ctrl_quiver, 'ctrl_dot': ctrl_dot,
        'fill_om': fill_om, 'fill_de': fill_de,
    })

# ── Animation ──
N_FRAMES = 100

def update(frame):
    progress = frame / (N_FRAMES - 1)
    for idx, gname in enumerate(gate_order):
        d = all_data[gname]
        el = cell_els[idx]
        n_fine = len(d['t_fine'])
        fi = min(int(progress * (n_fine - 1)), n_fine - 1)
        sl = slice(0, fi + 1)

        # Bloch trajectories
        for j, traj in enumerate(d['trajs']):
            el['bloch_lines'][j].set_data_3d(traj[sl, 0], traj[sl, 1], traj[sl, 2])
            el['bloch_pts'][j].set_data_3d([traj[fi, 0]], [traj[fi, 1]], [traj[fi, 2]])
        el['ax_b'].view_init(elev=22, azim=35 + frame * 0.5)
        el['time_text'].set_text(f"$t$ = {d['t_fine'][fi]:.2f}")

        # Pulses — use coarse grid (t_ctrl has ~121 points)
        n_ctrl = len(d['t'])
        ci = min(int(progress * (n_ctrl - 1)), n_ctrl - 1)
        cs = slice(0, ci + 1)
        el['pulse_om'].set_data(d['t'][cs], d['Om'][cs])
        el['pulse_de'].set_data(d['t'][cs], d['De'][cs])
        el['marker_om'].set_data([d['t'][ci]], [d['Om'][ci]])
        el['marker_de'].set_data([d['t'][ci]], [d['De'][ci]])

        hvec = np.array([d['Om'][ci], 0.0, d['De'][ci]], dtype=float)
        hnorm = np.linalg.norm(hvec)
        hdir = hvec / hnorm if hnorm > 1e-12 else hvec
        el['ctrl_quiver'][0].remove()
        el['ctrl_quiver'][0] = el['ax_b'].quiver(
            0, 0, 0, hdir[0], hdir[1], hdir[2], color="#e74c3c",
            linewidth=2.5, arrow_length_ratio=0.15, alpha=0.9, zorder=20)
        el['ctrl_dot'].set_data_3d([hdir[0]], [hdir[1]], [hdir[2]])

        # Fill
        if el['fill_om'][0] is not None:
            el['fill_om'][0].remove()
        el['fill_om'][0] = el['ax_om'].fill_between(
            d['t'][cs], 0, d['Om'][cs], color=colors_om, alpha=0.10)
        if el['fill_de'][0] is not None:
            el['fill_de'][0].remove()
        el['fill_de'][0] = el['ax_de'].fill_between(
            d['t'][cs], 0, d['De'][cs], color=colors_de, alpha=0.10)

    return []

anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=56, blit=False)
out = '/Users/strong/Desktop/research-workspace/projects/PINN/writing/manuscript/figures/SuppMovie1_rotation_XY_contrast.mp4'
anim.save(out, writer='ffmpeg', fps=18, dpi=120)
plt.close()
print(f"Saved: {out}")
