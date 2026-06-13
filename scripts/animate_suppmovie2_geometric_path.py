"""
SuppMovie2 (v2): Geometric Z gate — PPT style + orthogonal relationship.
Layout:
  Left 55%: Bloch sphere (probes solid + refs dashed, start dots, target circles)
  Right top:   Ω(t) with fill_between (PPT style)
  Right middle: Δ(t) with fill_between (PPT style)
  Right bottom: 2D perpendicularity check — control arrow vs ref projection arrow
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
    "font.size": 13,
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
})

RESULTS = '/Users/strong/Desktop/research-workspace/projects/PINN/geometric-gate/results'
BETA = 3  # best geometric gate
dname = f'{RESULTS}/outputs_turn_weighted_learnableT_env_beta{BETA}'

# ── Load data ──
ts = np.load(f'{dname}/timeseries_turn_weighted.npz')
t_ctrl = ts['t']; Om_ctrl = ts['Omega']; De_ctrl = ts['Delta']
T_val = t_ctrl[-1]

# ── RK4 ──
def rk4_full(r0, t_grid, Omega, Delta, n_steps=2000):
    T = t_grid[-1] - t_grid[0]; h = T / n_steps
    r = r0.copy(); tv = t_grid[0]; traj = [r.copy()]
    for _ in range(n_steps):
        def rhs(rv, ti):
            Om = float(np.interp(ti, t_grid, Omega))
            De = float(np.interp(ti, t_grid, Delta))
            return np.array([-De*rv[1], De*rv[0]-Om*rv[2], Om*rv[1]])
        k1 = rhs(r, tv); k2 = rhs(r+.5*h*k1, tv+.5*h)
        k3 = rhs(r+.5*h*k2, tv+.5*h); k4 = rhs(r+h*k3, tv+h)
        r += (h/6)*(k1+2*k2+2*k3+k4); tv += h; traj.append(r.copy())
    return np.array(traj)

N_STEPS = 2000
R_T = np.array([[-1,0,0],[0,-1,0],[0,0,1]])

probes = [
    (r"$|+x\rangle$", np.array([1.,0.,0.]), "#e74c3c", "o"),
    (r"$|+y\rangle$", np.array([0.,1.,0.]), "#27ae60", "s"),
    (r"$|+z\rangle$", np.array([0.,0.,1.]), "#2980b9", "^"),
]
refs = [
    (r"ref $+z$", np.array([0.,0.,1.]), "#f39c12", "D"),
    (r"ref $-z$", np.array([0.,0.,-1.]), "#8e44ad", "v"),
]

trajs_p = [rk4_full(p[1], t_ctrl, Om_ctrl, De_ctrl, N_STEPS) for p in probes]
trajs_r = [rk4_full(r[1], t_ctrl, Om_ctrl, De_ctrl, N_STEPS) for r in refs]
t_fine = np.linspace(t_ctrl[0], t_ctrl[-1], N_STEPS+1)

# ── Bloch sphere setup ──
def setup_bloch(ax):
    u = np.linspace(0, 2*np.pi, 50); v = np.linspace(0, np.pi, 25)
    xs = np.outer(np.cos(u), np.sin(v)); ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="#d0e4f5", alpha=0.06, edgecolor="none", shade=False)
    L = 1.35
    for d in [np.array([L,0,0]), np.array([0,L,0]), np.array([0,0,L])]:
        ax.plot([0, d[0]], [0, d[1]], [0, d[2]], "k-", lw=0.6, alpha=0.3)
        ax.plot([0, -d[0]], [0, -d[1]], [0, -d[2]], "k-", lw=0.6, alpha=0.3)
    th = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(th), np.sin(th), 0, "k-", lw=0.3, alpha=0.15)
    ax.plot(np.cos(th), 0, np.sin(th), "k-", lw=0.3, alpha=0.15)
    ax.plot(0, np.cos(th), np.sin(th), "k-", lw=0.3, alpha=0.15)
    ax.text(1.5, 0, 0, r"$x$", fontsize=14, ha="center")
    ax.text(0, 1.5, 0, r"$y$", fontsize=14, ha="center")
    ax.text(0, 0, 1.5, r"$z$", fontsize=14, ha="center")
    ax.set_xlim([-1.5, 1.5]); ax.set_ylim([-1.5, 1.5]); ax.set_zlim([-1.5, 1.5])
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.tick_params(length=0)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor("w")
    ax.view_init(elev=22, azim=35)

# ── Figure layout ──
fig = plt.figure(figsize=(16, 8), facecolor="white")

# Bloch sphere (left 50%)
ax_b = fig.add_axes([0.01, 0.05, 0.48, 0.88], projection="3d")
setup_bloch(ax_b)

ax_b.text2D(0.5, 0.97, r"Geometric $Z(\pi)$ Gate  ($\beta\!=\!3$)",
            transform=ax_b.transAxes, fontsize=15, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#2166ac", alpha=0.9))

# Static: start dots + target open circles
for plbl, pinit, pcol, pmarker in probes:
    tgt = R_T @ pinit
    ax_b.scatter(*tgt, facecolors="none", s=140, marker="o",
                 edgecolors=pcol, linewidths=1.5, zorder=9, depthshade=False, alpha=0.4)
    ax_b.scatter(*pinit, c=pcol, s=60, marker=pmarker,
                 edgecolors="black", linewidths=0.5, zorder=10, depthshade=False)
for rlbl, rinit, rcol, rmarker in refs:
    ax_b.scatter(*rinit, c=rcol, s=40, marker=rmarker,
                 edgecolors="black", linewidths=0.4, zorder=10, depthshade=False, alpha=0.6)

bloch_lines_p, bloch_pts_p = [], []
for plbl, pinit, pcol, pmarker in probes:
    line, = ax_b.plot([], [], [], color=pcol, lw=2.0, alpha=0.85)
    pt, = ax_b.plot([], [], [], "o", color=pcol, markersize=6,
                    markeredgecolor="black", markeredgewidth=0.5)
    bloch_lines_p.append(line); bloch_pts_p.append(pt)
bloch_lines_r = []
for rlbl, rinit, rcol, rmarker in refs:
    line, = ax_b.plot([], [], [], color=rcol, lw=1.2, alpha=0.7, ls="--")
    pt, = ax_b.plot([], [], [], "o", color=rcol, markersize=4,
                    markeredgecolor="black", markeredgewidth=0.3)
    bloch_lines_r.append((line, pt))

# Control field arrow on Bloch sphere (Hamiltonian axis direction)
# H ~ (Omega * sigma_x + Delta * sigma_z) → rotation axis n = (Ω, 0, Δ)/|Ω,Δ|
ctrl_quiver = [ax_b.quiver(0, 0, 0, 0, 0, 0, color="#e74c3c", linewidth=2.5,
                           arrow_length_ratio=0.15, alpha=0.9, zorder=20)]
ctrl_dot = ax_b.plot([], [], [], "o", color="#e74c3c", markersize=4, zorder=20)[0]

handles = []
for plbl, _, pcol, pmarker in probes:
    handles.append(Line2D([0],[0], color=pcol, marker=pmarker, ls="-",
                   markersize=5, label=plbl, markeredgecolor="black",
                   markeredgewidth=0.4, lw=1.5))
for rlbl, _, rcol, rmarker in refs:
    handles.append(Line2D([0],[0], color=rcol, marker=rmarker, ls="--",
                   markersize=4, label=rlbl, markeredgecolor="black",
                   markeredgewidth=0.3, lw=1.0))
ax_b.legend(handles=handles, loc="lower left", fontsize=9, ncol=2,
            bbox_to_anchor=(-0.02, -0.02), frameon=True, framealpha=0.85)
# Add control arrow to legend
handles_ctrl = handles + [Line2D([0],[0], color="#e74c3c", lw=2.5, label=r"$\mathbf{n}(t)$")]
ax_b.legend(handles=handles_ctrl, loc="lower left", fontsize=9, ncol=2,
            bbox_to_anchor=(-0.02, -0.02), frameon=True, framealpha=0.85)

time_text = ax_b.text2D(0.02, 0.02, "", transform=ax_b.transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="gray", alpha=0.8))

# ── Right column: Ω, Δ, and orthogonality ──
colors_om, colors_de = "#2166ac", "#b2182b"

# Ω panel (right top)
ax_om = fig.add_axes([0.57, 0.72, 0.40, 0.23])
ax_om.set_ylabel(r"$\Omega(t)$", fontsize=13)
ax_om.set_xlim([t_ctrl[0], t_ctrl[-1]])
om_range = max(abs(Om_ctrl.max()), abs(Om_ctrl.min())) * 1.15
ax_om.set_ylim([-om_range, om_range])
ax_om.axhline(0, color="gray", lw=0.4, ls="--")
ax_om.plot(t_ctrl, Om_ctrl, color=colors_om, lw=0.7, alpha=0.15)
ax_om.tick_params(direction="in", top=True, right=True, labelsize=10)
ax_om.set_xticklabels([])
ax_om.set_title("Control fields", fontsize=13, pad=4)

pulse_om, = ax_om.plot([], [], color=colors_om, lw=2.0)
marker_om, = ax_om.plot([], [], "o", color=colors_de, markersize=6,
                        markeredgecolor="black", markeredgewidth=0.5, zorder=10)

# Δ panel (right middle)
ax_de = fig.add_axes([0.57, 0.42, 0.40, 0.23])
ax_de.set_ylabel(r"$\Delta(t)$", fontsize=13)
ax_de.set_xlim([t_ctrl[0], t_ctrl[-1]])
de_range = max(abs(De_ctrl.max()), abs(De_ctrl.min())) * 1.15
ax_de.set_ylim([-de_range, de_range])
ax_de.axhline(0, color="gray", lw=0.4, ls="--")
ax_de.plot(t_ctrl, De_ctrl, color=colors_de, lw=0.7, alpha=0.15)
ax_de.tick_params(direction="in", top=True, right=True, labelsize=10)
ax_de.set_xticklabels([])

pulse_de, = ax_de.plot([], [], color=colors_de, lw=2.0)
marker_de, = ax_de.plot([], [], "o", color=colors_om, markersize=6,
                        markeredgecolor="black", markeredgewidth=0.5, zorder=10)

# Orthogonality panel (right bottom) — THE KEY PANEL
ax_orth = fig.add_axes([0.57, 0.05, 0.40, 0.30])
ax_orth.set_xlim(-1.5, 1.5); ax_orth.set_ylim(-1.5, 1.5)
ax_orth.set_aspect('equal')
ax_orth.axhline(0, color="gray", lw=0.3, alpha=0.5)
ax_orth.axvline(0, color="gray", lw=0.3, alpha=0.5)
# Unit circle
th = np.linspace(0, 2*np.pi, 100)
ax_orth.plot(np.cos(th), np.sin(th), color='#ccc', lw=0.5, alpha=0.5)
ax_orth.set_xlabel(r"$x\; / \;\hat\Omega$", fontsize=12)
ax_orth.set_ylabel(r"$z\; / \;\hat\Delta$", fontsize=12)
ax_orth.set_title("Parallel transport: $(\\Omega,\\Delta) \\perp (x_p, z_p)$", fontsize=11, fontweight='bold')
ax_orth.tick_params(labelsize=8)

# Control direction trajectory (faded)
ctrl_traj = np.column_stack([Om_ctrl, De_ctrl])
ctrl_norm = np.linalg.norm(ctrl_traj, axis=1, keepdims=True)
ctrl_norm = np.where(ctrl_norm > 1e-8, ctrl_norm, 1.0)
ctrl_dir = ctrl_traj / ctrl_norm  # normalized
ax_orth.plot(ctrl_dir[:, 0], ctrl_dir[:, 1], color=colors_om, lw=0.5, alpha=0.15)

# Reference state projection trajectory (faded)
ref_p_traj = trajs_r[0]  # ref_plus
rp_xz = ref_p_traj[:, [0, 2]]  # (x, z) projection
rp_norm = np.linalg.norm(rp_xz, axis=1, keepdims=True)
rp_norm = np.where(rp_norm > 1e-8, rp_norm, 1.0)
rp_dir = rp_xz / rp_norm
# Resample to control grid
from scipy.interpolate import interp1d
t_fine_full = t_fine
interp_x = interp1d(t_fine_full, rp_dir[:, 0], kind='linear')
interp_z = interp1d(t_fine_full, rp_dir[:, 1], kind='linear')
rp_dir_ctrl = np.column_stack([interp_x(t_ctrl), interp_z(t_ctrl)])
ax_orth.plot(rp_dir_ctrl[:, 0], rp_dir_ctrl[:, 1], color="#f39c12", lw=0.5, alpha=0.15)

# Animated arrows (will be redrawn each frame)
arrow_ctrl = [None]
arrow_ref = [None]
angle_text = ax_orth.text(0.02, 0.02, "", transform=ax_orth.transAxes, fontsize=11,
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                    edgecolor="gray", alpha=0.9))
dot_ctrl, = ax_orth.plot([], [], "o", color=colors_om, markersize=8,
                         markeredgecolor="black", markeredgewidth=0.5, zorder=10)
dot_ref, = ax_orth.plot([], [], "o", color="#f39c12", markersize=8,
                        markeredgecolor="black", markeredgewidth=0.5, zorder=10)
# Right angle marker
right_angle_line, = ax_orth.plot([], [], color="gray", lw=1.0, alpha=0.6)

# Legend for orth panel
leg_orth = [Line2D([0],[0], color=colors_om, lw=2, label=r"$(\hat\Omega, \hat\Delta)$"),
            Line2D([0],[0], color="#f39c12", lw=2, label=r"$(\hat x_p, \hat z_p)$")]
ax_orth.legend(handles=leg_orth, fontsize=9, loc="upper right", framealpha=0.85)

fill_om = [None]; fill_de = [None]

# ── Animation ──
n_ctrl = len(t_ctrl)
frame_idx = list(range(0, n_ctrl, 2))
if frame_idx[-1] != n_ctrl - 1:
    frame_idx.append(n_ctrl - 1)
n_frames = len(frame_idx)

def update(frame):
    ci = frame_idx[frame]
    progress = ci / (n_ctrl - 1)
    fi = min(int(progress * N_STEPS), N_STEPS)
    sl = slice(0, fi + 1)
    cs = slice(0, ci + 1)

    # Bloch trajectories
    for j, traj in enumerate(trajs_p):
        bloch_lines_p[j].set_data_3d(traj[sl, 0], traj[sl, 1], traj[sl, 2])
        bloch_pts_p[j].set_data_3d([traj[fi, 0]], [traj[fi, 1]], [traj[fi, 2]])
    for j, (line, pt) in enumerate(bloch_lines_r):
        tr = trajs_r[j]
        line.set_data_3d(tr[sl, 0], tr[sl, 1], tr[sl, 2])
        pt.set_data_3d([tr[fi, 0]], [tr[fi, 1]], [tr[fi, 2]])
    ax_b.view_init(elev=22, azim=35 + frame * 0.4)
    time_text.set_text(f"$t$ = {t_ctrl[ci]:.2f}")

    # Control arrow on Bloch sphere: axis direction n = (Ω, 0, Δ)/|Ω,Δ|
    Om_now = Om_ctrl[ci]; De_now = De_ctrl[ci]
    ctrl_mag = np.sqrt(Om_now**2 + De_now**2)
    if ctrl_mag > 1e-8:
        n_dir = np.array([Om_now, 0, De_now]) / ctrl_mag
        arrow_len = 1.2
        # Remove old quiver, draw new one
        ctrl_quiver[0].remove()
        ctrl_quiver[0] = ax_b.quiver(0, 0, 0, n_dir[0]*arrow_len, n_dir[1]*arrow_len, n_dir[2]*arrow_len,
                                      color="#e74c3c", linewidth=2.5, arrow_length_ratio=0.12,
                                      alpha=0.9, zorder=20)
        ctrl_dot.set_data_3d([n_dir[0]*arrow_len], [n_dir[1]*arrow_len], [n_dir[2]*arrow_len])
    else:
        ctrl_dot.set_data_3d([], [], [])

    # Pulses
    pulse_om.set_data(t_ctrl[cs], Om_ctrl[cs])
    pulse_de.set_data(t_ctrl[cs], De_ctrl[cs])
    marker_om.set_data([t_ctrl[ci]], [Om_ctrl[ci]])
    marker_de.set_data([t_ctrl[ci]], [De_ctrl[ci]])

    if fill_om[0] is not None: fill_om[0].remove()
    fill_om[0] = ax_om.fill_between(t_ctrl[cs], 0, Om_ctrl[cs], color=colors_om, alpha=0.10)
    if fill_de[0] is not None: fill_de[0].remove()
    fill_de[0] = ax_de.fill_between(t_ctrl[cs], 0, De_ctrl[cs], color=colors_de, alpha=0.10)

    # ── Orthogonality panel ──
    # Control direction
    Om_now = Om_ctrl[ci]; De_now = De_ctrl[ci]
    ctrl_mag = np.sqrt(Om_now**2 + De_now**2)
    if ctrl_mag > 1e-8:
        cd = np.array([Om_now, De_now]) / ctrl_mag
    else:
        cd = np.array([0.0, 0.0])

    # Reference state projection direction
    rp_now = trajs_r[0][fi]  # ref_plus at fine grid
    rp_xz_now = np.array([rp_now[0], rp_now[2]])
    rp_mag = np.linalg.norm(rp_xz_now)
    if rp_mag > 1e-8:
        rd = rp_xz_now / rp_mag
    else:
        rd = np.array([0.0, 0.0])

    # Draw arrows
    if arrow_ctrl[0] is not None:
        arrow_ctrl[0].remove()
    if arrow_ref[0] is not None:
        arrow_ref[0].remove()

    arrow_len = 1.1
    arrow_ctrl[0] = ax_orth.annotate("", xy=(cd[0]*arrow_len, cd[1]*arrow_len), xytext=(0, 0),
                                      arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.15",
                                                      color=colors_om, lw=2.5))
    arrow_ref[0] = ax_orth.annotate("", xy=(rd[0]*arrow_len, rd[1]*arrow_len), xytext=(0, 0),
                                     arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.15",
                                                     color="#f39c12", lw=2.5))
    dot_ctrl.set_data([cd[0]*arrow_len], [cd[1]*arrow_len])
    dot_ref.set_data([rd[0]*arrow_len], [rd[1]*arrow_len])

    # Angle between them
    cos_angle = np.clip(np.dot(cd, rd), -1, 1)
    angle_deg = np.degrees(np.arccos(abs(cos_angle)))
    E_now = 0.5 * (Om_now * rp_now[0] + De_now * rp_now[2])

    angle_text.set_text(f"$\\angle$ = {angle_deg:.1f}°   $E$ = {E_now:.2e}")

    # Right-angle marker (small square at intersection)
    if ctrl_mag > 0.1 and rp_mag > 0.1:
        sz = 0.12
        corner = cd * sz + rd * sz
        right_angle_line.set_data([cd[0]*sz, corner[0], rd[0]*sz],
                                  [cd[1]*sz, corner[1], rd[1]*sz])
    else:
        right_angle_line.set_data([], [])

    return []

anim = FuncAnimation(fig, update, frames=n_frames, interval=60, blit=False)
out = '/Users/strong/Desktop/research-workspace/projects/PINN/writing/manuscript/figures/SuppMovie2_geometric_path.mp4'
anim.save(out, writer='ffmpeg', fps=18, dpi=120)
plt.close()
print(f"Saved: {out}")
