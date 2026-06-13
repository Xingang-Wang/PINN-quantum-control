"""
General gates with MINIMAL constraints (only L_dyn + L_gate + L_phys + L_time)
Rerun Phase 1 to get T* closer to quantum speed limit.

Group A: x-z plane (ny=0), α = 0°,15°,30°,45°,60°,75°,90°
Group B: ny ≠ 0, 4 commutator gates
"""
import os, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pinn_dual_control_yz import (
    set_seed, Config, device, dtype,
    r0_init, r1_init, rx_init, ry_init,
    get_rates, stack_columns, stack_matrix_columns,
    grad_wrt_t, frobenius_norm_squared, vector_norm_squared,
    bloch_norm_penalty, PINNDualControl,
    choi_from_affine, choi_unitary, avg_gate_fidelity,
)

OUTDIR = "outputs_general_gates_minimal"
os.makedirs(OUTDIR, exist_ok=True)

Omega_max = 8.0

# ============================================================
# General rotation
# ============================================================

def rotation_bloch(n, theta):
    nx, ny, nz = n / np.linalg.norm(n)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c + (1-c)*nx*nx,      (1-c)*nx*ny - s*nz,  (1-c)*nx*nz + s*ny],
        [(1-c)*ny*nx + s*nz,   c + (1-c)*ny*ny,      (1-c)*ny*nz - s*nx],
        [(1-c)*nz*nx - s*ny,   (1-c)*nz*ny + s*nx,   c + (1-c)*nz*nz  ],
    ], dtype=np.float64)

def rotation_unitary(n, theta):
    nx, ny, nz = n / np.linalg.norm(n)
    c2, s2 = np.cos(theta/2), np.sin(theta/2)
    return np.array([
        [c2 - 1j*s2*nz,  -1j*s2*nx - s2*ny],
        [-1j*s2*nx + s2*ny,  c2 + 1j*s2*nz],
    ], dtype=np.complex128)

def T_direct_limit(n, theta, Omax=8.0):
    nx, ny, nz = n / np.linalg.norm(n)
    denom_parts = []
    if abs(nx) > 1e-10: denom_parts.append(Omax / abs(nx))
    if abs(nz) > 1e-10: denom_parts.append(Omax / abs(nz))
    if not denom_parts: return float('inf')
    r_max = min(denom_parts)
    return theta / r_max


# ============================================================
# PINN with learnable T, MINIMAL constraints
# ============================================================

class PINNLearnableT(nn.Module):
    def __init__(self, hidden_dim=96, hidden_layers=3, init_T=1.0):
        super().__init__()
        self.T_log = nn.Parameter(torch.tensor(math.log(init_T), dtype=dtype))
        self.net = PINNDualControl(hidden_dim, hidden_layers)

    @property
    def T_param(self):
        return torch.exp(self.T_log)

    def forward(self, t):
        return self.net(t)


def build(net, t_norm, cfg):
    T_phys = net.T_param
    t = t_norm * T_phys
    raw = net(t)
    u_Omega, u_Delta = raw[:, 0:1], raw[:, 1:2]
    Omega = cfg.Omega_max * torch.tanh(u_Omega)
    Delta = cfg.Delta_max * torch.tanh(u_Delta)
    g = 1.0 - torch.exp(-t)
    def traj(r_init, ux, uy, uz):
        return stack_columns(
            r_init[0] + g[:, 0] * ux[:, 0],
            r_init[1] + g[:, 0] * uy[:, 0],
            r_init[2] + g[:, 0] * uz[:, 0])
    r0 = traj(r0_init, raw[:, 2:3],  raw[:, 3:4],  raw[:, 4:5])
    r1 = traj(r1_init, raw[:, 5:6],  raw[:, 6:7],  raw[:, 7:8])
    rx = traj(rx_init, raw[:, 8:9],  raw[:, 9:10], raw[:, 10:11])
    ry = traj(ry_init, raw[:, 11:12], raw[:, 12:13], raw[:, 13:14])
    return Omega, Delta, r0, r1, rx, ry, t


def train_gate_minimal(R_bloch, gamma, steps=5000, init_T=0.7):
    set_seed(42)
    cfg = Config(target_gate='X', steps=steps, print_every=9999,
                 gamma_down=gamma, gamma_up=0.0)
    R_target = torch.tensor(R_bloch, dtype=dtype, device=device)

    net = PINNLearnableT(init_T=init_T).to(device)
    optimizer = optim.Adam([
        {'params': net.net.parameters(), 'lr': cfg.lr},
        {'params': [net.T_log], 'lr': 5e-3},
    ])
    t_norm = torch.linspace(0, 1, cfg.N_t, device=device, dtype=dtype).view(-1, 1)
    G1, G2 = get_rates(cfg)
    gd = cfg.gamma_down - cfg.gamma_up

    for step in range(steps):
        optimizer.zero_grad()
        Omega, Delta, r0, r1, rx, ry, t = build(net, t_norm, cfg)

        # L_dyn (KEEP)
        L_dyn = 0.0
        for r in [r0, r1, rx, ry]:
            x, y, z = r[:, 0:1], r[:, 1:2], r[:, 2:3]
            dx = grad_wrt_t(x, t, True)
            dy = grad_wrt_t(y, t, True)
            dz = grad_wrt_t(z, t, True)
            Rx_r = dx + Delta * y + G2 * x
            Ry_r = dy - Delta * x + Omega * z + G2 * y
            Rz_r = dz - Omega * y + G1 * z - gd
            L_dyn += torch.mean(Rx_r**2 + Ry_r**2 + Rz_r**2)

        # L_gate (KEEP)
        s0, s1, sx, sy = r0[-1, :], r1[-1, :], rx[-1, :], ry[-1, :]
        c = 0.5 * (s0 + s1)
        M = stack_matrix_columns(sx - c, sy - c, 0.5 * (s0 - s1))
        L_gate = frobenius_norm_squared(M - R_target) + vector_norm_squared(c)

        # L_phys (KEEP)
        L_phys = sum(bloch_norm_penalty(r) for r in [r0, r1, rx, ry])

        # L_time (KEEP)
        L_time = 1.0 * net.T_param

        # REMOVED: L_boundary, L_amp, L_smooth
        L_total = 1.0*L_dyn + 10.0*L_gate + 1e-3*L_phys + L_time
        L_total.backward()
        optimizer.step()

    return net, t_norm, cfg


# ============================================================
# RK4 validation
# ============================================================

def omega_interp(t_q, tg, og): return float(np.interp(t_q, tg, og))
def delta_interp(t_q, tg, dg): return float(np.interp(t_q, tg, dg))
def bloch_rhs_np(tv, r, tg, og, dg, G1, G2, gd):
    x, y, z = r
    O = omega_interp(tv, tg, og); D = delta_interp(tv, tg, dg)
    return np.array([-D*y-G2*x, D*x-O*z-G2*y, O*y-G1*z+gd], dtype=np.float64)

def rk4_prop(r_init, tg, og, dg, G1, G2, gd, n_steps=4000):
    t0, t1 = float(tg[0]), float(tg[-1]); h = (t1-t0)/n_steps
    r = r_init.astype(np.float64).copy(); tv = t0
    for _ in range(n_steps):
        k1=bloch_rhs_np(tv,r,tg,og,dg,G1,G2,gd)
        k2=bloch_rhs_np(tv+.5*h,r+.5*h*k1,tg,og,dg,G1,G2,gd)
        k3=bloch_rhs_np(tv+.5*h,r+.5*h*k2,tg,og,dg,G1,G2,gd)
        k4=bloch_rhs_np(tv+h,r+h*k3,tg,og,dg,G1,G2,gd)
        r+=(h/6)*(k1+2*k2+2*k3+k4); tv+=h
    return r


def validate(net, t_norm, cfg, R_bloch, U_target):
    T_opt = net.T_param.item()
    Omega, Delta, r0, r1, rx, ry, t = build(net, t_norm, cfg)
    Omega_np = Omega.detach().cpu().numpy().squeeze()
    Delta_np = Delta.detach().cpu().numpy().squeeze()
    t_np = t.detach().cpu().numpy().squeeze()
    G1, G2 = get_rates(cfg); gd = cfg.gamma_down - cfg.gamma_up

    probes = [np.array([0.,0.,1.]),np.array([0.,0.,-1.]),
              np.array([1.,0.,0.]),np.array([0.,1.,0.])]
    ends = [rk4_prop(p,t_np,Omega_np,Delta_np,G1,G2,gd) for p in probes]
    s0,s1,sx,sy = ends
    c_rk4 = 0.5*(s0+s1)
    M_rk4 = np.stack([sx-c_rk4, sy-c_rk4, 0.5*(s0-s1)], axis=1)

    F_proc = float(np.clip(np.real(np.trace(choi_unitary(U_target) @ choi_from_affine(M_rk4, c_rk4))), 0, 1))
    F_avg = avg_gate_fidelity(F_proc)

    Omega_max_val = float(np.max(np.abs(Omega_np)))
    Delta_max_val = float(np.max(np.abs(Delta_np)))

    return {
        "T_opt": T_opt,
        "F_avg": F_avg,
        "Omega_max": Omega_max_val,
        "Delta_max": Delta_max_val,
        "ratio_OD": Omega_max_val / (Delta_max_val + 1e-12),
    }


# ============================================================
# GATE SET
# ============================================================

gates = []

# Group A: x-z plane (ny=0)
for alpha_deg in [0, 15, 30, 45, 60, 75, 90]:
    alpha = np.radians(alpha_deg)
    n = np.array([np.cos(alpha), 0, np.sin(alpha)])
    label = f"R(α={alpha_deg}°,π)" if alpha_deg not in [0, 90] else \
            f"X(π)" if alpha_deg == 0 else f"Z(π)"
    gates.append({'label': label, 'n': n, 'theta': np.pi,
                  'alpha_deg': alpha_deg, 'ny_zero': True})

# Group B: ny ≠ 0
for axis, label in [((0,1,0), "Y(π)"), ((1,1,0), "R(xy,π)"),
                     ((0,1,1), "R(yz,π)"), ((1,1,1), "R(xyz,π)")]:
    n = np.array(axis, dtype=np.float64)
    n = n / np.linalg.norm(n)
    gates.append({'label': label, 'n': n, 'theta': np.pi,
                  'alpha_deg': None, 'ny_zero': False})


# ============================================================
# RUN
# ============================================================

gamma = 0.05
all_results = []

print(f"{'#'*70}")
print(f"  MINIMAL CONSTRAINTS: θ = π, vary rotation axis")
print(f"  Loss = L_dyn + 10·L_gate + 1e-3·L_phys + L_time")
print(f"{'#'*70}")

for g in gates:
    n = g['n']
    theta = g['theta']
    R_bloch = rotation_bloch(n, theta)
    U_target = rotation_unitary(n, theta)
    T_theory = T_direct_limit(n, theta, Omega_max)

    print(f"\n  {g['label']:>12}  n=({n[0]:.2f},{n[1]:.2f},{n[2]:.2f})  T_direct={T_theory:.4f}")
    net, t_norm, cfg = train_gate_minimal(R_bloch, gamma)
    r = validate(net, t_norm, cfg, R_bloch, U_target)
    r['label'] = g['label']
    r['n'] = n.tolist()
    r['theta'] = theta
    r['gamma'] = gamma
    r['ny_zero'] = g['ny_zero']
    r['T_direct'] = T_theory
    r['T_ratio'] = r['T_opt'] / T_theory if T_theory < 100 else float('inf')
    r['alpha_deg'] = g['alpha_deg']
    all_results.append(r)
    print(f"  >>> T*={r['T_opt']:.4f}  T_direct={T_theory:.4f}  ratio={r['T_ratio']:.3f}  "
          f"F={r['F_avg']:.6f}  |Ω|={r['Omega_max']:.3f}  |Δ|={r['Delta_max']:.3f}")

with open(f"{OUTDIR}/minimal_results.json", "w") as f:
    json.dump(all_results, f, indent=2)


# ============================================================
# SUMMARY
# ============================================================

print(f"\n\n{'='*90}")
print("MINIMAL CONSTRAINTS: T* vs quantum speed limit")
print(f"{'='*90}")

print(f"\n--- Group A: x-z plane (ny=0, direct generation) ---")
print(f"{'Gate':>12} | {'T*':>7} {'T_direct':>9} {'ratio':>6} | {'F_avg':>9} {'|Ω|/|Δ|':>7}")
print("-"*65)
for r in all_results:
    if r['ny_zero']:
        print(f"{r['label']:>12} | {r['T_opt']:7.4f} {r['T_direct']:9.4f} {r['T_ratio']:6.3f} | "
              f"{r['F_avg']:9.6f} {r['ratio_OD']:7.3f}")

print(f"\n--- Group B: ny ≠ 0 (commutator generation) ---")
print(f"{'Gate':>12} | {'T*':>7} {'T_direct':>9} {'ratio':>6} | {'F_avg':>9} {'|Ω|/|Δ|':>7}")
print("-"*65)
for r in all_results:
    if not r['ny_zero']:
        n = r['n']
        print(f"{r['label']:>12} | {r['T_opt']:7.4f} {r['T_direct']:9.4f} {r['T_ratio']:6.3f} | "
              f"{r['F_avg']:9.6f} {r['ratio_OD']:7.3f}")

print(f"\nResults saved to {OUTDIR}/")
