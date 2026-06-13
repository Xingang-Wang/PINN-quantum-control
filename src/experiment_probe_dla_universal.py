"""
Universal probe count law: DLA structure determines minimum probe count for ALL single-qubit gates.

Hypothesis:
- ny=0 (direct generation): 1 perpendicular probe → high fidelity
- ny≠0 (commutator generation): 1 perpendicular probe → fails

Uses perpendicular probe for each gate to isolate DLA effect from direction effect.
Validates with standard 4-probe RK4 tomography.
"""
import os, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pinn_dual_control_yz import (
    set_seed, Config, device, dtype,
    get_rates, stack_columns,
    grad_wrt_t, bloch_norm_penalty,
    choi_from_affine, choi_unitary, avg_gate_fidelity,
)

OUTDIR = "outputs_probe_dla_universal"
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# General rotation helpers
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

def perpendicular_probe(n):
    """Perpendicular probe in the x-z plane (n × ŷ).
    For x-z plane rotation axes, this gives probes that create
    significant x,z dynamics → well-determines Ω and Δ.
    Falls back to n × ẑ for y-axis gates."""
    n = n / np.linalg.norm(n)
    ref = np.array([0., 1., 0.])
    if np.abs(np.dot(n, ref)) > 0.99:
        ref = np.array([0., 0., 1.])
    v = np.cross(n, ref)
    return v / np.linalg.norm(v)

def T_direct_limit(n, theta, Omax=8.0):
    nx, ny, nz = n / np.linalg.norm(n)
    denom_parts = []
    if abs(nx) > 1e-10: denom_parts.append(Omax / abs(nx))
    if abs(nz) > 1e-10: denom_parts.append(Omax / abs(nz))
    if not denom_parts: return float('inf')
    return theta / min(denom_parts)


# ============================================================
# 1-probe PINN with learnable T, minimal constraints
# ============================================================

class PINN1Probe(nn.Module):
    """Compact PINN: 5-dim output (u_Ω, u_Δ, u_x, u_y, u_z)."""
    def __init__(self, hidden_dim=96, hidden_layers=3, init_T=1.0):
        super().__init__()
        self.T_log = nn.Parameter(torch.tensor(math.log(init_T), dtype=dtype))
        layers = []
        layers.append(nn.Linear(1, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 5))
        self.net = nn.Sequential(*layers)

    @property
    def T_param(self):
        return torch.exp(self.T_log)

    def forward(self, t):
        return self.net(t)


def build_1probe(net, t_norm, cfg, r_init):
    T_phys = net.T_param
    t = t_norm * T_phys
    raw = net(t)
    Omega = cfg.Omega_max * torch.tanh(raw[:, 0:1])
    Delta = cfg.Delta_max * torch.tanh(raw[:, 1:2])
    g = 1.0 - torch.exp(-t)
    r = stack_columns(
        r_init[0] + g[:, 0] * raw[:, 2],
        r_init[1] + g[:, 0] * raw[:, 3],
        r_init[2] + g[:, 0] * raw[:, 4],
    )
    return Omega, Delta, r, t


def train_1probe(R_bloch, r_init_np, gamma, steps=5000, init_T=0.7):
    set_seed(42)
    cfg = Config(target_gate='X', steps=steps, print_every=9999,
                 gamma_down=gamma, gamma_up=0.0)
    R_target = torch.tensor(R_bloch, dtype=dtype, device=device)
    r_init = torch.tensor(r_init_np, dtype=dtype, device=device)
    target_out = R_target @ r_init  # Expected final state: R · r_init

    net = PINN1Probe(init_T=init_T).to(device)
    optimizer = optim.Adam([
        {'params': net.net.parameters(), 'lr': cfg.lr},
        {'params': [net.T_log], 'lr': 5e-3},
    ])
    t_norm = torch.linspace(0, 1, cfg.N_t, device=device, dtype=dtype).view(-1, 1)
    G1, G2 = get_rates(cfg)
    gd = cfg.gamma_down - cfg.gamma_up

    for step in range(steps):
        optimizer.zero_grad()
        Omega, Delta, r, t = build_1probe(net, t_norm, cfg, r_init)

        # L_dyn
        x, y, z = r[:, 0:1], r[:, 1:2], r[:, 2:3]
        dx = grad_wrt_t(x, t, True)
        dy = grad_wrt_t(y, t, True)
        dz = grad_wrt_t(z, t, True)
        Rx = dx + Delta * y + G2 * x
        Ry = dy - Delta * x + Omega * z + G2 * y
        Rz = dz - Omega * y + G1 * z - gd
        L_dyn = torch.mean(Rx**2 + Ry**2 + Rz**2)

        # L_gate: single trajectory final-state matching
        s = r[-1, :]
        L_gate = torch.sum((s - target_out)**2)

        # Minimal constraints only
        L_phys = bloch_norm_penalty(r)
        L_time = 1.0 * net.T_param

        L_total = 1.0*L_dyn + 10.0*L_gate + 1e-3*L_phys + L_time
        L_total.backward()
        optimizer.step()

    return net, t_norm, cfg


# ============================================================
# RK4 validation (always 4 standard probes)
# ============================================================

def bloch_rhs_np(tv, r, tg, og, dg, G1, G2, gd):
    x, y, z = r
    O = float(np.interp(tv, tg, og))
    D = float(np.interp(tv, tg, dg))
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

def validate_4probe(net, t_norm, cfg, U_target):
    """Extract controls from trained net, validate with 4-probe RK4 tomography."""
    T_opt = net.T_param.item()
    with torch.no_grad():
        T_phys = net.T_param
        t = t_norm * T_phys
        raw = net(t)
        Omega_np = (cfg.Omega_max * torch.tanh(raw[:, 0])).cpu().numpy().flatten()
        Delta_np = (cfg.Delta_max * torch.tanh(raw[:, 1])).cpu().numpy().flatten()
        t_np = t.cpu().numpy().flatten()
    G1, G2 = get_rates(cfg); gd = cfg.gamma_down - cfg.gamma_up

    probes = [np.array([0.,0.,1.]), np.array([0.,0.,-1.]),
              np.array([1.,0.,0.]), np.array([0.,1.,0.])]
    ends = [rk4_prop(p, t_np, Omega_np, Delta_np, G1, G2, gd) for p in probes]
    s0, s1, sx, sy = ends
    c_rk4 = 0.5*(s0+s1)
    M_rk4 = np.stack([sx-c_rk4, sy-c_rk4, 0.5*(s0-s1)], axis=1)

    F_proc = float(np.clip(np.real(np.trace(
        choi_unitary(U_target) @ choi_from_affine(M_rk4, c_rk4))), 0, 1))
    F_avg = avg_gate_fidelity(F_proc)
    return T_opt, F_avg


# ============================================================
# GATE SET (11 axes from Discovery 6)
# ============================================================

gates = []

# Group A: x-z plane (ny=0, direct generation)
for alpha_deg in [0, 15, 30, 45, 60, 75, 90]:
    alpha = np.radians(alpha_deg)
    n = np.array([np.cos(alpha), 0, np.sin(alpha)])
    label = f"R(α={alpha_deg}°,π)" if alpha_deg not in [0, 90] else \
            "X(π)" if alpha_deg == 0 else "Z(π)"
    gates.append({'label': label, 'n': n, 'theta': np.pi,
                  'alpha_deg': alpha_deg, 'ny_zero': True})

# Group B: ny ≠ 0 (commutator generation)
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
print(f"  DLA UNIVERSAL PROBE LAW")
print(f"  1 perpendicular probe, minimal constraints")
print(f"  Validate with 4-probe RK4 tomography")
print(f"{'#'*70}")

for g in gates:
    n = g['n']
    ny = abs(n[1] / np.linalg.norm(n))
    theta = g['theta']
    R_bloch = rotation_bloch(n, theta)
    U_target = rotation_unitary(n, theta)
    T_theory = T_direct_limit(n, theta)

    # Perpendicular probe (isolates DLA effect from direction effect)
    r_probe = perpendicular_probe(n)
    dot_val = np.dot(n / np.linalg.norm(n), r_probe)

    print(f"\n  {g['label']:>12}  n=({n[0]:.2f},{n[1]:.2f},{n[2]:.2f})  ny={ny:.3f}")
    print(f"    probe=({r_probe[0]:.3f},{r_probe[1]:.3f},{r_probe[2]:.3f})  "
          f"n·probe={dot_val:.6f}")

    net, t_norm, cfg = train_1probe(R_bloch, r_probe, gamma)
    T_opt, F_avg = validate_4probe(net, t_norm, cfg, U_target)

    success = F_avg > 0.9
    all_results.append({
        'label': g['label'],
        'n': n.tolist(),
        'ny': ny,
        'theta': theta,
        'probe': r_probe.tolist(),
        'probe_dot_n': float(dot_val),
        'T_opt': T_opt,
        'T_direct': T_theory,
        'T_ratio': T_opt / T_theory if T_theory < 100 else float('inf'),
        'F_avg': F_avg,
        'ny_zero': g['ny_zero'],
        'alpha_deg': g['alpha_deg'],
        'success': success,
    })
    tag = "✓" if success else "✗"
    print(f"    >>> T*={T_opt:.4f}  F_avg={F_avg:.6f}  {tag}")

with open(f"{OUTDIR}/probe_dla_results.json", "w") as f:
    json.dump(all_results, f, indent=2)


# ============================================================
# SUMMARY
# ============================================================

print(f"\n\n{'='*70}")
print("DLA UNIVERSAL PROBE LAW: Summary")
print(f"{'='*70}")

print(f"\n{'Gate':>12} | {'ny':>5} | {'F_avg':>9} | {'T*':>7} | Result")
print("-"*55)
for r in all_results:
    tag = "SUCCESS" if r['success'] else "FAIL"
    print(f"{r['label']:>12} | {r['ny']:5.3f} | {r['F_avg']:9.6f} | {r['T_opt']:7.4f} | {tag}")

ny0 = [r for r in all_results if r['ny_zero']]
nyn = [r for r in all_results if not r['ny_zero']]

print(f"\n--- DLA Classification ---")
print(f"Direct generation (ny=0): {len(ny0)} gates, "
      f"{sum(1 for r in ny0 if r['success'])}/{len(ny0)} succeeded")
print(f"  F_avg range: [{min(r['F_avg'] for r in ny0):.4f}, {max(r['F_avg'] for r in ny0):.4f}]")
print(f"Commutator generation (ny≠0): {len(nyn)} gates, "
      f"{sum(1 for r in nyn if r['success'])}/{len(nyn)} succeeded")
print(f"  F_avg range: [{min(r['F_avg'] for r in nyn):.4f}, {max(r['F_avg'] for r in nyn):.4f}]")

print(f"\nResults saved to {OUTDIR}/")
