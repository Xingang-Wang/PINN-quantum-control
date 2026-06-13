"""
PINN geometric Z gate with turning-aware geometricity loss.

This script is the direct next-step implementation after:
1) mechanism validation for turning bottleneck,
2) fixed-T turning-aware L_geo,
3) optional learnable-T turning-aware L_geo.

It is intentionally close to the uploaded
- pinn_geometric_Z_gate_specialized.py
- pinn_geometric_Z_gate_learnable_T.py
so comparisons stay fair.
"""

import os
import json
import math
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

R_TARGET_Z = torch.tensor([
    [-1.0, 0.0, 0.0],
    [ 0.0,-1.0, 0.0],
    [ 0.0, 0.0, 1.0]
], device=device, dtype=dtype)

probe_x_init = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
probe_y_init = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
probe_z_init = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
ref_plus_init  = torch.tensor([0.0, 0.0,  1.0], device=device, dtype=dtype)
ref_minus_init = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype)


@dataclass
class Config:
    N_t: int = 201
    T: float = 1.0
    init_T: float = 1.0
    learnable_T: bool = False
    Omega_max: float = 12.0
    Delta_max: float = 12.0
    hidden_dim: int = 256
    hidden_layers: int = 6
    lr: float = 3e-4
    lr_T: float = 5e-3
    steps: int = 30000
    print_every: int = 500
    lambda_dyn: float = 80.0
    lambda_gate: float = 50.0
    lambda_cyc: float = 80.0
    lambda_geo: float = 300.0
    lambda_geo_init: float = 10.0
    lambda_ortho: float = 10.0
    lambda_purity: float = 5.0
    lambda_amp: float = 1e-4
    lambda_smooth: float = 1e-4
    lambda_T: float = 1.0
    beta_turn: float = 0.0
    turn_detach: bool = True
    turn_normalize_eps: float = 1e-8
    warmup_fraction: float = 0.3
    rk4_steps: int = 4000
    n_random_states: int = 200
    use_envelope: bool = True
    output_dir: str = 'outputs_turn_weighted'


def grad_wrt_t(y, t, *, create_graph):
    if y.ndim == 1:
        y = y.unsqueeze(1)
    return torch.autograd.grad(outputs=y.sum(), inputs=t, create_graph=create_graph, retain_graph=True, allow_unused=False)[0]


def stack_columns(x, y, z):
    if x.ndim == 2 and x.shape[1] == 1: x = x.squeeze(1)
    if y.ndim == 2 and y.shape[1] == 1: y = y.squeeze(1)
    if z.ndim == 2 and z.shape[1] == 1: z = z.squeeze(1)
    return torch.stack([x, y, z], dim=1)


def stack_matrix_columns(c1, c2, c3):
    return torch.stack([c1, c2, c3], dim=1)


def frobenius_norm_squared(M):
    return torch.sum(M * M)


def vector_norm_squared(v):
    return torch.sum(v * v)


def cross_with_control(r, Omega, Delta):
    x, y, z = r[:, 0:1], r[:, 1:2], r[:, 2:3]
    dx = -Delta * y
    dy = Delta * x - Omega * z
    dz = Omega * y
    return torch.cat([dx, dy, dz], dim=1)


def purity_penalty(r):
    norm2 = torch.sum(r * r, dim=1)
    return torch.mean((norm2 - 1.0) ** 2)


class PINNGeometricZGateFlexible(nn.Module):
    def __init__(self, hidden_dim=256, hidden_layers=6, init_T=1.0, learnable_T=False):
        super().__init__()
        self.learnable_T = learnable_T
        if learnable_T:
            self.T_log = nn.Parameter(torch.tensor(math.log(init_T), dtype=dtype))
        else:
            self.register_buffer('T_const', torch.tensor(init_T, dtype=dtype))
        layers: List[nn.Module] = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 17)]
        self.net = nn.Sequential(*layers)

    @property
    def T_param(self):
        return torch.exp(self.T_log) if self.learnable_T else self.T_const

    def forward(self, t):
        return self.net(t)


def build_controls_and_trajectories(net, t_input, cfg):
    T_phys = net.T_param
    t = t_input * T_phys if cfg.learnable_T else t_input
    raw = net(t)

    u_Omega = raw[:, 0:1]
    u_Delta = raw[:, 1:2]
    upx_x, upx_y, upx_z = raw[:, 2:3], raw[:, 3:4], raw[:, 4:5]
    upy_x, upy_y, upy_z = raw[:, 5:6], raw[:, 6:7], raw[:, 7:8]
    upz_x, upz_y, upz_z = raw[:, 8:9], raw[:, 9:10], raw[:, 10:11]
    urp_x, urp_y, urp_z = raw[:, 11:12], raw[:, 12:13], raw[:, 13:14]
    urm_x, urm_y, urm_z = raw[:, 14:15], raw[:, 15:16], raw[:, 16:17]

    if cfg.use_envelope:
        envelope = torch.sin(math.pi * t / T_phys)
        Omega = cfg.Omega_max * envelope * torch.tanh(u_Omega)
        Delta = cfg.Delta_max * envelope * torch.tanh(u_Delta)
    else:
        Omega = cfg.Omega_max * torch.tanh(u_Omega)
        Delta = cfg.Delta_max * torch.tanh(u_Delta)

    g = 1.0 - torch.exp(-t)
    probe_x = stack_columns(probe_x_init[0] + g[:, 0] * upx_x[:, 0], probe_x_init[1] + g[:, 0] * upx_y[:, 0], probe_x_init[2] + g[:, 0] * upx_z[:, 0])
    probe_y = stack_columns(probe_y_init[0] + g[:, 0] * upy_x[:, 0], probe_y_init[1] + g[:, 0] * upy_y[:, 0], probe_y_init[2] + g[:, 0] * upy_z[:, 0])
    probe_z = stack_columns(probe_z_init[0] + g[:, 0] * upz_x[:, 0], probe_z_init[1] + g[:, 0] * upz_y[:, 0], probe_z_init[2] + g[:, 0] * upz_z[:, 0])
    ref_plus = stack_columns(ref_plus_init[0] + g[:, 0] * urp_x[:, 0], ref_plus_init[1] + g[:, 0] * urp_y[:, 0], ref_plus_init[2] + g[:, 0] * urp_z[:, 0])
    ref_minus = stack_columns(ref_minus_init[0] + g[:, 0] * urm_x[:, 0], ref_minus_init[1] + g[:, 0] * urm_y[:, 0], ref_minus_init[2] + g[:, 0] * urm_z[:, 0])
    return {
        'raw': raw, 'Omega': Omega, 'Delta': Delta, 't_phys': t,
        'probe_x': probe_x, 'probe_y': probe_y, 'probe_z': probe_z,
        'ref_plus': ref_plus, 'ref_minus': ref_minus,
    }


def single_bloch_dynamical_loss(r, Omega, Delta, t, *, create_graph):
    x, y, z = r[:, 0:1], r[:, 1:2], r[:, 2:3]
    dx_dt = grad_wrt_t(x, t, create_graph=create_graph)
    dy_dt = grad_wrt_t(y, t, create_graph=create_graph)
    dz_dt = grad_wrt_t(z, t, create_graph=create_graph)
    rhs = cross_with_control(r, Omega, Delta)
    return torch.mean((dx_dt - rhs[:, 0:1])**2 + (dy_dt - rhs[:, 1:2])**2 + (dz_dt - rhs[:, 2:3])**2)


def compute_dynamical_loss(bundle, t, *, create_graph):
    Omega, Delta = bundle['Omega'], bundle['Delta']
    return sum(single_bloch_dynamical_loss(bundle[k], Omega, Delta, t, create_graph=create_graph)
               for k in ['probe_x', 'probe_y', 'probe_z', 'ref_plus', 'ref_minus'])


def reconstruct_rotation_from_probes(bundle):
    s_x = bundle['probe_x'][-1, :]
    s_y = bundle['probe_y'][-1, :]
    s_z = bundle['probe_z'][-1, :]
    R = stack_matrix_columns(s_x, s_y, s_z)
    return s_x, s_y, s_z, R


def compute_gate_loss(R):
    return frobenius_norm_squared(R - R_TARGET_Z)


def compute_cycle_loss(bundle):
    return vector_norm_squared(bundle['ref_plus'][-1, :] - ref_plus_init) + vector_norm_squared(bundle['ref_minus'][-1, :] - ref_minus_init)


def compute_turning_rate(bundle, t, *, create_graph):
    Omega = bundle['Omega']
    Delta = bundle['Delta']
    dOmega_dt = grad_wrt_t(Omega, t, create_graph=create_graph)
    dDelta_dt = grad_wrt_t(Delta, t, create_graph=create_graph)
    psi_dot = (Omega * dDelta_dt - Delta * dOmega_dt) / (Omega**2 + Delta**2 + 1e-9)
    kappa = torch.abs(psi_dot)
    return psi_dot, kappa, dOmega_dt, dDelta_dt


def compute_geometricity_loss(bundle, t, cfg, *, create_graph):
    Omega = bundle['Omega'][:, 0]
    Delta = bundle['Delta'][:, 0]
    ref_p, ref_m = bundle['ref_plus'], bundle['ref_minus']
    E_p = 0.5 * (Omega * ref_p[:, 0] + Delta * ref_p[:, 2])
    E_m = 0.5 * (Omega * ref_m[:, 0] + Delta * ref_m[:, 2])

    if cfg.beta_turn > 0.0:
        _, kappa, _, _ = compute_turning_rate(bundle, t, create_graph=create_graph)
        kappa_1d = kappa[:, 0]
        denom = torch.max(kappa_1d.detach()) + cfg.turn_normalize_eps
        kappa_norm = kappa_1d / denom
        if cfg.turn_detach:
            w = 1.0 + cfg.beta_turn * kappa_norm.detach()
        else:
            w = 1.0 + cfg.beta_turn * kappa_norm
        L_geo = torch.mean(w * (E_p**2 + E_m**2))
    else:
        kappa_1d = torch.zeros_like(E_p)
        L_geo = torch.mean(E_p**2 + E_m**2)

    return {
        'L_geo': L_geo,
        'E_plus': E_p,
        'E_minus': E_m,
        'mean_abs_E_plus': torch.mean(torch.abs(E_p)),
        'mean_abs_E_minus': torch.mean(torch.abs(E_m)),
        'max_abs_E_plus': torch.max(torch.abs(E_p)),
        'max_abs_E_minus': torch.max(torch.abs(E_m)),
        'kappa_turn': kappa_1d,
    }


def compute_orthogonality_loss(R):
    I3 = torch.eye(3, device=R.device, dtype=R.dtype)
    return frobenius_norm_squared(R.T @ R - I3) + (torch.det(R) - 1.0) ** 2


def compute_purity_loss(bundle):
    return sum(purity_penalty(bundle[k]) for k in ['probe_x', 'probe_y', 'probe_z', 'ref_plus', 'ref_minus'])


def compute_control_regularization(bundle, t, *, create_graph):
    _, _, dOmega_dt, dDelta_dt = compute_turning_rate(bundle, t, create_graph=create_graph)
    Omega, Delta = bundle['Omega'], bundle['Delta']
    L_amp = torch.mean(Omega**2 + Delta**2)
    L_smooth = torch.mean(dOmega_dt**2 + dDelta_dt**2)
    return L_amp, L_smooth


def compute_total_loss(net, t_input, cfg, *, create_graph):
    bundle = build_controls_and_trajectories(net, t_input, cfg)
    t_phys = bundle['t_phys']
    L_dyn = compute_dynamical_loss(bundle, t_phys, create_graph=create_graph)
    _, _, _, R = reconstruct_rotation_from_probes(bundle)
    L_gate = compute_gate_loss(R)
    L_cyc = compute_cycle_loss(bundle)
    geo = compute_geometricity_loss(bundle, t_phys, cfg, create_graph=create_graph)
    L_ortho = compute_orthogonality_loss(R)
    L_purity = compute_purity_loss(bundle)
    L_amp, L_smooth = compute_control_regularization(bundle, t_phys, create_graph=create_graph)
    L_T = net.T_param if cfg.learnable_T else torch.tensor(0.0, device=device, dtype=dtype)

    warm_bundle = {
        'R': R, 'L_dyn': L_dyn, 'L_gate': L_gate, 'L_cyc': L_cyc, 'L_ortho': L_ortho,
        'L_purity': L_purity, 'L_amp': L_amp, 'L_smooth': L_smooth, 'L_T': L_T,
    }
    warm_bundle.update(bundle)
    warm_bundle.update(geo)
    return warm_bundle


def interp_numpy(t_query, t_grid, value_grid):
    return float(np.interp(t_query, t_grid, value_grid))


def bloch_rhs_numpy(t_val, r, t_grid, omega_grid, delta_grid):
    Om = interp_numpy(t_val, t_grid, omega_grid)
    De = interp_numpy(t_val, t_grid, delta_grid)
    x, y, z = r
    return np.array([-De*y, De*x - Om*z, Om*y], dtype=np.float64)


def rk4_propagate_single(r_init, t_grid, omega_grid, delta_grid, n_steps):
    t0, t1 = float(t_grid[0]), float(t_grid[-1])
    h = (t1 - t0) / n_steps
    r = r_init.astype(np.float64).copy()
    t_val = t0
    for _ in range(n_steps):
        k1 = bloch_rhs_numpy(t_val, r, t_grid, omega_grid, delta_grid)
        k2 = bloch_rhs_numpy(t_val + 0.5*h, r + 0.5*h*k1, t_grid, omega_grid, delta_grid)
        k3 = bloch_rhs_numpy(t_val + 0.5*h, r + 0.5*h*k2, t_grid, omega_grid, delta_grid)
        k4 = bloch_rhs_numpy(t_val + h, r + h*k3, t_grid, omega_grid, delta_grid)
        r += (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t_val += h
    return r


def process_fidelity_from_rotation(R, R_target):
    return float(np.clip(np.real((np.trace(R_target.T @ R) + 1.0) / 4.0), 0, 1))


def average_gate_fidelity(F_proc, d=2):
    return float((d * F_proc + 1.0) / (d + 1.0))


def qubit_state_fidelity(r, s):
    nr2, ns2, rs = float(np.dot(r, r)), float(np.dot(s, s)), float(np.dot(r, s))
    return float(np.clip(0.5 * (1 + rs + math.sqrt(max(0.0, (1-nr2)*(1-ns2)))), 0.0, 1.0))


def random_state_validation(R_rk4, t_grid, omega_grid, delta_grid, n_steps, n_states=200):
    rng = np.random.default_rng(2026)
    vecs = rng.normal(size=(n_states, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    fids = []
    for r_in in vecs:
        r_out = rk4_propagate_single(r_in, t_grid, omega_grid, delta_grid, n_steps)
        fids.append(qubit_state_fidelity(r_out, R_TARGET_Z.detach().cpu().numpy() @ r_in))
    return {'mean_random_fid': float(np.mean(fids))}


def plot_training_curves(history: Dict[str, List[float]], filepath: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in ['L_total', 'L_dyn', 'L_gate', 'L_cyc', 'L_geo', 'L_ortho', 'T']:
        if k in history and len(history[k]) > 0:
            ax.plot(history[k], label=k)
    ax.set_yscale('log')
    ax.set_xlabel('step')
    ax.set_ylabel('value')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath, dpi=220)
    plt.close(fig)


def train(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)
    net = PINNGeometricZGateFlexible(cfg.hidden_dim, cfg.hidden_layers, cfg.init_T, cfg.learnable_T).to(device)
    if cfg.learnable_T:
        optimizer = optim.Adam([
            {'params': net.net.parameters(), 'lr': cfg.lr},
            {'params': [net.T_log], 'lr': cfg.lr_T},
        ])
    else:
        optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.steps, eta_min=1e-6)

    if cfg.learnable_T:
        t_train = torch.linspace(0.0, 1.0, cfg.N_t, device=device, dtype=dtype).view(-1, 1)
    else:
        t_train = torch.linspace(0.0, cfg.T, cfg.N_t, device=device, dtype=dtype).view(-1, 1)
    t_train.requires_grad_(True)

    history = {k: [] for k in ['L_total', 'L_dyn', 'L_gate', 'L_cyc', 'L_geo', 'L_ortho', 'T']}
    best_state, best_loss = None, float('inf')

    for step in range(1, cfg.steps + 1):
        warmup_steps = int(cfg.warmup_fraction * cfg.steps)
        if step <= warmup_steps:
            lambda_geo_eff = cfg.lambda_geo_init
        else:
            frac = (step - warmup_steps) / max(cfg.steps - warmup_steps, 1)
            lambda_geo_eff = cfg.lambda_geo_init + frac * (cfg.lambda_geo - cfg.lambda_geo_init)

        optimizer.zero_grad()
        out = compute_total_loss(net, t_train, cfg, create_graph=True)
        loss = (
            cfg.lambda_dyn * out['L_dyn'] +
            cfg.lambda_gate * out['L_gate'] +
            cfg.lambda_cyc * out['L_cyc'] +
            lambda_geo_eff * out['L_geo'] +
            cfg.lambda_ortho * out['L_ortho'] +
            cfg.lambda_purity * out['L_purity'] +
            cfg.lambda_amp * out['L_amp'] +
            cfg.lambda_smooth * out['L_smooth'] +
            cfg.lambda_T * out['L_T']
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        history['L_total'].append(float(loss.detach().cpu()))
        history['L_dyn'].append(float(out['L_dyn'].detach().cpu()))
        history['L_gate'].append(float(out['L_gate'].detach().cpu()))
        history['L_cyc'].append(float(out['L_cyc'].detach().cpu()))
        history['L_geo'].append(float(out['L_geo'].detach().cpu()))
        history['L_ortho'].append(float(out['L_ortho'].detach().cpu()))
        history['T'].append(float(net.T_param.detach().cpu()))

        if history['L_total'][-1] < best_loss:
            best_loss = history['L_total'][-1]
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        if step % cfg.print_every == 0 or step == 1 or step == cfg.steps:
            print(f"step={step:5d} | L={history['L_total'][-1]:.4e} | T={history['T'][-1]:.4f} | "
                  f"L_geo={history['L_geo'][-1]:.4e} | beta_turn={cfg.beta_turn:.2f}")

    if best_state is not None:
        net.load_state_dict(best_state)

    if cfg.learnable_T:
        t_eval = torch.linspace(0.0, 1.0, cfg.N_t, device=device, dtype=dtype).view(-1, 1)
    else:
        t_eval = torch.linspace(0.0, cfg.T, cfg.N_t, device=device, dtype=dtype).view(-1, 1)
    t_eval.requires_grad_(True)
    out = compute_total_loss(net, t_eval, cfg, create_graph=False)

    Omega = out['Omega'].detach().cpu().numpy().squeeze()
    Delta = out['Delta'].detach().cpu().numpy().squeeze()
    t_np = out['t_phys'].detach().cpu().numpy().squeeze()
    R_pinn = out['R'].detach().cpu().numpy()
    R_target_np = R_TARGET_Z.detach().cpu().numpy()
    ref_plus_np = out['ref_plus'].detach().cpu().numpy()
    ref_minus_np = out['ref_minus'].detach().cpu().numpy()
    E_plus = out['E_plus'].detach().cpu().numpy()
    E_minus = out['E_minus'].detach().cpu().numpy()
    kappa_turn = out['kappa_turn'].detach().cpu().numpy()

    R_rk4 = np.stack([
        rk4_propagate_single(np.array([1., 0., 0.]), t_np, Omega, Delta, cfg.rk4_steps),
        rk4_propagate_single(np.array([0., 1., 0.]), t_np, Omega, Delta, cfg.rk4_steps),
        rk4_propagate_single(np.array([0., 0., 1.]), t_np, Omega, Delta, cfg.rk4_steps),
    ], axis=1)
    ref_plus_T = rk4_propagate_single(ref_plus_init.detach().cpu().numpy(), t_np, Omega, Delta, cfg.rk4_steps)
    ref_minus_T = rk4_propagate_single(ref_minus_init.detach().cpu().numpy(), t_np, Omega, Delta, cfg.rk4_steps)
    random_stats = random_state_validation(R_rk4, t_np, Omega, Delta, cfg.rk4_steps, cfg.n_random_states)

    metrics = {
        'T_opt': float(net.T_param.detach().cpu()),
        'learnable_T': cfg.learnable_T,
        'use_envelope': cfg.use_envelope,
        'beta_turn': cfg.beta_turn,
        'turn_detach': cfg.turn_detach,
        'F_proc_pinn': process_fidelity_from_rotation(R_pinn, R_target_np),
        'F_proc_rk4': process_fidelity_from_rotation(R_rk4, R_target_np),
        'F_avg_pinn': average_gate_fidelity(process_fidelity_from_rotation(R_pinn, R_target_np)),
        'F_avg_rk4': average_gate_fidelity(process_fidelity_from_rotation(R_rk4, R_target_np)),
        'diff_R_pinn_vs_rk4': float(np.linalg.norm(R_pinn - R_rk4, ord='fro')),
        'cycle_plus_err_rk4': float(np.linalg.norm(ref_plus_T - ref_plus_init.detach().cpu().numpy())),
        'cycle_minus_err_rk4': float(np.linalg.norm(ref_minus_T - ref_minus_init.detach().cpu().numpy())),
        'mean_abs_E_plus': float(np.mean(np.abs(E_plus))),
        'mean_abs_E_minus': float(np.mean(np.abs(E_minus))),
        'max_abs_E_plus': float(np.max(np.abs(E_plus))),
        'max_abs_E_minus': float(np.max(np.abs(E_minus))),
        'mean_kappa_turn': float(np.mean(np.abs(kappa_turn))),
        'max_kappa_turn': float(np.max(np.abs(kappa_turn))),
        'final_L_geo': float(out['L_geo'].detach().cpu()),
        'final_L_gate': float(out['L_gate'].detach().cpu()),
        'final_L_dyn': float(out['L_dyn'].detach().cpu()),
        **random_stats,
    }

    with open(os.path.join(cfg.output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    np.savetxt(os.path.join(cfg.output_dir, 'controls.txt'), np.column_stack([t_np, Omega, Delta]), header='t Omega Delta', fmt='%.10f')
    np.savez(os.path.join(cfg.output_dir, 'timeseries_turn_weighted.npz'),
             t=t_np, Omega=Omega, Delta=Delta, E_plus=E_plus, E_minus=E_minus,
             kappa_turn=kappa_turn, ref_plus=ref_plus_np, ref_minus=ref_minus_np)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_np, Omega, label='Omega(t)')
    ax.plot(t_np, Delta, label='Delta(t)')
    ax.set_title(f'Turn-weighted controls (T*={metrics["T_opt"]:.4f})')
    ax.set_xlabel('t'); ax.set_ylabel('amplitude'); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(cfg.output_dir, 'controls_turn_weighted.png'), dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_np, E_plus, label='E_plus')
    ax.plot(t_np, E_minus, label='E_minus')
    ax.plot(t_np, np.abs(kappa_turn), label='kappa_turn', alpha=0.8)
    ax.set_title('Geometricity and turning-rate')
    ax.set_xlabel('t'); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(cfg.output_dir, 'turning_geometricity_overlay.png'), dpi=220); plt.close(fig)

    plot_training_curves(history, os.path.join(cfg.output_dir, 'training_curves_turn_weighted.png'))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return {'history': history, 'metrics': metrics}


def build_cfg_from_args(args) -> Config:
    return Config(
        N_t=args.N_t,
        T=args.T,
        init_T=args.init_T,
        learnable_T=args.learnable_T,
        Omega_max=args.Omega_max,
        Delta_max=args.Delta_max,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        lr=args.lr,
        lr_T=args.lr_T,
        steps=args.steps,
        print_every=args.print_every,
        lambda_dyn=args.lambda_dyn,
        lambda_gate=args.lambda_gate,
        lambda_cyc=args.lambda_cyc,
        lambda_geo=args.lambda_geo,
        lambda_geo_init=args.lambda_geo_init,
        lambda_ortho=args.lambda_ortho,
        lambda_purity=args.lambda_purity,
        lambda_amp=args.lambda_amp,
        lambda_smooth=args.lambda_smooth,
        lambda_T=args.lambda_T,
        beta_turn=args.beta_turn,
        turn_detach=not args.turn_no_detach,
        warmup_fraction=args.warmup_fraction,
        rk4_steps=args.rk4_steps,
        n_random_states=args.n_random_states,
        use_envelope=args.use_envelope,
        output_dir=args.output_dir,
    )


def main():
    p = argparse.ArgumentParser(description='PINN geometric Z gate with turning-aware geometricity loss.')
    p.add_argument('--learnable-T', action='store_true')
    p.add_argument('--use-envelope', action='store_true')
    p.add_argument('--beta-turn', type=float, default=0.0)
    p.add_argument('--turn-no-detach', action='store_true')
    p.add_argument('--T', type=float, default=1.0)
    p.add_argument('--init-T', type=float, default=1.0)
    p.add_argument('--N-t', dest='N_t', type=int, default=201)
    p.add_argument('--Omega-max', dest='Omega_max', type=float, default=12.0)
    p.add_argument('--Delta-max', dest='Delta_max', type=float, default=12.0)
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--hidden-layers', type=int, default=6)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--lr-T', dest='lr_T', type=float, default=5e-3)
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--print-every', dest='print_every', type=int, default=500)
    p.add_argument('--lambda-dyn', dest='lambda_dyn', type=float, default=80.0)
    p.add_argument('--lambda-gate', dest='lambda_gate', type=float, default=50.0)
    p.add_argument('--lambda-cyc', dest='lambda_cyc', type=float, default=80.0)
    p.add_argument('--lambda-geo', dest='lambda_geo', type=float, default=300.0)
    p.add_argument('--lambda-geo-init', dest='lambda_geo_init', type=float, default=10.0)
    p.add_argument('--lambda-ortho', dest='lambda_ortho', type=float, default=10.0)
    p.add_argument('--lambda-purity', dest='lambda_purity', type=float, default=5.0)
    p.add_argument('--lambda-amp', dest='lambda_amp', type=float, default=1e-4)
    p.add_argument('--lambda-smooth', dest='lambda_smooth', type=float, default=1e-4)
    p.add_argument('--lambda-T', dest='lambda_T', type=float, default=1.0)
    p.add_argument('--warmup-fraction', dest='warmup_fraction', type=float, default=0.3)
    p.add_argument('--rk4-steps', dest='rk4_steps', type=int, default=4000)
    p.add_argument('--n-random-states', dest='n_random_states', type=int, default=200)
    p.add_argument('--output-dir', dest='output_dir', type=str, default='outputs_turn_weighted')
    args = p.parse_args()
    cfg = build_cfg_from_args(args)
    train(cfg)


if __name__ == '__main__':
    main()
