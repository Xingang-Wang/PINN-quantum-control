"""
GRAPE + turning-aware geometricity loss.
与 PINN turning-aware 完全对称的实验。
"""
import os
import json
import csv
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)


@dataclass
class Config:
    T: float = 1.0
    N_t: int = 201
    n_slots: int = 200
    Omega_max: float = 12.0
    Delta_max: float = 12.0
    n_restarts: int = 5
    maxiter_stage1: int = 250
    maxiter_stage2: int = 350
    ftol: float = 1e-12
    gtol: float = 1e-8
    lambda_gate: float = 50.0
    lambda_cyc: float = 80.0
    lambda_geo_init: float = 10.0
    lambda_geo_final: float = 300.0
    lambda_ortho: float = 10.0
    lambda_amp: float = 1e-3
    lambda_smooth: float = 1.0
    beta_turn: float = 0.0       # turning-aware weight (0 = original GRAPE)
    no_env: bool = False         # if True, remove sine envelope
    rk4_steps: int = 4000
    n_random_states: int = 200
    output_dir: str = "outputs_grape_turn_weighted_beta0"


R_TARGET_Z = np.array([[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]])
probe_x_init = np.array([1.,0.,0.])
probe_y_init = np.array([0.,1.,0.])
probe_z_init = np.array([0.,0.,1.])
ref_plus_init = np.array([0.,0.,1.])
ref_minus_init = np.array([0.,0.,-1.])


def frobenius_norm_squared(M): return float(np.sum(M*M))
def vector_norm_squared(v): return float(np.sum(v*v))


def process_fidelity_from_rotation(R, R_target):
    val = (np.trace(R_target.T @ R) + 1.0) / 4.0
    return float(np.clip(np.real(val), 0., 1.))


def average_gate_fidelity(F_proc, d=2):
    return float((d*F_proc + 1.) / (d + 1.))


def rotation_checks(R):
    return {
        "R_orthogonality_error": float(np.linalg.norm(R.T @ R - np.eye(3), ord='fro')),
        "R_det_error": float(abs(np.linalg.det(R) - 1.0)),
    }


def time_nodes(cfg): return np.linspace(0., cfg.T, cfg.N_t)
def time_midpoints(cfg): return 0.5*(time_nodes(cfg)[:-1] + time_nodes(cfg)[1:])
def sine_envelope_midpoints(cfg): return np.sin(np.pi * time_midpoints(cfg) / cfg.T)


def unpack_controls(z, cfg):
    cO, cD = z[:cfg.n_slots], z[cfg.n_slots:]
    if cfg.no_env:
        return cfg.Omega_max * cO, cfg.Delta_max * cD
    env = sine_envelope_midpoints(cfg)
    return cfg.Omega_max * env * cO, cfg.Delta_max * env * cD


def bloch_generator(Om, De):
    return np.array([[0., -De, 0.],[De, 0., -Om],[0., Om, 0.]])


def propagate_single_piecewise(r_init, Omega, Delta, cfg):
    traj = np.zeros((cfg.N_t, 3))
    traj[0] = r_init.copy()
    dt = cfg.T / cfg.n_slots
    r = r_init.copy()
    for k in range(cfg.n_slots):
        r = expm(bloch_generator(Omega[k], Delta[k]) * dt) @ r
        traj[k+1] = r
    return traj


def propagate_all(Omega, Delta, cfg):
    return {k: propagate_single_piecewise(v, Omega, Delta, cfg)
            for k, v in [('probe_x', probe_x_init), ('probe_y', probe_y_init),
                         ('probe_z', probe_z_init), ('ref_plus', ref_plus_init),
                         ('ref_minus', ref_minus_init)]}


def reconstruct_rotation(bundle):
    return np.stack([bundle["probe_x"][-1], bundle["probe_y"][-1], bundle["probe_z"][-1]], axis=1)


def geometricity_terms(bundle, Omega, Delta, cfg):
    Om_n = np.concatenate([Omega, [Omega[-1]]])
    De_n = np.concatenate([Delta, [Delta[-1]]])
    rp, rm = bundle["ref_plus"], bundle["ref_minus"]
    E_p = 0.5*(Om_n*rp[:,0] + De_n*rp[:,2])
    E_m = 0.5*(Om_n*rm[:,0] + De_n*rm[:,2])
    return E_p, E_m


def compute_turning_rate(Omega, Delta):
    """Compute kappa = |d psi/dt| for PWC controls using finite differences."""
    psi = np.unwrap(np.arctan2(Delta, Omega))
    dpsi = np.gradient(psi)
    kappa = np.abs(dpsi)
    return kappa


def compute_losses(bundle, Omega, Delta, cfg, lambda_geo_eff):
    R = reconstruct_rotation(bundle)
    L_gate = frobenius_norm_squared(R - R_TARGET_Z)
    L_cyc = vector_norm_squared(bundle["ref_plus"][-1] - ref_plus_init) + \
            vector_norm_squared(bundle["ref_minus"][-1] - ref_minus_init)

    E_p, E_m = geometricity_terms(bundle, Omega, Delta, cfg)
    E2 = E_p**2 + E_m**2

    if cfg.beta_turn > 0:
        # turning-aware: compute kappa on slot midpoints, pad to nodes
        kappa = compute_turning_rate(Omega, Delta)
        kappa_nodes = np.concatenate([kappa, [kappa[-1]]])
        kappa_norm = kappa_nodes / (np.max(kappa_nodes) + 1e-9)
        w = 1.0 + cfg.beta_turn * kappa_norm
        L_geo = float(np.mean(w * E2))
    else:
        L_geo = float(np.mean(E2))

    rc = rotation_checks(R)
    L_ortho = rc["R_orthogonality_error"]**2 + rc["R_det_error"]**2
    L_amp = float(np.mean(Omega**2) + np.mean(Delta**2))
    dO, dD = np.diff(Omega), np.diff(Delta)
    L_smooth = float(np.mean(dO**2) + np.mean(dD**2)) if len(dO) > 0 else 0.

    L_total = (cfg.lambda_gate*L_gate + cfg.lambda_cyc*L_cyc +
               lambda_geo_eff*L_geo + cfg.lambda_ortho*L_ortho +
               cfg.lambda_amp*L_amp + cfg.lambda_smooth*L_smooth)

    return {
        "L_gate": L_gate, "L_cyc": L_cyc, "L_geo": L_geo,
        "L_ortho": L_ortho, "L_amp": L_amp, "L_smooth": L_smooth,
        "L_total": float(L_total),
        "mean_abs_E_plus": float(np.mean(np.abs(E_p))),
        "mean_abs_E_minus": float(np.mean(np.abs(E_m))),
        "max_abs_E_plus": float(np.max(np.abs(E_p))),
        "max_abs_E_minus": float(np.max(np.abs(E_m))),
    }


def rk4_propagate(r_init, Omega, Delta, cfg, n_steps):
    """RK4 with zero-order hold: GRAPE controls are piecewise-constant."""
    t0, t1 = 0., cfg.T
    h = (t1 - t0) / n_steps
    dt_slot = cfg.T / cfg.n_slots
    r = r_init.copy(); tv = t0
    def rhs(t, r):
        k = min(int(t / dt_slot), cfg.n_slots - 1)
        Om, De = Omega[k], Delta[k]
        return np.array([-De*r[1], De*r[0] - Om*r[2], Om*r[1]])
    for _ in range(n_steps):
        k1 = rhs(tv, r)
        k2 = rhs(tv+.5*h, r+.5*h*k1)
        k3 = rhs(tv+.5*h, r+.5*h*k2)
        k4 = rhs(tv+h, r+h*k3)
        r += (h/6)*(k1+2*k2+2*k3+k4); tv += h
    return r


def rk4_propagate_trajectory(r_init, Omega, Delta, cfg, n_steps):
    """Same as rk4_propagate but returns full trajectory for E(t) computation."""
    t0, t1 = 0., cfg.T
    h = (t1 - t0) / n_steps
    dt_slot = cfg.T / cfg.n_slots
    traj = np.zeros((n_steps + 1, 3))
    t_arr = np.zeros(n_steps + 1)
    r = r_init.copy(); tv = t0
    traj[0] = r; t_arr[0] = tv
    def rhs(t, r):
        k = min(int(t / dt_slot), cfg.n_slots - 1)
        Om, De = Omega[k], Delta[k]
        return np.array([-De*r[1], De*r[0] - Om*r[2], Om*r[1]])
    for i in range(n_steps):
        k1 = rhs(tv, r)
        k2 = rhs(tv+.5*h, r+.5*h*k1)
        k3 = rhs(tv+.5*h, r+.5*h*k2)
        k4 = rhs(tv+h, r+h*k3)
        r += (h/6)*(k1+2*k2+2*k3+k4); tv += h
        traj[i+1] = r; t_arr[i+1] = tv
    return traj, t_arr


def independent_validation(Omega, Delta, cfg):
    s_x = rk4_propagate(probe_x_init, Omega, Delta, cfg, cfg.rk4_steps)
    s_y = rk4_propagate(probe_y_init, Omega, Delta, cfg, cfg.rk4_steps)
    s_z = rk4_propagate(probe_z_init, Omega, Delta, cfg, cfg.rk4_steps)
    R = np.stack([s_x, s_y, s_z], axis=1)
    rp = rk4_propagate(ref_plus_init, Omega, Delta, cfg, cfg.rk4_steps)
    rm = rk4_propagate(ref_minus_init, Omega, Delta, cfg, cfg.rk4_steps)
    F_proc = process_fidelity_from_rotation(R, R_TARGET_Z)
    return {"R_rk4": R, "F_proc_rk4": F_proc, "F_avg_rk4": average_gate_fidelity(F_proc),
            "cycle_plus_err_rk4": float(np.linalg.norm(rp - ref_plus_init)),
            "cycle_minus_err_rk4": float(np.linalg.norm(rm - ref_minus_init))}


def objective_factory(cfg, lambda_geo_eff, history):
    def objective(z):
        Om, De = unpack_controls(z, cfg)
        bundle = propagate_all(Om, De, cfg)
        losses = compute_losses(bundle, Om, De, cfg, lambda_geo_eff)
        for k in ["L_total","L_gate","L_cyc","L_geo","L_ortho"]:
            history[k].append(losses[k])
        return losses["L_total"]
    return objective


def run_grape(cfg):
    bounds = [(-1.,1.)]*(2*cfg.n_slots)
    best = None; best_score = -1.

    for rid in range(cfg.n_restarts):
        rng = np.random.default_rng(42 + 97*rid)
        z0 = np.zeros(2*cfg.n_slots) if rid == 0 else rng.uniform(-0.2, 0.2, 2*cfg.n_slots)

        h1 = {"L_total":[],"L_gate":[],"L_cyc":[],"L_geo":[],"L_ortho":[]}
        h2 = {"L_total":[],"L_gate":[],"L_cyc":[],"L_geo":[],"L_ortho":[]}

        res1 = minimize(objective_factory(cfg, cfg.lambda_geo_init, h1), z0,
                        method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": cfg.maxiter_stage1, "ftol": cfg.ftol, "gtol": cfg.gtol, "maxls": 50})
        res2 = minimize(objective_factory(cfg, cfg.lambda_geo_final, h2), res1.x,
                        method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": cfg.maxiter_stage2, "ftol": cfg.ftol, "gtol": cfg.gtol, "maxls": 50})

        Om, De = unpack_controls(res2.x, cfg)
        bundle = propagate_all(Om, De, cfg)
        losses = compute_losses(bundle, Om, De, cfg, cfg.lambda_geo_final)
        val = independent_validation(Om, De, cfg)
        E_p, E_m = geometricity_terms(bundle, Om, De, cfg)
        R_train = reconstruct_rotation(bundle)

        # Compute E(t) on RK4 trajectory
        traj_rp, t_rk4 = rk4_propagate_trajectory(ref_plus_init, Om, De, cfg, cfg.rk4_steps)
        traj_rm, _ = rk4_propagate_trajectory(ref_minus_init, Om, De, cfg, cfg.rk4_steps)
        dt_slot = cfg.T / cfg.n_slots
        def zoh(t):
            k = min(int(t / dt_slot), cfg.n_slots - 1)
            return Om[k], De[k]
        E_p_rk4 = np.zeros(len(t_rk4))
        E_m_rk4 = np.zeros(len(t_rk4))
        for i, ti in enumerate(t_rk4):
            Om_i, De_i = zoh(ti)
            E_p_rk4[i] = 0.5 * (Om_i * traj_rp[i, 0] + De_i * traj_rp[i, 2])
            E_m_rk4[i] = 0.5 * (Om_i * traj_rm[i, 0] + De_i * traj_rm[i, 2])

        score = val["F_avg_rk4"]
        print(f"  restart {rid}: F_avg_rk4={score:.8f}  mean|E+|={losses['mean_abs_E_plus']:.6e}  max|E+|={losses['max_abs_E_plus']:.6e}")

        if score > best_score:
            best_score = score
            best = {"Omega": Om, "Delta": De, "bundle": bundle, "losses": losses,
                    "validation": val, "E_plus": E_p, "E_minus": E_m,
                    "E_plus_rk4": E_p_rk4, "E_minus_rk4": E_m_rk4,
                    "t_rk4": t_rk4,
                    "R_train": R_train, "restart_id": rid}

    return best


def save_results(best, cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    val = best["validation"]
    ls = best["losses"]

    metrics = {
        "beta_turn": cfg.beta_turn,
        "T": cfg.T, "n_slots": cfg.n_slots,
        "F_proc_rk4": val["F_proc_rk4"], "F_avg_rk4": val["F_avg_rk4"],
        "cycle_plus_err_rk4": val["cycle_plus_err_rk4"],
        "cycle_minus_err_rk4": val["cycle_minus_err_rk4"],
        "mean_abs_E_plus": ls["mean_abs_E_plus"],
        "mean_abs_E_minus": ls["mean_abs_E_minus"],
        "max_abs_E_plus": ls["max_abs_E_plus"],
        "max_abs_E_minus": ls["max_abs_E_minus"],
        "final_L_geo": ls["L_geo"], "final_L_gate": ls["L_gate"],
        "final_L_total": ls["L_total"],
    }
    with open(os.path.join(cfg.output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    np.savetxt(os.path.join(cfg.output_dir, "best_Omega.txt"), best["Omega"], fmt="%.10f")
    np.savetxt(os.path.join(cfg.output_dir, "best_Delta.txt"), best["Delta"], fmt="%.10f")
    np.savetxt(os.path.join(cfg.output_dir, "ref_plus_energy.txt"), best["E_plus"], fmt="%.10f")
    np.savetxt(os.path.join(cfg.output_dir, "ref_minus_energy.txt"), best["E_minus"], fmt="%.10f")
    np.savetxt(os.path.join(cfg.output_dir, "ref_plus_energy_rk4.txt"), best["E_plus_rk4"], fmt="%.10f")
    np.savetxt(os.path.join(cfg.output_dir, "ref_minus_energy_rk4.txt"), best["E_minus_rk4"], fmt="%.10f")
    np.savetxt(os.path.join(cfg.output_dir, "t_rk4.txt"), best["t_rk4"], fmt="%.10f")

    # Also save RK4 E(t) metrics
    E_rk4_mean = float(np.mean((np.abs(best["E_plus_rk4"]) + np.abs(best["E_minus_rk4"])) / 2))
    E_rk4_max = float(np.max((np.abs(best["E_plus_rk4"]) + np.abs(best["E_minus_rk4"])) / 2))
    metrics["mean_abs_E_rk4"] = E_rk4_mean
    metrics["max_abs_E_rk4"] = E_rk4_max
    with open(os.path.join(cfg.output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--beta-turn', type=float, default=0.0)
    p.add_argument('--output-dir', type=str, default='')
    p.add_argument('--n-restarts', type=int, default=5)
    p.add_argument('--T', type=float, default=1.0, help='Gate duration (default 1.0)')
    p.add_argument('--no-env', action='store_true', help='Remove sine envelope')
    args = p.parse_args()

    cfg = Config(beta_turn=args.beta_turn, n_restarts=args.n_restarts, T=args.T, no_env=args.no_env)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    else:
        cfg.output_dir = f"outputs_grape_turn_weighted_beta{int(args.beta_turn)}"

    print(f"GRAPE turning-aware: beta_turn={cfg.beta_turn}")
    best = run_grape(cfg)
    metrics = save_results(best, cfg)

    print(f"\nResults (beta_turn={cfg.beta_turn}):")
    print(f"  F_proc_rk4 = {metrics['F_proc_rk4']:.10f}")
    print(f"  mean|E+|   = {metrics['mean_abs_E_plus']:.6e}")
    print(f"  max|E+|    = {metrics['max_abs_E_plus']:.6e}")
    print(f"  cycle_err  = {metrics['cycle_plus_err_rk4']:.6e}")
    print(f"  Output: {cfg.output_dir}")
