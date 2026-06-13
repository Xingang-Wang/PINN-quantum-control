
import os
import csv
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize


def set_seed(seed: int = 42) -> None:
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
    rk4_steps: int = 4000
    n_random_states: int = 200
    output_dir: str = "outputs_grape_geometric_Z_gate_smooth_v2"


cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)

R_TARGET_Z = np.array([
    [-1.0, 0.0, 0.0],
    [ 0.0,-1.0, 0.0],
    [ 0.0, 0.0, 1.0]
], dtype=np.float64)

probe_x_init = np.array([1.0, 0.0, 0.0], dtype=np.float64)
probe_y_init = np.array([0.0, 1.0, 0.0], dtype=np.float64)
probe_z_init = np.array([0.0, 0.0, 1.0], dtype=np.float64)
ref_plus_init  = np.array([0.0, 0.0,  1.0], dtype=np.float64)
ref_minus_init = np.array([0.0, 0.0, -1.0], dtype=np.float64)


def frobenius_norm_squared(M: np.ndarray) -> float:
    return float(np.sum(M * M))


def vector_norm_squared(v: np.ndarray) -> float:
    return float(np.sum(v * v))


def sample_random_pure_bloch_vectors(n: int, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.astype(np.float64)


def qubit_state_fidelity_from_bloch(r: np.ndarray, s: np.ndarray) -> float:
    nr2 = float(np.dot(r, r))
    ns2 = float(np.dot(s, s))
    rs = float(np.dot(r, s))
    term = max(0.0, (1.0 - nr2) * (1.0 - ns2))
    F = 0.5 * (1.0 + rs + math.sqrt(term))
    return float(np.clip(F, 0.0, 1.0))


def process_fidelity_from_rotation(R: np.ndarray, R_target: np.ndarray) -> float:
    val = (np.trace(R_target.T @ R) + 1.0) / 4.0
    return float(np.clip(np.real(val), 0.0, 1.0))


def average_gate_fidelity_from_process_fidelity(F_proc: float, d: int = 2) -> float:
    return float((d * F_proc + 1.0) / (d + 1.0))


def rotation_checks(R: np.ndarray) -> Dict[str, float]:
    return {
        "R_orthogonality_error": float(np.linalg.norm(R.T @ R - np.eye(3), ord='fro')),
        "R_det_error": float(abs(np.linalg.det(R) - 1.0)),
    }


def time_nodes(cfg: Config) -> np.ndarray:
    return np.linspace(0.0, cfg.T, cfg.N_t, dtype=np.float64)


def time_midpoints(cfg: Config) -> np.ndarray:
    tg = time_nodes(cfg)
    return 0.5 * (tg[:-1] + tg[1:])


def sine_envelope_midpoints(cfg: Config) -> np.ndarray:
    tm = time_midpoints(cfg)
    return np.sin(np.pi * tm / cfg.T)


def unpack_controls(z: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    cO = z[:cfg.n_slots]
    cD = z[cfg.n_slots:]
    env = sine_envelope_midpoints(cfg)
    Omega = cfg.Omega_max * env * cO
    Delta = cfg.Delta_max * env * cD
    return Omega, Delta


def pulse_stats(Omega: np.ndarray, Delta: np.ndarray) -> Dict[str, float]:
    return {
        "Omega_max_abs": float(np.max(np.abs(Omega))),
        "Delta_max_abs": float(np.max(np.abs(Delta))),
        "ratio_peak": float(np.max(np.abs(Omega)) / (np.max(np.abs(Delta)) + 1e-12)),
        "ratio_rms": float(np.sqrt(np.mean(Omega**2)) / (np.sqrt(np.mean(Delta**2)) + 1e-12)),
        "allocation_int_abs_ratio": float(np.sum(np.abs(Omega)) / (np.sum(np.abs(Delta)) + 1e-12)),
    }


def bloch_generator(Omega: float, Delta: float) -> np.ndarray:
    return np.array([
        [ 0.0,   -Delta,  0.0],
        [ Delta,  0.0,   -Omega],
        [ 0.0,    Omega,  0.0],
    ], dtype=np.float64)


def propagate_single_piecewise(r_init: np.ndarray, Omega: np.ndarray, Delta: np.ndarray, cfg: Config) -> np.ndarray:
    traj = np.zeros((cfg.N_t, 3), dtype=np.float64)
    traj[0] = r_init.copy()
    dt = cfg.T / cfg.n_slots
    r = r_init.copy()
    for k in range(cfg.n_slots):
        A = bloch_generator(Omega[k], Delta[k])
        U = expm(A * dt)
        r = U @ r
        traj[k + 1] = r
    return traj


def propagate_all(Omega: np.ndarray, Delta: np.ndarray, cfg: Config) -> Dict[str, np.ndarray]:
    return {
        "probe_x":   propagate_single_piecewise(probe_x_init,   Omega, Delta, cfg),
        "probe_y":   propagate_single_piecewise(probe_y_init,   Omega, Delta, cfg),
        "probe_z":   propagate_single_piecewise(probe_z_init,   Omega, Delta, cfg),
        "ref_plus":  propagate_single_piecewise(ref_plus_init,  Omega, Delta, cfg),
        "ref_minus": propagate_single_piecewise(ref_minus_init, Omega, Delta, cfg),
    }


def reconstruct_rotation_from_probes(bundle: Dict[str, np.ndarray]):
    s_x = bundle["probe_x"][-1]
    s_y = bundle["probe_y"][-1]
    s_z = bundle["probe_z"][-1]
    R = np.stack([s_x, s_y, s_z], axis=1)
    return s_x, s_y, s_z, R


def geometricity_terms_on_nodes(bundle: Dict[str, np.ndarray], Omega: np.ndarray, Delta: np.ndarray, cfg: Config):
    Omega_nodes = np.concatenate([Omega, [Omega[-1]]])
    Delta_nodes = np.concatenate([Delta, [Delta[-1]]])
    rp = bundle["ref_plus"]
    rm = bundle["ref_minus"]
    E_plus = 0.5 * (Omega_nodes * rp[:, 0] + Delta_nodes * rp[:, 2])
    E_minus = 0.5 * (Omega_nodes * rm[:, 0] + Delta_nodes * rm[:, 2])
    return E_plus, E_minus


def compute_losses_from_bundle(bundle: Dict[str, np.ndarray], Omega: np.ndarray, Delta: np.ndarray, cfg: Config, lambda_geo_eff: float):
    _, _, _, R = reconstruct_rotation_from_probes(bundle)
    L_gate = frobenius_norm_squared(R - R_TARGET_Z)
    L_cyc = vector_norm_squared(bundle["ref_plus"][-1] - ref_plus_init) + vector_norm_squared(bundle["ref_minus"][-1] - ref_minus_init)
    E_plus, E_minus = geometricity_terms_on_nodes(bundle, Omega, Delta, cfg)
    L_geo = float(np.mean(E_plus**2 + E_minus**2))
    rot_chk = rotation_checks(R)
    L_ortho = rot_chk["R_orthogonality_error"]**2 + rot_chk["R_det_error"]**2
    L_amp = float(np.mean(Omega**2) + np.mean(Delta**2))
    dO = np.diff(Omega)
    dD = np.diff(Delta)
    L_smooth = float(np.mean(dO**2) + np.mean(dD**2)) if len(dO) > 0 else 0.0
    L_total = (
        cfg.lambda_gate * L_gate +
        cfg.lambda_cyc * L_cyc +
        lambda_geo_eff * L_geo +
        cfg.lambda_ortho * L_ortho +
        cfg.lambda_amp * L_amp +
        cfg.lambda_smooth * L_smooth
    )
    return {
        "L_gate": L_gate,
        "L_cyc": L_cyc,
        "L_geo": L_geo,
        "L_ortho": L_ortho,
        "L_amp": L_amp,
        "L_smooth": L_smooth,
        "L_total": float(L_total),
        "mean_abs_E_plus": float(np.mean(np.abs(E_plus))),
        "mean_abs_E_minus": float(np.mean(np.abs(E_minus))),
        "max_abs_E_plus": float(np.max(np.abs(E_plus))),
        "max_abs_E_minus": float(np.max(np.abs(E_minus))),
    }


def interp_numpy(t_query: float, t_grid: np.ndarray, value_grid: np.ndarray) -> float:
    return float(np.interp(t_query, t_grid, value_grid))


def control_grids_for_interp(Omega: np.ndarray, Delta: np.ndarray, cfg: Config):
    tm = time_midpoints(cfg)
    return tm, Omega, Delta


def bloch_rhs_numpy(t_val: float, r: np.ndarray, t_mid: np.ndarray, omega_grid: np.ndarray, delta_grid: np.ndarray) -> np.ndarray:
    Omega_t = interp_numpy(t_val, t_mid, omega_grid)
    Delta_t = interp_numpy(t_val, t_mid, delta_grid)
    x, y, z = r
    dx = -Delta_t * y
    dy =  Delta_t * x - Omega_t * z
    dz =  Omega_t * y
    return np.array([dx, dy, dz], dtype=np.float64)


def rk4_propagate_single(r_init: np.ndarray, Omega: np.ndarray, Delta: np.ndarray, cfg: Config, n_steps: int) -> np.ndarray:
    t0, t1 = 0.0, cfg.T
    h = (t1 - t0) / n_steps
    t_mid, omega_grid, delta_grid = control_grids_for_interp(Omega, Delta, cfg)
    r = r_init.astype(np.float64).copy()
    t_val = t0
    for _ in range(n_steps):
        k1 = bloch_rhs_numpy(t_val, r, t_mid, omega_grid, delta_grid)
        k2 = bloch_rhs_numpy(t_val + 0.5*h, r + 0.5*h*k1, t_mid, omega_grid, delta_grid)
        k3 = bloch_rhs_numpy(t_val + 0.5*h, r + 0.5*h*k2, t_mid, omega_grid, delta_grid)
        k4 = bloch_rhs_numpy(t_val + h, r + h*k3, t_mid, omega_grid, delta_grid)
        r = r + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t_val += h
    return r


def independent_rk4_validation(Omega: np.ndarray, Delta: np.ndarray, cfg: Config):
    s_x = rk4_propagate_single(probe_x_init, Omega, Delta, cfg, cfg.rk4_steps)
    s_y = rk4_propagate_single(probe_y_init, Omega, Delta, cfg, cfg.rk4_steps)
    s_z = rk4_propagate_single(probe_z_init, Omega, Delta, cfg, cfg.rk4_steps)
    R_rk4 = np.stack([s_x, s_y, s_z], axis=1)
    ref_plus_T = rk4_propagate_single(ref_plus_init, Omega, Delta, cfg, cfg.rk4_steps)
    ref_minus_T = rk4_propagate_single(ref_minus_init, Omega, Delta, cfg, cfg.rk4_steps)
    F_proc = process_fidelity_from_rotation(R_rk4, R_TARGET_Z)
    F_avg = average_gate_fidelity_from_process_fidelity(F_proc)
    return {
        "R_rk4": R_rk4,
        "F_proc_rk4": F_proc,
        "F_avg_rk4": F_avg,
        "ref_plus_T_rk4": ref_plus_T,
        "ref_minus_T_rk4": ref_minus_T,
        "cycle_plus_err_rk4": float(np.linalg.norm(ref_plus_T - ref_plus_init)),
        "cycle_minus_err_rk4": float(np.linalg.norm(ref_minus_T - ref_minus_init)),
        **rotation_checks(R_rk4),
    }


def random_state_validation(R_rk4: np.ndarray, Omega: np.ndarray, Delta: np.ndarray, cfg: Config):
    random_inputs = sample_random_pure_bloch_vectors(cfg.n_random_states, seed=2026)
    affine_errors = []
    fidelities_to_rotation = []
    fidelities_to_target = []
    target_errors = []
    for r_in in random_inputs:
        r_rk4 = rk4_propagate_single(r_in, Omega, Delta, cfg, cfg.rk4_steps)
        r_rot = R_rk4 @ r_in
        r_target = R_TARGET_Z @ r_in
        affine_errors.append(np.linalg.norm(r_rk4 - r_rot))
        fidelities_to_rotation.append(qubit_state_fidelity_from_bloch(r_rk4, r_rot))
        fidelities_to_target.append(qubit_state_fidelity_from_bloch(r_rk4, r_target))
        target_errors.append(np.linalg.norm(r_rk4 - r_target))
    return {
        "mean_rotation_consistency_error": float(np.mean(affine_errors)),
        "max_rotation_consistency_error": float(np.max(affine_errors)),
        "mean_fidelity_to_rotation_prediction": float(np.mean(fidelities_to_rotation)),
        "mean_fidelity_to_target_gate": float(np.mean(fidelities_to_target)),
        "mean_bloch_error_to_target_gate": float(np.mean(target_errors)),
    }


def random_initial_vector(cfg: Config, restart_id: int) -> np.ndarray:
    rng = np.random.default_rng(42 + 97 * restart_id)
    if restart_id == 0:
        return np.zeros(2 * cfg.n_slots, dtype=np.float64)
    scale = 0.20
    return rng.uniform(-scale, scale, size=2 * cfg.n_slots).astype(np.float64)


def objective_factory(cfg: Config, lambda_geo_eff: float, history: Dict[str, List[float]]):
    def objective(z: np.ndarray) -> float:
        Omega, Delta = unpack_controls(z, cfg)
        bundle = propagate_all(Omega, Delta, cfg)
        losses = compute_losses_from_bundle(bundle, Omega, Delta, cfg, lambda_geo_eff)
        history["L_total"].append(losses["L_total"])
        history["L_gate"].append(losses["L_gate"])
        history["L_cyc"].append(losses["L_cyc"])
        history["L_geo"].append(losses["L_geo"])
        history["L_ortho"].append(losses["L_ortho"])
        return losses["L_total"]
    return objective


def run_two_stage_lbfgsb(cfg: Config, restart_id: int):
    bounds = [(-1.0, 1.0)] * (2 * cfg.n_slots)
    z0 = random_initial_vector(cfg, restart_id)
    history_stage1 = {"L_total": [], "L_gate": [], "L_cyc": [], "L_geo": [], "L_ortho": []}
    history_stage2 = {"L_total": [], "L_gate": [], "L_cyc": [], "L_geo": [], "L_ortho": []}
    obj1 = objective_factory(cfg, cfg.lambda_geo_init, history_stage1)
    res1 = minimize(obj1, z0, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": cfg.maxiter_stage1, "ftol": cfg.ftol, "gtol": cfg.gtol, "maxls": 50})
    obj2 = objective_factory(cfg, cfg.lambda_geo_final, history_stage2)
    res2 = minimize(obj2, res1.x, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": cfg.maxiter_stage2, "ftol": cfg.ftol, "gtol": cfg.gtol, "maxls": 50})

    z_best = res2.x
    Omega, Delta = unpack_controls(z_best, cfg)
    bundle = propagate_all(Omega, Delta, cfg)
    losses = compute_losses_from_bundle(bundle, Omega, Delta, cfg, cfg.lambda_geo_final)
    _, _, _, R_train = reconstruct_rotation_from_probes(bundle)
    F_proc_train = process_fidelity_from_rotation(R_train, R_TARGET_Z)
    F_avg_train = average_gate_fidelity_from_process_fidelity(F_proc_train)
    validation = independent_rk4_validation(Omega, Delta, cfg)
    random_stats = random_state_validation(validation["R_rk4"], Omega, Delta, cfg)
    E_plus, E_minus = geometricity_terms_on_nodes(bundle, Omega, Delta, cfg)

    return {
        "restart_id": restart_id,
        "success_stage1": bool(res1.success),
        "success_stage2": bool(res2.success),
        "message_stage1": str(res1.message),
        "message_stage2": str(res2.message),
        "n_iter_stage1": int(res1.nit) if hasattr(res1, "nit") else -1,
        "n_iter_stage2": int(res2.nit) if hasattr(res2, "nit") else -1,
        "losses": losses,
        "F_proc_train": F_proc_train,
        "F_avg_train": F_avg_train,
        "R_train": R_train,
        "Omega": Omega,
        "Delta": Delta,
        "bundle": bundle,
        "E_plus": E_plus,
        "E_minus": E_minus,
        "history_stage1": history_stage1,
        "history_stage2": history_stage2,
        "pulse_stats": pulse_stats(Omega, Delta),
        "validation": validation,
        "random_stats": random_stats,
    }


def save_dict_csv(d: Dict[str, object], filepath: str) -> None:
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['key', 'value'])
        for k, v in d.items():
            writer.writerow([k, v])


def save_json(obj: Dict[str, object], filepath: str) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def plot_training_curves(history1: Dict[str, List[float]], history2: Dict[str, List[float]], filepath: str) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    for key in ["L_total", "L_gate", "L_cyc", "L_geo", "L_ortho"]:
        y = history1[key] + history2[key]
        ax.plot(y, label=key)
    ax.set_yscale('log')
    ax.set_xlabel('evaluation step')
    ax.set_ylabel('objective term')
    ax.set_title('GRAPE-style optimization history for geometric Z gate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=200)
    plt.close(fig)


def plot_controls(cfg: Config, Omega: np.ndarray, Delta: np.ndarray, filepath: str) -> None:
    tm = time_midpoints(cfg)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.step(tm, Omega, where='mid', label='Omega(t)')
    ax.step(tm, Delta, where='mid', label='Delta(t)')
    ax.set_xlabel('t')
    ax.set_ylabel('control amplitude')
    ax.set_title('GRAPE-style learned controls for geometric Z gate')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath, dpi=200)
    plt.close(fig)


def plot_trajectories(t_grid: np.ndarray, trajs: Dict[str, np.ndarray], filepath: str, title: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for label, r in trajs.items():
        axes[0].plot(t_grid, r[:, 0], label=label)
        axes[1].plot(t_grid, r[:, 1], label=label)
        axes[2].plot(t_grid, r[:, 2], label=label)
    axes[0].set_ylabel('x')
    axes[1].set_ylabel('y')
    axes[2].set_ylabel('z')
    axes[2].set_xlabel('t')
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filepath, dpi=200)
    plt.close(fig)


def plot_geometricity_terms(t_grid: np.ndarray, E_plus: np.ndarray, E_minus: np.ndarray, filepath: str) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(t_grid, E_plus, label=r'$<phi_+|H|phi_+>$')
    ax.plot(t_grid, E_minus, label=r'$<phi_-|H|phi_->$')
    ax.axhline(0.0, linestyle='--')
    ax.set_xlabel('t')
    ax.set_ylabel('energy expectation')
    ax.set_title('Geometricity check for geometric Z gate (GRAPE-style)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=200)
    plt.close(fig)


def summarize_result(best, cfg: Config):
    validation = best["validation"]
    losses = best["losses"]
    pulse = best["pulse_stats"]
    metrics = {
        "task": "geometric_Z_gate_grape_style",
        "T": cfg.T,
        "N_t": cfg.N_t,
        "n_slots": cfg.n_slots,
        "Omega_max_cfg": cfg.Omega_max,
        "Delta_max_cfg": cfg.Delta_max,
        "n_restarts": cfg.n_restarts,
        "lambda_gate": cfg.lambda_gate,
        "lambda_cyc": cfg.lambda_cyc,
        "lambda_geo_init": cfg.lambda_geo_init,
        "lambda_geo_final": cfg.lambda_geo_final,
        "lambda_ortho": cfg.lambda_ortho,
        "lambda_amp": cfg.lambda_amp,
        "lambda_smooth": cfg.lambda_smooth,
        "best_restart_id": best["restart_id"],
        "F_proc_train": best["F_proc_train"],
        "F_avg_train": best["F_avg_train"],
        "F_proc_rk4": validation["F_proc_rk4"],
        "F_avg_rk4": validation["F_avg_rk4"],
        "cycle_plus_err_rk4": validation["cycle_plus_err_rk4"],
        "cycle_minus_err_rk4": validation["cycle_minus_err_rk4"],
        "mean_rotation_consistency_error": best["random_stats"]["mean_rotation_consistency_error"],
        "max_rotation_consistency_error": best["random_stats"]["max_rotation_consistency_error"],
        "mean_fidelity_to_rotation_prediction": best["random_stats"]["mean_fidelity_to_rotation_prediction"],
        "mean_fidelity_to_target_gate": best["random_stats"]["mean_fidelity_to_target_gate"],
        "mean_bloch_error_to_target_gate": best["random_stats"]["mean_bloch_error_to_target_gate"],
        "final_L_gate": losses["L_gate"],
        "final_L_cyc": losses["L_cyc"],
        "final_L_geo": losses["L_geo"],
        "final_L_ortho": losses["L_ortho"],
        "final_L_amp": losses["L_amp"],
        "final_L_smooth": losses["L_smooth"],
        "final_L_total": losses["L_total"],
        "mean_abs_E_plus": losses["mean_abs_E_plus"],
        "mean_abs_E_minus": losses["mean_abs_E_minus"],
        "max_abs_E_plus": losses["max_abs_E_plus"],
        "max_abs_E_minus": losses["max_abs_E_minus"],
        **pulse,
        "success_stage1": best["success_stage1"],
        "success_stage2": best["success_stage2"],
        "n_iter_stage1": best["n_iter_stage1"],
        "n_iter_stage2": best["n_iter_stage2"],
        "message_stage1": best["message_stage1"],
        "message_stage2": best["message_stage2"],
        **{f"train_{k}": v for k, v in rotation_checks(best["R_train"]).items()},
        **{f"rk4_{k}": v for k, v in rotation_checks(validation["R_rk4"]).items()},
    }
    return metrics


def train_geometric_Z_gate_grape(cfg: Config):
    all_runs = []
    best = None
    best_key = -1.0
    for restart_id in range(cfg.n_restarts):
        print(f"[restart {restart_id}] optimizing geometric Z gate ...")
        res = run_two_stage_lbfgsb(cfg, restart_id)
        all_runs.append(res)
        score = res["validation"]["F_avg_rk4"]
        print(f"   -> F_avg_rk4 = {score:.8f}, mean|E+| = {res['losses']['mean_abs_E_plus']:.6e}, mean|E-| = {res['losses']['mean_abs_E_minus']:.6e}")
        if score > best_key:
            best_key = score
            best = res

    metrics = summarize_result(best, cfg)
    save_dict_csv(metrics, os.path.join(cfg.output_dir, "run_parameters_and_metrics_geometric_Z_gate_grape.csv"))
    save_json(metrics, os.path.join(cfg.output_dir, "run_parameters_and_metrics_geometric_Z_gate_grape.json"))

    brief_runs = []
    for r in all_runs:
        brief_runs.append({
            "restart_id": r["restart_id"],
            "F_proc_train": r["F_proc_train"],
            "F_avg_train": r["F_avg_train"],
            "F_proc_rk4": r["validation"]["F_proc_rk4"],
            "F_avg_rk4": r["validation"]["F_avg_rk4"],
            "mean_abs_E_plus": r["losses"]["mean_abs_E_plus"],
            "mean_abs_E_minus": r["losses"]["mean_abs_E_minus"],
            "L_total": r["losses"]["L_total"],
            "success_stage1": r["success_stage1"],
            "success_stage2": r["success_stage2"],
            "message_stage1": r["message_stage1"],
            "message_stage2": r["message_stage2"],
        })
    save_json({"all_runs": brief_runs}, os.path.join(cfg.output_dir, "all_restart_summary_geometric_Z_gate_grape.json"))

    t_grid = time_nodes(cfg)
    plot_training_curves(best["history_stage1"], best["history_stage2"],
                         os.path.join(cfg.output_dir, "training_losses_geometric_Z_gate_grape.png"))
    plot_controls(cfg, best["Omega"], best["Delta"],
                  os.path.join(cfg.output_dir, "controls_geometric_Z_gate_grape.png"))
    plot_trajectories(t_grid,
                      {"probe_x": best["bundle"]["probe_x"], "probe_y": best["bundle"]["probe_y"], "probe_z": best["bundle"]["probe_z"]},
                      os.path.join(cfg.output_dir, "probe_trajectories_geometric_Z_gate_grape.png"),
                      "Probe trajectories for geometric Z gate (GRAPE-style)")
    plot_trajectories(t_grid,
                      {"ref_plus(|0>)": best["bundle"]["ref_plus"], "ref_minus(|1>)": best["bundle"]["ref_minus"]},
                      os.path.join(cfg.output_dir, "reference_trajectories_geometric_Z_gate_grape.png"),
                      "Reference trajectories for geometric Z gate (GRAPE-style)")
    plot_geometricity_terms(t_grid, best["E_plus"], best["E_minus"],
                            os.path.join(cfg.output_dir, "geometricity_terms_geometric_Z_gate_grape.png"))

    np.savetxt(os.path.join(cfg.output_dir, "R_train_geometric_Z_gate_grape.txt"), best["R_train"], fmt="%.10f")
    np.savetxt(os.path.join(cfg.output_dir, "R_rk4_geometric_Z_gate_grape.txt"), best["validation"]["R_rk4"], fmt="%.10f")
    np.savetxt(os.path.join(cfg.output_dir, "R_target_geometric_Z_gate_grape.txt"), R_TARGET_Z, fmt="%.10f")
    np.savetxt(os.path.join(cfg.output_dir, "ref_plus_energy_expectation_geometric_Z_gate_grape.txt"), best["E_plus"], fmt="%.10f")
    np.savetxt(os.path.join(cfg.output_dir, "ref_minus_energy_expectation_geometric_Z_gate_grape.txt"), best["E_minus"], fmt="%.10f")
    np.savetxt(os.path.join(cfg.output_dir, "best_Omega_geometric_Z_gate_grape.txt"), best["Omega"], fmt="%.10f")
    np.savetxt(os.path.join(cfg.output_dir, "best_Delta_geometric_Z_gate_grape.txt"), best["Delta"], fmt="%.10f")
    return {"all_runs": all_runs, "best": best, "metrics": metrics}


if __name__ == "__main__":
    results = train_geometric_Z_gate_grape(cfg)
    print("\nGRAPE-style geometric Z baseline finished.")
    print("Output dir:", cfg.output_dir)
    print("Best RK4 F_proc:", results["metrics"]["F_proc_rk4"])
    print("Best RK4 F_avg :", results["metrics"]["F_avg_rk4"])
    print("mean |<phi_+|H|phi_+>| =", results["metrics"]["mean_abs_E_plus"])
    print("mean |<phi_-|H|phi_->| =", results["metrics"]["mean_abs_E_minus"])
    print("cycle_plus_err_rk4 =", results["metrics"]["cycle_plus_err_rk4"])
    print("cycle_minus_err_rk4 =", results["metrics"]["cycle_minus_err_rk4"])
