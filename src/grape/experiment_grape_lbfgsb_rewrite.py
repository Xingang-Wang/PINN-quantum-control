"""
GRAPE-style main baseline (L-BFGS-B version)

Purpose
-------
A more standard GRAPE-style baseline for formal comparison with the PINN results.

Key design choices
------------------
- fixed total time T per inner optimization
- piecewise-constant controls Ω_k, Δ_k
- optimize amplitudes directly with box bounds
- scipy.optimize.minimize(method="L-BFGS-B")
- outer sweep over T
- multi-start per T
- independent RK4 + 4-probe tomography validation

Recommended main use
--------------------
1. x-z plane 7-gate family under minimal constraints
2. representative 4-gate set for quick checks
3. no-envelope as a second-stage comparison once minimal is stable
"""

import os
import sys
import json
import math
import csv
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

# 确保 import 能找到父目录的 pinn_dual_control_yz
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.optimize import minimize

from pinn_dual_control_yz import (
    set_seed, Config, device, dtype,
    get_rates, choi_from_affine, choi_unitary, avg_gate_fidelity,
    rk4_propagate,
)

OUTDIR = "outputs_grape_lbfgsb_smooth"
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# Rotation helpers
# ============================================================

def rotation_bloch(n: np.ndarray, theta: float) -> np.ndarray:
    nx, ny, nz = n / np.linalg.norm(n)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c + (1-c)*nx*nx,      (1-c)*nx*ny - s*nz,  (1-c)*nx*nz + s*ny],
        [(1-c)*ny*nx + s*nz,   c + (1-c)*ny*ny,      (1-c)*ny*nz - s*nx],
        [(1-c)*nz*nx - s*ny,   (1-c)*nz*ny + s*nx,   c + (1-c)*nz*nz  ],
    ], dtype=np.float64)


def rotation_unitary(n: np.ndarray, theta: float) -> np.ndarray:
    nx, ny, nz = n / np.linalg.norm(n)
    c2, s2 = np.cos(theta/2), np.sin(theta/2)
    return np.array([
        [c2 - 1j*s2*nz,  -1j*s2*nx - s2*ny],
        [-1j*s2*nx + s2*ny,  c2 + 1j*s2*nz],
    ], dtype=np.complex128)


def T_direct_limit(n: np.ndarray, theta: float, Omax: float = 8.0) -> float:
    nx, ny, nz = n / np.linalg.norm(n)
    denom_parts = []
    if abs(nx) > 1e-10:
        denom_parts.append(Omax / abs(nx))
    if abs(nz) > 1e-10:
        denom_parts.append(Omax / abs(nz))
    if not denom_parts:
        return float("inf")
    return theta / min(denom_parts)


# ============================================================
# Gate sets
# ============================================================

def gate_set_representative() -> List[Dict]:
    specs = [
        ("X(pi)", np.array([1.0, 0.0, 0.0]), np.pi, 0.0, True),
        ("R(45deg,pi)", np.array([1.0, 0.0, 1.0]) / np.sqrt(2), np.pi, 45.0, True),
        ("Z(pi)", np.array([0.0, 0.0, 1.0]), np.pi, 90.0, True),
        ("R(xyz,pi)", np.array([1.0, 1.0, 1.0]) / np.sqrt(3), np.pi, None, False),
    ]
    out = []
    for label, n, theta, alpha_deg, ny_zero in specs:
        out.append({
            "label": label,
            "n": n,
            "theta": theta,
            "alpha_deg": alpha_deg,
            "ny_zero": ny_zero,
            "family": "representative",
        })
    return out


def gate_set_xz7() -> List[Dict]:
    out = []
    for alpha_deg in [0, 15, 30, 45, 60, 75, 90]:
        alpha = np.radians(alpha_deg)
        n = np.array([np.cos(alpha), 0.0, np.sin(alpha)], dtype=np.float64)
        label = (
            "X(pi)" if alpha_deg == 0 else
            "Z(pi)" if alpha_deg == 90 else
            f"R(alpha={alpha_deg}deg,pi)"
        )
        out.append({
            "label": label,
            "n": n,
            "theta": np.pi,
            "alpha_deg": alpha_deg,
            "ny_zero": True,
            "family": "xz7",
        })
    return out


# ============================================================
# Config
# ============================================================

@dataclass
class LBFGSBGRAPEConfig:
    gamma_down: float = 0.05
    gamma_up: float = 0.0
    gamma_phi: float = 0.0
    Omega_max: float = 8.0
    Delta_max: float = 8.0
    n_slots: int = 80
    n_restarts: int = 5
    maxiter: int = 350
    ftol: float = 1e-12
    gtol: float = 1e-8
    beta_gate: float = 10.0
    chi_amp: float = 1e-4
    zeta_smooth: float = 1e-4
    lambda_boundary: float = 1.0
    output_dir: str = OUTDIR
    t_multipliers: Tuple[float, ...] = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.35)
    t_fallback: Tuple[float, ...] = (0.28, 0.34, 0.40, 0.50, 0.65, 0.80)
    save_all_runs: bool = True


# ============================================================
# Affine propagation
# ============================================================

PROBES = torch.tensor([
    [0.0, 0.0,  1.0, 1.0],
    [0.0, 0.0, -1.0, 1.0],
    [1.0, 0.0,  0.0, 1.0],
    [0.0, 1.0,  0.0, 1.0],
], device=device, dtype=dtype)


def slot_generator_matrix(O: torch.Tensor, D: torch.Tensor, G1: float, G2: float, gd: float) -> torch.Tensor:
    K = torch.zeros((4, 4), device=device, dtype=dtype)
    K[0, 0] = -G2
    K[0, 1] = -D
    K[1, 0] =  D
    K[1, 1] = -G2
    K[1, 2] = -O
    K[2, 1] =  O
    K[2, 2] = -G1
    K[2, 3] =  gd
    return K


def propagate_piecewise_torch(Omega: torch.Tensor, Delta: torch.Tensor, T_fixed: float,
                              G1: float, G2: float, gd: float) -> torch.Tensor:
    dt = T_fixed / Omega.numel()
    states = PROBES.clone()
    for k in range(Omega.numel()):
        K = slot_generator_matrix(Omega[k], Delta[k], G1, G2, gd)
        E = torch.matrix_exp(K * dt)
        states = (E @ states.T).T
    return states


def affine_from_final_augmented(states: torch.Tensor):
    s0, s1, sx, sy = states[0, :3], states[1, :3], states[2, :3], states[3, :3]
    c = 0.5 * (s0 + s1)
    M = torch.stack([sx - c, sy - c, 0.5 * (s0 - s1)], dim=1)
    return M, c


# ============================================================
# Loss
# ============================================================

def gate_loss_from_affine(M: torch.Tensor, c: torch.Tensor, R_target: torch.Tensor) -> torch.Tensor:
    return torch.sum((M - R_target) ** 2) + torch.sum(c ** 2)


def smoothness_loss(x: torch.Tensor) -> torch.Tensor:
    if x.numel() <= 1:
        return torch.tensor(0.0, device=device, dtype=dtype)
    dx = x[1:] - x[:-1]
    return torch.mean(dx ** 2)


def unpack_controls(z: torch.Tensor, cfg: LBFGSBGRAPEConfig):
    u_Omega = z[:cfg.n_slots]
    u_Delta = z[cfg.n_slots:]
    Omega = cfg.Omega_max * torch.tanh(u_Omega)
    Delta = cfg.Delta_max * torch.tanh(u_Delta)
    return Omega, Delta


def pulse_ratios_np(Omega_np: np.ndarray, Delta_np: np.ndarray) -> Dict[str, float]:
    return {
        "ratio_peak": float(np.max(np.abs(Omega_np)) / (np.max(np.abs(Delta_np)) + 1e-12)),
        "ratio_rms": float(np.sqrt(np.mean(Omega_np**2)) / (np.sqrt(np.mean(Delta_np**2)) + 1e-12)),
    }


def build_loss_from_vector(z: torch.Tensor, cfg: LBFGSBGRAPEConfig, R_target: torch.Tensor,
                           T_fixed: float, mode: str) -> Dict[str, torch.Tensor]:
    Omega, Delta = unpack_controls(z, cfg)
    qcfg = Config(gamma_down=cfg.gamma_down, gamma_up=cfg.gamma_up, gamma_phi=cfg.gamma_phi)
    G1, G2 = get_rates(qcfg)
    gd = cfg.gamma_down - cfg.gamma_up

    states = propagate_piecewise_torch(Omega, Delta, T_fixed, G1, G2, gd)
    M, c = affine_from_final_augmented(states)
    L_gate = gate_loss_from_affine(M, c, R_target)

    if mode == "minimal":
        L_total = cfg.beta_gate * L_gate
        return {
            "Omega": Omega, "Delta": Delta, "M": M, "c": c,
            "L_gate": L_gate,
            "L_amp": torch.tensor(0.0, device=device, dtype=dtype),
            "L_smooth": torch.tensor(0.0, device=device, dtype=dtype),
            "L_boundary": torch.tensor(0.0, device=device, dtype=dtype),
            "L_total": L_total,
        }

    if mode == "no_envelope":
        L_amp = torch.mean(Omega**2) + torch.mean(Delta**2)
        L_smooth = smoothness_loss(Omega) + smoothness_loss(Delta)
        L_boundary = Omega[0]**2 + Omega[-1]**2 + Delta[0]**2 + Delta[-1]**2
        L_total = cfg.beta_gate * L_gate + cfg.chi_amp * L_amp + cfg.zeta_smooth * L_smooth + cfg.lambda_boundary * L_boundary
        return {
            "Omega": Omega, "Delta": Delta, "M": M, "c": c,
            "L_gate": L_gate, "L_amp": L_amp, "L_smooth": L_smooth,
            "L_boundary": L_boundary, "L_total": L_total,
        }

    raise ValueError(f"Unknown mode: {mode}")


# ============================================================
# Validation
# ============================================================

def validate_controls(Omega_np: np.ndarray, Delta_np: np.ndarray, T_fixed: float,
                      gamma_down: float, gamma_up: float, gamma_phi: float,
                      U_target: np.ndarray) -> Dict[str, float]:
    cfg = Config(gamma_down=gamma_down, gamma_up=gamma_up, gamma_phi=gamma_phi, T=T_fixed)
    t_grid = np.linspace(0.0, T_fixed, len(Omega_np), dtype=np.float64)
    probes = [
        np.array([0., 0., 1.]),
        np.array([0., 0., -1.]),
        np.array([1., 0., 0.]),
        np.array([0., 1., 0.]),
    ]
    ends = [rk4_propagate(p, t_grid, Omega_np, Delta_np, cfg, cfg.rk4_steps) for p in probes]
    s0, s1, sx, sy = ends
    c = 0.5 * (s0 + s1)
    M = np.stack([sx - c, sy - c, 0.5 * (s0 - s1)], axis=1)
    F_proc = float(np.clip(np.real(np.trace(choi_unitary(U_target) @ choi_from_affine(M, c))), 0, 1))
    F_avg = avg_gate_fidelity(F_proc)
    return {
        "F_proc": F_proc,
        "F_avg": F_avg,
        "Omega_max_abs": float(np.max(np.abs(Omega_np))),
        "Delta_max_abs": float(np.max(np.abs(Delta_np))),
        **pulse_ratios_np(Omega_np, Delta_np),
    }


# ============================================================
# SciPy objective wrapper
# ============================================================

def random_initial_vector(cfg: LBFGSBGRAPEConfig, restart_id: int) -> np.ndarray:
    rng = np.random.default_rng(42 + 97 * restart_id)
    if restart_id == 0:
        # Start with small values so tanh(u) is in linear regime (~0)
        z = 0.1 * np.ones(2 * cfg.n_slots, dtype=np.float64)
        z[:cfg.n_slots] = 0.1   # small positive -> tanh(0.1) ≈ 0.1, Omega ≈ 0.8
        z[cfg.n_slots:] = 0.1
        return z
    scale = 1.5  # tanh(1.5) ≈ 0.9, still has usable gradient
    z = np.zeros(2 * cfg.n_slots, dtype=np.float64)
    z[:] = rng.uniform(-scale, scale, size=2 * cfg.n_slots)
    return z


def scipy_objective_factory(cfg: LBFGSBGRAPEConfig, R_target_t: torch.Tensor,
                            T_fixed: float, mode: str, history: Dict[str, list]):
    def fun_and_grad(z_np: np.ndarray):
        z = torch.tensor(z_np, device=device, dtype=dtype, requires_grad=True)
        out = build_loss_from_vector(z, cfg, R_target_t, T_fixed, mode)
        loss = out["L_total"]
        loss.backward()
        grad = z.grad.detach().cpu().numpy().astype(np.float64)

        history["L_total"].append(float(out["L_total"].detach().cpu().item()))
        history["L_gate"].append(float(out["L_gate"].detach().cpu().item()))
        history["L_amp"].append(float(out["L_amp"].detach().cpu().item()))
        history["L_smooth"].append(float(out["L_smooth"].detach().cpu().item()))
        history["L_boundary"].append(float(out["L_boundary"].detach().cpu().item()))
        return float(loss.detach().cpu().item()), grad
    return fun_and_grad


# ============================================================
# Optimization
# ============================================================

def one_lbfgsb_run(gate: Dict, cfg: LBFGSBGRAPEConfig, T_fixed: float, mode: str, restart_id: int) -> Dict:
    set_seed(42 + 97 * restart_id)
    R_target_t = torch.tensor(rotation_bloch(gate["n"], gate["theta"]), device=device, dtype=dtype)
    U_target = rotation_unitary(gate["n"], gate["theta"])

    z0 = random_initial_vector(cfg, restart_id)
    history = {"L_total": [], "L_gate": [], "L_amp": [], "L_smooth": [], "L_boundary": []}
    fg = scipy_objective_factory(cfg, R_target_t, T_fixed, mode, history)

    def fun(z):
        f, g = fg(z)
        fun._jac = g
        return f

    def jac(z):
        return fun._jac

    res = minimize(
        fun,
        z0,
        method="L-BFGS-B",
        jac=jac,
        options={"maxiter": cfg.maxiter, "ftol": cfg.ftol, "gtol": cfg.gtol, "maxls": 50},
    )

    z_best = torch.tensor(res.x, device=device, dtype=dtype)
    out = build_loss_from_vector(z_best, cfg, R_target_t, T_fixed, mode)
    Omega_np = out["Omega"].detach().cpu().numpy().astype(np.float64)
    Delta_np = out["Delta"].detach().cpu().numpy().astype(np.float64)
    val = validate_controls(Omega_np, Delta_np, T_fixed, cfg.gamma_down, cfg.gamma_up, cfg.gamma_phi, U_target)

    return {
        "gate": gate["label"],
        "family": gate["family"],
        "alpha_deg": gate["alpha_deg"],
        "ny_zero": gate["ny_zero"],
        "theta": gate["theta"],
        "n": gate["n"].tolist(),
        "mode": mode,
        "optimizer": "L-BFGS-B",
        "restart_id": restart_id,
        "T_fixed": float(T_fixed),
        "success": bool(res.success),
        "message": str(res.message),
        "n_iter": int(res.nit) if hasattr(res, "nit") else -1,
        "loss_final": float(out["L_total"].detach().cpu().item()),
        "L_gate": float(out["L_gate"].detach().cpu().item()),
        "L_amp": float(out["L_amp"].detach().cpu().item()),
        "L_smooth": float(out["L_smooth"].detach().cpu().item()),
        "L_boundary": float(out["L_boundary"].detach().cpu().item()),
        **val,
        "Omega": Omega_np.tolist(),
        "Delta": Delta_np.tolist(),
        "history": history,
    }


def default_T_candidates(gate: Dict, cfg: LBFGSBGRAPEConfig) -> List[float]:
    T0 = T_direct_limit(gate["n"], gate["theta"], cfg.Omega_max)
    if np.isfinite(T0):
        return [round(float(T0 * m), 4) for m in cfg.t_multipliers if T0 * m > 0.03]
    # Y-axis gate needs larger T range
    if gate.get("family") == "y_axis":
        return [0.50, 0.65, 0.80, 0.95, 1.10, 1.25]
    return list(cfg.t_fallback)


def run_experiment(cfg: LBFGSBGRAPEConfig, gates: List[Dict], modes: Tuple[str, ...] = ("minimal",)) -> List[Dict]:
    os.makedirs(cfg.output_dir, exist_ok=True)
    best_over_T = []
    all_runs = []

    for gate in gates:
        T_candidates = default_T_candidates(gate, cfg)
        gate_runs = []
        print("\n" + "=" * 88)
        print(f"Gate: {gate['label']:<18} family={gate['family']:<14} T candidates = {T_candidates}")

        for mode in modes:
            mode_runs = []
            print(f"  Mode: {mode}")
            for T_fixed in T_candidates:
                for restart_id in range(cfg.n_restarts):
                    res = one_lbfgsb_run(gate, cfg, T_fixed, mode, restart_id)
                    mode_runs.append(res)
                    gate_runs.append(res)
                    print(f"    T={T_fixed:7.4f} restart={restart_id}  F_avg={res['F_avg']:.6f}  peak={res['ratio_peak']:.3f}  rms={res['ratio_rms']:.3f}  ok={res['success']}")

            best_mode = max(mode_runs, key=lambda x: x["F_avg"])
            best_over_T.append({k: v for k, v in best_mode.items() if k not in ("Omega", "Delta", "history")})

            stem = gate["label"].replace("(", "_").replace(")", "").replace(",", "_").replace("/", "_").replace(" ", "_")
            np.save(os.path.join(cfg.output_dir, f"best_{stem}_{mode}_omega.npy"), np.array(best_mode["Omega"], dtype=np.float64))
            np.save(os.path.join(cfg.output_dir, f"best_{stem}_{mode}_delta.npy"), np.array(best_mode["Delta"], dtype=np.float64))

        if cfg.save_all_runs:
            gate_json = os.path.join(cfg.output_dir, f"{gate['label'].replace('(', '_').replace(')', '').replace(',', '_').replace('/', '_').replace(' ', '_')}.json")
            with open(gate_json, "w") as f:
                json.dump(gate_runs, f, indent=2)
        all_runs.extend(gate_runs)

    with open(os.path.join(cfg.output_dir, "lbfgsb_grape_best_over_T_summary.json"), "w") as f:
        json.dump({"config": asdict(cfg), "results": best_over_T}, f, indent=2)

    with open(os.path.join(cfg.output_dir, "lbfgsb_grape_best_over_T_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "gate", "family", "mode", "alpha_deg", "ny_zero",
            "T_fixed", "F_avg", "F_proc", "ratio_peak", "ratio_rms",
            "Omega_max_abs", "Delta_max_abs", "optimizer", "restart_id", "success", "n_iter"
        ])
        writer.writeheader()
        for r in best_over_T:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

    return best_over_T


if __name__ == "__main__":
    cfg = LBFGSBGRAPEConfig(
        gamma_down=0.05,
        gamma_up=0.0,
        gamma_phi=0.0,
        Omega_max=8.0,
        Delta_max=8.0,
        n_slots=80,
        n_restarts=5,
        maxiter=500,
        beta_gate=10.0,
        chi_amp=0.05,
        zeta_smooth=0.5,
        lambda_boundary=1.0,
        output_dir=OUTDIR,
    )

    gates = gate_set_representative() + gate_set_xz7() + [
        {
            "label": "Y(pi)",
            "n": np.array([0.0, 1.0, 0.0]),
            "theta": np.pi,
            "alpha_deg": None,
            "ny_zero": False,
            "family": "y_axis",
        },
    ]
    results = run_experiment(cfg, gates=gates, modes=("no_envelope",))

    print("\n" + "=" * 100)
    print("L-BFGS-B GRAPE-STYLE MAIN SUMMARY (best over T and restarts)")
    print("=" * 100)
    for r in results:
        print(f"{r['gate']:<18} {r['mode']:<12} T={r['T_fixed']:.4f}  F_avg={r['F_avg']:.6f}  peak={r['ratio_peak']:.3f}  rms={r['ratio_rms']:.3f}  ok={r['success']}")
