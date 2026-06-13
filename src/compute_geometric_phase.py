"""
Geometric-phase accounting for the geometric Z gate.

For each set of learned controls, independently propagate state vectors
|0> and |1> via RK4, then compute:
  - total phase: gamma_a^total = arg <psi_a(0)|psi_a(T)>
  - dynamical phase: gamma_a^dyn = -int_0^T <psi_a|H|psi_a> dt
  - geometric phase: gamma_a^geom = gamma_a^total - gamma_a^dyn
  - relative geometric phase: Delta_gamma_geom = wrap(gamma_1^geom - gamma_0^geom)
  - phase error: eps_phase = |wrap(Delta_gamma_geom - pi)|
"""
import numpy as np
import json
from pathlib import Path

RESULTS = Path("/Users/strong/Desktop/research-workspace/projects/PINN/geometric-gate/results")

def rhs_statevec(t, psi, t_grid, omega_grid, delta_grid):
    """d/dt |psi> = -i H |psi>, H = (1/2)(Omega sigma_x + Delta sigma_z)"""
    Om = float(np.interp(t, t_grid, omega_grid))
    De = float(np.interp(t, t_grid, delta_grid))
    # psi = [alpha, beta] (complex)
    # H = (1/2)[[Delta, Omega], [Omega, -Delta]]
    alpha, beta = psi
    dalpha = -1j * (De/2 * alpha + Om/2 * beta)
    dbeta  = -1j * (Om/2 * alpha - De/2 * beta)
    return np.array([dalpha, dbeta])

def rhs_statevec_zoh(t, psi, Omega_arr, Delta_arr, dt_slot, n_slots):
    """Zero-order hold version for GRAPE piecewise-constant controls."""
    k = min(int(t / dt_slot), n_slots - 1)
    Om, De = Omega_arr[k], Delta_arr[k]
    alpha, beta = psi
    dalpha = -1j * (De/2 * alpha + Om/2 * beta)
    dbeta  = -1j * (Om/2 * alpha - De/2 * beta)
    return np.array([dalpha, dbeta])

def rk4_statevec(psi0, t_grid, omega_grid, delta_grid, n_steps):
    """RK4 state-vector propagation with linear interpolation (PINN)."""
    t0, t1 = t_grid[0], t_grid[-1]
    h = (t1 - t0) / n_steps
    traj = np.zeros((n_steps + 1, 2), dtype=complex)
    t_arr = np.linspace(t0, t1, n_steps + 1)
    psi = psi0.copy()
    traj[0] = psi
    tv = t0
    for i in range(n_steps):
        k1 = rhs_statevec(tv, psi, t_grid, omega_grid, delta_grid)
        k2 = rhs_statevec(tv + .5*h, psi + .5*h*k1, t_grid, omega_grid, delta_grid)
        k3 = rhs_statevec(tv + .5*h, psi + .5*h*k2, t_grid, omega_grid, delta_grid)
        k4 = rhs_statevec(tv + h, psi + h*k3, t_grid, omega_grid, delta_grid)
        psi = psi + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        tv += h
        traj[i+1] = psi
    return traj, t_arr

def rk4_statevec_zoh(psi0, Omega, Delta, T, n_slots, n_steps):
    """RK4 state-vector propagation with zero-order hold (GRAPE)."""
    h = T / n_steps
    dt_slot = T / n_slots
    traj = np.zeros((n_steps + 1, 2), dtype=complex)
    t_arr = np.linspace(0, T, n_steps + 1)
    psi = psi0.copy()
    traj[0] = psi
    tv = 0.0
    for i in range(n_steps):
        k1 = rhs_statevec_zoh(tv, psi, Omega, Delta, dt_slot, n_slots)
        k2 = rhs_statevec_zoh(tv + .5*h, psi + .5*h*k1, Omega, Delta, dt_slot, n_slots)
        k3 = rhs_statevec_zoh(tv + .5*h, psi + .5*h*k2, Omega, Delta, dt_slot, n_slots)
        k4 = rhs_statevec_zoh(tv + h, psi + h*k3, Omega, Delta, dt_slot, n_slots)
        psi = psi + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        tv += h
        traj[i+1] = psi
    return traj, t_arr

def wrap(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

RK4_STEPS = 4000

psi0 = np.array([1.0 + 0j, 0.0 + 0j])  # |0>
psi1 = np.array([0.0 + 0j, 1.0 + 0j])  # |1>

results = {}

# === PINN (envelope) ===
for beta in [0, 1, 3, 10, 30, 50]:
    dname = f"outputs_turn_weighted_learnableT_env_beta{beta}"
    d = RESULTS / dname
    ts_path = d / "timeseries_turn_weighted.npz"
    if not ts_path.exists():
        print(f"PINN beta={beta}: no data"); continue
    ts = np.load(ts_path)
    t, Om, De = ts['t'], ts['Omega'], ts['Delta']
    T = t[-1]

    traj0, t_rk4 = rk4_statevec(psi0, t, Om, De, RK4_STEPS)
    traj1, _     = rk4_statevec(psi1, t, Om, De, RK4_STEPS)

    # Total phase: arg <psi_a(0)|psi_a(T)>
    gamma0_total = np.angle(np.vdot(psi0, traj0[-1]))
    gamma1_total = np.angle(np.vdot(psi1, traj1[-1]))

    # Cycle closure: 1 - |<psi(0)|psi(T)>|^2
    C0 = 1 - abs(np.vdot(psi0, traj0[-1]))**2
    C1 = 1 - abs(np.vdot(psi1, traj1[-1]))**2

    # Dynamical phase: -int <psi|H|psi> dt (trapezoidal)
    # <psi|H|psi> = (1/2)(Delta*rz + Omega*rx)
    # For state vector [alpha, beta]: rx = 2*Re(alpha* beta), rz = |alpha|^2 - |beta|^2
    def energy_density(traj, t_arr, t_grid, omega_grid, delta_grid):
        E = np.zeros(len(t_arr))
        for i, ti in enumerate(t_arr):
            Om_i = float(np.interp(ti, t_grid, omega_grid))
            De_i = float(np.interp(ti, t_grid, delta_grid))
            alpha, beta = traj[i]
            rx = 2 * np.real(np.conj(alpha) * beta)
            rz = abs(alpha)**2 - abs(beta)**2
            E[i] = 0.5 * (Om_i * rx + De_i * rz)
        return E

    E0 = energy_density(traj0, t_rk4, t, Om, De)
    E1 = energy_density(traj1, t_rk4, t, Om, De)
    gamma0_dyn = -np.trapezoid(E0, t_rk4)
    gamma1_dyn = -np.trapezoid(E1, t_rk4)

    gamma0_geom = gamma0_total - gamma0_dyn
    gamma1_geom = gamma1_total - gamma1_dyn
    dgamma_geom = wrap(gamma1_geom - gamma0_geom)
    eps_phase = abs(wrap(dgamma_geom - np.pi))

    label = f"PINN_env_b{beta}"
    results[label] = {
        "T": T, "C0": C0, "C1": C1,
        "gamma0_total": gamma0_total, "gamma1_total": gamma1_total,
        "gamma0_dyn": gamma0_dyn, "gamma1_dyn": gamma1_dyn,
        "gamma0_geom": gamma0_geom, "gamma1_geom": gamma1_geom,
        "dgamma_geom": dgamma_geom, "eps_phase": eps_phase,
        "mean_abs_E0": float(np.mean(np.abs(E0))),
        "mean_abs_E1": float(np.mean(np.abs(E1))),
    }

    # Save to metrics
    m = json.load(open(d / "metrics.json"))
    m["phase_C0_rk4"] = float(C0)
    m["phase_C1_rk4"] = float(C1)
    m["phase_gamma0_total"] = float(gamma0_total)
    m["phase_gamma1_total"] = float(gamma1_total)
    m["phase_gamma0_dyn"] = float(gamma0_dyn)
    m["phase_gamma1_dyn"] = float(gamma1_dyn)
    m["phase_gamma0_geom"] = float(gamma0_geom)
    m["phase_gamma1_geom"] = float(gamma1_geom)
    m["phase_dgamma_geom"] = float(dgamma_geom)
    m["phase_eps_phase"] = float(eps_phase)
    with open(d / "metrics.json", 'w') as f:
        json.dump(m, f, indent=2)

    print(f"PINN env beta={beta:2d}: T={T:.4f}, C0={C0:.2e}, C1={C1:.2e}")
    print(f"  gamma0: total={gamma0_total:+.6f}, dyn={gamma0_dyn:+.6e}, geom={gamma0_geom:+.6f}")
    print(f"  gamma1: total={gamma1_total:+.6f}, dyn={gamma1_dyn:+.6e}, geom={gamma1_geom:+.6f}")
    print(f"  Delta_gamma_geom = {dgamma_geom:+.6f}, eps_phase = {eps_phase:.6e}")

# === GRAPE (envelope, T=1.5) ===
for beta in [0, 3]:
    dname = f"outputs_grape_turn_beta{beta}_T1.5"
    d = RESULTS / dname
    if not (d / "best_Omega.txt").exists():
        print(f"GRAPE env beta={beta}: no data"); continue
    Om = np.loadtxt(d / "best_Omega.txt")
    De = np.loadtxt(d / "best_Delta.txt")
    T = 1.5
    n_slots = len(Om)

    traj0, t_rk4 = rk4_statevec_zoh(psi0, Om, De, T, n_slots, RK4_STEPS)
    traj1, _     = rk4_statevec_zoh(psi1, Om, De, T, n_slots, RK4_STEPS)

    gamma0_total = np.angle(np.vdot(psi0, traj0[-1]))
    gamma1_total = np.angle(np.vdot(psi1, traj1[-1]))
    C0 = 1 - abs(np.vdot(psi0, traj0[-1]))**2
    C1 = 1 - abs(np.vdot(psi1, traj1[-1]))**2

    def energy_density_zoh(traj, t_arr, Omega, Delta, T, n_slots):
        dt_slot = T / n_slots
        E = np.zeros(len(t_arr))
        for i, ti in enumerate(t_arr):
            k = min(int(ti / dt_slot), n_slots - 1)
            Om_i, De_i = Omega[k], Delta[k]
            alpha, beta = traj[i]
            rx = 2 * np.real(np.conj(alpha) * beta)
            rz = abs(alpha)**2 - abs(beta)**2
            E[i] = 0.5 * (Om_i * rx + De_i * rz)
        return E

    E0 = energy_density_zoh(traj0, t_rk4, Om, De, T, n_slots)
    E1 = energy_density_zoh(traj1, t_rk4, Om, De, T, n_slots)
    gamma0_dyn = -np.trapezoid(E0, t_rk4)
    gamma1_dyn = -np.trapezoid(E1, t_rk4)

    gamma0_geom = gamma0_total - gamma0_dyn
    gamma1_geom = gamma1_total - gamma1_dyn
    dgamma_geom = wrap(gamma1_geom - gamma0_geom)
    eps_phase = abs(wrap(dgamma_geom - np.pi))

    label = f"GRAPE_env_b{beta}"
    results[label] = {
        "T": T, "C0": C0, "C1": C1,
        "gamma0_total": gamma0_total, "gamma1_total": gamma1_total,
        "gamma0_dyn": gamma0_dyn, "gamma1_dyn": gamma1_dyn,
        "gamma0_geom": gamma0_geom, "gamma1_geom": gamma1_geom,
        "dgamma_geom": dgamma_geom, "eps_phase": eps_phase,
    }

    m = json.load(open(d / "metrics.json"))
    m["phase_C0_rk4"] = float(C0)
    m["phase_C1_rk4"] = float(C1)
    m["phase_gamma0_total"] = float(gamma0_total)
    m["phase_gamma1_total"] = float(gamma1_total)
    m["phase_gamma0_dyn"] = float(gamma0_dyn)
    m["phase_gamma1_dyn"] = float(gamma1_dyn)
    m["phase_gamma0_geom"] = float(gamma0_geom)
    m["phase_gamma1_geom"] = float(gamma1_geom)
    m["phase_dgamma_geom"] = float(dgamma_geom)
    m["phase_eps_phase"] = float(eps_phase)
    with open(d / "metrics.json", 'w') as f:
        json.dump(m, f, indent=2)

    print(f"GRAPE env beta={beta}: T={T:.4f}, C0={C0:.2e}, C1={C1:.2e}")
    print(f"  gamma0: total={gamma0_total:+.6f}, dyn={gamma0_dyn:+.6e}, geom={gamma0_geom:+.6f}")
    print(f"  gamma1: total={gamma1_total:+.6f}, dyn={gamma1_dyn:+.6e}, geom={gamma1_geom:+.6f}")
    print(f"  Delta_gamma_geom = {dgamma_geom:+.6f}, eps_phase = {eps_phase:.6e}")

print("\n=== Summary Table ===")
print(f"{'Method':<20} {'C0':>10} {'C1':>10} {'gamma0_dyn':>12} {'gamma1_dyn':>12} {'dgamma_geom':>12} {'eps_phase':>12}")
print("-"*90)
for label, r in results.items():
    print(f"{label:<20} {r['C0']:10.2e} {r['C1']:10.2e} {r['gamma0_dyn']:12.4e} {r['gamma1_dyn']:12.4e} {r['dgamma_geom']:12.6f} {r['eps_phase']:12.4e}")

print("\nDone.")
