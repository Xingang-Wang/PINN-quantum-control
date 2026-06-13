"""
Compute PINN geometricity error E(t) on RK4 trajectory.
Load saved controls (201 points), interpolate (linear, appropriate for smooth PINN output),
propagate ref_plus/ref_minus via RK4, compute E(t) at each RK4 step.
"""
import numpy as np
import json
from pathlib import Path

RESULTS = Path("/Users/strong/Desktop/research-workspace/projects/PINN/geometric-gate/results")

def bloch_rhs(t, r, t_grid, omega_grid, delta_grid):
    Om = float(np.interp(t, t_grid, omega_grid))
    De = float(np.interp(t, t_grid, delta_grid))
    return np.array([-De*r[1], De*r[0] - Om*r[2], Om*r[1]])

def rk4_trajectory(r_init, t_grid, omega_grid, delta_grid, n_steps):
    t0, t1 = t_grid[0], t_grid[-1]
    h = (t1 - t0) / n_steps
    traj = np.zeros((n_steps + 1, 3))
    t_arr = np.linspace(t0, t1, n_steps + 1)
    r = r_init.copy()
    tv = t0
    traj[0] = r
    for i in range(n_steps):
        k1 = bloch_rhs(tv, r, t_grid, omega_grid, delta_grid)
        k2 = bloch_rhs(tv + 0.5*h, r + 0.5*h*k1, t_grid, omega_grid, delta_grid)
        k3 = bloch_rhs(tv + 0.5*h, r + 0.5*h*k2, t_grid, omega_grid, delta_grid)
        k4 = bloch_rhs(tv + h, r + h*k3, t_grid, omega_grid, delta_grid)
        r = r + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        tv += h
        traj[i+1] = r
    return traj, t_arr

def compute_E_on_rk4(traj_rp, traj_rm, t_arr, t_grid, omega_grid, delta_grid):
    E_p = np.zeros(len(t_arr))
    E_m = np.zeros(len(t_arr))
    for i, ti in enumerate(t_arr):
        Om = float(np.interp(ti, t_grid, omega_grid))
        De = float(np.interp(ti, t_grid, delta_grid))
        E_p[i] = 0.5 * (Om * traj_rp[i, 0] + De * traj_rp[i, 2])
        E_m[i] = 0.5 * (Om * traj_rm[i, 0] + De * traj_rm[i, 2])
    return E_p, E_m

ref_plus_init = np.array([0., 0., 1.])
ref_minus_init = np.array([0., 0., -1.])
RK4_STEPS = 4000

betas = [0, 1, 3, 10, 30, 50]
for beta in betas:
    dirname = f"outputs_turn_weighted_learnableT_env_beta{beta}"
    ts_path = RESULTS / dirname / "timeseries_turn_weighted.npz"
    if not ts_path.exists():
        print(f"  beta={beta}: no data, skipping")
        continue

    ts = np.load(ts_path)
    m = json.load(open(RESULTS / dirname / "metrics.json"))
    t = ts['t']
    Omega = ts['Omega']
    Delta = ts['Delta']
    T_opt = m['T_opt']

    # RK4 propagate ref_plus and ref_minus
    traj_rp, t_rk4 = rk4_trajectory(ref_plus_init, t, Omega, Delta, RK4_STEPS)
    traj_rm, _ = rk4_trajectory(ref_minus_init, t, Omega, Delta, RK4_STEPS)

    # Compute E(t) on RK4 trajectory
    E_p_rk4, E_m_rk4 = compute_E_on_rk4(traj_rp, traj_rm, t_rk4, t, Omega, Delta)
    E_rk4 = 0.5 * (np.abs(E_p_rk4) + np.abs(E_m_rk4))

    # Compare with training E(t)
    E_train = 0.5 * (np.abs(ts['E_plus']) + np.abs(ts['E_minus']))

    # Save
    out_dir = RESULTS / dirname
    np.savetxt(out_dir / "ref_plus_energy_rk4.txt", E_p_rk4, fmt="%.10f")
    np.savetxt(out_dir / "ref_minus_energy_rk4.txt", E_m_rk4, fmt="%.10f")
    np.savetxt(out_dir / "t_rk4.txt", t_rk4, fmt="%.10f")

    m["mean_abs_E_rk4"] = float(E_rk4.mean())
    m["max_abs_E_rk4"] = float(E_rk4.max())
    with open(out_dir / "metrics.json", 'w') as f:
        json.dump(m, f, indent=2)

    print(f"beta={beta:2d}: T*={T_opt:.4f}, train mean|E|={E_train.mean():.6e}, RK4 mean|E|={E_rk4.mean():.6e}, "
          f"train max|E|={E_train.max():.6e}, RK4 max|E|={E_rk4.max():.6e}")

print("\nDone.")
