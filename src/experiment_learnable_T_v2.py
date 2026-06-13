"""
Approach B improved: more steps, better init, separate lr for T_log
"""
import os, json, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinn_dual_control_yz import (
    set_seed, Config, device, dtype,
    GATE_BLOCH, GATE_UNITARY, make_target,
    r0_init, r1_init, rx_init, ry_init,
    get_rates, stack_columns, stack_matrix_columns,
    grad_wrt_t, frobenius_norm_squared, vector_norm_squared,
    bloch_norm_penalty, PINNDualControl,
)

OUTDIR = "outputs_learnable_T_v2"
FIGDIR = "figures_learnable_T_v2"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)


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


def build_learnable(net, t_norm, cfg):
    T_phys = net.T_param
    t = t_norm * T_phys
    raw = net(t)
    u_Omega, u_Delta = raw[:, 0:1], raw[:, 1:2]
    Omega = cfg.Omega_max * torch.sin(math.pi * t / T_phys) * torch.tanh(u_Omega)
    Delta = cfg.Delta_max * torch.sin(math.pi * t / T_phys) * torch.tanh(u_Delta)
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


def compute_loss_T(net, t_norm, cfg, R_target, lambda_T, create_graph):
    Omega, Delta, r0, r1, rx, ry, t = build_learnable(net, t_norm, cfg)
    G1, G2 = get_rates(cfg)
    gd = cfg.gamma_down - cfg.gamma_up

    L_dyn = 0.0
    for r in [r0, r1, rx, ry]:
        x, y, z = r[:, 0:1], r[:, 1:2], r[:, 2:3]
        dx = grad_wrt_t(x, t, create_graph)
        dy = grad_wrt_t(y, t, create_graph)
        dz = grad_wrt_t(z, t, create_graph)
        Rx = dx + Delta * y + G2 * x
        Ry = dy - Delta * x + Omega * z + G2 * y
        Rz = dz - Omega * y + G1 * z - gd
        L_dyn = L_dyn + torch.mean(Rx**2 + Ry**2 + Rz**2)

    s0, s1, sx, sy = r0[-1, :], r1[-1, :], rx[-1, :], ry[-1, :]
    c = 0.5 * (s0 + s1)
    M = stack_matrix_columns(sx - c, sy - c, 0.5 * (s0 - s1))
    L_gate = frobenius_norm_squared(M - R_target) + vector_norm_squared(c)

    L_amp = torch.mean(Omega**2) + torch.mean(Delta**2)
    dO = grad_wrt_t(Omega, t, create_graph)
    dD = grad_wrt_t(Delta, t, create_graph)
    L_smooth = torch.mean(dO**2) + torch.mean(dD**2)
    L_phys = sum(bloch_norm_penalty(r) for r in [r0, r1, rx, ry])

    L_time = lambda_T * net.T_param
    L_total = (1.0*L_dyn + 10.0*L_gate + 1e-4*L_amp + 1e-4*L_smooth + 1e-3*L_phys + L_time)
    return {"L_total": L_total, "L_dyn": L_dyn, "L_gate": L_gate,
            "T_param": net.T_param.item()}


# ---- RK4 validation ----
def omega_interp(t_q, tg, og):
    return float(np.interp(t_q, tg, og))
def delta_interp(t_q, tg, dg):
    return float(np.interp(t_q, tg, dg))
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

def validate(net, t_norm, cfg):
    T_opt = net.T_param.item()
    Omega, Delta, r0, r1, rx, ry, t = build_learnable(net, t_norm, cfg)
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
    R_np = np.array(GATE_BLOCH[cfg.target_gate], dtype=np.float64)

    # Process fidelity
    I2=np.eye(2,dtype=np.complex128)
    SX=np.array([[0,1],[1,0]],dtype=np.complex128)
    SY=np.array([[0,-1j],[1j,0]],dtype=np.complex128)
    SZ=np.array([[1,0],[0,-1]],dtype=np.complex128)
    paulis=[SX,SY,SZ]
    E_I=I2+sum(c_rk4[i]*paulis[i] for i in range(3))
    E=[E_I]+[sum(M_rk4[j,i]*paulis[j] for j in range(3)) for i in range(3)]
    kets=[0.5*(I2+SZ),0.5*(I2-SZ),0.5*(SX+1j*SY),0.5*(SX-1j*SY)]
    Es=[0.5*(E[0]+E[3]),0.5*(E[0]-E[3]),0.5*(E[1]+1j*E[2]),0.5*(E[1]-1j*E[2])]
    basis=[np.array([[1,0],[0,0]]),np.array([[0,0],[0,1]]),
           np.array([[0,1],[0,0]]),np.array([[0,0],[1,0]])]
    J=sum(np.kron(basis[i],Es[i]) for i in range(4))/2.0
    U=GATE_UNITARY[cfg.target_gate]
    psi=U.reshape(-1,order='F')/math.sqrt(U.shape[0])
    J_target=np.outer(psi,np.conjugate(psi))
    F_proc=float(np.clip(np.real(np.trace(J_target@J)),0,1))
    F_avg=float((2*F_proc+1)/3)

    rng=np.random.default_rng(2025)
    vecs=rng.normal(size=(200,3))
    vecs=(vecs/np.linalg.norm(vecs,axis=1,keepdims=True)).astype(np.float64)
    fids=[]
    for r_in in vecs:
        r_out=rk4_prop(r_in,t_np,Omega_np,Delta_np,G1,G2,gd)
        r_ideal=R_np@r_in
        nr2=np.dot(r_out,r_out);ns2=np.dot(r_ideal,r_ideal);rs=np.dot(r_out,r_ideal)
        fids.append(float(np.clip(0.5*(1+rs+math.sqrt(max(0,(1-nr2)*(1-ns2)))),0,1)))
    return {"T_opt":T_opt,"F_avg":F_avg,"F_proc":F_proc,
            "err_M":float(np.linalg.norm(M_rk4-R_np,'fro')),
            "err_c":float(np.linalg.norm(c_rk4)),
            "random_mean_fid":float(np.mean(fids))}


def train_one(gamma, lambda_T, steps=5000, init_T=0.7, lr_T=5e-3):
    """Improved: more steps, better init, separate lr for T_log."""
    set_seed(42)
    R_target, _ = make_target('X')
    cfg = Config(target_gate='X', steps=steps, print_every=9999, gamma_down=gamma)

    net = PINNLearnableT(init_T=init_T).to(device)
    # Separate param groups: network weights vs T_log
    optimizer = optim.Adam([
        {'params': net.net.parameters(), 'lr': cfg.lr},         # 1e-3
        {'params': [net.T_log],         'lr': lr_T},             # higher for T
    ])

    t_norm = torch.linspace(0, 1, cfg.N_t, device=device, dtype=dtype).view(-1, 1)

    T_history = []
    for step in range(steps):
        optimizer.zero_grad()
        out = compute_loss_T(net, t_norm, cfg, R_target, lambda_T, True)
        out["L_total"].backward()
        optimizer.step()
        T_history.append(out["T_param"])
        if step % 1000 == 0 or step == steps - 1:
            print(f"  step {step:4d}  L={out['L_total'].item():.3e}  "
                  f"T={out['T_param']:.4f}")

    result = validate(net, t_norm, cfg)
    result["lambda_T"] = lambda_T
    result["init_T"] = init_T
    result["lr_T"] = lr_T
    result["steps"] = steps
    result["gamma"] = gamma
    return result


# ============================================================
# Key comparisons
# ============================================================

configs = [
    # (name, gamma, lambda_T, steps, init_T, lr_T)
    # v1 baseline (original B settings)
    ("v1_baseline",       0.05, 1.0, 2000, 1.0, 1e-3),
    ("v1_baseline",       0.30, 1.0, 2000, 1.0, 1e-3),
    # v2: all improvements
    ("v2_best",           0.05, 1.0, 5000, 0.7, 5e-3),
    ("v2_best",           0.30, 1.0, 5000, 0.7, 5e-3),
    # v2 with different lambda_T
    ("v2_lam0.5",         0.05, 0.5, 5000, 0.7, 5e-3),
    ("v2_lam0.5",         0.30, 0.5, 5000, 0.7, 5e-3),
]

all_results = []
for name, gamma, lam, steps, init_T, lr_T in configs:
    print(f"\n{'='*55}")
    print(f"  {name}: γ={gamma}, λ_T={lam}, steps={steps}, init_T={init_T}, lr_T={lr_T}")
    print(f"{'='*55}")
    r = train_one(gamma, lam, steps=steps, init_T=init_T, lr_T=lr_T)
    r["name"] = name
    r["gamma"] = gamma
    r["steps"] = steps
    all_results.append(r)
    print(f"  >>> T*={r['T_opt']:.4f}, F_avg={r['F_avg']:.6f}")

# Save
with open(f"{OUTDIR}/improved_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

# ---- Summary table ----
print("\n" + "="*80)
print("SUMMARY: Approach B improvements")
print("="*80)
print(f"{'Config':<18} {'γ':>5} {'λ_T':>5} {'steps':>5} {'init_T':>7} {'lr_T':>6} "
      f"{'T*':>6} {'F_avg':>8} {'F_proc':>8}")
print("-"*80)
for r in all_results:
    print(f"{r['name']:<18} {r['gamma']:>5.2f} {r['lambda_T']:>5.1f} "
          f"{r['steps']:>5d} {r['init_T']:>7.1f} "
          f"{'%.0e'%r.get('lr_T',1e-3):>6} "
          f"{r['T_opt']:>6.3f} {r['F_avg']:>8.6f} {r['F_proc']:>8.6f}")

# Approach A references
print("\nApproach A reference (sweep):")
print("  γ=0.05: T=0.6, F_avg=0.988885  (best sweep)")
print("  γ=0.30: T=0.6, F_avg=0.942424  (best sweep)")

print(f"\nResults saved to {OUTDIR}/")
