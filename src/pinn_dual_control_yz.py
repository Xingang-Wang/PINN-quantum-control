import os
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ============================================================
# 说明
# ============================================================
# 双控制场 PINN：Omega(t) + Delta(t)
#   Omega 耦合 y-z 平面旋转，Delta 耦合 x-y 平面旋转
# 两个独立控制 → 可实现任意单比特门
# 目标门通过 config.target_gate 参数选择：'X', 'Y', 'Z'


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(42)


@dataclass
class Config:
    # 物理参数
    Delta0: float = 0.0
    gamma_down: float = 0.05
    gamma_up: float = 0.0
    gamma_phi: float = 0.0
    # 门时间与采样
    T: float = 1.0
    N_t: int = 121
    # 脉冲幅度尺度
    Omega_max: float = 8.0
    Delta_max: float = 8.0
    # 网络
    hidden_dim: int = 96
    hidden_layers: int = 3
    # 训练
    lr: float = 1e-3
    steps: int = 3000
    print_every: int = 200
    # 损失权重
    alpha_dyn: float = 1.0
    beta_gate: float = 10.0
    chi_amp: float = 1e-4
    zeta_smooth: float = 1e-4
    eta_phys: float = 1e-3
    lambda_M: float = 1.0
    lambda_c: float = 1.0
    # 验证
    rk4_steps: int = 4000
    n_random_states: int = 200
    # 目标门
    target_gate: str = 'Y'
    output_dir: str = ""

    def __post_init__(self):
        if not self.output_dir:
            self.output_dir = f"outputs_pinn_dual_{self.target_gate.lower()}_gate"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


# ============================================================
# 目标门定义
# ============================================================

GATE_BLOCH = {
    'X': [[ 1, 0, 0], [0,-1, 0], [0, 0,-1]],   # (x,y,z)→(x,-y,-z)
    'Y': [[-1, 0, 0], [0, 1, 0], [0, 0,-1]],   # (x,y,z)→(-x,y,-z)
    'Z': [[-1, 0, 0], [0,-1, 0], [0, 0, 1]],   # (x,y,z)→(-x,-y,z)
}
GATE_UNITARY = {
    'X': np.array([[0,1],[1,0]], dtype=np.complex128),
    'Y': np.array([[0,-1j],[1j,0]], dtype=np.complex128),
    'Z': np.array([[1,0],[0,-1]], dtype=np.complex128),
}

def make_target(gate: str):
    R = torch.tensor(GATE_BLOCH[gate], device=device, dtype=dtype)
    c = torch.zeros(3, device=device, dtype=dtype)
    return R, c

R_target, c_target = make_target('Y')  # 默认值，train_and_evaluate 会覆盖

# 四个探针态
r0_init = torch.tensor([0.,0., 1.], device=device, dtype=dtype)
r1_init = torch.tensor([0.,0.,-1.], device=device, dtype=dtype)
rx_init = torch.tensor([1.,0., 0.], device=device, dtype=dtype)
ry_init = torch.tensor([0.,1., 0.], device=device, dtype=dtype)


# ============================================================
# 辅助函数
# ============================================================

def get_rates(cfg):
    G1 = cfg.gamma_down + cfg.gamma_up
    G2 = 0.5*(cfg.gamma_down + cfg.gamma_up) + cfg.gamma_phi
    return G1, G2

def make_time_grid(cfg, requires_grad=True):
    t = torch.linspace(0, cfg.T, cfg.N_t, device=device, dtype=dtype).view(-1,1)
    t.requires_grad_(requires_grad)
    return t

def stack_columns(x, y, z):
    for v in [x,y,z]:
        if v.ndim==2 and v.shape[1]==1: pass
    if x.ndim==2 and x.shape[1]==1: x=x.squeeze(1)
    if y.ndim==2 and y.shape[1]==1: y=y.squeeze(1)
    if z.ndim==2 and z.shape[1]==1: z=z.squeeze(1)
    return torch.stack([x,y,z], dim=1)

def stack_matrix_columns(c1,c2,c3):
    return torch.stack([c1,c2,c3], dim=1)

def grad_wrt_t(y, t, create_graph):
    if y.ndim==1: y=y.unsqueeze(1)
    return torch.autograd.grad(y.sum(), t, create_graph=create_graph,
                               retain_graph=True, allow_unused=False)[0]

def frobenius_norm_squared(M): return torch.sum(M*M)
def vector_norm_squared(v): return torch.sum(v*v)
def bloch_norm_penalty(r):
    norm_r = torch.sqrt(torch.sum(r*r, dim=1) + 1e-12)
    return torch.mean(torch.relu(norm_r - 1.0)**2)


# ============================================================
# PINN 网络：14 输出（Omega + Delta + 4×3 探针态）
# ============================================================

class PINNDualControl(nn.Module):
    """
    输出 14 通道：
      0: u_Omega
      1: u_Delta       ← 新增
      2,3,4:   u0_x, u0_y, u0_z
      5,6,7:   u1_x, u1_y, u1_z
      8,9,10:  ux_x, ux_y, ux_z
      11,12,13: uy_x, uy_y, uy_z
    """
    def __init__(self, hidden_dim=96, hidden_layers=3):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers-1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 14))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)


# ============================================================
# 构造双脉冲和轨迹
# ============================================================

def build_controls_and_trajectories(net, t, cfg):
    raw = net(t)  # (N_t, 14)

    u_Omega = raw[:, 0:1]
    u_Delta = raw[:, 1:2]

    # 两个脉冲都用 hard constraint：sin 包络 × tanh 缩放
    # 保证 t=0 和 t=T 时自动为 0
    Omega = cfg.Omega_max * torch.sin(math.pi * t / cfg.T) * torch.tanh(u_Omega)
    Delta = cfg.Delta_max * torch.sin(math.pi * t / cfg.T) * torch.tanh(u_Delta)

    g = 1.0 - torch.exp(-t)

    def make_traj(r_init, ux, uy, uz):
        return stack_columns(
            r_init[0] + g[:,0]*ux[:,0],
            r_init[1] + g[:,0]*uy[:,0],
            r_init[2] + g[:,0]*uz[:,0])

    r0 = make_traj(r0_init, raw[:,2:3],  raw[:,3:4],  raw[:,4:5])
    r1 = make_traj(r1_init, raw[:,5:6],  raw[:,6:7],  raw[:,7:8])
    rx = make_traj(rx_init, raw[:,8:9],  raw[:,9:10], raw[:,10:11])
    ry = make_traj(ry_init, raw[:,11:12], raw[:,12:13], raw[:,13:14])

    return Omega, Delta, r0, r1, rx, ry


# ============================================================
# Bloch 方程残差（含 Delta）
# ============================================================

def bloch_residual(r, Omega, Delta, t, cfg, create_graph):
    Gamma1, Gamma2 = get_rates(cfg)
    x, y, z = r[:,0:1], r[:,1:2], r[:,2:3]
    dx_dt = grad_wrt_t(x, t, create_graph=create_graph)
    dy_dt = grad_wrt_t(y, t, create_graph=create_graph)
    dz_dt = grad_wrt_t(z, t, create_graph=create_graph)

    Rx = dx_dt + Delta * y + Gamma2 * x
    Ry = dy_dt - Delta * x + Omega * z + Gamma2 * y
    Rz = dz_dt - Omega * y + Gamma1 * z - (cfg.gamma_down - cfg.gamma_up)
    return Rx, Ry, Rz

def compute_dynamical_loss(r0, r1, rx, ry, Omega, Delta, t, cfg, create_graph):
    loss = 0.0
    for r in [r0, r1, rx, ry]:
        Rx, Ry, Rz = bloch_residual(r, Omega, Delta, t, cfg, create_graph)
        loss = loss + torch.mean(Rx**2 + Ry**2 + Rz**2)
    return loss


# ============================================================
# 仿射映射重建 + 门损失
# ============================================================

def reconstruct_affine_map(r0, r1, rx, ry):
    s0, s1, sx, sy = r0[-1,:], r1[-1,:], rx[-1,:], ry[-1,:]
    c = 0.5*(s0+s1)
    M = stack_matrix_columns(sx-c, sy-c, 0.5*(s0-s1))
    return s0, s1, sx, sy, M, c

def compute_gate_loss(M, c, cfg):
    return cfg.lambda_M * frobenius_norm_squared(M - R_target) + cfg.lambda_c * vector_norm_squared(c)

def compute_total_loss(net, t, cfg, create_graph):
    Omega, Delta, r0, r1, rx, ry = build_controls_and_trajectories(net, t, cfg)
    L_dyn = compute_dynamical_loss(r0, r1, rx, ry, Omega, Delta, t, cfg, create_graph)
    s0, s1, sx, sy, M, c = reconstruct_affine_map(r0, r1, rx, ry)
    L_gate = compute_gate_loss(M, c, cfg)
    L_amp_O = torch.mean(Omega**2)
    L_amp_D = torch.mean(Delta**2)
    L_amp = L_amp_O + L_amp_D
    dO = grad_wrt_t(Omega, t, create_graph=create_graph)
    dD = grad_wrt_t(Delta, t, create_graph=create_graph)
    L_smooth = torch.mean(dO**2) + torch.mean(dD**2)
    L_phys = sum(bloch_norm_penalty(r) for r in [r0,r1,rx,ry])
    L_total = cfg.alpha_dyn*L_dyn + cfg.beta_gate*L_gate + cfg.chi_amp*L_amp + cfg.zeta_smooth*L_smooth + cfg.eta_phys*L_phys
    return {"Omega": Omega, "Delta": Delta,
            "r0": r0, "r1": r1, "rx": rx, "ry": ry,
            "M": M, "c": c,
            "L_dyn": L_dyn, "L_gate": L_gate, "L_amp": L_amp,
            "L_smooth": L_smooth, "L_phys": L_phys, "L_total": L_total}


# ============================================================
# RK4 独立验证
# ============================================================

def omega_interp(t_q, t_grid, omega_grid):
    return float(np.interp(t_q, t_grid, omega_grid))

def delta_interp(t_q, t_grid, delta_grid):
    return float(np.interp(t_q, t_grid, delta_grid))

def bloch_rhs(t_val, r, t_grid, omega_grid, delta_grid, cfg):
    Gamma1, Gamma2 = get_rates(cfg)
    x, y, z = r
    O = omega_interp(t_val, t_grid, omega_grid)
    D = delta_interp(t_val, t_grid, delta_grid)
    dx = -D * y - Gamma2 * x
    dy =  D * x - O * z - Gamma2 * y
    dz =  O * y - Gamma1 * z + (cfg.gamma_down - cfg.gamma_up)
    return np.array([dx, dy, dz], dtype=np.float64)

def rk4_propagate(r_init, t_grid, omega_grid, delta_grid, cfg, n_steps):
    t0, t1 = float(t_grid[0]), float(t_grid[-1])
    h = (t1-t0)/n_steps
    r = r_init.astype(np.float64).copy()
    tv = t0
    for _ in range(n_steps):
        k1 = bloch_rhs(tv, r, t_grid, omega_grid, delta_grid, cfg)
        k2 = bloch_rhs(tv+0.5*h, r+0.5*h*k1, t_grid, omega_grid, delta_grid, cfg)
        k3 = bloch_rhs(tv+0.5*h, r+0.5*h*k2, t_grid, omega_grid, delta_grid, cfg)
        k4 = bloch_rhs(tv+h, r+h*k3, t_grid, omega_grid, delta_grid, cfg)
        r += (h/6.0)*(k1+2*k2+2*k3+k4)
        tv += h
    return r

def rk4_validation(omega_grid, delta_grid, t_grid, cfg, M_pinn, c_pinn):
    probes = [np.array([0.,0.,1.]), np.array([0.,0.,-1.]),
              np.array([1.,0.,0.]), np.array([0.,1.,0.])]
    ends = [rk4_propagate(p, t_grid, omega_grid, delta_grid, cfg, cfg.rk4_steps) for p in probes]
    s0, s1, sx, sy = ends
    c_rk4 = 0.5*(s0+s1)
    M_rk4 = np.stack([sx-c_rk4, sy-c_rk4, 0.5*(s0-s1)], axis=1)
    R_np = np.array(GATE_BLOCH[cfg.target_gate], dtype=np.float64)
    return {
        "M_rk4": M_rk4, "c_rk4": c_rk4,
        "diff_M": np.linalg.norm(M_pinn-M_rk4, 'fro'),
        "diff_c": np.linalg.norm(c_pinn-c_rk4),
        "err_M": np.linalg.norm(M_rk4-R_np, 'fro'),
        "err_c": np.linalg.norm(c_rk4),
    }


# ============================================================
# 过程保真度
# ============================================================

I2 = np.eye(2, dtype=np.complex128)
SX = np.array([[0,1],[1,0]], dtype=np.complex128)
SY = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
SZ = np.array([[1,0],[0,-1]], dtype=np.complex128)

def choi_from_affine(M, c):
    paulis = [SX, SY, SZ]
    E_I = I2 + sum(c[i]*paulis[i] for i in range(3))
    E = [E_I] + [sum(M[j,i]*paulis[j] for j in range(3)) for i in range(3)]
    kets = [0.5*(I2+SZ), 0.5*(I2-SZ), 0.5*(SX+1j*SY), 0.5*(SX-1j*SY)]
    Es = [0.5*(E[0]+E[3]), 0.5*(E[0]-E[3]), 0.5*(E[1]+1j*E[2]), 0.5*(E[1]-1j*E[2])]
    basis = [np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]]),
             np.array([[0,1],[0,0]]), np.array([[0,0],[1,0]])]
    J = sum(np.kron(basis[i], Es[i]) for i in range(4))
    return J / 2.0

def choi_unitary(U):
    psi = U.reshape(-1, order='F') / math.sqrt(U.shape[0])
    return np.outer(psi, np.conjugate(psi))

def process_fidelity(M, c, U_target):
    F = np.real(np.trace(choi_unitary(U_target) @ choi_from_affine(M, c)))
    return float(np.clip(F, 0, 1))

def avg_gate_fidelity(F_proc, d=2):
    return float((d*F_proc+1)/(d+1))


# ============================================================
# 随机态验证
# ============================================================

def bloch_fidelity(r, s):
    nr2, ns2 = float(np.dot(r,r)), float(np.dot(s,s))
    rs = float(np.dot(r,s))
    return float(np.clip(0.5*(1+rs+math.sqrt(max(0,(1-nr2)*(1-ns2)))), 0, 1))

def random_validation(M_rk4, c_rk4, t_grid, omega_grid, delta_grid, cfg):
    rng = np.random.default_rng(2025)
    vecs = rng.normal(size=(cfg.n_random_states, 3))
    vecs = (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float64)
    R_np = np.array(GATE_BLOCH[cfg.target_gate], dtype=np.float64)
    fids, errs = [], []
    for r_in in vecs:
        r_out = rk4_propagate(r_in, t_grid, omega_grid, delta_grid, cfg, cfg.rk4_steps)
        r_ideal = R_np @ r_in
        fids.append(bloch_fidelity(r_out, r_ideal))
        errs.append(np.linalg.norm(r_out - r_ideal))
    return {"mean_fidelity": float(np.mean(fids)), "mean_error": float(np.mean(errs))}


# ============================================================
# 训练主函数
# ============================================================

def train_and_evaluate(cfg):
    global R_target, c_target
    R_target, c_target = make_target(cfg.target_gate)
    os.makedirs(cfg.output_dir, exist_ok=True)
    net = PINNDualControl(hidden_dim=cfg.hidden_dim, hidden_layers=cfg.hidden_layers).to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    t_train = make_time_grid(cfg, requires_grad=True)

    print("="*60)
    print(f"  Noisy {cfg.target_gate} Gate — Dual Control (Omega + Delta)")
    print("="*60)

    loss_history = {k: [] for k in ["L_total","L_dyn","L_gate","L_amp","L_smooth","L_phys"]}

    for step in range(cfg.steps):
        optimizer.zero_grad()
        out = compute_total_loss(net, t_train, cfg, create_graph=True)
        out["L_total"].backward()
        optimizer.step()

        for k in loss_history:
            loss_history[k].append(out[k].item())

        if step % cfg.print_every == 0 or step == cfg.steps - 1:
            print(f"step {step:4d}  L={out['L_total'].item():.4e}  "
                  f"L_dyn={out['L_dyn'].item():.4e}  L_gate={out['L_gate'].item():.4e}")

    # 评估
    t_eval = make_time_grid(cfg, requires_grad=True)
    out = compute_total_loss(net, t_eval, cfg, create_graph=False)

    Omega_star = out["Omega"].detach().cpu().numpy().squeeze()
    Delta_star = out["Delta"].detach().cpu().numpy().squeeze()
    M_star = out["M"].detach().cpu().numpy()
    c_star = out["c"].detach().cpu().numpy()
    t_np = t_eval.detach().cpu().numpy().squeeze()

    # RK4 验证
    rk4 = rk4_validation(Omega_star, Delta_star, t_np, cfg, M_star, c_star)
    U_target = GATE_UNITARY[cfg.target_gate]
    F_proc = process_fidelity(rk4["M_rk4"], rk4["c_rk4"], U_target)
    F_avg = avg_gate_fidelity(F_proc)
    rand = random_validation(rk4["M_rk4"], rk4["c_rk4"], t_np, Omega_star, Delta_star, cfg)

    print("\n" + "="*60)
    print(f"  RESULTS: {cfg.target_gate} Gate (gamma={cfg.gamma_down})")
    print("="*60)
    print(f"M_RK4 =\n{rk4['M_rk4']}")
    print(f"c_RK4 = {rk4['c_rk4']}")
    R_np = np.array(GATE_BLOCH[cfg.target_gate], dtype=np.float64)
    print(f"Target R_{cfg.target_gate} =\n{R_np}")
    print(f"||M - R||_F           = {rk4['err_M']:.6e}")
    print(f"||c||                 = {rk4['err_c']:.6e}")
    print(f"Process fidelity      = {F_proc:.8f}")
    print(f"Average gate fidelity  = {F_avg:.8f}")
    print(f"Random mean fidelity   = {rand['mean_fidelity']:.8f}")
    print("="*60)

    # 保存
    report = {"config": asdict(cfg), "M_RK4": rk4["M_rk4"].tolist(), "c_RK4": rk4["c_rk4"].tolist(),
              "process_fidelity": F_proc, "average_gate_fidelity": F_avg, "random_validation": rand}
    with open(os.path.join(cfg.output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    np.save(os.path.join(cfg.output_dir, "time_grid.npy"), t_np)
    np.save(os.path.join(cfg.output_dir, "pulse_omega.npy"), Omega_star)
    np.save(os.path.join(cfg.output_dir, "pulse_delta.npy"), Delta_star)
    for key in ["r0","r1","rx","ry"]:
        np.save(os.path.join(cfg.output_dir, f"{key}.npy"), out[key].detach().cpu().numpy())

    # 保存损失历史
    for k, v in loss_history.items():
        np.save(os.path.join(cfg.output_dir, f"loss_{k.replace('L_','')}.npy"), np.array(v))

    # 画图：双脉冲
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(t_np, Omega_star, 'b-', lw=2)
    ax1.set_xlabel('t'); ax1.set_ylabel('Ω(t)')
    ax1.set_title(f'Omega(t) — {cfg.target_gate} gate'); ax1.grid(True, alpha=0.3)
    ax2.plot(t_np, Delta_star, 'r-', lw=2)
    ax2.set_xlabel('t'); ax2.set_ylabel('Δ(t)')
    ax2.set_title(f'Delta(t) — {cfg.target_gate} gate'); ax2.grid(True, alpha=0.3)
    fig.suptitle(f'Dual Control Pulses for {cfg.target_gate} Gate (γ={cfg.gamma_down})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "dual_pulses.png"), dpi=180)
    plt.close()

    return report


# ============================================================
# 主入口：依次运行 Y 门和 Z 门
# ============================================================
if __name__ == "__main__":
    print("\n" + "#"*60)
    print("#  Running Z Gate Optimization")
    print("#"*60 + "\n")
    set_seed(42)
    cfg_z = Config(target_gate='Z', steps=3000)
    report_z = train_and_evaluate(cfg_z)
