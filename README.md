# Evolution-Level Quantum Optimal Control of Single-Qubit Gates with Physics-Informed Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-B31B1B.svg)](https://arxiv.org/abs/xxxx.xxxxx)

Companion code and data for **"Evolution-Level Quantum Optimal Control of Single-Qubit Gates with Physics-Informed Neural Networks"** ([arXiv:xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx)).

---

## Overview

This repository implements an **evolution-level representation** for single-qubit gate design using physics-informed neural networks. Instead of optimizing discrete pulse amplitudes, a neural network generates the control fields, state trajectories, and gate duration as a single differentiable controlled process. The Bloch-equation residual, terminal gate objective, and physical constraints are imposed within a unified loss. The resulting controls are verified by independent time evolution.

We use two settings to expose two levels of structure:

| Setting | What the representation reveals |
|---|---|
| **Rotation gates** ($X$, $Z$, $Y$, intermediate axes) | Endpoint-constrained control: the learned processes recover the expected minimum-time organization of bounded single-qubit control. Gate time, saturated pulse form, and channel division follow target-axis geometry without being prescribed. |
| **Geometric $Z$ gate** | Path-constrained control: the endpoint map is not sufficient — the reference path must suppress dynamical phase while accumulating geometric phase. The representation identifies rapid control-direction turns as the bottleneck and converts this diagnosis into a training weight that refines the process. |

For the geometric gate, the PINN with turn-weighted loss achieves $F_{\rm proc} > 0.999999$ with negligible dynamical phase, and the residual geometricity error is reduced at the turning intervals without rising elsewhere.

---

## Repository structure

```
PINN-quantum-control/
├── src/                        Core modules
│   ├── pinn_dual_control_yz.py         PINN framework (network, loss, Bloch dynamics)
│   ├── experiment_general_gates.py     Rotation-gate sweep (control geometry, Fig. 3–5)
│   ├── experiment_general_gates_minimal.py  Minimal-constraint sweep (time law, Fig. 6)
│   ├── experiment_probe_dla_universal.py    Probe-count law (Fig. 7)
│   ├── pinn_geometric_Z_gate_turn_weighted.py  Geometric gate with turn-weighted loss
│   ├── compute_pinn_rk4_E.py           Independent RK4 validation
│   ├── compute_geometric_phase.py      Geometric-phase decomposition
│   └── grape/                          GRAPE baselines (L-BFGS-B)
├── scripts/                    Figure & animation generation
│   ├── generate_all_figures.py         Generate all paper figures → figures/
│   ├── make_method_overview_loop.py    Method overview figure
│   ├── animate_suppmovie1_rotation_gates.py   Supplemental Movie S1
│   └── animate_suppmovie2_geometric_path.py   Supplemental Movie S2
├── data/                       Pre-computed results (metrics, controls)
│   ├── rotation_gates/
│   └── geometric_gate/
├── figures/                    Final figure PDFs (as in the manuscript)
├── supplementary/              Supplemental Material
│   ├── Movie_S1_rotation_gate_trajectories.mp4
│   ├── Movie_S2_geometric_gate_path_dynamics.mp4
│   └── supplemental_description.md
├── manuscript/                 LaTeX source
└── pyproject.toml              Dependencies
```

---

## Quick start

### Install

```bash
pip install -e .
```

Dependencies: Python ≥ 3.9, PyTorch ≥ 2.0, NumPy, SciPy, Matplotlib.

### Reproduce paper figures

```bash
# Generate all figures → figures/
python scripts/generate_all_figures.py

# Method overview figure
python scripts/make_method_overview_loop.py

# Supplemental animations → supplementary/
python scripts/animate_suppmovie1_rotation_gates.py
python scripts/animate_suppmovie2_geometric_path.py
```

### Run experiments

```bash
# Rotation gates (Fig. 3–5)
python src/experiment_general_gates.py

# Minimal-constraint rotation gates / time law (Fig. 6)
python src/experiment_general_gates_minimal.py

# Probe-count law (Fig. 7)
python src/experiment_probe_dla_universal.py

# Geometric gate with turn-weighted loss
python src/pinn_geometric_Z_gate_turn_weighted.py

# GRAPE baseline comparison
python src/grape/experiment_grape_lbfgsb_rewrite.py          # rotation gates
python src/grape/grape_geometric_Z_gate_baseline.py          # geometric gate
```

Pre-computed data in `data/` allows figure generation without re-running training.

---

## Supplemental Material

| File | Content | Section |
|---|---|---|
| `Movie_S1_rotation_gate_trajectories.mp4` | Time-resolved Bloch trajectories and control fields for $X(\pi)$ (direct) and $Y(\pi)$ (indirect) gates | §III A — Rotation gates |
| `Movie_S2_geometric_gate_path_dynamics.mp4` | Bloch-sphere path dynamics, control direction, and synchronized fields for the geometric $Z(\pi)$ gate | §III B — Geometric gates |

Each animation shows synchronized Bloch-sphere trajectories, $\Omega(t)$ and $\Delta(t)$ panels, and a moving time marker. Movie S2 additionally displays the instantaneous control-direction arrow and the reference-state loops that carry the geometric constraint.

A full description is in [`supplementary/supplemental_description.md`](supplementary/supplemental_description.md).

---

## Key methods

### Evolution-level PINN representation

A fully connected MLP (three-layer, width 96 for rotation gates; six-layer, width 256 for geometric gates, $\tanh$ activations) takes physical time $t \in [0, T]$ as input and outputs bounded control fields $\Omega(t), \Delta(t)$ and probe/reference trajectories simultaneously. The total gate duration $T = e^\tau$ is a learnable scalar optimized jointly with the network parameters.

Loss terms:
- **Dynamics loss** $\mathcal{L}_{\rm dyn}$: Bloch-equation residual at collocation points
- **Gate loss** $\mathcal{L}_{\rm gate}$: channel-matrix reconstruction error from terminal probe states
- **Time penalty** $\mathcal{L}_T = T$: biases toward shortest compatible duration
- **Regularization**: amplitude, smoothness, and boundary terms
- **Path constraints** (geometric gate only): geometricity, cycle closure, purity, orthogonality

### Piecewise-constant baseline (geometric gate)

Piecewise-constant controls with $N=200$ bins, matrix-exponential propagation, and L-BFGS-B optimization with 5 random restarts. Matched duration $T=1.5$, same sine-envelope and amplitude bounds as the PINN.

### Validation

All results are validated by **independent RK4 propagation** (4000 steps) with no reliance on network-internal trajectories. Process fidelity $F_{\rm proc} = \mathrm{Tr}(\chi_{\rm tar} \chi_{\rm imp})$ is computed from the reconstructed Pauli-basis process matrix.

---

## Citation

If you use this code, please cite:

```bibtex
@article{Du2026EvolutionLevel,
  title   = {Evolution-Level Quantum Optimal Control of Single-Qubit Gates with Physics-Informed Neural Networks},
  author  = {Du, Yao and Cheng, Jian-Jian and Zhang, Lin and Hu, Ming-Liang and Wang, Xingang},
  year    = {2026},
  eprint  = {xxxx.xxxxx},
  archivePrefix = {arXiv}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

---

## License

- **Code**: [MIT License](LICENSE)
- **Figures and manuscript text**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (Grant Nos. 12275165 and 12275212), the Young Talent Support Program for Doctoral Students of the China Association for Science and Technology (CAST), the Natural Science Foundation of Shaanxi Province (Grant No. 2025JC-YBQN-055), and the Youth Innovation Team of Shaanxi Universities (Grant No. 24JP177).
