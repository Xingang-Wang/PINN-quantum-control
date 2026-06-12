# Continuous Physics-Informed Representations for Quantum Rotation and Geometric Gates

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-B31B1B.svg)](https://arxiv.org/abs/xxxx.xxxxx)

Companion code and data for **"Continuous Physics-Informed Representations for Quantum Rotation and Geometric Gates"** ([arXiv:xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx)).

---

## Overview

This repository implements a **continuous physics-informed representation** for single-qubit gate design. Instead of optimizing discrete pulse amplitudes, a neural network generates the control fields, state trajectories, and gate duration as a single differentiable controlled process. The Bloch-equation residual, terminal gate objective, and physical constraints are imposed within a unified loss. The resulting controls are verified by independent time evolution.

We use two settings to expose two levels of structure:

| Setting | What the representation reveals |
|---|---|
| **Rotation gates** ($X$, $Z$, $Y$, intermediate axes) | Endpoint-constrained control: the learned processes recover the expected minimum-time organization of bounded single-qubit control. Gate time, saturated pulse form, and channel division follow target-axis geometry without being prescribed. |
| **Geometric $Z$ gate** | Path-constrained control: the endpoint map is not sufficient — the reference path must suppress dynamical phase while accumulating geometric phase. The representation identifies rapid control-direction turns as the bottleneck and converts this diagnosis into a training weight that refines the process. |

For the geometric gate, the PINN with turn-weighted loss achieves $\varepsilon_{\rm phase} = 1.3 \times 10^{-4}$ rad (geometric-phase closure error), compared to $4.8 \times 10^{-3}$ rad for a GRAPE-type baseline at matched duration.

---

## Repository structure

```
PINN-quantum-control/
├── src/                    Core modules (PINN framework, GRAPE baseline, RK4 validation)
├── scripts/                One-command reproduction scripts
├── data/                   Pre-computed results (metrics, controls, time series)
│   ├── rotation_gates/
│   └── geometric_gate/
├── figures/                Final figure PDFs (as in the manuscript)
├── supplementary/          Supplemental Material animations and description
│   ├── Movie_S1_rotation_gate_trajectories.mp4
│   ├── Movie_S2_geometric_gate_path_dynamics.mp4
│   └── supplemental_description.md
├── manuscript/             LaTeX source
└── pyproject.toml          Dependencies
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

# Individual experiments
python scripts/reproduce_rotation_gates.py      # Fig. 2–7
python scripts/reproduce_geometric_gate.py      # Fig. 8–10
python scripts/reproduce_grape_baseline.py      # GRAPE comparison data
```

Pre-computed data in `data/` allows figure generation without re-running training. Each script can also run the full optimization from scratch (see `--train` flag).

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

### Continuous PINN representation

A six-layer MLP (width 256, $\tanh$ activations) takes physical time $t \in [0, T]$ as input and outputs bounded control fields $\Omega(t), \Delta(t)$ and four probe trajectories $\bm r^{(k)}(t)$ simultaneously. The total gate duration $T = e^\tau$ is a learnable scalar optimized jointly with the network parameters.

Loss terms:
- **Dynamics loss** $\mathcal{L}_{\rm dyn}$: Bloch-equation residual at collocation points
- **Gate loss** $\mathcal{L}_{\rm gate}$: channel-matrix reconstruction error from terminal probe states
- **Time penalty** $\mathcal{L}_T = T$: biases toward shortest compatible duration
- **Regularization**: amplitude, smoothness, and boundary terms

### GRAPE-type baseline

Piecewise-constant controls with $N=80$ bins, matrix-exponential propagation, and L-BFGS-B optimization with 5 random restarts per gate. Optimal duration determined by external time scan over 8 candidates.

### Validation

All results are validated by **independent RK4 propagation** (4000 steps) with no reliance on network-internal trajectories. Process fidelity $F_{\rm proc} = \mathrm{Tr}(\chi_{\rm tar} \chi_{\rm imp})$ is computed from the reconstructed Pauli-basis process matrix.

---

## Citation

If you use this code, please cite:

```bibtex
@article{Du2025ContinuousPhysicsInformed,
  title   = {Continuous Physics-Informed Representations for Quantum Rotation and Geometric Gates},
  author  = {Du, Yao and Cheng, Jian-Jian and Zhang, Lin and Hu, Ming-Liang and Wang, Xingang},
  year    = {2025},
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

This work was supported by the National Natural Science Foundation of China (NNSFC) under Grant No. 12275165.
