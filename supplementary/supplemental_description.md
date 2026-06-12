# Supplemental Material

This directory contains the animations referenced in the manuscript
"Continuous Controlled-Process Representations for Quantum Rotation and Geometric Gates."

---

## Movie S1 — Rotation-gate trajectories and control fields

**File:** `Movie_S1_rotation_gate_trajectories.mp4`

**Content:** Time-resolved Bloch-sphere trajectories and synchronized control fields for two representative rotation gates:

- **Left panel:** $X(\pi)$ gate — a directly generated rotation using a nearly saturated $\Omega(t)$ channel with $\Delta(t) \simeq 0$.
- **Right panel:** $Y(\pi)$ gate — an indirectly synthesized rotation requiring coordinated noncommuting action of both $\Omega(t)$ and $\Delta(t)$.

Each panel shows:
- A Bloch sphere with two probe-state trajectories (solid lines) and their current positions (filled dots). Open circles mark the ideal target states.
- The instantaneous control-field direction $\hat{\bm h}(t) = (\Omega, 0, \Delta)/|(\Omega, \Delta)|$ shown as a red arrow on the Bloch sphere.
- Synchronized $\Omega(t)$ (blue) and $\Delta(t)$ (red) panels with fill, tracking the control amplitude at each time step.

**Key observation:** The $X(\pi)$ gate uses a single saturated control channel, while the $Y(\pi)$ gate requires both channels to cooperate through the commutator structure $[\sigma_x, \sigma_z] = 2i\sigma_y$, resulting in a longer optimized duration.

**Corresponding section:** §III A — Rotation gates. Referenced near Fig. 4 and Fig. 7.

---

## Movie S2 — Geometric gate path dynamics

**File:** `Movie_S2_geometric_gate_path_dynamics.mp4`

**Content:** Time-resolved Bloch-sphere trajectories, synchronized control fields, and instantaneous control-direction arrow for the optimized geometric $Z(\pi)$ gate.

The animation shows:
- A Bloch sphere with probe-state trajectories implementing the target $Z$ rotation ($|{+}x\rangle \to |{-}x\rangle$, $|{+}y\rangle \to |{-}y\rangle$) and reference-state loops that carry the geometric phase.
- The instantaneous control direction $\hat{\bm h}(t)$ displayed as a red arrow on the Bloch sphere, illustrating the coordination between the control field and the reference path.
- Synchronized $\Omega(t)$ and $\Delta(t)$ panels with fill.

**Key observation:** The control direction rotates smoothly through the $\Omega$–$\Delta$ plane, maintaining the parallel-transport condition $\hat{\bm h}(t) \perp \hat{\bm n}(t)$ along the reference paths. The reference loops close at $t = T$, confirming cycle completion.

**Corresponding section:** §III B — Geometric gates. Referenced near Fig. 8.

---

## Technical details

| Property | Value |
|----------|-------|
| Format | MP4 (H.264) |
| Resolution | 1920 × 960 px |
| Frame rate | 18 fps |
| Duration | ~5.6 s each |
| Audio | None |

Both files can be played by any standard media player supporting MP4/H.264 (VLC, QuickTime, browser).
