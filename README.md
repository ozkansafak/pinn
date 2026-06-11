# PINN — Lid-Driven Cavity Flow

A neural network trained to solve a fluid flow problem by satisfying the governing physics equations directly — no mesh, no simulation.

<p align="center">
  <img src="images/lid_driven_cavity_diagram.png" width="252"/>
</p>
<p align="center"><em>The top lid moves at u = 1, driving a recirculating vortex inside the unit square. Three walls are stationary (no-slip).</em></p>

<p align="center">
  <img src="videos/flow_Re100_uniformU_1xhidden.gif"/>
</p>
<p align="center"><em>Training animation — Re=100, uniform lid, 1x network (12,867 parameters). Vorticity + streamlines, pressure + −∇p vectors, streamfunction isolines over 40,000 epochs.</em></p>

---

## The Problem

**Setup:** A unit square box filled with viscous fluid. The top wall (the "lid") slides horizontally at speed u=1. The other three walls are fixed. The fluid inside circulates in a vortex — a standard test case in computational fluid dynamics (CFD) called the *lid-driven cavity*.

**Goal:** Find the velocity field `(u, v)` and pressure field `p` everywhere inside the box at steady state. `u` and `v` are the x- and y-direction components of velocity; `p` is the scalar pressure at each point.

**Physics constraint:** The solution must satisfy the incompressible Navier-Stokes (NS) equations — conservation of momentum and mass for a viscous fluid. The Reynolds number Re=100 (ν=0.01) sets how viscous the fluid is; at Re=100 the flow is smooth and laminar.

**Our approach:** Instead of discretizing the domain onto a mesh (as traditional CFD solvers do), we train a neural network `f(x, y) → (u, v, p)` to satisfy the NS equations at every point simultaneously. See [How it works →](DESIGN.md#loss-function)

---

## Network Architecture

```
(x, y)  →  [Linear → tanh/sin] × 4  →  Linear  →  (u, v, p)
```

Default: `[2, 64, 64, 64, 64, 3]` — 12,867 parameters. ReLU cannot be used because the NS residual requires second-order spatial derivatives, which vanish for ReLU. We experiment with tanh and sin activations.

<p align="center">
  <img src="assets/network_architecture.png" width="820"/>
</p>

---

## Width Sweep Results

Sin network errors decrease consistently with width; tanh degrades at width ≥ 128 due to neuron saturation. Width 1024 is the best result so far — 7× lower error than the tanh baseline at the same width. Full tables and figure in [DESIGN.md](DESIGN.md#width-sweep-design).

---

## Results

<p align="center">
  <img src="images/epoch=70_000%20Re=100.png"/>
</p>
<p align="center"><em>
<strong>Row 1:</strong> Vorticity field with streamlines (left); pressure field with −∇p vectors (right).<br>
<strong>Row 2:</strong> Pointwise PDE residual (left); boundary condition error on all four walls (right).<br>
<strong>Row 3:</strong> Cross-sectional u and v profiles at y = 0.001, 0.5, 0.999.<br>
<strong>Row 4:</strong> Training and evaluation loss curves (log scale).
</em></p>

At epoch 27,000 the network predicts u = 0.7225 at (x=0.5, y=0.9609), within 2% of the Ghia et al. (1982) reference value of 0.73722.

---

## Setup

```bash
uv sync
jupyter notebook
```

## File Structure

```
pinn.py                # tanh and sin network models, NS residual, boundary data
train.py               # Full training run
train_width.py         # Single width-sweep run
sweep_width.sh         # Orchestrator: 10 doubling widths in sequence
make_arch_diagram.py   # Generate assets/network_architecture.png
make_activation_fig.py # Generate assets/activation_comparison.png
make_error_fig.py      # Generate assets/error_vs_width.png
parse_sweep_logs.py    # Parse log files into CSVs
plot_sweep.py          # Plot loss & LR curves for all widths
notebook.ipynb         # Training driver and visualisation
results/               # width_sweep.csv, per-run logs, loss curves
pyproject.toml         # Dependencies
```

See [DESIGN.md](DESIGN.md) for full technical details.
