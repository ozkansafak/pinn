# PINN — Lid-Driven Cavity Flow

A Physics-Informed Neural Network (PINN) that solves the 2D incompressible Navier-Stokes equations for the classic lid-driven cavity benchmark, without a grid, directly from the governing equations and boundary conditions.

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

The lid-driven cavity is a canonical computational fluid dynamics (CFD) benchmark. The domain is [0,1]² with a viscous incompressible fluid. The top lid moves at u=1; three walls are no-slip. This drives a recirculating vortex. We solve at Re=100 (ν=0.01).

Traditional CFD solvers discretize the domain onto a mesh and march forward in time. Here we take a different approach: a neural network is trained to satisfy the Navier-Stokes equations and boundary conditions simultaneously, without a mesh or time-stepping. The network takes `(x, y)` as input and outputs `(u, v, p)` — x- and y-direction velocity and pressure — at any point in the domain. Spatial derivatives are computed exactly via PyTorch autograd, and the NS residual is minimized as part of the training loss.

See [DESIGN.md](DESIGN.md) for the full formulation: NS equations, loss function, boundary conditions, collocation points, and autograd.

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
parse_sweep_logs.py    # Parse log files into CSVs
plot_sweep.py          # Plot loss & LR curves for all widths
notebook.ipynb         # Training driver and visualisation
results/               # width_sweep.csv, per-run logs, loss curves
pyproject.toml         # Dependencies
```

See [DESIGN.md](DESIGN.md) for full technical details.
