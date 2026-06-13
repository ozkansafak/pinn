# PINN, Lid Driven Cavity Flow

A neural network trained to solve a fluid flow problem by satisfying the governing physics equations directly — no mesh, no simulation.

<p align="center">
  <img src="images/lid_driven_cavity_diagram.png" width="252"/>
</p>
<p align="center"><em>The top lid moves at u = 1, driving a recirculating asymettric vortex inside the flow domain. The three wall boundaries are stationary (no slip condition).</em></p>

<p align="center">
  <img src="videos/flow_Re100_uniformU_1xhidden.gif"/>
</p>
<p align="center"><em>Training animation — sin network, W=64, 12,867 parameters, Re=100, uniform lid, 36,000 epochs. 
Left: vorticity field + velocity vectors.  Center: pressure field + −∇p vectors. Right: streamfunction isolines.</em></p>

---

## The Problem

**Setup:** A unit square box filled with viscous fluid. The top wall (the "lid") slides horizontally at speed u=1. The other three walls are fixed. The fluid inside circulates in a vortex — a standard test case in computational fluid dynamics (CFD) called the *lid-driven cavity*.

**Goal:** Find the velocity field `(u, v)` and pressure field `p` everywhere inside the box at steady state. `u` and `v` are the x- and y-direction velocity components; `p` is scalar pressure.

**Our approach:** Traditional CFD solvers discretize the domain onto a mesh and solve the equations numerically at each grid point. Instead, we train a neural network `f(x, y) → (u, v, p)` to satisfy the governing equations at selected points inside the flow domain simultaneously.

---

## Governing Equations

The solution must satisfy the incompressible Navier-Stokes equations at Re=100. (Note that Re characterize the ratio of inertial to viscous forces.  Re=100 places the flow firmly in the laminar regime. At Re~5,000–10,000 the flow becomes turbulent and far harder to solve numerically.)

**x direction momentum:**
$$u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{\partial p}{\partial x} + \nu \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

**y direction momentum:**
$$u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{\partial p}{\partial y} + \nu \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)$$

**Mass conservation (incompressibility):**
$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

**Boundary conditions:** no-slip on all four walls — fluid at each wall matches the wall velocity. Three walls are fixed (u=v=0); fluid at the lid moves at u=1, v=0.

---

## The PINN Approach

The network is trained by minimizing a residual with three terms:

$$\mathcal{L} = 10 \cdot \mathcal{L}_{BC} + \mathcal{L}_{PDE} + 10 \cdot \mathcal{L}_{p}$$

- **`L_BC`** — boundary condition error on all four walls (`N_b = 1,000` wall points)
- **`L_PDE`** — NS residual at collocation points: $\frac{1}{N_f}\sum(r_x^2 + r_y^2 + r_c^2)$, where `r_x`, `r_y`, `r_c` are the x-momentum, y-momentum, and continuity residuals
- **`L_p`** — pressure gauge: pins `p(0.5, 0.5) = 0` to fix the free constant (incompressible NS only determines pressure up to an additive constant)

The factor of 10 on `L_BC` and `L_p` prioritizes boundary satisfaction over interior accuracy early in training.

`L_PDE` requires evaluating the NS residual at `N_f = 10,000` random interior *collocation points* each epoch. This in turn requires spatial derivatives of the network outputs — we use PyTorch autograd to compute them exactly:

```python
x_f = torch.rand(N_f, 1, requires_grad=True)   # collocation points
u_x  = grad(u.sum(), x_f, create_graph=True)[0] # ∂u/∂x
u_xx = grad(u_x.sum(), x_f, create_graph=True)[0] # ∂²u/∂x²
```

<details>
<summary><strong>Note: why <code>requires_grad=True</code> on collocation points doesn't update them</strong></summary>

The autograd computation graph and the optimizer are two separate objects:

- **Autograd:** Builds a computation graph to record the sequence of operations that produces each tensor. `.backward()` traverses this graph in reverse via the chain rule to compute gradients for every tensor in the graph, including `x_f` and the network weights.
- **Optimizer:** a separate object with a list of tensors to update. `Adam(model.parameters())` only knows about the weights.

So `x_f.grad` is populated after `.backward()` but never used by the optimizer. The collocation points are resampled fresh every epoch anyway.
</details>

---

## Network Architecture

```
(x, y)  →  [sin] × 4  →  (u, v, p)
```

Default: `[2, 64, 64, 64, 64, 3]` — 12,867 parameters. ReLU cannot be used because the NS residual requires second-order spatial derivatives, which vanish for ReLU. We experiment with tanh and sin activations.

<p align="center">
  <img src="assets/network_architecture.png" width="574"/>
</p>

### Sinusoidal Activation

```
tanh network:  x  →  tanh(W x + b)
sin  network:  x  →  sin(ω₀ · W x + b)
```

In wide tanh networks, pre-activations scale like $\sqrt{\text{width}}$, pushing neurons into saturation. Saturated neurons have near-zero derivatives — computing $\partial^2 u / \partial x^2$ via autograd chains through activation derivatives, so when those collapse the viscous term drops out of the residual and `L_PDE` cannot be minimized.

The sin activation avoids this via a matching initialization that keeps pre-activations in `[−π, π]` at any width. The first layer uses `U(−1/n_in, 1/n_in)`; hidden layers use $U\left(-\frac{\sqrt{6/n_\text{in}}}{\omega_0},\ +\frac{\sqrt{6/n_\text{in}}}{\omega_0}\right)$. We use `ω₀ = 1.0`. Every derivative of sin is also a sin or cos, so the gradients flow cleanly throughout the network.

<p align="center">
  <img src="assets/activation_comparison.png" width="900"/>
</p>
<p align="center"><em>tanh saturates for large inputs. sin stays oscillatory at all inputs.</em></p>

---

## Results

### Width Sweep

We train 10 networks by doubling the layer widths from 4 to 2048, with both tanh and sin activations, and measure how accuracy scales with model size.

- LR scales as `lr = 1e-3 × (64/W)` — μP-inspired, keeps effective update magnitude constant across widths
- Training terminates automatically via `ReduceLROnPlateau(factor=0.3, patience=3000 epochs)` when `lr < lr_initial × 1e-3`

<p align="center">
  <img src="assets/error_vs_width.png" width="900"/>
</p>
<p align="center"><em>Left: percent velocity error vs width — tanh degrades at w>64; sin peaks at w=1024 (0.23% error). 
Right: PDE residual loss vs width — sin's L_PDE worsens after w=128, likely due to insufficient N_f=10,000 collocation points for larger models.</em></p>

Error decreases slowly past w=32 because N_f is held fixed at 10,000 while model size grows ~4× per step; by Chinchilla scaling, $N_f \propto N_\text{params}$, so the collocation budget should grow proportionally with the model.

#### sin network sweep

| Width | Model size | N_f | Epochs | Train time | eval_L_pde | L_pde_max | u_ghia | \|Δu\| |
|------:|-----------:|----:|-------:|-----------:|-----------:|----------:|-------:|------:|
| 4 | 87 | 10,000 | 26,500 | 10 min | 0.01968 | 1.064 | 0.6338 | 0.1035 |
| 8 | 267 | 10,000 | 33,500 | 12 min | 0.01566 | 1.382 | 0.6374 | 0.0998 |
| 16 | 915 | 10,000 | 60,000 | 21 min | 0.00274 | 0.636 | 0.7333 | 0.0039 |
| 32 | 3.4K | 10,000 | 40,500 | 18 min | 0.00167 | 0.425 | 0.7355 | 0.0018 |
| 64 | 12.9K | 10,000 | 50,000 | 17 min | 0.00091 | 0.499 | 0.7402 | 0.0030 |
| 128 | 50.3K | 40,000 | 44,500 | 106 min | 0.00060 | 0.268 | 0.7416 | 0.0044 |
| 256 | 199K | 155,000 | — | — | — | — | — | — |
| 512 | 791K | 600,000 | — | — | — | — | — | — |

All runs on L4 GPU via [Modal](https://modal.com) cloud compute. `N_f` scales proportionally with model size, anchored at w=64 → N_f=10,000. w=256 and w=512 runs in progress. `u_ghia` is the predicted u-velocity at `(x=0.5, y=0.9609)`, compared against the Ghia et al. (1982) reference value of 0.73722 at Re=100.

`eval_L_pde` and `L_pde_max` are both evaluated on the same uniform 128×128 interior grid — mean and max of `r_x² + r_y² + r_c²` respectively. From a physics standpoint `L_pde_max` is the more meaningful metric: the mean can look small even when the solution violates the equations badly at isolated points. With proportionally scaled N_f, `L_pde_max` drops to 0.268 at w=128 — a 3× improvement over the fixed N_f=10,000 baseline (0.879), confirming that the collocation budget was the bottleneck.

#### Future Work

- Add `L_pde_max` directly to the training loss (e.g. as a minimax term) so the optimizer is explicitly penalized for worst-case residuals rather than average ones.

---

## Appendix

### Training Dashboard

<p align="center">
  <img src="videos/dashboard_Re100_uniformU_1xhidden.gif"/>
</p>
<p align="center"><em>Dashboard animation — sin network, W=64, 12,867 parameters, Re=100, uniform lid, 36,000 epochs. Row 1: vorticity + pressure fields. Row 2: pointwise PDE residual + BC error. Row 3: cross-sectional u and v profiles. Row 4: loss curves and learning rate schedule.</em></p>

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

---

## References

- Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B., & Wetzstein, G. (2020), *Implicit Neural Representations with Periodic Activation Functions.*, NeurIPS 2020. [arXiv:2006.09661](https://arxiv.org/abs/2006.09661)
- Ghia, U., Ghia, K. N., & Shin, C. T. (1982), *High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method.*, Journal of Computational Physics, 48(3), 387–411. [pdf link](https://www.msaidi.ir/upload/Ghia1982.pdf?i=1)
- Botella, O. & Peyret, R. (1998), *Benchmark Spectral Results on the Lid-Driven Cavity Flow.*, Computers & Fluids, 27, 421–433. [pdf link](https://cats2d.com/documentation/botellapeyret98.pdf) (higher-accuracy spectral benchmark for the same problem)
- Yang, G., Hu, E. J., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., ... & Gao, J. (2022), *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.*, arXiv:2203.03466. [arXiv:2203.03466](https://arxiv.org/pdf/2203.03466)
