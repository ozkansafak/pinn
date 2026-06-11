# PINN Design Notes

Technical reference for the lid-driven cavity PINN. For results and setup see [README.md](README.md).

---

## The CFD Problem

The lid-driven cavity is a canonical benchmark in computational fluid dynamics. The domain is a unit square filled with a viscous, incompressible fluid. The top lid (y=1) moves horizontally at unit velocity (u=1), while the three remaining walls are stationary. This drives an asymmetric circulating vortex inside the cavity.

The governing equations are the incompressible Navier-Stokes equations:

**x-momentum:**
$$u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{\partial p}{\partial x} + \nu \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

**y-momentum:**
$$u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{\partial p}{\partial y} + \nu \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)$$

**Mass conservation:**
$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

where `u`, `v` are velocity components, `p` is pressure, and `ν = 0.01` (Re = 100).

---

## Boundary Conditions

| Wall | Condition |
|------|-----------|
| Bottom (y=0) | u=0, v=0 |
| Top (y=1) — the lid | u=1, v=0 |
| Left (x=0) | u=0, v=0 |
| Right (x=1) | u=0, v=0 |

`N_b = 1,000` points are sampled uniformly along the four walls. The BC loss is:

$$\mathcal{L}_{BC} = \frac{1}{N_b} \sum_i \left[ (u_{pred} - u_{BC})^2 + (v_{pred} - v_{BC})^2 \right]$$

---

## Loss Function

The total training loss combines three terms:

$$\mathcal{L} = 10 \cdot \mathcal{L}_{BC} + \mathcal{L}_{PDE} + 10 \cdot \mathcal{L}_{p}$$

**BC loss** `L_BC` — enforces boundary conditions on the walls.

**PDE loss** `L_PDE` — enforces the Navier-Stokes equations at collocation points in the interior.

**Pressure gauge** `L_p` — the incompressible NS equations only determine pressure up to an additive constant. We pin `p(0.5, 0.5) = 0`:

$$\mathcal{L}_{p} = p(0.5, 0.5)^2$$

The factor of 10 on `L_BC` and `L_p` ensures boundary conditions are satisfied before the interior.

---

## L_PDE and Collocation Points

`N_f = 10,000` collocation points are sampled randomly in the interior. A  residual is calculated as a measure of how far the predicted velocity and pressure fields deviate from satisfying the NS equations at each point:

$$\mathcal{L}_{PDE} = \frac{1}{N_f} \sum_i \left( r_x^2 + r_y^2 + r_c^2 \right)$$

where `r_x`, `r_y`, `r_c` are the residuals of x-momentum, y-momentum, and mass continuity.

### Evaluation L_PDE

After each training step, `L_PDE` is also evaluated on a hold out set of `N_eval = 2,000` points.

---

## How Autograd Works on Collocation Points

NS residuals require spatial derivatives of the network outputs — `∂u/∂x`, `∂²u/∂x²`, `∂p/∂x`, etc. These are computed via PyTorch autograd.

Collocation points are created with `requires_grad=True`:

```python
x_f = torch.rand(N_f, 1, requires_grad=True)
y_f = torch.rand(N_f, 1, requires_grad=True)
```

First and second derivatives are then computed exactly:

```python
u_x  = grad(u.sum(), x_f, create_graph=True)[0]   # ∂u/∂x
u_xx = grad(u_x.sum(), x_f, create_graph=True)[0]  # ∂²u/∂x²
```

`create_graph=True` keeps the graph of `u_x` so it can be differentiated again. During the backward pass, gradients flow through the derivative computation back to the weights.

> **Note — why `requires_grad=True` on `x_f` doesn't cause `x_f` to be updated during training:**
>
> The autograd computation graph and the optimizer are two separate things:
>
> - **Autograd / computation graph** — tracks how tensors were computed from other tensors. Any tensor with `requires_grad=True` participates. `.backward()` traverses this graph to compute gradients (`∂loss/∂tensor`) for every participating tensor — including `x_f` and the network weights.
> - **Optimizer** — a separate object that holds an explicit list of tensors to update. It calls `.grad` on each of *its* tensors and applies the update rule (Adam, SGD, etc.). It has no knowledge of the computation graph.
>
> So after `.backward()`, both `x_f.grad` and `weight.grad` are populated. But only the weights were passed to `Adam(net.parameters())`, so only the weights get updated by `.step()`. The gradient on `x_f` sits there unused. The collocation points are resampled fresh every epoch anyway.

### Inference note

For prediction, use `torch.no_grad()`. For evaluating `L_PDE` at inference time, `torch.no_grad()` cannot be used — the NS residual still requires autograd through the input coordinates. The weights are not updated; no `.backward()` or `.step()` is called. But the inputs must remain in the computation graph.

---

## SIREN — Sinusoidal Representation Networks

> Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B., & Wetzstein, G. (2020), *Implicit Neural Representations with Periodic Activation Functions.*, NeurIPS 2020. [arXiv:2006.09661](https://arxiv.org/abs/2006.09661)

Sitzmann et al. show that periodic activations — specifically `sin(ω₀·Wx+b)` with a carefully chosen initialization — allow networks to represent signals and their higher-order derivatives far more accurately than ReLU or Tanh. For PINNs, this is directly relevant: the NS residual requires second-order spatial derivatives, and SIREN's smooth, nonzero derivatives at every order give the optimizer a clean gradient signal all the way to the weights.

A SIREN replaces the `tanh` activation with a `sin`:

```
tanh  layer:  x  →  tanh(W x + b)
SIREN layer:  x  →  sin(ω₀ · W x + b)
```

The key difference is the **initialization scheme**. For a standard tanh network, weights are drawn from Xavier/He. For SIREN, weights are initialized so the input to `sin(·)` is uniformly distributed over `[−π, π]` — preserving the activation distribution through depth. The first layer uses `U(−1/n_in, 1/n_in)` and subsequent layers use `U(−√(6/n_in)/ω₀, +√(6/n_in)/ω₀)`.

**Why it matters for PINNs:**

- The NS equations require 2nd-order derivatives. For `sin`, every derivative is also a `sin` or `cos` — nonzero everywhere, so gradient signal flows cleanly.
- In wide tanh networks, pre-activations grow like `√width`, pushing neurons into the saturated ±1 region. SIREN's init keeps the pre-activations in the oscillatory regime regardless of width.
- We use `ω₀ = 1.0`. The cavity flow solution is smooth; low-frequency bias is appropriate.

<p align="center">
  <img src="assets/activation_comparison.png" width="900"/>
</p>
<p align="center"><em>tanh saturates to ±1 for large inputs — its 1st and 2nd derivatives collapse to zero in the flat region, starving gradient flow. sin stays oscillatory at all amplitudes.</em></p>

---

## Width Sweep Design

`sweep_width.sh` trains 10 networks doubling width from 4 → 2048:

```bash
bash sweep_width.sh              # SIREN, widths 4..2048, cap 60k epochs
bash sweep_width.sh 4 60000 tanh # tanh baseline
```

Each run (`train_width.py --width W --activation {siren|tanh}`):
- Builds `[2, W, W, W, W, 3]`
- LR scales as `lr = 1e-3 × (64/W)` — μP-inspired, keeps effective update magnitude constant across widths
- `ReduceLROnPlateau(factor=0.3, patience=3000 epochs)` — self-terminating when `lr < lr_initial × 1e-3` (7 plateau fires)
- Appends one row to `results/width_sweep.csv`
- Logs stdout to `results/logs/width_{W}_{activation}.log`
- Writes per-epoch loss and LR to `results/loss_curves/width_{W}_{activation}.csv`

**Note on collocation points:** `N_f = 10,000` is fixed across all widths in this sweep — it was not scaled with model capacity. A future sweep could scale `N_f ∝ sqrt(params)` to better match the model's representational capacity.

<p align="center">
  <img src="assets/error_vs_width.png" width="820"/>
</p>
<p align="center"><em>Normalized velocity error vs width for tanh and sin activations. tanh peaks at w=64 then collapses due to neuron saturation. The sin network's errors decrease consistently, reaching 0.23% at w=1024 — 5× better than the best tanh result.</em></p>

**Planned plots:**
- `eval_L_pde` vs `n_params` (log-log) — residual quality vs. capacity
- Loss and LR curves per width on one figure (`python plot_sweep.py`)
