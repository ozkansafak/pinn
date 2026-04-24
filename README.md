# PINN — Lid-Driven Cavity Flow



![Training animation Re=100](images/Re=100.gif)

Training Animation for Re=100, that shows the evolution of the flow field and the streamline function


A Physics Informed Neural Network (PINN) that solves the 2D incompressible Navier Stokes equations for the classic lid driven cavity problem, without finite difference or finte volume grid, directly from the governing equations and the boundary conditions.


![Flow field at epoch 70,000 Re=100](images/epoch=70_000%20Re=100.png)

[Click for training video: Re=100_compressed.mp4](images/Re=100_compressed.mp4)

---

## The CFD Problem

The lid-driven cavity is a canonical benchmark in computational fluid dynamics. The domain is a unit square [0,1]² filled with a viscous, incompressible fluid. The top wall (y=1) moves horizontally at unit velocity (u=1), while the three remaining walls are stationary. This drives an asymmetric circulating vortex inside the cavity.

The governing equations are the incompressible Navier-Stokes equations:

**x-dir momentum conservation:**
$$u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \nu \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = 0$$

**y-dir  momentum conservation:**
$$u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + \frac{\partial p}{\partial y} - \nu \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right) = 0$$

**Mass conservation:**
$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

where `u`, `v` are the velocity components, `p` is pressure, and `ν = 0.01` is the kinematic viscosity (Re = 100).

---

## The PINN Network

A fully-connected neural network takes a spatial coordinate `(x, y)` as input and outputs the three flow fields simultaneously:

```
(x, y)  →  [Linear → Tanh] × 4  →  Linear  →  (u, v, p)
```

Architecture: `[2, 64, 64, 64, 64, 3]` (12,867 parameters) — two input neurons, four hidden layers of 64 neurons with Tanh activations, three output neurons. Tanh is chosen over ReLU because the NS equations involve second-order derivatives, which vanish for ReLU activations.

The network is a continuous function approximator — it represents the flow field at every point in the domain, not just on a fixed mesh.

---

## Boundary Conditions

The boundary condition loss penalizes the network for violating the no-slip and lid conditions on all four walls:

| Wall | Condition |
|------|-----------|
| Bottom (y=0) | u=0, v=0 |
| Top (y=1) — the lid | u=1, v=0 |
| Left (x=0) | u=0, v=0 |
| Right (x=1) | u=0, v=0 |

`N_b = 1,000` points are sampled uniformly along the four walls. The BC loss is:

$$\mathcal{L}_{BC} = \frac{1}{N_b} \sum_i \left[ (u_{pred} - u_{BC})^2 + (v_{pred} - v_{BC})^2 \right]$$

---

## Loss Functions

The total training loss combines three terms:

$$\mathcal{L} = 10 \cdot \mathcal{L}_{BC} + \mathcal{L}_{PDE} + 10 \cdot \mathcal{L}_{p}$$

**BC loss** `L_BC` — enforces boundary conditions on the walls (above).

**PDE loss** `L_PDE` — enforces the Navier-Stokes equations at collocation points in the interior (see below).

**Pressure gauge fix** `L_p` — the incompressible NS equations only determine pressure up to an additive constant (if `p(x,y)` is a solution, so is `p(x,y) + k`). We pin `p(0.5, 0.5) = 0`:

$$\mathcal{L}_{p} = p(0.5, 0.5)^2$$

The factor of 10 on `L_BC` and `L_p` weights them more heavily than the PDE residual, ensuring the network first satisfies the boundary conditions before fitting the interior.

---

## L_PDE and Collocation Points

`N_f = 10,000` collocation points are sampled randomly in the interior of [0,1]². These points carry **no labels** — there is no ground truth velocity or pressure to compare against. Instead, the network is penalised for failing to satisfy the PDE at each point:

$$\mathcal{L}_{PDE} = \frac{1}{N_f} \sum_i \left( r_x^2 + r_y^2 + r_c^2 \right)$$

where `r_x`, `r_y`, `r_c` are the residuals of the x-momentum, y-momentum, and mass conservation equations evaluated at each collocation point. A perfect solution to the NS equations would give `L_PDE = 0`.

### Evaluation L_PDE

After each training step, `L_PDE` is also evaluated on a fresh set of `N_eval = 2,000` randomly sampled points (independent of the training collocation points). This gives an unbiased estimate of how well the network satisfies the PDE across the domain — analogous to a validation loss.

---

## How Autograd Works on Collocation Points

Computing the NS residual requires spatial derivatives of the network outputs — `∂u/∂x`, `∂²u/∂x²`, `∂p/∂x`, etc. These are obtained via PyTorch's automatic differentiation, not finite differences.

The collocation points are created with `requires_grad=True`:

```python
x_f = torch.rand(N_f, 1, requires_grad=True)
y_f = torch.rand(N_f, 1, requires_grad=True)
```

This tells PyTorch to track all operations involving `x_f` and `y_f` in a computation graph. When the network computes `u = net(x_f, y_f)`, the graph records the dependency of `u` on `x_f`. First and second derivatives are then computed exactly:

```python
u_x  = grad(u.sum(), x_f, create_graph=True)[0]   # ∂u/∂x
u_xx = grad(u_x.sum(), x_f, create_graph=True)[0]  # ∂²u/∂x²
```

`create_graph=True` is required for second derivatives; it tells PyTorch to keep the graph of `u_x` so that it can be differentiated again to get `u_xx`.

During the training backward pass, gradients flow all the way through the derivative computation back to the network weights, so the optimizer can minimize the PDE residual by adjusting the network parameters.

---

## Inference

Once trained, the network is a meshfree surrogate for the flow field. Any point `(x, y) ∈ [0,1]²` can be queried instantly:

```python
with torch.no_grad():
    u, v, p = net(torch.tensor([[0.5]]), torch.tensor([[0.8]]))
```

`torch.no_grad()` is used here because we only want the network's predictions — no derivatives w.r.t. the weights are needed.

To evaluate `L_PDE` at inference time (e.g. on a large point cloud for a high-fidelity accuracy estimate), `torch.no_grad()` cannot be used because the NS residual still requires autograd through the input coordinates. The network weights are not updated; no `.backward()` or `.step()` is called. But the input coordinates must remain in the computation graph.

---

## File Structure

```
pinn.py          # PINN model, NS residual, data helpers
notebook.ipynb   # Training driver, visualisation
pyproject.toml   # Dependencies
uv.lock          # Pinned package versions
```

## Setup

```bash
uv sync
jupyter notebook
```

---

## Notes

**Compress the mp4 video file:**

```bash
ffmpeg -y -i images/Re=100.mp4  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2"  -c:v libx264 -crf 32 -preset slow -pix_fmt yuv420p  images/Re=100_compressed.mp4
```

**Convert to GIF (plays once, no loop):**

```bash
ffmpeg -y -i images/Re=100_compressed.mp4  -vf "fps=6,scale=360:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"  -loop -1  images/Re=100.gif
```
