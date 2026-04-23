import torch
import torch.nn as nn
from torch.autograd import grad


class PINN(nn.Module):
    def __init__(self, layers=[2, 64, 64, 64, 64, 3]):
        super().__init__()
        seq = []
        for i in range(len(layers) - 1):
            seq += [nn.Linear(layers[i], layers[i + 1])]
            if i < len(layers) - 2:
                seq += [nn.Tanh()]
        self.net = nn.Sequential(*seq)

    def forward(self, x, y):
        out = self.net(torch.cat([x, y], dim=1))
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]  # u, v, p


def ns_residual(net, x, y, nu):
    u, v, p = net(x, y)
    u_x  = grad(u.sum(), x, create_graph=True)[0]
    u_y  = grad(u.sum(), y, create_graph=True)[0]
    u_xx = grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = grad(u_y.sum(), y, create_graph=True)[0]
    v_x  = grad(v.sum(), x, create_graph=True)[0]
    v_y  = grad(v.sum(), y, create_graph=True)[0]
    v_xx = grad(v_x.sum(), x, create_graph=True)[0]
    v_yy = grad(v_y.sum(), y, create_graph=True)[0]
    p_x  = grad(p.sum(), x, create_graph=True)[0]
    p_y  = grad(p.sum(), y, create_graph=True)[0]

    r_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    r_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    r_c = u_x + v_y

    return r_x, r_y, r_c


def make_boundary_data(N_b=200):
    t = torch.linspace(0, 1, N_b // 4).unsqueeze(1)
    zeros, ones = torch.zeros_like(t), torch.ones_like(t)
    x_bc = torch.cat([t, t, zeros, ones])
    y_bc = torch.cat([zeros, ones, t, t])
    u_bc = torch.cat([zeros, ones, zeros, zeros])  # lid (y=1): u=1
    v_bc = torch.zeros_like(x_bc)
    return x_bc, y_bc, u_bc, v_bc


def make_collocation_points(N_f=10_000):
    x_f = torch.rand(N_f, 1, requires_grad=True)
    y_f = torch.rand(N_f, 1, requires_grad=True)
    return x_f, y_f


def eval_all_losses(net, nu, N_eval=2_000):
    # L_PDE: fresh interior collocation points
    x = torch.rand(N_eval, 1, requires_grad=True)
    y = torch.rand(N_eval, 1, requires_grad=True)
    r_x, r_y, r_c = ns_residual(net, x, y, nu)
    l_pde = (r_x**2 + r_y**2 + r_c**2).mean().item()

    # L_BC: fresh boundary points
    x_bc, y_bc, u_bc, v_bc = make_boundary_data(N_eval // 4)
    with torch.no_grad():
        u_p, v_p, _ = net(x_bc, y_bc)
    l_bc = ((u_p - u_bc)**2 + (v_p - v_bc)**2).mean().item()

    # L_p: fixed point — same as training
    with torch.no_grad():
        _, _, p_mid = net(torch.tensor([[0.5]]), torch.tensor([[0.5]]))
    l_p = p_mid.item() ** 2

    return l_pde, l_bc, l_p


def visualize(net, epoch, histories, N=64, step=4, s=10):
    """Plot loss curves, cross-sections, and flow field in one figure.

    histories: dict with keys
        epochs_train, train_pde_loss, train_bc_loss, train_p_loss,
        epochs_eval,  eval_pde_loss,  eval_bc_loss,  eval_p_loss
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from IPython.display import clear_output

    black_red = LinearSegmentedColormap.from_list("black_red", ["black", "red"])

    # Flow field
    xs, ys, U, V, P = eval_flow_field(net, N=N)
    # Cross-section at y = 0.5
    N_sec = 10_000
    x_sec = torch.linspace(0, 1, N_sec).unsqueeze(1)
    y_sec = torch.full((N_sec, 1), 0.5)
    with torch.no_grad():
        u_sec, v_sec, _ = net(x_sec, y_sec)
    x_sec = x_sec.squeeze().numpy()
    u_sec = u_sec.squeeze().numpy()
    v_sec = v_sec.squeeze().numpy()

    et = histories['epochs_train']
    ep = histories['epochs_eval']

    clear_output(wait=True)
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 6, hspace=0.45, wspace=0.35)
    fig.suptitle(f"Epoch {epoch}", fontsize=14)

    # Row 0: Loss curves
    for col, (key_tr, key_ev, title) in enumerate([
        ('train_pde_loss', 'eval_pde_loss', 'L_PDE'),
        ('train_bc_loss',  'eval_bc_loss',  'L_BC'),
        ('train_p_loss',   'eval_p_loss',   'L_p'),
    ]):
        ax = fig.add_subplot(gs[0, col*2:(col+1)*2])
        if et:
            ax.semilogy(et[::s], histories[key_tr][::s], c='k', alpha=0.6, label='train')
        if ep:
            ax.semilogy(ep[::s], histories[key_ev][::s], c='r', alpha=0.6, label='eval')
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.margins(x=0)
        ax.legend(fontsize=8)

    # Row 1: Cross-sections
    ax_u = fig.add_subplot(gs[1, 1:3])
    ax_u.plot(x_sec, u_sec, c='k', alpha=0.6)
    ax_u.set_title("u(x, y=0.5) — cross-sectional slice")
    ax_u.set_xlabel("x")
    ax_u.set_ylabel("u")
    ax_u.margins(x=0)

    ax_v = fig.add_subplot(gs[1, 4:6])
    ax_v.plot(x_sec, v_sec, c='k', alpha=0.6)
    ax_v.set_title("v(x, y=0.5) — cross-sectional slice")
    ax_v.set_xlabel("x")
    ax_v.set_ylabel("v")
    ax_v.margins(x=0)

    # Vorticity via autograd: ω = dv/dx - du/dy
    x_flat = torch.tensor(xs.repeat(N).reshape(N*N, 1), dtype=torch.float32, requires_grad=True)
    y_flat = torch.tensor(np.tile(ys, N).reshape(N*N, 1), dtype=torch.float32, requires_grad=True)
    u_out, v_out, _ = net(x_flat, y_flat)
    du_dy = grad(u_out.sum(), y_flat, retain_graph=True, create_graph=False)[0]
    dv_dx = grad(v_out.sum(), x_flat, create_graph=False)[0]
    omega = (dv_dx - du_dy).reshape(N, N).detach().numpy()

    # Row 2: Quiver+vorticity | Pressure+∇p | Streamlines
    ax_q = fig.add_subplot(gs[2, 0:2])
    cf_w = ax_q.contourf(xs, ys, omega.T, levels=50, cmap="RdBu_r", alpha=0.7)
    plt.colorbar(cf_w, ax=ax_q, label="ω")
    ax_q.quiver(
        xs[::step], ys[::step],
        U[::step, ::step].T, V[::step, ::step].T,
        color='k', alpha=0.6,
    )
    ax_q.set_title("Vorticity ω(x,y) + velocity vectors")
    ax_q.set_xlabel("x")
    ax_q.set_ylabel("y")
    ax_q.set_xlim(0, 1)
    ax_q.set_ylim(0, 1)
    ax_q.set_aspect("equal")

    dP_dx, dP_dy = np.gradient(P, xs, ys)
    mag = np.sqrt(dP_dx**2 + dP_dy**2)

    ax_p = fig.add_subplot(gs[2, 2:4])
    cf = ax_p.contourf(xs, ys, P.T, levels=50, cmap="RdBu_r")
    plt.colorbar(cf, ax=ax_p, label="p")
    ax_p.quiver(
        xs[::step], ys[::step],
        -dP_dx[::step, ::step].T, -dP_dy[::step, ::step].T,
        mag[::step, ::step].T.flatten(),
        cmap=black_red, alpha=0.7,
    )
    ax_p.set_title("Pressure + (−∇p)")
    ax_p.set_xlabel("x")
    ax_p.set_ylabel("y")
    ax_p.set_xlim(0, 1)
    ax_p.set_ylim(0, 1)
    ax_p.set_aspect("equal")

    ax_s = fig.add_subplot(gs[2, 4:6])
    ax_s.streamplot(xs, ys, U.T, V.T, color='k', density=2, linewidth=0.8, arrowsize=0)
    ax_s.set_title("Streamlines")
    ax_s.set_xlabel("x")
    ax_s.set_ylabel("y")
    ax_s.set_xlim(0, 1)
    ax_s.set_ylim(0, 1)
    ax_s.set_aspect("equal")

    plt.tight_layout()
    plt.show()


def eval_flow_field(net, N=64):
    """Evaluate u, v, p on an N×N grid over [0,1]².
    Returns xs, ys (1-D numpy), U, V, P (2-D numpy, shape N×N).
    """
    xs = torch.linspace(0, 1, N)
    ys = torch.linspace(0, 1, N)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    with torch.no_grad():
        U, V, P = net(X.reshape(-1, 1), Y.reshape(-1, 1))
    U = U.reshape(N, N).numpy()
    V = V.reshape(N, N).numpy()
    P = P.reshape(N, N).numpy()
    return xs.numpy(), ys.numpy(), U, V, P
