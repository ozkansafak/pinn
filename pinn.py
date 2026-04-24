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
    t = torch.rand(N_b // 4, 1)
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


def _plot_streamfunction(ax, xs, ys, U, V, n_levels=30):
    """Plot streamfunction isolines with one directional arrow per line."""
    import numpy as np
    dy = ys[1] - ys[0]
    psi = np.cumsum(U * dy, axis=1)  # ψ(x,y) = ∫₀ʸ u dy,  shape (N, N)
    cs = ax.contour(xs, ys, psi.T, levels=n_levels, colors='k', linewidths=0.7, alpha=0.8, linestyles='solid')

    # Place one arrow per contour segment, direction corrected against (U, V)
    for segs in cs.allsegs:
        for seg in segs:
            if len(seg) < 6:
                continue
            mid = len(seg) // 2
            x0, y0 = seg[mid]
            x1, y1 = seg[min(mid + 3, len(seg) - 1)]
            ix = int(np.clip(np.searchsorted(xs, x0) - 1, 0, len(xs) - 2))
            iy = int(np.clip(np.searchsorted(ys, y0) - 1, 0, len(ys) - 2))
            if (x1 - x0) * U[ix, iy] + (y1 - y0) * V[ix, iy] < 0:
                x0, y0, x1, y1 = x1, y1, x0, y0
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color='k',
                                        lw=0.8, mutation_scale=10))


def visualize(net, epoch, histories, nu, N=64, N_res=32, step=4, s=10):
    """Plot flow field, residuals, cross-sections, and loss curves in one figure.

    histories: dict with keys
        epochs_train, train_pde_loss, train_bc_loss, train_p_loss,
        epochs_eval,  eval_pde_loss,  eval_bc_loss,  eval_p_loss
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap


    black_red = LinearSegmentedColormap.from_list("black_red", ["black", "red"])
    white_red = LinearSegmentedColormap.from_list("white_red", ["white", "red"])
    fs = 16  # base font size

    # ── Flow field on N×N grid ──────────────────────────────────────────────
    xs, ys, U, V, P = eval_flow_field(net, N=N)

    # Vorticity via autograd
    x_flat = torch.tensor(xs.repeat(N).reshape(N*N, 1), dtype=torch.float32, requires_grad=True)
    y_flat = torch.tensor(np.tile(ys, N).reshape(N*N, 1), dtype=torch.float32, requires_grad=True)
    u_out, v_out, _ = net(x_flat, y_flat)
    du_dy = grad(u_out.sum(), y_flat, retain_graph=True, create_graph=False)[0]
    dv_dx = grad(v_out.sum(), x_flat, create_graph=False)[0]
    omega = (dv_dx - du_dy).reshape(N, N).detach().numpy()

    dP_dx, dP_dy = np.gradient(P, xs, ys)
    mag = np.sqrt(dP_dx**2 + dP_dy**2)
    mag_clipped = np.clip(mag, 0, np.percentile(mag, 95))

    # ── PDE residual on N_res×N_res interior grid ──────────────────────────
    xs_r = torch.linspace(0.02, 0.98, N_res)
    ys_r = torch.linspace(0.02, 0.98, N_res)
    Xr, Yr = torch.meshgrid(xs_r, ys_r, indexing='ij')
    x_r = Xr.reshape(-1, 1).requires_grad_(True)
    y_r = Yr.reshape(-1, 1).requires_grad_(True)
    r_x, r_y, r_c = ns_residual(net, x_r, y_r, nu)
    pde_res = (r_x**2 + r_y**2 + r_c**2).reshape(N_res, N_res).detach().numpy()

    # ── BC residual on all four walls ──────────────────────────────────────
    N_wall = 200
    t = torch.linspace(0, 1, N_wall).unsqueeze(1)
    z, o = torch.zeros_like(t), torch.ones_like(t)
    walls = [
        (t, z, z, z, "bottom (y=0)"),
        (t, o, o, z, "top lid (y=1)"),
        (z, t, z, z, "left (x=0)"),
        (o, t, z, z, "right (x=1)"),
    ]

    # ── Cross-section at y=0.5 ─────────────────────────────────────────────
    N_sec = 5_000
    x_sec = torch.linspace(0, 1, N_sec).unsqueeze(1)
    y_sec = torch.full((N_sec, 1), 0.5)
    with torch.no_grad():
        u_sec, v_sec, _ = net(x_sec, y_sec)
    x_sec = x_sec.squeeze().numpy()
    u_sec = u_sec.squeeze().numpy()
    v_sec = v_sec.squeeze().numpy()

    et = histories['epochs_train']
    ep = histories['epochs_eval']

    # ── Figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 22), layout='constrained')
    fig.get_layout_engine().set(rect=[0, 0, 1, 0.99])
    gs = fig.add_gridspec(4, 12, hspace=0.5, wspace=0.35)
    fig.suptitle(f"Epoch {epoch}    Re = {round(1/nu)}", fontsize=fs+3)

    # ── Row 0: Vorticity+quiver+streamlines | Pressure+∇p ─────────────────
    ax_q = fig.add_subplot(gs[0, 0:6])
    cf_w = ax_q.contourf(xs, ys, omega.T, levels=50, cmap="RdBu_r", alpha=0.7)
    plt.colorbar(cf_w, ax=ax_q, label="ω").ax.tick_params(labelsize=fs-1)
    _plot_streamfunction(ax_q, xs, ys, U, V)
    ax_q.set_title("Vorticity ω + streamlines", fontsize=fs)
    ax_q.set_xlabel("x", fontsize=fs); ax_q.set_ylabel("y", fontsize=fs)
    ax_q.tick_params(labelsize=fs-1)
    ax_q.set_xlim(0, 1); ax_q.set_ylim(0, 1); ax_q.set_aspect("equal"); ax_q.set_anchor('C')

    ax_p = fig.add_subplot(gs[0, 6:12])
    cf = ax_p.contourf(xs, ys, P.T, levels=50, cmap="RdBu_r")
    plt.colorbar(cf, ax=ax_p, label="p").ax.tick_params(labelsize=fs-1)
    with np.errstate(invalid='ignore'):
        ax_p.quiver(xs[::step], ys[::step],
                    -dP_dx[::step, ::step].T, -dP_dy[::step, ::step].T,
                    mag_clipped[::step, ::step].T.flatten(), cmap=white_red, alpha=0.7)
    ax_p.set_title("Pressure + (−∇p)", fontsize=fs)
    ax_p.set_xlabel("x", fontsize=fs); ax_p.set_ylabel("y", fontsize=fs)
    ax_p.tick_params(labelsize=fs-1)
    ax_p.set_xlim(0, 1); ax_p.set_ylim(0, 1); ax_p.set_aspect("equal"); ax_p.set_anchor('C')

    # ── Row 1: PDE residual colorplot | BC residual scatter ────────────────
    ax_pde_r = fig.add_subplot(gs[1, 0:6])
    cf_r = ax_pde_r.contourf(xs_r.numpy(), ys_r.numpy(), pde_res.T, levels=50, cmap=white_red)
    plt.colorbar(cf_r, ax=ax_pde_r, label="r²").ax.tick_params(labelsize=fs-1)
    ax_pde_r.set_title("L_PDE residual  r²(x,y) = rₓ² + r_y² + r_c²", fontsize=fs)
    ax_pde_r.set_xlabel("x", fontsize=fs); ax_pde_r.set_ylabel("y", fontsize=fs)
    ax_pde_r.tick_params(labelsize=fs-1)
    ax_pde_r.set_xlim(0, 1); ax_pde_r.set_ylim(0, 1); ax_pde_r.set_aspect("equal"); ax_pde_r.set_anchor('C')

    ax_bc_r = fig.add_subplot(gs[1, 6:12])
    colors_wall = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']
    with torch.no_grad():
        for (xw, yw, u_bc, v_bc, label), col in zip(walls, colors_wall):
            u_p, v_p, _ = net(xw, yw)
            bc_err = ((u_p - u_bc)**2 + (v_p - v_bc)**2).squeeze().numpy()
            xw_np = xw.squeeze().numpy()
            yw_np = yw.squeeze().numpy()
            sc = ax_bc_r.scatter(xw_np, yw_np, c=bc_err, cmap=white_red, s=6,
                                 vmin=0, label=label)
    plt.colorbar(sc, ax=ax_bc_r, label="(u_err² + v_err²)").ax.tick_params(labelsize=fs-1)
    ax_bc_r.set_title("L_BC residual on boundary", fontsize=fs)
    ax_bc_r.set_xlabel("x", fontsize=fs); ax_bc_r.set_ylabel("y", fontsize=fs)
    ax_bc_r.tick_params(labelsize=fs-1)
    ax_bc_r.set_xlim(-0.05, 1.05); ax_bc_r.set_ylim(-0.05, 1.05); ax_bc_r.set_aspect("equal"); ax_bc_r.set_anchor('C')

    # ── Row 2: Cross-sections ──────────────────────────────────────────────
    ax_u = fig.add_subplot(gs[2, 0:5])
    ax_u.plot(x_sec, u_sec, c='k', alpha=0.6)
    ax_u.set_title("u(x, y=0.5) — cross-sectional slice", fontsize=fs)
    ax_u.set_xlabel("x", fontsize=fs); ax_u.set_ylabel("u", fontsize=fs)
    ax_u.tick_params(labelsize=fs-1)
    ax_u.margins(x=0)

    ax_v = fig.add_subplot(gs[2, 7:12])
    ax_v.plot(x_sec, v_sec, c='k', alpha=0.6)
    ax_v.set_title("v(x, y=0.5) — cross-sectional slice", fontsize=fs)
    ax_v.set_xlabel("x", fontsize=fs); ax_v.set_ylabel("v", fontsize=fs)
    ax_v.tick_params(labelsize=fs-1)
    ax_v.margins(x=0)

    # ── Row 3: Loss curves ─────────────────────────────────────────────────
    for col, (key_tr, key_ev, title) in enumerate([
        ('train_pde_loss', 'eval_pde_loss', 'L_PDE  (NS residual)'),
        ('train_bc_loss',  'eval_bc_loss',  'L_BC  (boundary condition)'),
        ('train_p_loss',   'eval_p_loss',   'L_p  (pressure gauge fix)'),
    ]):
        ax = fig.add_subplot(gs[3, col*4:(col+1)*4])
        if et:
            ax.semilogy(et[::s], histories[key_tr][::s], c='k', alpha=0.6, label='train')
        if ep:
            ax.semilogy(ep[::s], histories[key_ev][::s], c='r', alpha=0.6, label='eval')
        ax.set_title(title, fontsize=fs)
        ax.set_xlabel("Epoch", fontsize=fs)
        ax.tick_params(labelsize=fs-1)
        ax.margins(x=0)
        if key_tr != 'train_bc_loss':
            ax.legend(fontsize=fs-1)

    plt.show()


def plot_flow_field(net, epoch, nu, axes=None, N=64, step=4):
    """Render vorticity+quiver | pressure+(-∇p) | streamlines into `axes`.

    If axes is None a new (1×3) figure is created and returned.
    axes must be a sequence of three matplotlib Axes when provided.
    Returns (fig, axes).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from torch.autograd import grad

    black_red = LinearSegmentedColormap.from_list("black_red", ["black", "red"])

    xs, ys, U, V, P = eval_flow_field(net, N=N)

    # Vorticity via autograd
    x_flat = torch.tensor(xs.repeat(N).reshape(N * N, 1), dtype=torch.float32, requires_grad=True)
    y_flat = torch.tensor(np.tile(ys, N).reshape(N * N, 1), dtype=torch.float32, requires_grad=True)
    u_out, v_out, _ = net(x_flat, y_flat)
    du_dy = grad(u_out.sum(), y_flat, retain_graph=True, create_graph=False)[0]
    dv_dx = grad(v_out.sum(), x_flat, create_graph=False)[0]
    omega = (dv_dx - du_dy).reshape(N, N).detach().numpy()

    dP_dx, dP_dy = np.gradient(P, xs, ys)
    mag = np.sqrt(dP_dx ** 2 + dP_dy ** 2)
    mag_clipped = np.clip(mag, 0, np.percentile(mag, 95))

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout='constrained')
    else:
        fig = axes[0].get_figure()

    ax_q, ax_p, ax_s = axes

    for ax in axes:
        ax.cla()

    cf_w = ax_q.contourf(xs, ys, omega.T, levels=50, cmap="RdBu_r", alpha=0.7)
    plt.colorbar(cf_w, ax=ax_q, label="ω")
    with np.errstate(invalid='ignore'):
        ax_q.quiver(xs[::step], ys[::step], U[::step, ::step].T, V[::step, ::step].T, color='k', alpha=0.6)
    fig.suptitle(f"Re = {round(1/nu)}    Epoch {epoch}", fontsize=16)
    ax_q.set_title("Vorticity ω + velocity vectors")
    ax_q.set_xlabel("x"); ax_q.set_ylabel("y")
    ax_q.set_xlim(0, 1); ax_q.set_ylim(0, 1); ax_q.set_aspect("equal")

    cf = ax_p.contourf(xs, ys, P.T, levels=50, cmap="RdBu_r")
    plt.colorbar(cf, ax=ax_p, label="p")
    with np.errstate(invalid='ignore'):
        ax_p.quiver(xs[::step], ys[::step],
                    -dP_dx[::step, ::step].T, -dP_dy[::step, ::step].T,
                    mag_clipped[::step, ::step].T.flatten(), cmap=black_red, alpha=0.7)
    ax_p.set_title("Pressure + (−∇p)")
    ax_p.set_xlabel("x"); ax_p.set_ylabel("y")
    ax_p.set_xlim(0, 1); ax_p.set_ylim(0, 1); ax_p.set_aspect("equal")

    _plot_streamfunction(ax_s, xs, ys, U, V)
    ax_s.set_title("Streamlines (ψ isolines)")
    ax_s.set_xlabel("x"); ax_s.set_ylabel("y")
    ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1); ax_s.set_aspect("equal")

    return fig, axes


def make_animation(snapshots, layers, nu, N=64, step=4, fps=5):
    """Create a FuncAnimation from a list of (epoch, state_dict) snapshots.

    snapshots : list of (epoch, state_dict) collected during training
    layers    : PINN layer spec, e.g. [2, 64, 64, 64, 64, 3]
    Returns a matplotlib FuncAnimation — call .save() or display in notebook.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    net_anim = PINN(layers)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout='constrained')

    def update(frame):
        epoch, state_dict = snapshots[frame]
        net_anim.load_state_dict(state_dict)
        plot_flow_field(net_anim, epoch, nu, axes=axes, N=N, step=step)

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=int(1000 / fps))
    plt.close(fig)
    return ani


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
