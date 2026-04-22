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


def eval_pde_loss(net, nu, N_eval=2_000):
    x = torch.rand(N_eval, 1, requires_grad=True)
    y = torch.rand(N_eval, 1, requires_grad=True)
    r_x, r_y, r_c = ns_residual(net, x, y, nu)
    return (r_x**2 + r_y**2 + r_c**2).mean().item()
