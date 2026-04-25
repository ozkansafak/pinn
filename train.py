"""
train.py — standalone training script for PINN lid-driven cavity
Usage:
    python train.py --lid uniform --network 1x
    python train.py --lid sigmoid --network 2x
"""
import argparse
import copy
import os
import time

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script use
import matplotlib.pyplot as plt
import torch

from pinn import (
    PINN, ns_residual, make_boundary_data, make_collocation_points,
    eval_all_losses, visualize, plot_flow_field,
)

# ── CLI arguments ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--lid',     choices=['uniform', 'sigmoid'], required=True)
parser.add_argument('--network', choices=['1x', '2x'],           required=True)
parser.add_argument('--epochs',  type=int, default=10_000)
args = parser.parse_args()

# ── Hyperparameters ────────────────────────────────────────────────────────
N_b       = 1_000
N_f       = 10_000
N_eval_pde = 2_000
nu        = 0.01
lr        = 1e-3
num_epochs = args.epochs
log_every  = 100

smooth_lid  = (args.lid == 'sigmoid')
network_tag = args.network
lid_tag     = 'sigmoidU' if smooth_lid else 'uniformU'
layers      = [2, 128, 128, 128, 128, 3] if network_tag == '2x' else [2, 64, 64, 64, 64, 3]

Re        = round(1 / nu)
run_name  = f"Re{Re}_{lid_tag}_{network_tag}hidden"
run_label = f"Re={Re}, {lid_tag}, {network_tag}hidden"
img_dir   = f"images/flow_{run_name}"
viz_dir   = f"images/dashboard_{run_name}"

os.makedirs(img_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)

ghia_u   = {100: 0.73722, 400: 0.61756, 1000: 0.51117, 3200: 0.46547,
            5000: 0.45992, 7500: 0.47323, 10000: 0.48070}
ghia_ref = ghia_u.get(Re, float('nan'))

total_params = sum(p.numel() for p in PINN(layers).parameters())
print(f"Run        : {run_label}")
print(f"Epochs     : {num_epochs}")
print(f"Parameters : {total_params:,}")
print(f"Flow dir   : {img_dir}")
print(f"Dash dir   : {viz_dir}")
print()

# ── State ──────────────────────────────────────────────────────────────────
net = PINN(layers)
opt = torch.optim.Adam(net.parameters(), lr=lr)

epochs_train, train_pde_loss, train_bc_loss, train_p_loss = [], [], [], []
epochs_eval,  eval_pde_loss,  eval_bc_loss,  eval_p_loss  = [], [], [], []
lr_history = []

histories = {
    'epochs_train':   epochs_train,
    'train_pde_loss': train_pde_loss,
    'train_bc_loss':  train_bc_loss,
    'train_p_loss':   train_p_loss,
    'epochs_eval':    epochs_eval,
    'eval_pde_loss':  eval_pde_loss,
    'eval_bc_loss':   eval_bc_loss,
    'eval_p_loss':    eval_p_loss,
    'lr_history':     lr_history,
}


# ── Helpers ────────────────────────────────────────────────────────────────
def save_frame(epoch):
    l_pde, l_bc, l_p = eval_all_losses(net, nu, N_eval=N_eval_pde, smooth_lid=smooth_lid)
    epochs_eval.append(epoch)
    eval_pde_loss.append(l_pde)
    eval_bc_loss.append(l_bc)
    eval_p_loss.append(l_p)

    fig, _ = plot_flow_field(net, epoch, nu, run_label=run_label)
    fig.savefig(f"{img_dir}/epoch={epoch:06d}.png", dpi=80, bbox_inches='tight')
    plt.close(fig)

    if epoch % 100 == 0:
        fig2 = visualize(net, epoch, histories, nu, show=False, run_label=run_label, num_epochs=num_epochs)
        fig2.savefig(f"{viz_dir}/epoch={epoch:06d}.png", dpi=60, bbox_inches='tight')
        plt.close(fig2)


def log_stats(epoch, loss_bc, loss_pde, loss_p):
    print(f"\nepoch {epoch:>6d} | L_bc {loss_bc.item():.3e} | L_pde {loss_pde.item():.3e} | "
          f"L_p {loss_p.item():.3e} | eval L_pde {eval_pde_loss[-1]:.3e}")
    with torch.no_grad():
        u_pred, v_pred, p_pred = net(torch.tensor([[0.5]]), torch.tensor([[0.9609]]))
        t_w = torch.linspace(0, 1, 1_000).unsqueeze(1)
        z_w, o_w = torch.zeros_like(t_w), torch.ones_like(t_w)
        u_b, v_b, _ = net(t_w, z_w)
        u_l, v_l, _ = net(z_w, t_w)
        u_r, v_r, _ = net(o_w, t_w)
    print(f"  u(0.5, 0.9609) = {u_pred.item():.4f}   (expect {ghia_ref:.5f} — Ghia et al. Re={Re})")
    for name, u, v in [('bottom (y=0)', u_b, v_b), ('left   (x=0)', u_l, v_l), ('right  (x=1)', u_r, v_r)]:
        print(f"  {name} : u mean={u.mean():.4f}  std={u.std():.4f}   "
              f"v mean={v.mean():.4f}  std={v.std():.4f}")
    print(flush=True)


# ── Training loop ──────────────────────────────────────────────────────────
save_frame(0)

t_start = time.perf_counter()
epoch = 0

while epoch < num_epochs:
    epoch += 1

    opt.zero_grad()
    x_bc, y_bc, u_bc, v_bc = make_boundary_data(N_b, smooth_lid=smooth_lid)
    x_f, y_f = make_collocation_points(N_f)

    u_p, v_p, _ = net(x_bc, y_bc)
    loss_bc = ((u_p - u_bc)**2 + (v_p - v_bc)**2).mean()

    r_x, r_y, r_c = ns_residual(net, x_f, y_f, nu)
    loss_pde = (r_x**2 + r_y**2 + r_c**2).mean()

    _, _, p_mid = net(torch.tensor([[0.5]]), torch.tensor([[0.5]]))
    loss_p = p_mid**2

    loss = 10 * loss_bc + loss_pde + 10 * loss_p
    loss.backward()
    opt.step()

    epochs_train.append(epoch)
    train_pde_loss.append(loss_pde.item())
    train_bc_loss.append(loss_bc.item())
    train_p_loss.append(loss_p.item())
    lr_history.append(opt.param_groups[0]['lr'])

    if epoch in (15_000, 25_000, 35_000):
        for g in opt.param_groups:
            g['lr'] /= 10
        print(f"\n*** lr dropped to {opt.param_groups[0]['lr']:.2e} at epoch {epoch} ***", flush=True)

    if epoch % 10 == 0:
        save_frame(epoch)

    if epoch % 1000 == 0:
        elapsed = (time.perf_counter() - t_start) / 60
        print(f"epoch:{epoch}  ({elapsed:.1f} min)", flush=True)
        log_stats(epoch, loss_bc, loss_pde, loss_p)
    elif epoch % 100 == 0:
        print(epoch, end=' ', flush=True)
    elif epoch % 10 == 0:
        print('.', end='', flush=True)

elapsed = (time.perf_counter() - t_start) / 60
print(f"\nTraining complete — {run_label} — epoch:{epoch}  ({elapsed:.1f} min)")
