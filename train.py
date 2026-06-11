"""
train.py — standalone training script for PINN lid-driven cavity
Usage:
    python train.py --lid uniform --network 1x
    python train.py --lid sigmoid --network 2x
"""
import argparse
import os
import time

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script use
import matplotlib.pyplot as plt
import torch

from pinn import (
    PINN, SIREN, ns_residual, make_boundary_data, make_collocation_points,
    eval_all_losses, visualize, plot_flow_field,
)

# ── CLI arguments ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--lid',        choices=['uniform', 'sigmoid'], required=True)
parser.add_argument('--network',    choices=['1x', '2x'],           required=True)
parser.add_argument('--epochs',     type=int,   default=10_000)
parser.add_argument('--activation', choices=['tanh', 'siren'],      default='tanh')
parser.add_argument('--width',      type=int,   default=None)
parser.add_argument('--device',     type=str,   default=None,
                    help='Force device: cpu, mps, cuda. Default: auto-detect.')
args = parser.parse_args()

# ── Device ─────────────────────────────────────────────────────────────────
if args.device:
    DEVICE = torch.device(args.device)
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f"Device     : {DEVICE}")

# ── Hyperparameters ────────────────────────────────────────────────────────
N_b        = 1_000
N_f        = 10_000
N_eval_pde = 2_000
EVAL_EVERY = 500
nu         = 0.01
num_epochs = args.epochs
log_every  = 100

smooth_lid  = (args.lid == 'sigmoid')
network_tag = args.network
lid_tag     = 'sigmoidU' if smooth_lid else 'uniformU'
ACTIVATION  = args.activation

if args.width:
    default_w = args.width
else:
    default_w = 128 if network_tag == '2x' else 64
layers     = [2, default_w, default_w, default_w, default_w, 3]
lr_initial = 1e-3 * (64 / default_w)
lr_min     = lr_initial * 1e-3

Re        = round(1 / nu)
run_name  = f"Re{Re}_{lid_tag}_{network_tag}hidden"
act_label = f"sin (SIREN), W={default_w}" if ACTIVATION == 'siren' else f"tanh, W={default_w}"
run_label = f"Re={Re}, {lid_tag}, {act_label}"
img_dir   = f"images/flow_{run_name}"
viz_dir   = f"images/dashboard_{run_name}"

os.makedirs(img_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)

ghia_u   = {100: 0.73722, 400: 0.61756, 1000: 0.51117, 3200: 0.46547,
            5000: 0.45992, 7500: 0.47323, 10000: 0.48070}
ghia_ref = ghia_u.get(Re, float('nan'))

total_params = sum(p.numel() for p in PINN(layers).parameters())
print(f"Run        : {run_label}")
print(f"Activation : {ACTIVATION}")
print(f"Epochs     : {num_epochs}")
print(f"Parameters : {total_params:,}")
print(f"Flow dir   : {img_dir}")
print(f"Dash dir   : {viz_dir}")
print()

# ── State ──────────────────────────────────────────────────────────────────
model = (SIREN(layers) if ACTIVATION == 'siren' else PINN(layers)).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=lr_initial)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='min', factor=0.3, patience=3_000 // EVAL_EVERY
)

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
    l_pde, l_bc, l_p = eval_all_losses(model, nu, N_eval=N_eval_pde, smooth_lid=smooth_lid)
    epochs_eval.append(epoch)
    eval_pde_loss.append(l_pde)
    eval_bc_loss.append(l_bc)
    eval_p_loss.append(l_p)

    fig, _ = plot_flow_field(model, epoch, nu, run_label=run_label)
    fig.savefig(f"{img_dir}/epoch={epoch:06d}.png", dpi=80, bbox_inches='tight')
    plt.close(fig)

    if epoch % 100 == 0:
        fig2 = visualize(model, epoch, histories, nu, show=False, run_label=run_label, num_epochs=num_epochs)
        fig2.savefig(f"{viz_dir}/epoch={epoch:06d}.png", dpi=60, bbox_inches='tight')
        plt.close(fig2)


def log_stats(epoch, loss_bc, loss_pde, loss_p):
    print(f"\nepoch {epoch:>6d} | L_bc {loss_bc.item():.3e} | L_pde {loss_pde.item():.3e} | "
          f"L_p {loss_p.item():.3e} | eval L_pde {eval_pde_loss[-1]:.3e}")
    with torch.no_grad():
        u_pred, _, _ = model(torch.tensor([[0.5]], device=DEVICE), torch.tensor([[0.9609]], device=DEVICE))
        t_w = torch.linspace(0, 1, 1_000, device=DEVICE).unsqueeze(1)
        z_w, o_w = torch.zeros_like(t_w), torch.ones_like(t_w)
        u_b, v_b, _ = model(t_w, z_w)
        u_l, v_l, _ = model(z_w, t_w)
        u_r, v_r, _ = model(o_w, t_w)
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
    x_bc, y_bc = x_bc.to(DEVICE), y_bc.to(DEVICE)
    u_bc, v_bc = u_bc.to(DEVICE), v_bc.to(DEVICE)
    x_f, y_f = make_collocation_points(N_f)
    x_f, y_f = x_f.to(DEVICE), y_f.to(DEVICE)

    u_p, v_p, _ = model(x_bc, y_bc)
    loss_bc = ((u_p - u_bc)**2 + (v_p - v_bc)**2).mean()

    r_x, r_y, r_c = ns_residual(model, x_f, y_f, nu)
    loss_pde = (r_x**2 + r_y**2 + r_c**2).mean()

    _, _, p_mid = model(torch.tensor([[0.5]], device=DEVICE), torch.tensor([[0.5]], device=DEVICE))
    loss_p = p_mid**2

    loss = 10 * loss_bc + loss_pde + 10 * loss_p
    loss.backward()
    opt.step()

    epochs_train.append(epoch)
    train_pde_loss.append(loss_pde.item())
    train_bc_loss.append(loss_bc.item())
    train_p_loss.append(loss_p.item())
    current_lr = opt.param_groups[0]['lr']
    lr_history.append(current_lr)

    if epoch % EVAL_EVERY == 0:
        l_pde_eval, _, _ = eval_all_losses(model, nu, N_eval=N_eval_pde, smooth_lid=smooth_lid)
        scheduler.step(l_pde_eval)
        current_lr = opt.param_groups[0]['lr']
        elapsed = (time.perf_counter() - t_start) / 60
        print(f"\nepoch {epoch:>6d} | eval_pde {l_pde_eval:.3e} | lr {current_lr:.2e} | {elapsed:.1f} min",
              flush=True)
        log_stats(epoch, loss_bc, loss_pde, loss_p)
        if current_lr < lr_min:
            print(f"  → converged (lr {current_lr:.2e} < {lr_min:.2e})", flush=True)
            save_frame(epoch)
            break

    if epoch % 10 == 0:
        save_frame(epoch)

elapsed = (time.perf_counter() - t_start) / 60
print(f"\nTraining complete — {run_label} — epoch:{epoch}  ({elapsed:.1f} min)")
