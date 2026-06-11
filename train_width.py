"""
train_width.py — single width-sweep run for PINN lid-driven cavity

Usage:
    python train_width.py --width 64
    python train_width.py --width 128 --max-epochs 30000 --output results/width_sweep.csv
"""
import argparse
import csv
import os
import socket
import subprocess
import time
from datetime import datetime

import torch

from pinn import PINN, SIREN, ns_residual, make_boundary_data

# ── Device ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


def _make_collocation(n, device):
    x = torch.rand(n, 1, device=device).requires_grad_(True)
    y = torch.rand(n, 1, device=device).requires_grad_(True)
    return x, y


def _eval_losses(model, nu, n=2_000, device=DEVICE):
    x, y = _make_collocation(n, device)
    r_x, r_y, r_c = ns_residual(model, x, y, nu)
    l_pde = (r_x**2 + r_y**2 + r_c**2).mean().item()

    # L∞ norm on dense grid
    N_grid = 128
    xs = torch.linspace(0.01, 0.99, N_grid, device=device)
    ys = torch.linspace(0.01, 0.99, N_grid, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing='ij')
    x_g = Xg.reshape(-1, 1).requires_grad_(True)
    y_g = Yg.reshape(-1, 1).requires_grad_(True)
    r_x_g, r_y_g, r_c_g = ns_residual(model, x_g, y_g, nu)
    l_pde_max = (r_x_g**2 + r_y_g**2 + r_c_g**2).max().item()

    x_bc, y_bc, u_bc, v_bc = make_boundary_data(n // 4, smooth_lid=False)
    x_bc, y_bc = x_bc.to(device), y_bc.to(device)
    u_bc, v_bc = u_bc.to(device), v_bc.to(device)
    with torch.no_grad():
        u_p, v_p, _ = model(x_bc, y_bc)
    l_bc = ((u_p - u_bc)**2 + (v_p - v_bc)**2).mean().item()

    with torch.no_grad():
        t = torch.tensor([[0.5]], device=device)
        _, _, p_mid = model(t, t)
    l_p = p_mid.item() ** 2

    return l_pde, l_pde_max, l_bc, l_p

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--width',      type=int,   required=True)
parser.add_argument('--max-epochs', type=int,   default=60_000)
parser.add_argument('--output',     type=str,   default='results/width_sweep.csv')
parser.add_argument('--activation', choices=['tanh', 'siren'], default='siren')
args = parser.parse_args()

WIDTH      = args.width
MAX_EPOCHS = args.max_epochs
CSV_PATH   = args.output
ACTIVATION = args.activation

# ── Fixed hyperparameters ──────────────────────────────────────────────────────
N_b        = 1_000
N_f        = 10_000
N_eval     = 2_000
EVAL_EVERY = 500          # epochs between eval_L_pde checks (feeds scheduler + stop)
nu         = 0.01
LID        = 'uniform'
layers     = [2, WIDTH, WIDTH, WIDTH, WIDTH, 3]
lr_initial = 1e-3 * (64 / WIDTH)   # μP-inspired: constant effective update size across widths
lr_min     = lr_initial * 1e-3     # stop when LR drops this low

Re         = round(1 / nu)
ghia_ref   = 0.73722               # Ghia et al. (1982), Re=100, u at (0.5, 0.9609)

# ── Git SHA ────────────────────────────────────────────────────────────────────
try:
    git_sha = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    git_sha = 'unknown'

# ── Derive run_id from existing CSV rows ──────────────────────────────────────
def next_run_id(path):
    if not os.path.exists(path):
        return 1
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return (max(int(r['run_id']) for r in rows) + 1) if rows else 1

run_id       = next_run_id(CSV_PATH)
n_params     = sum(p.numel() for p in PINN(layers).parameters())
LOSS_CSV     = f"results/loss_curves/width_{WIDTH}_{ACTIVATION}.csv"
os.makedirs("results/loss_curves", exist_ok=True)
LOSS_FIELDS  = ["epoch", "train_pde", "train_bc", "eval_pde", "lr"]

print(f"run_id     : {run_id}")
print(f"device     : {DEVICE}")
print(f"activation : {ACTIVATION}")
print(f"width      : {WIDTH}  →  layers {layers}")
print(f"n_params   : {n_params:,}")
print(f"lr_initial : {lr_initial:.3e}   lr_min : {lr_min:.3e}")
print(f"max_epochs : {MAX_EPOCHS}")
print(f"git_sha    : {git_sha}")
print(flush=True)

# ── Model & optimiser ──────────────────────────────────────────────────────────
model = (SIREN(layers) if ACTIVATION == 'siren' else PINN(layers)).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=lr_initial)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='min', factor=0.3, patience=3_000 // EVAL_EVERY  # 3k-epoch stall window
)

# ── Training loop ──────────────────────────────────────────────────────────────
t_start      = time.perf_counter()
epoch        = 0
converged    = False
eval_L_pde   = float('nan')

# running loss accumulators (averaged over the EVAL_EVERY window)
acc_pde, acc_bc, acc_p = 0.0, 0.0, 0.0

while epoch < MAX_EPOCHS:
    epoch += 1

    opt.zero_grad()

    x_bc, y_bc, u_bc, v_bc = make_boundary_data(N_b, smooth_lid=False)
    x_bc, y_bc = x_bc.to(DEVICE), y_bc.to(DEVICE)
    u_bc, v_bc = u_bc.to(DEVICE), v_bc.to(DEVICE)
    x_f, y_f = _make_collocation(N_f, DEVICE)

    u_p, v_p, _ = model(x_bc, y_bc)
    loss_bc = ((u_p - u_bc)**2 + (v_p - v_bc)**2).mean()

    r_x, r_y, r_c = ns_residual(model, x_f, y_f, nu)
    loss_pde = (r_x**2 + r_y**2 + r_c**2).mean()

    t = torch.tensor([[0.5]], device=DEVICE)
    _, _, p_mid = model(t, t)
    loss_p = p_mid**2

    loss = 10 * loss_bc + loss_pde + 10 * loss_p
    loss.backward()
    opt.step()

    acc_pde += loss_pde.item()
    acc_bc  += loss_bc.item()
    acc_p   += loss_p.item()

    if epoch % EVAL_EVERY == 0:
        eval_L_pde, eval_L_pde_max, eval_L_bc, eval_L_p = _eval_losses(model, nu, n=N_eval, device=DEVICE)
        scheduler.step(eval_L_pde)
        current_lr = opt.param_groups[0]['lr']

        avg_pde = acc_pde / EVAL_EVERY
        avg_bc  = acc_bc  / EVAL_EVERY
        avg_p   = acc_p   / EVAL_EVERY
        acc_pde = acc_bc = acc_p = 0.0

        elapsed = (time.perf_counter() - t_start) / 60
        print(
            f"epoch {epoch:>6d} | train_pde {avg_pde:.3e} | train_bc {avg_bc:.3e} "
            f"| eval_pde {eval_L_pde:.3e} | lr {current_lr:.2e} | {elapsed:.1f} min",
            flush=True
        )

        write_header = not os.path.exists(LOSS_CSV) or os.path.getsize(LOSS_CSV) == 0
        with open(LOSS_CSV, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=LOSS_FIELDS)
            if write_header:
                w.writeheader()
            w.writerow({"epoch": epoch, "train_pde": f"{avg_pde:.6e}",
                        "train_bc": f"{avg_bc:.6e}", "eval_pde": f"{eval_L_pde:.6e}",
                        "lr": f"{current_lr:.6e}"})

        if current_lr < lr_min:
            converged = True
            print(f"  → converged (lr {current_lr:.2e} < {lr_min:.2e})", flush=True)
            break

# ── Final evaluation ───────────────────────────────────────────────────────────
final_L_pde, final_L_pde_max, final_L_bc, final_L_p = _eval_losses(model, nu, n=N_eval, device=DEVICE)
with torch.no_grad():
    u_ghia_pred, _, _ = model(
        torch.tensor([[0.5]], device=DEVICE),
        torch.tensor([[0.9609]], device=DEVICE),
    )
u_ghia = u_ghia_pred.item()

elapsed_min = (time.perf_counter() - t_start) / 60
lr_final    = opt.param_groups[0]['lr']

print(f"\n{'='*60}")
print(f"width={WIDTH}  epochs={epoch}  converged={converged}")
print(f"final eval_L_pde : {final_L_pde:.4e}")
print(f"final L_bc       : {final_L_bc:.4e}")
print(f"u_ghia           : {u_ghia:.5f}  (ref {ghia_ref:.5f},  err {abs(u_ghia - ghia_ref):.5f})")
print(f"elapsed          : {elapsed_min:.2f} min")
print(f"{'='*60}", flush=True)

# ── Append to CSV ──────────────────────────────────────────────────────────────
FIELDS = [
    'run_id', 'timestamp', 'activation', 'width', 'n_params', 'epochs_run', 'converged',
    'lr_initial', 'lr_final',
    'final_L_pde', 'final_L_pde_max', 'final_L_bc', 'final_L_p', 'eval_L_pde',
    'u_ghia', 'ghia_ref', 'elapsed_min',
    'lid', 'git_sha',
]

row = {
    'run_id':           run_id,
    'timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'activation':       ACTIVATION,
    'width':            WIDTH,
    'n_params':         n_params,
    'epochs_run':       epoch,
    'converged':        converged,
    'lr_initial':       f'{lr_initial:.6e}',
    'lr_final':         f'{lr_final:.6e}',
    'final_L_pde':      f'{final_L_pde:.6e}',
    'final_L_pde_max':  f'{final_L_pde_max:.6e}',
    'final_L_bc':       f'{final_L_bc:.6e}',
    'final_L_p':        f'{final_L_p:.6e}',
    'eval_L_pde':       f'{final_L_pde:.6e}',
    'u_ghia':       f'{u_ghia:.6f}',
    'ghia_ref':     f'{ghia_ref:.6f}',
    'elapsed_min':  f'{elapsed_min:.3f}',
    'lid':          LID,
    'git_sha':      git_sha,
}

write_header = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
os.makedirs(os.path.dirname(CSV_PATH) or '.', exist_ok=True)
with open(CSV_PATH, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if write_header:
        writer.writeheader()
    writer.writerow(row)

print(f"\nAppended to {CSV_PATH}", flush=True)
