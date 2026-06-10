"""Plot eval_L_pde and LR curves for all completed width-sweep runs."""
import csv
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

CURVE_DIR = "results/loss_curves"
OUT_DIR   = "results"

def load_curve(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    epochs    = [int(r["epoch"])         for r in rows]
    eval_pde  = [float(r["eval_pde"])    for r in rows]
    train_pde = [float(r["train_pde"])   for r in rows]
    lr        = [float(r["lr"])          for r in rows]
    return epochs, train_pde, eval_pde, lr


def main():
    files = sorted(glob.glob(f"{CURVE_DIR}/width_*.csv"),
                   key=lambda p: int(os.path.basename(p).replace("width_","").replace(".csv","")))

    if not files:
        print("No loss curve CSVs found. Run parse_sweep_logs.py first.")
        return

    widths = [int(os.path.basename(p).replace("width_","").replace(".csv","")) for p in files]
    colors = cm.viridis(np.linspace(0.1, 0.9, len(files)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax_loss, ax_lr = axes
    fig.suptitle("Width Sweep — Loss & LR Curves", fontsize=14, fontweight="bold")

    for path, width, color in zip(files, widths, colors):
        epochs, train_pde, eval_pde, lr = load_curve(path)
        label = f"w={width}"
        ax_loss.semilogy(epochs, eval_pde,  color=color, lw=1.5, label=label)
        ax_lr.semilogy  (epochs, lr,         color=color, lw=1.5, label=label)

    ax_loss.set_title("eval_L_pde  (PDE residual, independent points)", fontsize=12)
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("eval_L_pde", fontsize=11)
    ax_loss.legend(fontsize=9, ncol=2)
    ax_loss.margins(x=0)
    ax_loss.grid(True, which="both", alpha=0.3)

    ax_lr.set_title("Learning Rate", fontsize=12)
    ax_lr.set_xlabel("Epoch", fontsize=11)
    ax_lr.set_ylabel("LR", fontsize=11)
    ax_lr.legend(fontsize=9, ncol=2)
    ax_lr.margins(x=0)
    ax_lr.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = f"{OUT_DIR}/sweep_loss_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
