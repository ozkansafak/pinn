"""Generate a neural network architecture diagram for the PINN and save to images/."""
import numpy as np
import matplotlib.pyplot as plt

# ── Layout constants ───────────────────────────────────────────────────────────
LAYER_X = [0.0, 1.8, 3.4, 5.0, 6.6, 8.4]          # x position of each layer
LAYER_SIZES = [2, 64, 64, 64, 64, 3]              # true neuron counts (64 shown compressed)
SHOW_N = 5       # visible neurons in hidden layers
NEURON_R = 0.18  # neuron circle radius
FIG_W, FIG_H = 16, 7

COLORS = {
    "input":  "#4C72B0",
    "hidden": "#55A868",
    "output": "#C44E52",
    "edge":   "#AAAAAA",
    "tanh":   "#8172B2",
}

LAYER_LABELS = ["Input", "Hidden 1", "Hidden 2", "Hidden 3", "Hidden 4", "Output"]
ACT_LABELS   = [None, "sin", "sin", "sin", "sin", None]

INPUT_NAMES  = ["y", "x"]   # x on top, y on bottom (ys goes bottom→top)
OUTPUT_NAMES = ["p", "v", "u"]  # u on top, p on bottom


def neuron_ys(n_visible, fig_h=FIG_H):
    """Evenly-spaced y positions for n_visible neurons in a column."""
    span = 0.72 * fig_h
    return np.linspace(fig_h / 2 - span / 2, fig_h / 2 + span / 2, n_visible)


def draw_neurons(ax, lx, ys, color, labels=None, size=1120):
    for i, y in enumerate(ys):
        ax.scatter(lx, y, s=size, zorder=5, color=color,
                   edgecolors="white", linewidths=1.4)
        if labels:
            ax.text(lx, y, labels[i], ha="center", va="center",
                    fontsize=25, color="white", fontweight="bold", zorder=6)


def draw_dots(ax, lx, y_center):
    """Three vertical dots to indicate more neurons."""
    for dy in (-0.22, 0, 0.22):
        ax.scatter(lx, y_center + dy, s=30, color="#888888", zorder=5)


def draw_edges(ax, x0, ys0, x1, ys1, alpha=0.10):
    for y0 in ys0:
        for y1 in ys1:
            ax.plot([x0, x1], [y0, y1], color=COLORS["edge"],
                    lw=0.55, alpha=alpha, zorder=1)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(-0.8, 9.6)
    ax.set_ylim(0.2, FIG_H - 0.2)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F8F8")

    # ── Pre-compute visible neuron positions ──────────────────────────────────
    mid = FIG_H / 2
    all_ys = []
    for li, lx in enumerate(LAYER_X):
        if li == 0:
            gap = 0.55
            ys = np.array([mid - gap / 2, mid + gap / 2])
        elif li == len(LAYER_X) - 1:
            gap = 0.55
            ys = np.array([mid - gap, mid, mid + gap])
        else:
            ys = neuron_ys(SHOW_N)
        all_ys.append(ys)

    # ── Edges (draw first so they sit behind neurons) ─────────────────────────
    for li in range(len(LAYER_X) - 1):
        draw_edges(ax, LAYER_X[li], all_ys[li], LAYER_X[li + 1], all_ys[li + 1])

    # ── Neurons ───────────────────────────────────────────────────────────────
    for li, lx in enumerate(LAYER_X):
        ys = all_ys[li]
        if li == 0:
            color = COLORS["input"]
            draw_neurons(ax, lx, ys, color, INPUT_NAMES)
        elif li == len(LAYER_X) - 1:
            color = COLORS["output"]
            draw_neurons(ax, lx, ys, color, OUTPUT_NAMES)
        else:
            color = COLORS["hidden"]
            draw_neurons(ax, lx, ys[:2], color)
            draw_dots(ax, lx, ys[2])
            draw_neurons(ax, lx, ys[3:], color)

    # ── Layer labels (below) ──────────────────────────────────────────────────
    for li, (lx, lbl) in enumerate(zip(LAYER_X, LAYER_LABELS)):
        ax.text(lx, 0.45, lbl, ha="center", va="top", fontsize=22,
                color="#333333", fontweight="bold")

    # ── Activation labels (above, between layers) ─────────────────────────────
    for li in range(1, len(LAYER_X) - 1):
        mid_x = LAYER_X[li]
        y_top = all_ys[li][-1] + 0.55
        ax.text(mid_x, y_top, ACT_LABELS[li], ha="center", va="bottom",
                fontsize=22, color=COLORS["tanh"], style="italic",
                bbox=dict(boxstyle="round,pad=0.25", fc="#EDE7F6", ec=COLORS["tanh"],
                          alpha=0.85, lw=0.8))

    # ── Arrow after last hidden, before output ────────────────────────────────
    ax.annotate("", xy=(LAYER_X[-1] - 0.28, FIG_H / 2),
                xytext=(LAYER_X[-2] + 0.28, FIG_H / 2),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.4))

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        "PINN Architecture  ·  [2 → 64 → 64 → 64 → 64 → 3]  ·  12,867 parameters",
        fontsize=20, fontweight="bold", pad=12, color="#222222"
    )

    plt.tight_layout()
    out = "assets/network_architecture.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
