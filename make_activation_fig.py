"""Generate a side-by-side comparison of tanh vs SIREN (sin) activations
showing the function and its first two derivatives."""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 1000)

# ── tanh and derivatives ───────────────────────────────────────────────────────
tanh    = np.tanh(x)
tanh_d1 = 1 - np.tanh(x)**2          # sech²(x)
tanh_d2 = -2 * np.tanh(x) * (1 - np.tanh(x)**2)

# ── sin (SIREN, ω₀=1) and derivatives ─────────────────────────────────────────
omega = 1.0
sin    =  np.sin(omega * x)
sin_d1 =  omega * np.cos(omega * x)
sin_d2 = -omega**2 * np.sin(omega * x)

TANH_COLOR = "#C44E52"
SIN_COLOR  = "black"

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.patch.set_facecolor("#F8F8F8")
fig.suptitle("tanh vs sin", fontsize=14,
             fontweight="bold", color="#222222")

titles   = ["f(x)", "f ′(x)  — 1st derivative", "f ″(x)  — 2nd derivative"]
tanh_data = [tanh, tanh_d1, tanh_d2]
sin_data  = [sin,  sin_d1,  sin_d2]

for ax, title, td, sd in zip(axes, titles, tanh_data, sin_data):
    ax.axhline(0, color="#CCCCCC", lw=0.8, zorder=0)
    ax.axvline(0, color="#CCCCCC", lw=0.8, zorder=0)
    ax.plot(x, td, color=TANH_COLOR, lw=2.0, label="tanh", linestyle="--")
    ax.plot(x, sd, color=SIN_COLOR,  lw=2.0, label="sin")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x", fontsize=11)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-1.5, 1.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_facecolor("#FAFAFA")


plt.tight_layout()
out = "assets/activation_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
plt.close(fig)
