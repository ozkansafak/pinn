"""Generate assets/error_vs_width.png — normalized velocity error vs width."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

GHIA_REF = 0.737220

# ── tanh baseline ──────────────────────────────────────────────────────────────
tanh_widths = [4, 8, 16, 32, 64, 128, 256]
tanh_ughi   = [0.618789, 0.636039, 0.712834, 0.717778, 0.728309, 0.692204, 0.595371]

# ── sin sweep ──────────────────────────────────────────────────────────────────
sin_widths = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
sin_ughi   = [0.612545, 0.669994, 0.728651, 0.741875, 0.740412,
              0.739961, 0.741628, 0.739332, 0.738904]

def norm_err_pct(u_pred):
    return abs(u_pred - GHIA_REF) / GHIA_REF * 100

tanh_err = [norm_err_pct(u) for u in tanh_ughi]
sin_err  = [norm_err_pct(u) for u in sin_ughi]

TANH_COLOR = "#C44E52"
SIN_COLOR  = "black"

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#F8F8F8")
ax.set_facecolor("#FAFAFA")

ax.plot(tanh_widths, tanh_err, color=TANH_COLOR, lw=2.0, linestyle="--",
        marker="o", markersize=6, label="tanh")
ax.plot(sin_widths,  sin_err,  color=SIN_COLOR,  lw=2.0, linestyle="-",
        marker="o", markersize=6, label="sin")

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_ylim(bottom=0.09)
ax.set_xlim(left=3, right=2500)

ax.set_xticks([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: str(int(v))))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))

ax.set_xlabel("Width", fontsize=12)
ax.set_ylabel("Normalized error  |u − u_ref| / u_ref", fontsize=11)
ax.set_title("Percent Error in Velocity vs Layer Width  (Re=100, Ghia ref)",
             fontsize=13, fontweight="bold", color="#222222")
ax.legend(fontsize=11)
ax.grid(True, which="both", alpha=0.25)

plt.tight_layout()
out = "assets/error_vs_width.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
plt.close(fig)
