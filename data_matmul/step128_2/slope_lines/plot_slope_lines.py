import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ── load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("../merged_verification_results_mnk.csv")
x = df["Predicted_result"].values.astype(float)
y = df["Actual_Duration_us"].values.astype(float)

# ── linear regression  y = reg_slope * x + intercept ─────────────────────────
reg_slope, intercept, r, p, se = stats.linregress(x, y)
print(f"Linear regression: y = {reg_slope:.6e} * x + {intercept:.6f}")
print(f"  R² = {r**2:.4f}")

# O is the regression line's y-intercept: (0, intercept)
O = (0.0, intercept)
print(f"Origin O = (0, {intercept:.6f})")

# ── slope from O to every data point: (y - O_y) / (x - O_x) ─────────────────
# O_x = 0, so denominator is just x
slopes = (y - intercept) / x

# only consider points with x > 300000 for choosing M and S
mask = x > 200000
idx_M = np.where(mask, slopes, -np.inf).argmax()   # largest slope  → point M
idx_S = np.where(mask, slopes,  np.inf).argmin()   # smallest slope → point S

slope_M = slopes[idx_M]
slope_S = slopes[idx_S]

print(f"Point M  →  x={x[idx_M]:.3f}, y={y[idx_M]:.6f},  slope from O={slope_M:.6e}")
print(f"Point S  →  x={x[idx_S]:.3f}, y={y[idx_S]:.6f},  slope from O={slope_S:.6e}")

# ── 10 intermediate slopes evenly spaced between S and M ──────────────────────
n_mid = 5
mid_slopes = np.linspace(slope_S, slope_M, n_mid + 2)[1:-1]
all_slopes  = [slope_S] + list(mid_slopes) + [slope_M]
n_lines     = len(all_slopes)   # 12 total

# ── colour palette ─────────────────────────────────────────────────────────────
cmap   = plt.colormaps.get_cmap("plasma").resampled(n_lines)
colors = [cmap(i) for i in range(n_lines)]

# ── x-range for guide lines ───────────────────────────────────────────────────
x_max  = x.max() * 1.05
x_line = np.array([0.0, x_max])

# ── plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))

# scatter (all data)
ax.scatter(x, y, s=8, alpha=0.35, color="steelblue", zorder=2, label="Data points")

# regression line
ax.plot(x_line, reg_slope * x_line + intercept, color="gray",
        linewidth=1.5, linestyle="-.", zorder=3,
        label=f"Regression (R²={r**2:.3f})")

# highlight M and S
ax.scatter(x[idx_M], y[idx_M], s=80, color="red",   zorder=5, label="M (max slope from O)")
ax.scatter(x[idx_S], y[idx_S], s=80, color="green", zorder=5, label="S (min slope from O)")

# origin O  (regression intercept)
ax.scatter(*O, s=100, color="black", zorder=6)
ax.annotate(f"O (intercept={intercept:.3f})", O,
            textcoords="offset points", xytext=(8, -14),
            fontsize=9, fontweight="bold")

# slope lines through O
for i, (s, c) in enumerate(zip(all_slopes, colors)):
    # line: y = s*(x - O_x) + O_y  →  y = s*x + intercept  (since O_x = 0)
    y_line = s * x_line + intercept

    if i == 0:
        lw, ls, label_tag = 2.0, "--", "S"
    elif i == n_lines - 1:
        lw, ls, label_tag = 2.0, "--", "M"
    else:
        lw, ls, label_tag = 1.2, "-",  str(i)

    ax.plot(x_line, y_line, color=c, linewidth=lw, linestyle=ls,
            label=f"slope {label_tag}: {s:.3e}", zorder=3)

    # label near right end of the line
    x_lbl = x_max * 0.85
    y_lbl = s * x_lbl + intercept
    ax.text(x_lbl, y_lbl, label_tag, color=c, fontsize=8,
            va="bottom", ha="left", fontweight="bold")

# annotate M and S data points
ax.annotate(f"M\nslope={slope_M:.3e}", (x[idx_M], y[idx_M]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8, color="red", fontweight="bold")
ax.annotate(f"S\nslope={slope_S:.3e}", (x[idx_S], y[idx_S]),
            textcoords="offset points", xytext=(6, -18),
            fontsize=8, color="green", fontweight="bold")

ax.set_xlabel("Predicted result (cycles)", fontsize=12)
ax.set_ylabel("Actual Duration (µs)", fontsize=12)
ax.set_title("Predicted vs Actual – slope guide lines from regression intercept O", fontsize=13)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.8)
ax.grid(True, linestyle=":", alpha=0.4)

plt.tight_layout()
out = "scatter_slope_lines.png"
plt.savefig(out, dpi=300)
print(f"Saved → {out}")
plt.close()
