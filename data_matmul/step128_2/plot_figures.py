import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "merged_verification_results.csv"

df = pd.read_csv(CSV_PATH)

x = df["Predicted_result"].values
y = df["Actual_Duration_us"].values

# ── Figure 1: Scatter plot with linear fit ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(x, y, s=12, alpha=0.5, edgecolors="none", c="#1f77b4")

slope, intercept = np.polyfit(x, y, 1)
x_fit = np.linspace(x.min(), x.max(), 500)
ax.plot(x_fit, slope * x_fit + intercept, color="red", linewidth=1.5,
        label=f"Linear fit: y = {slope:.4e}x + {intercept:.4f}")

ax.set_xlabel("Predicted Result (cycles)", fontsize=13)
ax.set_ylabel("Actual Duration (µs)", fontsize=13)
ax.set_title("Predicted Result vs Actual Duration", fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(SCRIPT_DIR / "scatter_predicted_vs_actual.png", dpi=200)
print("Saved scatter_predicted_vs_actual.png")

# ── Figure 2: Log-log scatter ───────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.scatter(x, y, s=12, alpha=0.5, edgecolors="none", c="#2ca02c")
ax2.set_xscale("log")
ax2.set_yscale("log")

ax2.set_xlabel("Predicted Result (cycles, log)", fontsize=13)
ax2.set_ylabel("Actual Duration (µs, log)", fontsize=13)
ax2.set_title("Predicted vs Actual (Log-Log Scale)", fontsize=15)
ax2.grid(True, alpha=0.3, which="both")
fig2.tight_layout()
fig2.savefig(SCRIPT_DIR / "scatter_loglog.png", dpi=200)
print("Saved scatter_loglog.png")

# ── Figure 3: Residual plot ─────────────────────────────────────────────
y_pred_linear = slope * x + intercept
residuals = y - y_pred_linear

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.scatter(x, residuals, s=10, alpha=0.4, edgecolors="none", c="#d62728")
ax3.axhline(0, color="black", linewidth=0.8)
ax3.set_xlabel("Predicted Result (cycles)", fontsize=13)
ax3.set_ylabel("Residual (µs)", fontsize=13)
ax3.set_title("Residuals of Linear Fit", fontsize=15)
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig(SCRIPT_DIR / "residuals.png", dpi=200)
print("Saved residuals.png")

# ── Figure 4: Hexbin density plot ───────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(10, 8))
hb = ax4.hexbin(x, y, gridsize=50, cmap="YlOrRd", mincnt=1)
fig4.colorbar(hb, ax=ax4, label="Count")
ax4.set_xlabel("Predicted Result (cycles)", fontsize=13)
ax4.set_ylabel("Actual Duration (µs)", fontsize=13)
ax4.set_title("Density: Predicted vs Actual", fontsize=15)
fig4.tight_layout()
fig4.savefig(SCRIPT_DIR / "density_hexbin.png", dpi=200)
print("Saved density_hexbin.png")

plt.show()
print("Done.")
