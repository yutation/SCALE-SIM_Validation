import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR.parent / "merged_verification_results.csv"
df = pd.read_csv(CSV_PATH)

def parse_mnk(shape_str):
    nums = list(map(int, re.findall(r"\d+", shape_str)))
    M, K, _, N = nums
    return M, N, K

df[["M", "N", "K"]] = df["Input_Shapes"].apply(lambda s: pd.Series(parse_mnk(s)))
df["MNK"] = df["M"] * df["N"] * df["K"]

x = df["MNK"].values
y = df["Predicted_result"].values

# ── Scatter plot ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(x, y, s=12, alpha=0.5, edgecolors="none", c="#1f77b4")

slope, intercept = np.polyfit(x, y, 1)
x_fit = np.linspace(x.min(), x.max(), 500)
ax.plot(x_fit, slope * x_fit + intercept, color="red", linewidth=1.5,
        label=f"Linear fit: y = {slope:.4e}x + {intercept:.2f}")

ax.set_xlabel("M × N × K", fontsize=13)
ax.set_ylabel("Predicted Result (cycles)", fontsize=13)
ax.set_title("Predicted Result vs M×N×K", fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(SCRIPT_DIR / "predict_vs_mnk.png", dpi=200)
print("Saved predict_vs_mnk.png")
plt.close(fig)

# ── Log-log scatter ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(x, y, s=12, alpha=0.5, edgecolors="none", c="#2ca02c")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("M × N × K (log)", fontsize=13)
ax.set_ylabel("Predicted Result (cycles, log)", fontsize=13)
ax.set_title("Predicted Result vs M×N×K (Log-Log)", fontsize=15)
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(SCRIPT_DIR / "predict_vs_mnk_loglog.png", dpi=200)
print("Saved predict_vs_mnk_loglog.png")
plt.close(fig)

# ── Actual: scatter plot ────────────────────────────────────────────────
ya = df["Actual_Duration_us"].values

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(x, ya, s=12, alpha=0.5, edgecolors="none", c="#ff7f0e")

slope_a, intercept_a = np.polyfit(x, ya, 1)
ax.plot(x_fit, slope_a * x_fit + intercept_a, color="red", linewidth=1.5,
        label=f"Linear fit: y = {slope_a:.4e}x + {intercept_a:.2f}")

ax.set_xlabel("M × N × K", fontsize=13)
ax.set_ylabel("Actual Duration (µs)", fontsize=13)
ax.set_title("Actual Duration vs M×N×K", fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(SCRIPT_DIR / "actual_vs_mnk.png", dpi=200)
print("Saved actual_vs_mnk.png")
plt.close(fig)

# ── Actual: log-log scatter ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(x, ya, s=12, alpha=0.5, edgecolors="none", c="#d62728")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("M × N × K (log)", fontsize=13)
ax.set_ylabel("Actual Duration (µs, log)", fontsize=13)
ax.set_title("Actual Duration vs M×N×K (Log-Log)", fontsize=15)
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(SCRIPT_DIR / "actual_vs_mnk_loglog.png", dpi=200)
print("Saved actual_vs_mnk_loglog.png")
plt.close(fig)

print("Done.")
