import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
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

x = df["Predicted_result"].values
y = df["Actual_Duration_us"].values
ratio = y / x  # Actual / Predicted

# ── Figure 1: Overall ratio scatter ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(x, ratio, s=12, alpha=0.5, edgecolors="none", c="#1f77b4")
ax.axhline(np.median(ratio), color="red", linewidth=1.2, linestyle="--",
           label=f"Median ratio = {np.median(ratio):.4e}")
ax.set_xlabel("Predicted Result (cycles)", fontsize=13)
ax.set_ylabel("Actual / Predicted", fontsize=13)
ax.set_title("Ratio Diagnosis: Actual / Predicted vs Predicted", fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(SCRIPT_DIR / "diagnosis_ratio.png", dpi=200)
print("Saved diagnosis_ratio.png")
plt.close(fig)

# ── Figures 2-4: Ratio colored by M, N, K ───────────────────────────────
for dim_name in ["M", "N", "K"]:
    dim_vals = df[dim_name].values
    unique_vals = np.sort(df[dim_name].unique())
    norm = mcolors.Normalize(vmin=unique_vals.min(), vmax=unique_vals.max())

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(x, ratio, c=dim_vals, cmap=cm.viridis, norm=norm,
                    s=14, alpha=0.6, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(f"{dim_name} value", fontsize=12)

    ax.axhline(np.median(ratio), color="red", linewidth=1, linestyle="--",
               alpha=0.6, label=f"Median = {np.median(ratio):.4e}")
    ax.set_xlabel("Predicted Result (cycles)", fontsize=13)
    ax.set_ylabel("Actual / Predicted", fontsize=13)
    ax.set_title(f"Ratio Diagnosis — colored by {dim_name}", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = SCRIPT_DIR / f"diagnosis_ratio_{dim_name}.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path.name}")
    plt.close(fig)

# ── Figure 5: Log-scale ratio ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(x, ratio, s=12, alpha=0.5, edgecolors="none", c="#2ca02c")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Predicted Result (cycles, log)", fontsize=13)
ax.set_ylabel("Actual / Predicted (log)", fontsize=13)
ax.set_title("Ratio Diagnosis (Log-Log)", fontsize=15)
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(SCRIPT_DIR / "diagnosis_ratio_loglog.png", dpi=200)
print("Saved diagnosis_ratio_loglog.png")
plt.close(fig)

# ── Figure 6: Ratio histogram ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(ratio, bins=80, edgecolor="white", linewidth=0.4, color="#ff7f0e")
ax.axvline(np.median(ratio), color="red", linewidth=1.5, linestyle="--",
           label=f"Median = {np.median(ratio):.4e}")
ax.axvline(np.mean(ratio), color="blue", linewidth=1.5, linestyle="--",
           label=f"Mean = {np.mean(ratio):.4e}")
ax.set_xlabel("Actual / Predicted", fontsize=13)
ax.set_ylabel("Count", fontsize=13)
ax.set_title("Distribution of Actual / Predicted Ratio", fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(SCRIPT_DIR / "diagnosis_ratio_hist.png", dpi=200)
print("Saved diagnosis_ratio_hist.png")
plt.close(fig)

print("Done.")
