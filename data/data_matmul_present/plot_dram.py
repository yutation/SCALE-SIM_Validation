import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

df = pd.read_csv("merged_verification_results.csv")

# Parse all integer dimensions from the Input_Shapes string
def max_dim(shapes_str):
    return max(int(x) for x in shapes_str.replace("(", "").replace(")", "")
               .replace("[", "").replace("]", "").split(",") if x.strip().isdigit())

# Bucket by matrix dimension:
#   small  – all dims < 128
#   medium – max dim in [128, 1024]
#   large  – max dim > 1024
def size_label(row):
    m = max_dim(row["Input_Shapes"])
    if m < 128:
        return "small (max dim < 128)"
    elif m <= 1024:
        return "medium (128 ≤ max dim ≤ 1024)"
    else:
        return "large (max dim > 1024)"

df["size_bucket"] = df.apply(size_label, axis=1)

bucket_order = ["small (max dim < 128)", "medium (128 ≤ max dim ≤ 1024)", "large (max dim > 1024)"]
colors = {"small (max dim < 128)": "#4e9af1",
          "medium (128 ≤ max dim ≤ 1024)": "#f4a11d",
          "large (max dim > 1024)": "#e05c5c"}

fig, ax = plt.subplots(figsize=(8, 7))

for bucket in bucket_order:
    sub = df[df["size_bucket"] == bucket]
    ax.scatter(
        sub["Actual_Duration_us"],
        sub["Predicted_Latency_us"],
        label=bucket,
        color=colors[bucket],
        edgecolors="white",
        linewidths=0.4,
        s=55,
        alpha=0.85,
        zorder=3,
    )

# Perfect-prediction reference line
all_vals = pd.concat([df["Actual_Duration_us"], df["Predicted_Latency_us"]])
lo, hi = all_vals.min() * 0.9, all_vals.max() * 1.05
ax.plot([lo, hi], [lo, hi], color="#888888", linewidth=1.2, linestyle="--",
        label="Perfect prediction", zorder=2)

# Axes labels and formatting
ax.set_xlabel("Actual Duration (µs)", fontsize=13)
ax.set_ylabel("Predicted Latency (µs)", fontsize=13)
ax.set_title("GEMM: Predicted vs. Actual", fontsize=14, fontweight="bold")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
ax.tick_params(labelsize=11)

legend = ax.legend(fontsize=10, framealpha=0.85, edgecolor="#cccccc",
                   title="Workload size", title_fontsize=10)

# Annotate R² and MAPE
actual = df["Actual_Duration_us"].values
pred = df["Predicted_Latency_us"].values
ss_res = np.sum((actual - pred) ** 2)
ss_tot = np.sum((actual - actual.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
mape = np.mean(np.abs((actual - pred) / actual)) * 100

stats_text = f"R² = {r2:.3f}\nMAPE = {mape:.1f}%\nn = {len(df)}"
ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9))

plt.tight_layout()
out_path = "dram_predicted_vs_actual.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
