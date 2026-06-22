import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("filtered_events.csv")

sub_df = df[df["event_type"] == "sub"].copy()

def extract_size(kernel_name):
    m = re.search(r"\((\d+),?\)", kernel_name)
    return int(m.group(1)) if m else None

sub_df["size"] = sub_df["kernel_name"].apply(extract_size)

result = (
    sub_df.groupby("size")["dur(us)"]
    .mean()
    .reset_index()
    .rename(columns={"dur(us)": "avg_sub_dur(us)"})
    .sort_values("size")
)

output_path = "sub_avg_duration_pivot.csv"
result.to_csv(output_path, index=False)
print(f"Saved to {output_path}")

x = result["size"].values
y = result["avg_sub_dur(us)"].values

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
y_pred = slope * x + intercept

print(f"\nLinear Regression:")
print(f"  slope     = {slope:.6e}  (us per element)")
print(f"  intercept = {intercept:.6f} us")
print(f"  R²        = {r_value**2:.6f}")
print(f"  p-value   = {p_value:.3e}")

fig, ax = plt.subplots(figsize=(10, 5))

ax.scatter(x, y, s=18, alpha=0.7, color="steelblue", label="avg sub duration")
ax.plot(x, y_pred, color="crimson", linewidth=1.8,
        label=f"linear fit: y = {slope:.2e}·x + {intercept:.3f}  (R²={r_value**2:.4f})")

ax.set_xlabel("Size", fontsize=12)
ax.set_ylabel("Avg sub duration (µs)", fontsize=12)
ax.set_title("Sub kernel avg duration vs. size — linear regression", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
fig_path = "sub_avg_duration_regression.png"
plt.savefig(fig_path, dpi=150)
print(f"Figure saved to {fig_path}")
