import pandas as pd
import re

df = pd.read_csv("filtered_events.csv")

sub_df = df[df["event_type"] == "sub"].copy()

def extract_dims(kernel_name):
    m = re.search(r"\((\d+),\s*(\d+)\)", kernel_name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

sub_df[["dim0", "dim1"]] = sub_df["kernel_name"].apply(
    lambda x: pd.Series(extract_dims(x))
)

result = (
    sub_df.groupby(["dim0", "dim1"])["dur(us)"]
    .mean()
    .reset_index()
    .rename(columns={"dur(us)": "avg_dur(us)"})
    .sort_values(["dim0", "dim1"])
)

result.insert(2, "size", result["dim0"] * result["dim1"])

output_path = "sub_avg_duration_2d.csv"
result.to_csv(output_path, index=False)
print(f"Saved to {output_path}")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(result["size"], result["avg_dur(us)"], s=15, alpha=0.6, color="steelblue")
ax.set_xlabel("Size (dim0 × dim1)", fontsize=12)
ax.set_ylabel("Avg sub duration (µs)", fontsize=12)
ax.set_title("2D reduce: avg sub duration vs. size", fontsize=13)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
fig_path = "sub_avg_duration_2d_scatter.png"
plt.savefig(fig_path, dpi=150)
print(f"Figure saved to {fig_path}")

# Second figure: x=dim0, y=avg_dur, color=dim1
import matplotlib.cm as cm
import matplotlib.colors as mcolors

dim1_vals = sorted(result["dim1"].unique())
cmap = cm.get_cmap("viridis", len(dim1_vals))
norm = mcolors.BoundaryNorm(boundaries=dim1_vals + [dim1_vals[-1] + 32], ncolors=len(dim1_vals))

fig2, ax2 = plt.subplots(figsize=(11, 6))
sc = ax2.scatter(
    result["dim0"], result["avg_dur(us)"],
    c=result["dim1"], cmap="viridis", s=20, alpha=0.75
)
cbar = fig2.colorbar(sc, ax=ax2)
cbar.set_label("dim1", fontsize=11)
ax2.set_xlabel("dim0", fontsize=12)
ax2.set_ylabel("Avg sub duration (µs)", fontsize=12)
ax2.set_title("2D reduce: avg sub duration vs. dim0  (color = dim1)", fontsize=13)
ax2.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
fig2_path = "sub_avg_duration_2d_dim0_color.png"
plt.savefig(fig2_path, dpi=150)
print(f"Figure saved to {fig2_path}")

# Third figure: x=dim1, y=avg_dur, color=dim0
fig3, ax3 = plt.subplots(figsize=(11, 6))
sc3 = ax3.scatter(
    result["dim1"], result["avg_dur(us)"],
    c=result["dim0"], cmap="viridis", s=20, alpha=0.75
)
cbar3 = fig3.colorbar(sc3, ax=ax3)
cbar3.set_label("dim0", fontsize=11)
ax3.set_xlabel("dim1", fontsize=12)
ax3.set_ylabel("Avg sub duration (µs)", fontsize=12)
ax3.set_title("2D reduce: avg sub duration vs. dim1  (color = dim0)", fontsize=13)
ax3.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
fig3_path = "sub_avg_duration_2d_dim1_color.png"
plt.savefig(fig3_path, dpi=150)
print(f"Figure saved to {fig3_path}")
