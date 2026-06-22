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
    # [(M, K), (K, N)]
    M, K, _, N = nums
    return M, N, K

df[["M", "N", "K"]] = df["Input_Shapes"].apply(lambda s: pd.Series(parse_mnk(s)))

x = df["Predicted_result"].values
y = df["Actual_Duration_us"].values

for dim_name in ["M", "N", "K"]:
    dim_vals = df[dim_name].values
    unique_vals = np.sort(df[dim_name].unique())

    norm = mcolors.Normalize(vmin=unique_vals.min(), vmax=unique_vals.max())
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(x, y, c=dim_vals, cmap=cmap, norm=norm,
                    s=14, alpha=0.6, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(f"{dim_name} value", fontsize=12)

    ax.set_xlabel("Predicted Result (cycles)", fontsize=13)
    ax.set_ylabel("Actual Duration (µs)", fontsize=13)
    ax.set_title(f"Predicted vs Actual — colored by {dim_name}", fontsize=15)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = SCRIPT_DIR / f"scatter_color_{dim_name}.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path.name}")
    plt.close(fig)

print("Done.")
