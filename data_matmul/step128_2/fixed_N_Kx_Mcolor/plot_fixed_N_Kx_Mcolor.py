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

unique_M = np.sort(df["M"].unique())
norm = mcolors.Normalize(vmin=unique_M.min(), vmax=unique_M.max())
cmap = cm.viridis

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

for n_val in sorted(df["N"].unique()):
    sub = df[df["N"] == n_val].sort_values(["M", "K"])

    fig, (ax_pred, ax_act) = plt.subplots(2, 1, figsize=(11, 10), sharex=True)

    for m_val in unique_M:
        group = sub[sub["M"] == m_val].sort_values("K")
        color = cmap(norm(m_val))
        ax_pred.plot(group["K"], group["Predicted_result"], marker="o", markersize=4,
                     linestyle="-", linewidth=1.2, color=color, alpha=0.8)
        ax_act.plot(group["K"], group["Actual_Duration_us"], marker="s", markersize=4,
                    linestyle="-", linewidth=1.2, color=color, alpha=0.8)

    ax_pred.set_ylabel("Predicted Result (cycles)", fontsize=13)
    ax_pred.set_title(f"N = {n_val}", fontsize=15)
    ax_pred.grid(True, alpha=0.25)

    ax_act.set_xlabel("K", fontsize=13)
    ax_act.set_ylabel("Actual Duration (µs)", fontsize=13)
    ax_act.grid(True, alpha=0.25)

    fig.subplots_adjust(left=0.08, right=0.82, top=0.94, bottom=0.07, hspace=0.12)
    cbar_ax = fig.add_axes([0.85, 0.07, 0.025, 0.87])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("M value", fontsize=12)

    out = SCRIPT_DIR / f"fixed_N_{n_val}.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out.name}")
    plt.close(fig)

print("Done.")
