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

unique_vals = np.sort(df["M"].unique())  # same set as N values

norm_N = mcolors.Normalize(vmin=unique_vals.min(), vmax=unique_vals.max())
norm_M = mcolors.Normalize(vmin=unique_vals.min(), vmax=unique_vals.max())
cmap = cm.viridis

sm_N = cm.ScalarMappable(cmap=cmap, norm=norm_N)
sm_N.set_array([])
sm_M = cm.ScalarMappable(cmap=cmap, norm=norm_M)
sm_M.set_array([])

for val in unique_vals:
    # ── Left: fixed M=val, color=N ──────────────────────────────────────
    sub_M = df[df["M"] == val]
    # ── Right: fixed N=val, color=M ─────────────────────────────────────
    sub_N = df[df["N"] == val]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex="col")
    ax_pred_M, ax_pred_N = axes[0]
    ax_act_M,  ax_act_N  = axes[1]

    # Left column: fixed M, color by N
    for n_val in unique_vals:
        group = sub_M[sub_M["N"] == n_val].sort_values("K")
        color = cmap(norm_N(n_val))
        ax_pred_M.plot(group["K"], group["Predicted_result"],
                       marker="o", markersize=3, linewidth=1.2, color=color, alpha=0.85)
        ax_act_M.plot(group["K"], group["Actual_Duration_us"],
                      marker="s", markersize=3, linewidth=1.2, color=color, alpha=0.85)

    ax_pred_M.set_title(f"Fixed M = {val}  |  color = N", fontsize=12)
    ax_pred_M.set_ylabel("Predicted Result (cycles)", fontsize=11)
    ax_pred_M.grid(True, alpha=0.25)
    ax_act_M.set_xlabel("K", fontsize=11)
    ax_act_M.set_ylabel("Actual Duration (µs)", fontsize=11)
    ax_act_M.grid(True, alpha=0.25)

    # Right column: fixed N, color by M
    for m_val in unique_vals:
        group = sub_N[sub_N["M"] == m_val].sort_values("K")
        color = cmap(norm_M(m_val))
        ax_pred_N.plot(group["K"], group["Predicted_result"],
                       marker="o", markersize=3, linewidth=1.2, color=color, alpha=0.85)
        ax_act_N.plot(group["K"], group["Actual_Duration_us"],
                      marker="s", markersize=3, linewidth=1.2, color=color, alpha=0.85)

    ax_pred_N.set_title(f"Fixed N = {val}  |  color = M", fontsize=12)
    ax_pred_N.set_ylabel("Predicted Result (cycles)", fontsize=11)
    ax_pred_N.grid(True, alpha=0.25)
    ax_act_N.set_xlabel("K", fontsize=11)
    ax_act_N.set_ylabel("Actual Duration (µs)", fontsize=11)
    ax_act_N.grid(True, alpha=0.25)

    fig.suptitle(f"Value = {val}", fontsize=14, y=0.99)
    fig.subplots_adjust(left=0.07, right=0.85, top=0.94, bottom=0.08,
                        hspace=0.25, wspace=0.35)

    # Colorbar for left (N color)
    cbar_ax_L = fig.add_axes([0.87, 0.08, 0.018, 0.86])
    cbar_L = fig.colorbar(sm_N, cax=cbar_ax_L)
    cbar_L.set_label("N value (left col)", fontsize=10)

    # Colorbar for right (M color) — share the same range so one bar is enough,
    # but label it for clarity
    cbar_ax_R = fig.add_axes([0.93, 0.08, 0.018, 0.86])
    cbar_R = fig.colorbar(sm_M, cax=cbar_ax_R)
    cbar_R.set_label("M value (right col)", fontsize=10)

    out = SCRIPT_DIR / f"combined_{val}.png"
    fig.savefig(out, dpi=180)
    print(f"Saved {out.name}")
    plt.close(fig)

print("Done.")
