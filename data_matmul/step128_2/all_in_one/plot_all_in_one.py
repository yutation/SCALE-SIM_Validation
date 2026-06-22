import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
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
unique_N = np.sort(df["N"].unique())

MARKERS = ["o", "s", "^", "v", "D", "p", "*", "h", "<", ">",
           "P", "X", "d", "H", "8", "+"]
marker_map = {n: MARKERS[i] for i, n in enumerate(unique_N)}

norm = mcolors.Normalize(vmin=unique_M.min(), vmax=unique_M.max())
cmap = cm.viridis

fig, (ax_pred, ax_act) = plt.subplots(2, 1, figsize=(22, 18), sharex=True)

for m_val in unique_M:
    color = cmap(norm(m_val))
    for n_val in unique_N:
        group = df[(df["M"] == m_val) & (df["N"] == n_val)].sort_values("K")
        mk = marker_map[n_val]
        ax_pred.plot(group["K"], group["Predicted_result"],
                     marker=mk, markersize=5, linewidth=1.0,
                     color=color, alpha=0.75)
        ax_act.plot(group["K"], group["Actual_Duration_us"],
                    marker=mk, markersize=5, linewidth=1.0,
                    color=color, alpha=0.75)

ax_pred.set_ylabel("Predicted Result (cycles)", fontsize=14)
ax_pred.set_title("Predicted Result  |  X = K,  color = M,  shape = N", fontsize=15)
ax_pred.grid(True, alpha=0.2)

ax_act.set_xlabel("K", fontsize=14)
ax_act.set_ylabel("Actual Duration (µs)", fontsize=14)
ax_act.set_title("Actual Duration  |  X = K,  color = M,  shape = N", fontsize=15)
ax_act.grid(True, alpha=0.2)

# ── Colorbar for M ──────────────────────────────────────────────────────
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.subplots_adjust(left=0.06, right=0.80, top=0.95, bottom=0.06, hspace=0.15)
cbar_ax = fig.add_axes([0.82, 0.06, 0.018, 0.89])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("M value", fontsize=13)
cbar.set_ticks(unique_M)
cbar.set_ticklabels([str(v) for v in unique_M], fontsize=8)

# ── Legend for N (marker shapes) ───────────────────────────────────────
legend_handles = [
    mlines.Line2D([0], [0], marker=marker_map[n], color="gray",
                  markersize=7, linewidth=1.0, label=f"N = {n}")
    for n in unique_N
]
fig.legend(handles=legend_handles, title="N value (shape)",
           loc="center right", bbox_to_anchor=(1.0, 0.5),
           fontsize=9, title_fontsize=10, framealpha=0.9,
           ncol=1)

out = SCRIPT_DIR / "all_in_one.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out.name}")
plt.close(fig)
print("Done.")
