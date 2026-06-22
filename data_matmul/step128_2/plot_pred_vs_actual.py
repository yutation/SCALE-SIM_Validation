import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

OUT_DIR = 'pred_vs_actual_panels'
os.makedirs(OUT_DIR, exist_ok=True)

# ── load & parse ──────────────────────────────────────────────────────────
df = pd.read_csv('merged_verification_results.csv')

def parse_shapes(s):
    nums = re.findall(r'\d+', s)
    return int(nums[0]), int(nums[1]), int(nums[3])   # M, K, N

df[['M', 'K', 'N']] = df['Input_Shapes'].apply(lambda s: pd.Series(parse_shapes(s)))

# ── shared encoding (same palette / markers used across all three panels) ──
ALL_VALS = sorted(df['M'].unique())   # same 16 values for M, K, N

MARKERS = ['o', 's', 'D', '^', 'v', '<', '>', 'p',
           '*', 'h', 'H', '8', 'P', 'X', 'd', (4, 1, 45)]

cmap = matplotlib.colormaps.get_cmap('tab20').resampled(len(ALL_VALS))
COLORS  = [cmap(i) for i in range(len(ALL_VALS))]

val_to_color  = {v: COLORS[i]  for i, v in enumerate(ALL_VALS)}
val_to_marker = {v: MARKERS[i] for i, v in enumerate(ALL_VALS)}

dims = ['M', 'N', 'K']

for dim in dims:
    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(1, 2, width_ratios=[3, 1],
                           wspace=0.05,
                           left=0.08, right=0.97,
                           top=0.91, bottom=0.09)
    ax  = fig.add_subplot(gs[0, 0])
    lax = fig.add_subplot(gs[0, 1])
    lax.axis('off')

    # ── scatter ───────────────────────────────────────────────────────────
    for val in ALL_VALS:
        sub = df[df[dim] == val]
        ax.scatter(
            sub['Predicted_result'],
            sub['Actual_Duration_us'],
            c=[val_to_color[val]],
            marker=val_to_marker[val],
            s=50,
            alpha=0.62,
            linewidths=0.0,
            label=str(val),
        )

    ax.set_xlabel('Predicted result (model units)', fontsize=13)
    ax.set_ylabel('Actual Duration (µs)', fontsize=13)
    ax.set_title(
        f'Predicted vs Actual Duration\nColored & shaped by  {dim}',
        fontsize=14, fontweight='bold'
    )
    ax.tick_params(labelsize=11)
    ax.grid(True, ls='--', lw=0.4, alpha=0.5)

    # ── legend ────────────────────────────────────────────────────────────
    handles = [
        mlines.Line2D([], [],
                      color=val_to_color[v],
                      marker=val_to_marker[v],
                      linestyle='None',
                      markersize=9,
                      label=str(v))
        for v in ALL_VALS
    ]
    lax.legend(
        handles=handles,
        title=f'{dim}  value',
        title_fontsize=11,
        loc='center left',
        bbox_to_anchor=(0.0, 0.5),
        frameon=True,
        fontsize=10,
        ncol=2,
    )

    out_path = os.path.join(OUT_DIR, f'pred_vs_actual_{dim}.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out_path}')
