import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('filtered_events_with_cycles.csv')

# Filter only main events that have both duration and cycles
df['cycles'] = pd.to_numeric(df['cycles'], errors='coerce')
main_events = df[(df['event_type'] == 'main') & (df['cycles'].notna())]

# Define kernel groups
kernel_groups = ['matmul', 'dot_product', 'matmul_dot_product', 'convolution']

for group in kernel_groups:
    # Use boolean indexing on DataFrame with Series.str.startswith()
    mask = main_events['kernel_name'].astype(str).str.startswith(group)
    group_events = main_events.loc[mask]
    durations = group_events['dur(us)'].to_numpy()
    cycles = group_events['cycles'].to_numpy()
    if len(durations) > 1 and len(cycles) > 1:
        corr = np.corrcoef(durations, cycles)[0, 1]
        print(f"Correlation for {group}: {corr:.6f}")
        print(f"  Number of events: {len(durations)}")
        print(f"  Duration range: {durations.min():.6f} - {durations.max():.6f} μs")
        print(f"  Cycles range: {cycles.min()} - {cycles.max()}")
    else:
        print(f"Not enough data for {group} to calculate correlation.")
    print("") 