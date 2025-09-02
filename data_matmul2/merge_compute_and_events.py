import pandas as pd

# Load the compute report
compute_df = pd.read_csv('validation/data_matmul2/COMPUTE_REPORT.csv')
# Load the events
events_df = pd.read_csv('validation/data_matmul2/filtered_events.csv')

# Extract kernel_name from events
kernel_names = events_df['kernel_name'].unique()

# Prepare output rows
output_rows = []

for kernel in kernel_names:
    # Get total cycles from compute report (try to match by order)
    # The compute report does not have kernel_name, so we assume order matches
    # Find the index of this kernel in the unique kernel list
    kernel_idx = list(kernel_names).index(kernel)
    if kernel_idx < len(compute_df):
        total_cycles = compute_df.iloc[kernel_idx][' Total Cycles'] if ' Total Cycles' in compute_df.columns else compute_df.iloc[kernel_idx]['Total Cycles']
    else:
        total_cycles = None
    # Get event times for this kernel
    kernel_events = events_df[events_df['kernel_name'] == kernel]
    # Sum durations for each event type
    fusion_time = kernel_events[(kernel_events['event_type'] == 'sub') & (kernel_events['event_name'] == 'fusion')]['dur(us)'].sum()
    main_time = kernel_events[(kernel_events['event_type'] == 'main')]['dur(us)'].sum()
    copy_start_time = kernel_events[(kernel_events['event_type'] == 'sub') & (kernel_events['event_name'] == 'copy-start')]['dur(us)'].sum()
    copy_done_time = kernel_events[(kernel_events['event_type'] == 'sub') & (kernel_events['event_name'] == 'copy-done')]['dur(us)'].sum()
    output_rows.append({
        'kernel_name': kernel,
        'total_cycles': total_cycles,
        'fusion_time': fusion_time,
        'main_time': main_time,
        'copy_start_time': copy_start_time,
        'copy_done_time': copy_done_time
    })

# Output to CSV
output_df = pd.DataFrame(output_rows)
output_df.to_csv('merged_kernel_report.csv', index=False)
print('Merged CSV written to merged_kernel_report.csv') 