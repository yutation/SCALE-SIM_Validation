kernel_group_corr = {
    'matmul': 0.999679,
    'dot_product': 0.156216,
    'matmul_dot_product': 0.805722,
    'convolution': 0.999987
}

import pandas as pd

df = pd.read_csv('filtered_events_with_cycles.csv')

def get_group(kernel_name):
    for group in kernel_group_corr:
        if str(kernel_name).startswith(group):
            return group
    return None

group_corrs = []
for idx, row in df.iterrows():
    if row['event_type'] == 'main' and pd.notna(row['cycles']):
        group = get_group(row['kernel_name'])
        if group:
            group_corrs.append(kernel_group_corr[group])
        else:
            group_corrs.append('')
    else:
        group_corrs.append('')

df['group_corr'] = group_corrs
df.to_csv('filtered_events_with_group_corr.csv', index=False) 