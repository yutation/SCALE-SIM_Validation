import pandas as pd
import re

# Load the merged report
df = pd.read_csv('validation/data_matmul2/merged_kernel_report.csv')

# Function to extract M and N from kernel_name
pattern = re.compile(r'matmul_(\d+)x(\d+)x(\d+)')
def m_leq_n(kernel_name):
    match = pattern.match(kernel_name)
    if match:
        M, N, K = map(int, match.groups())
        return M <= N
    return False

# Filter rows
filtered_df = df[df['kernel_name'].apply(m_leq_n)]

# Write to new CSV
filtered_df.to_csv('validation/data_matmul2/filtered_m_leq_n.csv', index=False)
print('Filtered report written to filtered_m_leq_n.csv') 