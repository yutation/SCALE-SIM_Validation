import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the merged report
df = pd.read_csv('validation/data_matmul2/filtered_m_leq_n.csv')

# Ensure numeric types
df['fusion_time'] = pd.to_numeric(df['fusion_time'], errors='coerce')
df['total_cycles'] = pd.to_numeric(df['total_cycles'], errors='coerce')

# Calculate correlation
corr = df['fusion_time'].corr(df['total_cycles'])
print(f'Correlation between fusion_time and total_cycles: {corr}')

# Plot (x: total_cycles, y: fusion_time)
plt.figure(figsize=(8,6))
sns.regplot(x='total_cycles', y='fusion_time', data=df, scatter_kws={'s': 20}, line_kws={'color': 'red'})
plt.xlabel('ScaleSim Total Cycles')
plt.ylabel('Fusion Time (us)')
plt.title(f'Total Cycles vs Fusion Time\nCorrelation: {corr:.3f}')
plt.tight_layout()
plt.savefig('validation/data_matmul2/fusion_vs_cycles_filtered.png')
plt.show() 