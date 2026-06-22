#!/usr/bin/env python3
"""
Scatter plot comparing measured TPU latency vs estimated latency from learned model.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('./comparsion_2d_add.csv', skipinitialspace=True)

# Extract measured and predicted latencies
# Strip whitespace from column names
df.columns = df.columns.str.strip()

measured_latency = df['actual_latency'].values
estimated_latency = df['predicted_latency'].values

# Calculate error metrics for display on plot
absolute_errors = np.abs(measured_latency - estimated_latency)
relative_errors = np.abs((measured_latency - estimated_latency) / measured_latency) * 100
squared_errors = (measured_latency - estimated_latency) ** 2

mae = np.mean(absolute_errors)
median_ae = np.median(absolute_errors)
median_re = np.median(relative_errors)

# R-squared (coefficient of determination)
ss_res = np.sum(squared_errors)
ss_tot = np.sum((measured_latency - np.mean(measured_latency)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Create the figure
plt.figure(figsize=(10, 8))

# Scatter plot
plt.scatter(measured_latency, estimated_latency, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)

# Add y=x dashed line (perfect prediction line)
min_val = min(measured_latency.min(), estimated_latency.min())
max_val = max(measured_latency.max(), estimated_latency.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y = x (Perfect Prediction)')

# Labels and title
plt.xlabel('Measured TPU Latency (μs)', fontsize=14, fontweight='bold')
plt.ylabel('Estimated Latency (μs)', fontsize=14, fontweight='bold')
plt.title('Estimated vs Measured Latency for Elementwise Operations', fontsize=16, fontweight='bold')

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle='--')

# Add text box with error statistics
textstr = f'R² = {r_squared:.4f}\n'
textstr += f'MAE = {mae:.2f} μs\n'
textstr += f'Median Abs. Error = {median_ae:.2f} μs\n'
textstr += f'Median Rel. Error = {median_re:.2f}%'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

# Add legend
plt.legend(fontsize=12, loc='lower right')

# Make axes equal to better see deviations from y=x
plt.axis('equal')
plt.xlim(min_val * 0.95, max_val * 1.05)
plt.ylim(min_val * 0.95, max_val * 1.05)

# Tight layout
plt.tight_layout()

# Save the figure
output_filename = './latency_comparison_scatter_2d_add.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_filename}")

# Also save as PDF for publication quality
# output_pdf = 'asplos/latency_comparison_scatter.pdf'
# plt.savefig(output_pdf, bbox_inches='tight')
# print(f"Plot saved to: {output_pdf}")

# Show the plot
plt.show()

# Print comprehensive error statistics
print("\n" + "="*60)
print("PREDICTION ERROR STATISTICS")
print("="*60)

print(f"\n📊 Dataset Summary:")
print(f"  Number of data points: {len(measured_latency)}")
print(f"  Measured latency range: [{measured_latency.min():.2f}, {measured_latency.max():.2f}] μs")
print(f"  Estimated latency range: [{estimated_latency.min():.2f}, {estimated_latency.max():.2f}] μs")

# Calculate additional error metrics
rmse = np.sqrt(np.mean(squared_errors))
mape = np.mean(relative_errors)
max_ae = np.max(absolute_errors)
std_ae = np.std(absolute_errors)

# Pearson correlation coefficient
correlation = np.corrcoef(measured_latency, estimated_latency)[0, 1]

print(f"\n📈 Absolute Error Metrics:")
print(f"  Mean Absolute Error (MAE):        {mae:.2f} μs")
print(f"  Median Absolute Error:            {median_ae:.2f} μs")
print(f"  Root Mean Squared Error (RMSE):   {rmse:.2f} μs")
print(f"  Max Absolute Error:               {max_ae:.2f} μs")
print(f"  Std Dev of Absolute Error:        {std_ae:.2f} μs")

print(f"\n📊 Relative Error Metrics:")
print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"  Median Relative Error:                 {median_re:.2f}%")
print(f"  Max Relative Error:                    {np.max(relative_errors):.2f}%")

print(f"\n🎯 Goodness of Fit:")
print(f"  R² (Coefficient of Determination): {r_squared:.4f}")
print(f"  Pearson Correlation Coefficient:   {correlation:.4f}")

# Error distribution percentiles
print(f"\n📉 Error Distribution (Percentiles):")
print(f"  25th percentile (Q1):  {np.percentile(absolute_errors, 25):.2f} μs")
print(f"  50th percentile (Q2):  {np.percentile(absolute_errors, 50):.2f} μs")
print(f"  75th percentile (Q3):  {np.percentile(absolute_errors, 75):.2f} μs")
print(f"  95th percentile:       {np.percentile(absolute_errors, 95):.2f} μs")
print(f"  99th percentile:       {np.percentile(absolute_errors, 99):.2f} μs")

# Accuracy within thresholds
within_5pct = np.sum(relative_errors <= 5) / len(relative_errors) * 100
within_10pct = np.sum(relative_errors <= 10) / len(relative_errors) * 100
within_20pct = np.sum(relative_errors <= 20) / len(relative_errors) * 100

print(f"\n✓ Prediction Accuracy:")
print(f"  Within  5% error:  {within_5pct:.1f}% of predictions")
print(f"  Within 10% error:  {within_10pct:.1f}% of predictions")
print(f"  Within 20% error:  {within_20pct:.1f}% of predictions")

print("\n" + "="*60)

