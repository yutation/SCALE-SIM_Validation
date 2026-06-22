#!/usr/bin/env python3
"""
Script to plot Predicted Latency vs Actual Duration.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define file paths
input_file = "merged_combined.csv"
output_file = "latency_comparison.png"

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, input_file)
output_path = os.path.join(script_dir, output_file)

# Read the CSV file
print(f"Reading {input_file}...")
df = pd.read_csv(input_path)
print(f"  - Total rows: {len(df)}")

# Extract X and Y data
X = df['Predicted_Latency_us']
Y = df['Actual_Duration_us']

print(f"\nData statistics:")
print(f"  Predicted Latency: min={X.min():.2f}, max={X.max():.2f}, mean={X.mean():.2f}")
print(f"  Actual Duration: min={Y.min():.2f}, max={Y.max():.2f}, mean={Y.mean():.2f}")

# Create the figure
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot
scatter = ax.scatter(X, Y, alpha=0.6, s=30, c='blue', edgecolors='navy', linewidth=0.5)

# Add trend line (linear regression)
z = np.polyfit(X, Y, 1)
p = np.poly1d(z)
x_trend = np.linspace(X.min(), X.max(), 100)
ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, 
        label=f'Trend line: y={z[0]:.4f}x+{z[1]:.2f}', alpha=0.7)

# Calculate and display correlation
correlation = np.corrcoef(X, Y)[0, 1]
print(f"  Correlation coefficient: {correlation:.4f}")
print(f"  Linear fit: y = {z[0]:.4f}x + {z[1]:.2f}")

# Labels and title
ax.set_xlabel('Predicted Latency (Model Output)', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual Duration (μs)', fontsize=14, fontweight='bold')
ax.set_title('Predicted Latency vs Actual Duration\nMatrix Multiplication Operations', 
             fontsize=16, fontweight='bold', pad=20)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--')

# Add correlation text box
textstr = f'Correlation: {correlation:.4f}\nData points: {len(df)}\nSlope: {z[0]:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

# Legend
ax.legend(loc='lower right', fontsize=11)

# Tight layout
plt.tight_layout()

# Save the figure
print(f"\nSaving figure to {output_file}...")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure saved successfully!")
print(f"  Output: {output_path}")

# Optionally display the plot
# plt.show()
