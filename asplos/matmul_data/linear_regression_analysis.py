#!/usr/bin/env python3
"""
Linear regression analysis for predicted vs actual latency
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Read the data
df = pd.read_csv('merged_verification_results.csv')

# Extract x and y values
x = df['Predicted_Latency_us'].values.reshape(-1, 1)
y = df['Actual_Duration_us'].values

# Perform linear regression
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

# Calculate metrics
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
mape = np.mean(np.abs((y - y_pred) / y)) * 100

# Print regression results
print("=" * 60)
print("Linear Regression Analysis Results")
print("=" * 60)
print(f"Regression Equation: y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f} μs")
print(f"MAE: {mae:.4f} μs")
print(f"MAPE: {mape:.4f}%")
print("=" * 60)

# Create scatter plot with regression line
plt.figure(figsize=(12, 8))

# Scatter plot
plt.scatter(x, y, alpha=0.6, s=50, color='blue', edgecolors='black', linewidth=0.5, label='Data Points')

# Regression line
x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'Linear Fit: y = {model.coef_[0]:.4f}x + {model.intercept_:.2f}')

# Perfect prediction line (y = x)
plt.plot([x.min(), x.max()], [x.min(), x.max()], 'g--', linewidth=2, alpha=0.7, label='Perfect Prediction (y = x)')

# Labels and title
plt.xlabel('Predicted Latency (μs)', fontsize=12, fontweight='bold')
plt.ylabel('Actual Duration (μs)', fontsize=12, fontweight='bold')
plt.title('Predicted vs Actual Latency for Matrix Multiplication Operations', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='upper left')
plt.grid(True, alpha=0.3, linestyle='--')

# Add text box with statistics
textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.2f} μs\nMAE = {mae:.2f} μs\nMAPE = {mape:.2f}%\nn = {len(x)}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.98, 0.02, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment='right', bbox=props)

# Make plot square-ish for better comparison
plt.tight_layout()

# Save the figure
output_file = 'latency_regression_scatter.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nScatter plot saved as: {output_file}")

# Show the plot
plt.show()

# Additional analysis: residuals
plt.figure(figsize=(12, 8))

residuals = y - y_pred

# Residual plot
plt.subplot(2, 1, 1)
plt.scatter(x, residuals, alpha=0.6, s=50, color='purple', edgecolors='black', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Latency (μs)', fontsize=12, fontweight='bold')
plt.ylabel('Residuals (μs)', fontsize=12, fontweight='bold')
plt.title('Residual Plot', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')

# Histogram of residuals
plt.subplot(2, 1, 2)
plt.hist(residuals, bins=20, color='purple', alpha=0.7, edgecolor='black')
plt.xlabel('Residuals (μs)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Distribution of Residuals', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
residual_file = 'latency_regression_residuals.png'
plt.savefig(residual_file, dpi=300, bbox_inches='tight')
print(f"Residual plots saved as: {residual_file}")

plt.show()

print("\nAnalysis complete!")




