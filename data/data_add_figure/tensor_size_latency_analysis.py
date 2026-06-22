#!/usr/bin/env python3
"""
Tensor Size vs Average Latency Analysis
Creates separate visualizations for 1D and 2D tensor operations
with linear regression analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import os

def load_and_prepare_data(csv_file):
    """
    Load and prepare data for analysis
    
    Args:
        csv_file (str): Path to the CSV file
        
    Returns:
        tuple: (X, y, df) where X is tuple_product, y is avg_duration_us, df is the dataframe
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for required columns
    if 'tuple_product' not in df.columns or 'avg_duration_us' not in df.columns:
        raise ValueError("Required columns 'tuple_product' and 'avg_duration_us' not found in CSV")
    
    # Remove any rows with NaN values in the columns of interest
    df_clean = df.dropna(subset=['tuple_product', 'avg_duration_us'])
    print(f"After removing NaN values: {len(df_clean)} rows (removed {len(df) - len(df_clean)} rows)")
    
    # Extract X and y variables
    X = df_clean['tuple_product'].values.reshape(-1, 1)  # reshape for sklearn
    y = df_clean['avg_duration_us'].values
    
    print(f"X (tuple_product/tensor size) range: [{X.min():.0f}, {X.max():.0f}]")
    print(f"y (avg_duration_us/latency) range: [{y.min():.6f}, {y.max():.6f}]")
    
    return X, y, df_clean

def perform_linear_regression(X, y):
    """
    Perform linear regression analysis
    
    Args:
        X (array): Independent variable (tuple_product/tensor size)
        y (array): Dependent variable (avg_duration_us/latency)
        
    Returns:
        dict: Dictionary containing regression results
    """
    print("Performing linear regression analysis...")
    
    # Sklearn linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate statistics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Scipy stats for additional statistics
    X_flat = X.flatten()
    slope, intercept, r_value, p_value, std_err = stats.linregress(X_flat, y)
    
    # Confidence intervals (95%)
    n = len(X)
    t_val = stats.t.ppf(0.975, n-2)  # 95% confidence interval
    slope_ci = std_err * t_val
    
    results = {
        'model': model,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r2,
        'r_value': r_value,
        'p_value': p_value,
        'std_error': std_err,
        'slope_ci': slope_ci,
        'mse': mse,
        'rmse': rmse,
        'y_pred': y_pred,
        'n_samples': n
    }
    
    return results

def create_single_plot(X, y, results, title, output_file):
    """
    Create a single plot for tensor size vs latency with linear regression
    
    Args:
        X (array): Independent variable (tensor size)
        y (array): Dependent variable (latency)
        results (dict): Regression results
        title (str): Plot title
        output_file (str): Output file path
    """
    print(f"Creating visualization for {title}...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    X_flat = X.flatten()
    y_pred = results['y_pred']
    
    # Scatter plot with regression line
    ax.scatter(X_flat, y, alpha=0.6, color='blue', s=50, label='Data points', edgecolors='black', linewidth=0.5)
    ax.plot(X_flat, y_pred, color='red', linewidth=2.5, label=f'Linear regression (R² = {results["r_squared"]:.4f})')
    
    # Labels and title
    ax.set_xlabel('Tensor Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Latency (μs)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add regression equation and statistics text box
    equation = f'y = {results["slope"]:.2e}·x + {results["intercept"]:.4f}'
    stats_text = f"""Linear Regression Analysis
    
Equation: {equation}

Statistics:
  • R² = {results['r_squared']:.4f}
  • R = {results['r_value']:.4f}
  • p-value = {results['p_value']:.2e}
  • RMSE = {results['rmse']:.4f} μs
  • Slope = {results['slope']:.6f} ± {results['slope_ci']:.6f}
  • Intercept = {results['intercept']:.4f} μs
  • N = {results['n_samples']} samples"""
    
    # Position the text box
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=11, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.85, edgecolor='black', linewidth=1.5))
    
    # Interpretation text
    if results['p_value'] < 0.05:
        direction = "increases" if results['slope'] > 0 else "decreases"
        interpretation = f"Significant correlation: Latency {direction} with tensor size"
    else:
        interpretation = "No significant correlation found"
    
    ax.text(0.5, -0.12, interpretation, transform=ax.transAxes,
            fontsize=12, ha='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Show plot
    plt.show()
    
    return fig

def print_results_summary(data_type, results):
    """
    Print summary of regression analysis results
    
    Args:
        data_type (str): Type of data (1D or 2D)
        results (dict): Regression results
    """
    print("\n" + "="*80)
    print(f"{data_type} DATA - LINEAR REGRESSION SUMMARY")
    print("="*80)
    
    print(f"\nModel Equation:")
    print(f"  avg_latency = {results['slope']:.6f} × tensor_size + {results['intercept']:.6f}")
    
    print(f"\nKey Statistics:")
    print(f"  R² (coefficient of determination): {results['r_squared']:.4f}")
    print(f"  p-value: {results['p_value']:.2e}")
    print(f"  RMSE: {results['rmse']:.6f} μs")
    print(f"  Sample size: {results['n_samples']}")
    
    if results['p_value'] < 0.05:
        print(f"\n  ✓ Statistically significant relationship (p < 0.05)")
        print(f"  ✓ The model explains {results['r_squared']*100:.1f}% of the variance")
    else:
        print(f"\n  ✗ No statistically significant relationship found")

def main():
    """
    Main function to run the analysis for both 1D and 2D data
    """
    # File paths
    file_1d = "kernel_statistics_filtered_1d.csv"
    file_2d = "kernel_statistics_filtered_2d.csv"
    output_1d = "tensor_size_vs_latency_1d.png"
    output_2d = "tensor_size_vs_latency_2d.png"
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if input files exist
    file_1d_path = os.path.join(script_dir, file_1d)
    file_2d_path = os.path.join(script_dir, file_2d)
    
    if not os.path.exists(file_1d_path):
        print(f"Error: Input file '{file_1d}' not found in {script_dir}!")
        return
    
    if not os.path.exists(file_2d_path):
        print(f"Error: Input file '{file_2d}' not found in {script_dir}!")
        return
    
    try:
        # Process 1D data
        print("\n" + "="*80)
        print("PROCESSING 1D TENSOR DATA")
        print("="*80)
        X_1d, y_1d, df_1d = load_and_prepare_data(file_1d_path)
        results_1d = perform_linear_regression(X_1d, y_1d)
        print_results_summary("1D", results_1d)
        output_1d_path = os.path.join(script_dir, output_1d)
        create_single_plot(X_1d, y_1d, results_1d, 
                          "1D Tensor Elementwise Addition: Tensor Size vs Average Latency",
                          output_1d_path)
        
        # Process 2D data
        print("\n" + "="*80)
        print("PROCESSING 2D TENSOR DATA")
        print("="*80)
        X_2d, y_2d, df_2d = load_and_prepare_data(file_2d_path)
        results_2d = perform_linear_regression(X_2d, y_2d)
        print_results_summary("2D", results_2d)
        output_2d_path = os.path.join(script_dir, output_2d)
        create_single_plot(X_2d, y_2d, results_2d,
                          "2D Tensor Elementwise Addition: Tensor Size vs Average Latency",
                          output_2d_path)
        
        # Comparison summary
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(f"\n1D Tensors:")
        print(f"  • Tensor size range: {X_1d.min():.0f} - {X_1d.max():.0f}")
        print(f"  • Latency range: {y_1d.min():.4f} - {y_1d.max():.4f} μs")
        print(f"  • R² = {results_1d['r_squared']:.4f}")
        print(f"  • Slope = {results_1d['slope']:.6f} μs per element")
        
        print(f"\n2D Tensors:")
        print(f"  • Tensor size range: {X_2d.min():.0f} - {X_2d.max():.0f}")
        print(f"  • Latency range: {y_2d.min():.4f} - {y_2d.max():.4f} μs")
        print(f"  • R² = {results_2d['r_squared']:.4f}")
        print(f"  • Slope = {results_2d['slope']:.6f} μs per element")
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"\nGenerated files:")
        print(f"  • {output_1d}")
        print(f"  • {output_2d}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

