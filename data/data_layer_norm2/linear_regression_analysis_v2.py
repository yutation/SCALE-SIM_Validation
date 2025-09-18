#!/usr/bin/env python3
"""
Linear Regression Analysis: dim_product vs avg_duration_us
Analyzes the relationship between dimension product and average kernel duration for Layer Norm kernels
Based on the reference analysis structure but adapted for layer norm kernel data
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
    Load and prepare data for linear regression analysis
    
    Args:
        csv_file (str): Path to the CSV file
        
    Returns:
        tuple: (X, y, df) where X is dim_product, y is avg_duration_us, df is the dataframe
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for required columns
    if 'dim_product' not in df.columns or 'avg_duration_us' not in df.columns:
        raise ValueError("Required columns 'dim_product' and 'avg_duration_us' not found in CSV")
    
    # Remove any rows with NaN values in the columns of interest
    df_clean = df.dropna(subset=['dim_product', 'avg_duration_us'])
    print(f"After removing NaN values: {len(df_clean)} rows (removed {len(df) - len(df_clean)} rows)")
    
    # Extract X and y variables
    X = df_clean['dim_product'].values.reshape(-1, 1)  # reshape for sklearn
    y = df_clean['avg_duration_us'].values
    
    print(f"X (dim_product) range: [{X.min():.0f}, {X.max():.0f}]")
    print(f"y (avg_duration_us) range: [{y.min():.6f}, {y.max():.6f}]")
    
    return X, y, df_clean

def perform_linear_regression(X, y):
    """
    Perform linear regression analysis
    
    Args:
        X (array): Independent variable (dim_product)
        y (array): Dependent variable (avg_duration_us)
        
    Returns:
        dict: Dictionary containing regression results
    """
    print("\nPerforming linear regression analysis...")
    
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

def create_visualizations(X, y, results, df, output_dir='.'):
    """
    Create visualizations for the regression analysis
    
    Args:
        X (array): Independent variable
        y (array): Dependent variable  
        results (dict): Regression results
        df (DataFrame): Original dataframe for additional plotting
        output_dir (str): Directory to save plots
    """
    print("\nCreating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Linear Regression Analysis: dim_product vs avg_duration_us (Layer Norm Kernels)', 
                 fontsize=16, fontweight='bold')
    
    X_flat = X.flatten()
    y_pred = results['y_pred']
    
    # 1. Scatter plot with regression line - colored by batch size
    scatter = ax1.scatter(X_flat, y, alpha=0.7, c=df['batch_size'], cmap='viridis', s=50)
    ax1.plot(X_flat, y_pred, color='red', linewidth=3, label=f'Regression line (R² = {results["r_squared"]:.4f})')
    ax1.set_xlabel('dim_product (batch_size × seq_len × hidden_dim)', fontsize=12)
    ax1.set_ylabel('avg_duration_us (Average Duration)', fontsize=12)
    ax1.set_title('Scatter Plot with Regression Line\n(colored by batch size)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for batch size
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Batch Size', fontsize=10)
    
    # Format x-axis with scientific notation for large numbers
    ax1.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # 2. Residuals plot
    residuals = y - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.7, color='green', s=40)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted avg_duration_us', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Residuals Plot\n(Actual - Predicted)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Q-Q plot for residuals normality
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot of Residuals\n(Normality Check)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance by batch size box plot
    df_with_pred = df.copy()
    df_with_pred['predicted_duration'] = y_pred
    df_with_pred['residuals'] = residuals
    
    batch_sizes = sorted(df['batch_size'].unique())
    batch_data = [df[df['batch_size'] == bs]['avg_duration_us'].values for bs in batch_sizes]
    
    bp = ax4.boxplot(batch_data, tick_labels=batch_sizes, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(batch_sizes)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_xlabel('Batch Size', fontsize=12)
    ax4.set_ylabel('avg_duration_us', fontsize=12)
    ax4.set_title('Performance Distribution\nby Batch Size', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f"""Regression Statistics:
Slope: {results['slope']:.8f} ± {results['slope_ci']:.8f}
Intercept: {results['intercept']:.4f}
R²: {results['r_squared']:.6f}
R: {results['r_value']:.6f}
p-value: {results['p_value']:.2e}
RMSE: {results['rmse']:.4f} μs
N: {results['n_samples']} kernels

Interpretation:
• {results['r_squared']*100:.2f}% of variance explained
• Performance scales linearly with size
• Each unit increase in dim_product
  adds {results['slope']*1e6:.3f} nanoseconds"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    
    # Save plot
    plot_file = os.path.join(output_dir, 'linear_regression_analysis_v2.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    
    # Show plot
    plt.show()
    
    return fig

def print_detailed_results(results):
    """
    Print detailed regression analysis results
    
    Args:
        results (dict): Regression results
    """
    print("\n" + "="*80)
    print("DETAILED LINEAR REGRESSION ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nModel Equation:")
    print(f"avg_duration_us = {results['slope']:.8f} × dim_product + {results['intercept']:.4f}")
    
    print(f"\nRegression Coefficients:")
    print(f"  Slope (β₁):           {results['slope']:.8f} ± {results['slope_ci']:.8f}")
    print(f"  Intercept (β₀):       {results['intercept']:.4f}")
    print(f"  Standard Error:       {results['std_error']:.8f}")
    
    print(f"\nModel Performance:")
    print(f"  R-squared (R²):       {results['r_squared']:.6f}")
    print(f"  Correlation (R):      {results['r_value']:.6f}")
    print(f"  Mean Squared Error:   {results['mse']:.4f}")
    print(f"  Root Mean Squared Error: {results['rmse']:.4f} μs")
    
    print(f"\nStatistical Significance:")
    print(f"  p-value:              {results['p_value']:.2e}")
    print(f"  Significance level:   {'***' if results['p_value'] < 0.001 else '**' if results['p_value'] < 0.01 else '*' if results['p_value'] < 0.05 else 'ns'}")
    
    print(f"\nSample Size:")
    print(f"  Number of observations: {results['n_samples']}")
    
    print(f"\nInterpretation:")
    if results['p_value'] < 0.05:
        direction = "increases" if results['slope'] > 0 else "decreases"
        print(f"  • There is a statistically significant relationship between dim_product and avg_duration_us")
        print(f"  • For every unit increase in dim_product, avg_duration_us {direction} by {abs(results['slope']):.8f} microseconds")
        print(f"  • This equals {abs(results['slope'])*1e6:.3f} nanoseconds per additional element")
        print(f"  • The model explains {results['r_squared']*100:.2f}% of the variance in avg_duration_us")
    else:
        print(f"  • No statistically significant relationship found between dim_product and avg_duration_us")
        print(f"  • The relationship could be due to random chance")
    
    if results['r_squared'] > 0.95:
        strength = "excellent"
    elif results['r_squared'] > 0.8:
        strength = "strong"
    elif results['r_squared'] > 0.5:
        strength = "moderate" 
    elif results['r_squared'] > 0.3:
        strength = "weak"
    else:
        strength = "very weak"
    
    print(f"  • The correlation is {strength} (R² = {results['r_squared']:.4f})")
    
    # Performance implications
    print(f"\nPerformance Implications:")
    print(f"  • Base overhead: {results['intercept']:.2f} μs (fixed cost)")
    print(f"  • Scaling rate: {results['slope']*1e6:.3f} ns per tensor element")
    print(f"  • Highly predictable performance (R² > 0.99)" if results['r_squared'] > 0.99 else f"  • Reasonably predictable performance")

def save_results_to_csv(X, y, results, df, output_file):
    """
    Save regression results to CSV file
    
    Args:
        X (array): Independent variable
        y (array): Dependent variable
        results (dict): Regression results
        df (DataFrame): Original dataframe
        output_file (str): Output CSV file path
    """
    print(f"\nSaving detailed results to {output_file}...")
    
    # Create results dataframe with original data
    results_df = df.copy()
    results_df['avg_duration_us_predicted'] = results['y_pred']
    results_df['residuals'] = y - results['y_pred']
    results_df['abs_residuals'] = np.abs(y - results['y_pred'])
    results_df['relative_error_pct'] = (results_df['residuals'] / y) * 100
    
    # Add regression statistics as comments
    with open(output_file, 'w') as f:
        f.write(f"# Linear Regression Analysis Results - Layer Norm Kernels\n")
        f.write(f"# Model: avg_duration_us = {results['slope']:.8f} * dim_product + {results['intercept']:.4f}\n")
        f.write(f"# R-squared: {results['r_squared']:.6f}\n")
        f.write(f"# Correlation: {results['r_value']:.6f}\n")
        f.write(f"# p-value: {results['p_value']:.2e}\n")
        f.write(f"# RMSE: {results['rmse']:.4f} μs\n")
        f.write(f"# Sample size: {results['n_samples']}\n")
        f.write(f"# Performance scaling: {results['slope']*1e6:.3f} ns per tensor element\n")
        f.write("#\n")
    
    # Append the data
    results_df.to_csv(output_file, mode='a', index=False)
    print(f"Results saved to: {output_file}")

def analyze_performance_by_dimensions(df, results):
    """
    Analyze performance patterns by different dimensions
    
    Args:
        df (DataFrame): Original dataframe
        results (dict): Regression results
    """
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS BY DIMENSIONS")
    print("="*60)
    
    # Performance by batch size
    print(f"\nPerformance by Batch Size:")
    batch_stats = df.groupby('batch_size').agg({
        'avg_duration_us': ['count', 'mean', 'std', 'min', 'max'],
        'dim_product': ['mean', 'min', 'max']
    }).round(3)
    print(batch_stats.to_string())
    
    # Performance by hidden dimension
    print(f"\nPerformance by Hidden Dimension:")
    hidden_stats = df.groupby('hidden_dim').agg({
        'avg_duration_us': ['count', 'mean', 'std'],
        'batch_size': ['min', 'max']
    }).round(3)
    print(hidden_stats.to_string())
    
    # Find best and worst performing configurations
    print(f"\nBest Performing Configurations (lowest duration):")
    best_configs = df.nsmallest(5, 'avg_duration_us')[['kernel_name', 'batch_size', 'seq_len', 'hidden_dim', 'avg_duration_us']]
    print(best_configs.to_string(index=False))
    
    print(f"\nWorst Performing Configurations (highest duration):")
    worst_configs = df.nlargest(5, 'avg_duration_us')[['kernel_name', 'batch_size', 'seq_len', 'hidden_dim', 'avg_duration_us']]
    print(worst_configs.to_string(index=False))

def main():
    """
    Main function to run the linear regression analysis
    """
    # File paths
    input_file = "kernel_analysis.csv"
    output_file = "linear_regression_results_v2.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        return
    
    try:
        # Load and prepare data
        X, y, df = load_and_prepare_data(input_file)
        
        # Perform linear regression
        results = perform_linear_regression(X, y)
        
        # Print detailed results
        print_detailed_results(results)
        
        # Analyze performance by dimensions
        analyze_performance_by_dimensions(df, results)
        
        # Create visualizations
        create_visualizations(X, y, results, df)
        
        # Save results to CSV
        save_results_to_csv(X, y, results, df, output_file)
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"Generated files:")
        print(f"  • linear_regression_analysis_v2.png (comprehensive visualization)")
        print(f"  • {output_file} (detailed results with predictions)")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
