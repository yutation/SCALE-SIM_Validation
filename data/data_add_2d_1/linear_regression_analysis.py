#!/usr/bin/env python3
"""
Linear Regression Analysis: tuple_product vs avg_duration_us
Analyzes the relationship between tensor tuple product and average kernel duration
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
    
    print(f"X (tuple_product) range: [{X.min():.0f}, {X.max():.0f}]")
    print(f"y (avg_duration_us) range: [{y.min():.6f}, {y.max():.6f}]")
    
    return X, y, df_clean

def perform_linear_regression(X, y):
    """
    Perform linear regression analysis
    
    Args:
        X (array): Independent variable (tuple_product)
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

def create_visualizations(X, y, results, output_dir='.'):
    """
    Create visualizations for the regression analysis
    
    Args:
        X (array): Independent variable
        y (array): Dependent variable  
        results (dict): Regression results
        output_dir (str): Directory to save plots
    """
    print("\nCreating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Linear Regression Analysis: tuple_product vs avg_duration_us', fontsize=16, fontweight='bold')
    
    X_flat = X.flatten()
    y_pred = results['y_pred']
    
    # 1. Scatter plot with regression line
    ax1.scatter(X_flat, y, alpha=0.6, color='blue', s=30, label='Data points')
    ax1.plot(X_flat, y_pred, color='red', linewidth=2, label=f'Regression line (R² = {results["r_squared"]:.4f})')
    ax1.set_xlabel('tuple_product (Tensor Size)')
    ax1.set_ylabel('avg_duration_us (Average Duration)')
    ax1.set_title('Scatter Plot with Regression Line')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals plot
    residuals = y - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, color='green', s=30)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted avg_duration_us')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Q-Q plot for residuals normality
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot of Residuals (Normality Check)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Histogram of residuals
    ax4.hist(residuals, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Residuals')
    ax4.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f"""Regression Statistics:
    Slope: {results['slope']:.6f} ± {results['slope_ci']:.6f}
    Intercept: {results['intercept']:.6f}
    R²: {results['r_squared']:.4f}
    R: {results['r_value']:.4f}
    p-value: {results['p_value']:.2e}
    RMSE: {results['rmse']:.6f}
    N: {results['n_samples']}"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save plot
    plot_file = os.path.join(output_dir, 'linear_regression_analysis.png')
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
    print(f"avg_duration_us = {results['slope']:.6f} × tuple_product + {results['intercept']:.6f}")
    
    print(f"\nRegression Coefficients:")
    print(f"  Slope (β₁):           {results['slope']:.6f} ± {results['slope_ci']:.6f}")
    print(f"  Intercept (β₀):       {results['intercept']:.6f}")
    print(f"  Standard Error:       {results['std_error']:.6f}")
    
    print(f"\nModel Performance:")
    print(f"  R-squared (R²):       {results['r_squared']:.4f}")
    print(f"  Correlation (R):      {results['r_value']:.4f}")
    print(f"  Mean Squared Error:   {results['mse']:.6f}")
    print(f"  Root Mean Squared Error: {results['rmse']:.6f}")
    
    print(f"\nStatistical Significance:")
    print(f"  p-value:              {results['p_value']:.2e}")
    print(f"  Significance level:   {'***' if results['p_value'] < 0.001 else '**' if results['p_value'] < 0.01 else '*' if results['p_value'] < 0.05 else 'ns'}")
    
    print(f"\nSample Size:")
    print(f"  Number of observations: {results['n_samples']}")
    
    print(f"\nInterpretation:")
    if results['p_value'] < 0.05:
        direction = "increases" if results['slope'] > 0 else "decreases"
        print(f"  • There is a statistically significant relationship between tuple_product and avg_duration_us")
        print(f"  • For every unit increase in tuple_product, avg_duration_us {direction} by {abs(results['slope']):.6f} microseconds")
        print(f"  • The model explains {results['r_squared']*100:.1f}% of the variance in avg_duration_us")
    else:
        print(f"  • No statistically significant relationship found between tuple_product and avg_duration_us")
        print(f"  • The relationship could be due to random chance")
    
    if results['r_squared'] > 0.8:
        strength = "strong"
    elif results['r_squared'] > 0.5:
        strength = "moderate" 
    elif results['r_squared'] > 0.3:
        strength = "weak"
    else:
        strength = "very weak"
    
    print(f"  • The correlation is {strength} (R² = {results['r_squared']:.3f})")

def save_results_to_csv(X, y, results, output_file):
    """
    Save regression results to CSV file
    
    Args:
        X (array): Independent variable
        y (array): Dependent variable
        results (dict): Regression results
        output_file (str): Output CSV file path
    """
    print(f"\nSaving detailed results to {output_file}...")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'tuple_product': X.flatten(),
        'avg_duration_us_actual': y,
        'avg_duration_us_predicted': results['y_pred'],
        'residuals': y - results['y_pred'],
        'abs_residuals': np.abs(y - results['y_pred'])
    })
    
    # Add regression statistics as comments
    with open(output_file, 'w') as f:
        f.write(f"# Linear Regression Analysis Results\n")
        f.write(f"# Model: avg_duration_us = {results['slope']:.6f} * tuple_product + {results['intercept']:.6f}\n")
        f.write(f"# R-squared: {results['r_squared']:.4f}\n")
        f.write(f"# p-value: {results['p_value']:.2e}\n")
        f.write(f"# RMSE: {results['rmse']:.6f}\n")
        f.write(f"# Sample size: {results['n_samples']}\n")
        f.write("#\n")
    
    # Append the data
    results_df.to_csv(output_file, mode='a', index=False)
    print(f"Results saved to: {output_file}")

def main():
    """
    Main function to run the linear regression analysis
    """
    # File paths
    input_file = "kernel_statistics_filtered.csv"
    output_file = "linear_regression_results.csv"
    
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
        
        # Create visualizations
        create_visualizations(X, y, results)
        
        # Save results to CSV
        save_results_to_csv(X, y, results, output_file)
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
