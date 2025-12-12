#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import math
import sys
import os
import glob
from pathlib import Path

# Add the parent directory to sys.path to import linear_models
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
try:
    from linear_models import matmul_scale_sim_model
except ImportError:
    # If import fails, define the function directly
    def matmul_scale_sim_model(m: int, n: int, k: int, systolic_array_size: int = 128) -> int:
        v1 = (2*systolic_array_size + systolic_array_size + m - 2) * math.ceil(n / systolic_array_size) * math.ceil(k / systolic_array_size)
        m, n = n, m
        v2 = (2*systolic_array_size + systolic_array_size + m - 2) * math.ceil(n / systolic_array_size) * math.ceil(k / systolic_array_size)
        return min(v1, v2)

def add_scale_sim_column(df):
    """Add scale sim model column to the dataframe"""
    scale_sim_values = []
    
    for _, row in df.iterrows():
        m, n, k = int(row['dim_m']), int(row['dim_n']), int(row['dim_k'])
        scale_sim_value = matmul_scale_sim_model(m, n, k)
        scale_sim_values.append(scale_sim_value)
    
    df['scale_sim_cycles'] = scale_sim_values
    return df

def perform_regression_analysis(df):
    """Perform linear regression analysis"""
    # Features and target
    X = df[['scale_sim_cycles']].values
    y = df['fusion_avg'].values
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Print results
    print("=== Linear Regression Results ===")
    print(f"Coefficient (slope): {model.coef_[0]:.8f}")
    print(f"Intercept: {model.intercept_:.6f}")
    print(f"R² Score: {r2:.6f}")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Root Mean Squared Error: {rmse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"\nLinear Model Equation:")
    print(f"fusion_avg = {model.coef_[0]:.8f} * scale_sim_cycles + {model.intercept_:.6f}")
    
    return model, y_pred, r2, mse, mae, rmse

def create_visualizations(df, model, y_pred, output_filename='scale_sim_regression_analysis.png'):
    """Create visualization plots"""
    
    # Calculate statistics
    y_true = df['fusion_avg'].values
    residuals = y_true - y_pred
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot with regression line
    ax1 = axes[0, 0]
    ax1.scatter(df['scale_sim_cycles'], df['fusion_avg'], alpha=0.6, s=50, label='Data')
    ax1.plot(df['scale_sim_cycles'], y_pred, 'r-', linewidth=2, label='Regression Line')
    ax1.set_xlabel('Scale Sim Cycles', fontsize=11)
    ax1.set_ylabel('Fusion Average (μs)', fontsize=11)
    ax1.set_title('Scale Sim Cycles vs Fusion Average', fontsize=12, fontweight='bold')
    
    # Add regression equation and statistics
    equation_text = f'y = {model.coef_[0]:.6f}x + {model.intercept_:.6f}'
    stats_text = f'R² = {r2:.6f}\nRMSE = {rmse:.6f}\nMAE = {mae:.6f}'
    
    # Position text box in upper left
    textstr = f'{equation_text}\n{stats_text}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Residual plot
    ax2 = axes[0, 1]
    ax2.scatter(y_pred, residuals, alpha=0.6, s=50, c='blue')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Line')
    ax2.set_xlabel('Predicted Values', fontsize=11)
    ax2.set_ylabel('Residuals', fontsize=11)
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    
    # Add residual statistics
    residual_stats = f'Mean: {np.mean(residuals):.6f}\nStd: {np.std(residuals):.6f}'
    ax2.text(0.05, 0.95, residual_stats, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Actual vs Predicted
    ax3 = axes[1, 0]
    min_val = min(df['fusion_avg'].min(), y_pred.min())
    max_val = max(df['fusion_avg'].max(), y_pred.max())
    ax3.scatter(df['fusion_avg'], y_pred, alpha=0.6, s=50, c='green', label='Predictions')
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('Actual Values', fontsize=11)
    ax3.set_ylabel('Predicted Values', fontsize=11)
    ax3.set_title('Actual vs Predicted Values', fontsize=12, fontweight='bold')
    
    # Add R² and correlation
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    pred_stats = f'R² = {r2:.6f}\nCorr = {corr:.6f}\nn = {len(y_true)}'
    ax3.text(0.05, 0.95, pred_stats, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribution of residuals
    ax4 = axes[1, 1]
    n, bins, patches = ax4.hist(residuals, bins=30, alpha=0.7, edgecolor='black', color='purple')
    ax4.set_xlabel('Residuals', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    
    # Add normal distribution overlay
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    ax4.axvline(mu, color='r', linestyle='--', linewidth=2, label=f'Mean: {mu:.6f}')
    
    # Add distribution statistics
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)
    dist_stats = f'Skewness: {skewness:.4f}\nKurtosis: {kurtosis:.4f}'
    ax4.text(0.65, 0.95, dist_stats, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # Add overall title with key statistics
    fig.suptitle(f'Linear Regression Analysis (R² = {r2:.4f}, RMSE = {rmse:.4f})', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved as: {output_filename}")

def analyze_by_dimension_groups(df):
    """Analyze regression by different dimension groups"""
    print("\n=== Analysis by Dimension Groups ===")
    
    # Group by M dimension
    m_groups = df.groupby('dim_m')
    print("\nAnalysis by M dimension:")
    for m_val, group in m_groups:
        if len(group) >= 5:  # Only analyze groups with sufficient data
            X = group[['scale_sim_cycles']].values
            y = group['fusion_avg'].values
            model = LinearRegression()
            model.fit(X, y)
            r2 = r2_score(y, model.predict(X))
            print(f"M={m_val}: R²={r2:.4f}, Slope={model.coef_[0]:.8f}, Intercept={model.intercept_:.4f}")
    
    # Group by N dimension
    n_groups = df.groupby('dim_n')
    print("\nAnalysis by N dimension:")
    for n_val, group in n_groups:
        if len(group) >= 5:
            X = group[['scale_sim_cycles']].values
            y = group['fusion_avg'].values
            model = LinearRegression()
            model.fit(X, y)
            r2 = r2_score(y, model.predict(X))
            print(f"N={n_val}: R²={r2:.4f}, Slope={model.coef_[0]:.8f}, Intercept={model.intercept_:.4f}")
    
    # Group by K dimension
    k_groups = df.groupby('dim_k')
    print("\nAnalysis by K dimension:")
    for k_val, group in k_groups:
        if len(group) >= 5:
            X = group[['scale_sim_cycles']].values
            y = group['fusion_avg'].values
            model = LinearRegression()
            model.fit(X, y)
            r2 = r2_score(y, model.predict(X))
            print(f"K={k_val}: R²={r2:.4f}, Slope={model.coef_[0]:.8f}, Intercept={model.intercept_:.4f}")

def generate_output_filenames(input_file):
    """
    Generate output filenames based on input filename.
    
    Args:
        input_file (str): Path to input CSV file
        
    Returns:
        tuple: (output_csv_path, output_png_path)
    """
    input_path = Path(input_file)
    base_name = input_path.stem
    
    # Generate output CSV filename
    if 'fusion_statistics_report' in base_name:
        output_csv_name = base_name.replace('fusion_statistics_report', 'fusion_statistics_report_with_scale_sim')
    else:
        output_csv_name = f'{base_name}_with_scale_sim'
    
    # Generate output PNG filename
    if 'fusion_statistics_report' in base_name:
        # Extract the suffix (e.g., _128, _1024, _4096)
        suffix = base_name.replace('fusion_statistics_report', '')
        output_png_name = f'scale_sim_regression_analysis{suffix}'
    else:
        output_png_name = f'scale_sim_regression_analysis_{base_name}'
    
    return f'{output_csv_name}.csv', f'{output_png_name}.png'


def process_single_file(csv_path):
    """
    Process a single input file.
    
    Args:
        csv_path (str): Path to the input CSV file
        
    Returns:
        tuple: (df, model) or (None, None) on error
    """
    print(f"\n{'='*60}")
    print(f"Loading data from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows")
        print("Original columns:", list(df.columns))
        
        # Add scale sim column
        print("\nAdding scale sim cycles column...")
        df = add_scale_sim_column(df)
        print("Scale sim column added successfully!")
        
        # Display basic statistics
        print("\n=== Basic Statistics ===")
        print(f"Scale Sim Cycles - Min: {df['scale_sim_cycles'].min()}, Max: {df['scale_sim_cycles'].max()}")
        print(f"Fusion Average - Min: {df['fusion_avg'].min():.6f}, Max: {df['fusion_avg'].max():.6f}")
        
        # Generate output filenames
        output_csv_path, output_png_path = generate_output_filenames(csv_path)
        
        # Save updated CSV
        df.to_csv(output_csv_path, index=False)
        print(f"\nUpdated CSV saved as: {output_csv_path}")
        
        # Perform regression analysis
        print("\nPerforming regression analysis...")
        model, y_pred, r2, mse, mae, rmse = perform_regression_analysis(df)
        
        # Create visualizations
        print("\nCreating visualizations...")
        create_visualizations(df, model, y_pred, output_png_path)
        
        # Analyze by dimension groups
        analyze_by_dimension_groups(df)
        
        # Show correlation matrix
        print("\n=== Correlation Analysis ===")
        correlation_cols = ['dim_m', 'dim_n', 'dim_k', 'scale_sim_cycles', 'fusion_avg']
        corr_matrix = df[correlation_cols].corr()
        print("\nCorrelation Matrix:")
        print(corr_matrix.round(4))
        
        # Display sample of data with new column
        print("\n=== Sample Data with Scale Sim Column ===")
        sample_cols = ['kernel_name', 'dim_m', 'dim_n', 'dim_k', 'scale_sim_cycles', 'fusion_avg']
        print(df[sample_cols].head(10))
        
        return df, model
        
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found!")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main function to process fusion statistics and add scale sim analysis."""
    # Check for command-line arguments
    if len(sys.argv) > 1:
        # Use provided file paths
        input_files = sys.argv[1:]
    else:
        # Default: find all fusion_statistics_report*.csv files in current directory
        input_files = sorted(glob.glob('fusion_statistics_report_*.csv'))
        
        if not input_files:
            print("Error: No fusion_statistics_report_*.csv files found in current directory!")
            print("Usage: python add_scale_sim_analysis.py [input_file1.csv input_file2.csv ...]")
            return
    
    print(f"Found {len(input_files)} file(s) to process")
    
    # Process each file
    success_count = 0
    results = []
    
    for input_file in input_files:
        df, model = process_single_file(input_file)
        if df is not None and model is not None:
            success_count += 1
            results.append((input_file, df, model))
    
    print(f"\n{'='*60}")
    print(f"Processing complete: {success_count}/{len(input_files)} file(s) successfully processed")
    
    return results


if __name__ == "__main__":
    results = main()
