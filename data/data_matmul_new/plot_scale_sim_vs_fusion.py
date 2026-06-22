#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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

def create_plot(df, output_filename='scale_sim_vs_fusion.png'):
    """Create scatter plot with regression line"""
    
    # Prepare data for regression
    X = df[['scale_sim_cycles']].values
    y = df['fusion_avg'].values
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot with regression line
    ax.scatter(df['scale_sim_cycles'], df['fusion_avg'], alpha=0.6, s=80, label='Data', color='steelblue')
    ax.plot(df['scale_sim_cycles'], y_pred, 'r-', linewidth=3, label='Regression Line')
    ax.set_xlabel('SCALE-Sim Cycles', fontsize=18)
    ax.set_ylabel('Average Latency (μs)', fontsize=18)
    ax.set_title('SCALE-Sim Cycles vs Average Latency', fontsize=20, fontweight='bold')
    
    # Add regression equation and statistics
    equation_text = f'y = {model.coef_[0]:.6f}x + {model.intercept_:.6f}'
    stats_text = f'R² = {r2:.6f}\nRMSE = {rmse:.6f}\nMAE = {mae:.6f}\nn = {len(y)}'
    
    # Position text box in upper left
    textstr = f'{equation_text}\n{stats_text}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props, family='monospace')
    
    ax.legend(loc='lower right', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n=== Regression Results ===")
    print(f"Coefficient (slope): {model.coef_[0]:.8f}")
    print(f"Intercept: {model.intercept_:.6f}")
    print(f"R² Score: {r2:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"\nPlot saved as: {output_filename}")

def generate_output_filename(input_file):
    """Generate output filename based on input filename"""
    input_path = Path(input_file)
    base_name = input_path.stem
    
    # Extract suffix if it exists (e.g., _128, _1024, _4096)
    if 'fusion_statistics_report' in base_name:
        suffix = base_name.replace('fusion_statistics_report', '')
        suffix = suffix.replace('_with_scale_sim', '')
        output_name = f'scale_sim_vs_fusion{suffix}'
    else:
        output_name = f'scale_sim_vs_fusion_{base_name}'
    
    return f'{output_name}.png'

def process_file(csv_path):
    """Process a single input file"""
    print(f"\n{'='*60}")
    print(f"Loading data from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows")
        
        # Add scale sim column if not present
        if 'scale_sim_cycles' not in df.columns:
            print("Adding scale sim cycles column...")
            df = add_scale_sim_column(df)
        else:
            print("Using existing scale_sim_cycles column...")
        
        # Generate output filename
        output_png = generate_output_filename(csv_path)
        
        # Create plot
        print("\nCreating plot...")
        create_plot(df, output_png)
        
        return True
        
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found!")
        return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    # Check for command-line arguments
    if len(sys.argv) > 1:
        # Use provided file paths
        input_files = sys.argv[1:]
    else:
        # Default: find all fusion_statistics_report*.csv files
        patterns = ['fusion_statistics_report_*.csv', 'fusion_statistics_report_with_scale_sim_*.csv']
        input_files = []
        for pattern in patterns:
            input_files.extend(glob.glob(pattern))
        
        # Remove duplicates and sort
        input_files = sorted(set(input_files))
        
        if not input_files:
            print("Error: No fusion_statistics_report*.csv files found in current directory!")
            print("Usage: python plot_scale_sim_vs_fusion.py [input_file1.csv input_file2.csv ...]")
            return
    
    print(f"Found {len(input_files)} file(s) to process")
    
    # Process each file
    success_count = 0
    for input_file in input_files:
        if process_file(input_file):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete: {success_count}/{len(input_files)} file(s) successfully processed")

if __name__ == "__main__":
    main()

