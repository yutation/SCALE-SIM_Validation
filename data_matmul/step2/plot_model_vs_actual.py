#!/usr/bin/env python3
"""
Generate a single scatter plot showing model_output vs actual_duration.
Different colors are used for each (K, N) combination.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

# Set matplotlib style for academic paper quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def parse_input_shapes(shape_str):
    """
    Parse input shapes string to extract M, K, N values.
    Example: "[(32, 128), (128, 128)]" -> M=32, K=128, N=128
    """
    matches = re.findall(r'\d+', shape_str)
    if len(matches) >= 4:
        M = int(matches[0])
        K = int(matches[1])
        N = int(matches[3])
        return M, K, N
    return None, None, None

def load_and_process_data(csv_path):
    """Load CSV and extract M, K, N, actual_duration, model_output."""
    df = pd.read_csv(csv_path)
    
    # Parse shapes
    shapes = df['Input_Shapes'].apply(parse_input_shapes)
    df['M'] = shapes.apply(lambda x: x[0])
    df['K'] = shapes.apply(lambda x: x[1])
    df['N'] = shapes.apply(lambda x: x[2])
    
    # Rename columns for clarity
    df['actual_duration'] = df['Actual_Duration_us']
    df['model_output'] = df['Model_Output']
    
    # Remove any rows with missing values
    df = df.dropna(subset=['M', 'K', 'N', 'actual_duration', 'model_output'])
    
    return df

def main():
    # Get script directory
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'merged_verification_results_with_model2.csv'
    output_path = script_dir / 'model_vs_actual_scatter.png'
    
    print(f"Loading data from: {csv_path}")
    df = load_and_process_data(csv_path)
    
    print(f"Data loaded: {len(df)} rows")
    print(f"M range: {df['M'].min()} to {df['M'].max()}")
    print(f"K values: {sorted(df['K'].unique())}")
    print(f"N values: {sorted(df['N'].unique())}")
    
    # Define (K, N) combinations and colors
    kn_combinations = [
        (128, 128),
        (128, 1024),
        (1024, 128),
        (1024, 1024)
    ]
    
    # Define colors for each combination (using distinct, colorblind-friendly palette)
    colors = {
        (128, 128): '#1f77b4',    # Blue
        (128, 1024): '#ff7f0e',   # Orange
        (1024, 128): '#2ca02c',   # Green
        (1024, 1024): '#d62728'   # Red
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each (K, N) combination with different color
    for K, N in kn_combinations:
        # Filter data for this (K, N) combination
        mask = (df['K'] == K) & (df['N'] == N)
        subset = df[mask].copy()
        
        if len(subset) == 0:
            print(f"Warning: No data for K={K}, N={N}")
            continue
        
        # Sort by model_output for better visualization
        subset = subset.sort_values('model_output')
        
        print(f"Plotting K={K}, N={N}: {len(subset)} points")
        
        # Scatter plot
        ax.scatter(subset['model_output'], subset['actual_duration'], 
                  color=colors[(K, N)], alpha=0.6, s=20, 
                  label=f'K={K}, N={N}', edgecolors='none')
    
    # Add labels and title
    ax.set_xlabel('Model Output', fontweight='bold')
    ax.set_ylabel('Actual Duration (μs)', fontweight='bold')
    ax.set_title('Model Output vs Actual Duration for Different GEMM Configurations', 
                fontweight='bold', pad=15)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add legend
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='black')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    print(f"\nSaving figure to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure save