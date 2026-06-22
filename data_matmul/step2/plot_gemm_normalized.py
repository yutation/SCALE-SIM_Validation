#!/usr/bin/env python3
"""
Generate 2x2 subplot visualization of normalized GEMM performance results.
Each subplot shows normalized actual_duration and model_output vs M for different (K, N) combinations.
Values are normalized independently within each (K, N) group.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

# Set matplotlib style for academic paper quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def parse_input_shapes(shape_str):
    """
    Parse input shapes string to extract M, K, N values.
    Example: "[(32, 128), (128, 128)]" -> M=32, K=128, N=128
    """
    # Extract numbers from the string
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

def normalize_within_group(df, K, N):
    """
    Normalize actual_duration and model_output within a (K, N) group.
    Normalization: value_normalized = value / max(value in that group)
    
    Args:
        df: dataframe with all data
        K: K dimension value
        N: N dimension value
    
    Returns:
        subset dataframe with normalized values
    """
    # Filter data for this (K, N) combination
    mask = (df['K'] == K) & (df['N'] == N)
    subset = df[mask].copy()
    
    if len(subset) == 0:
        return subset
    
    # Normalize independently
    max_actual = subset['actual_duration'].max()
    max_model = subset['model_output'].max()
    
    subset['actual_normalized'] = subset['actual_duration'] / max_actual
    subset['model_normalized'] = subset['model_output'] / max_model
    
    print(f"  K={K}, N={N}: max_actual={max_actual:.2f}, max_model={max_model:.2f}")
    
    return subset

def create_subplot(ax, data, K, N):
    """
    Create a single subplot with normalized values.
    
    Args:
        ax: matplotlib axis object
        data: filtered and normalized dataframe for this (K, N) combination
        K: K dimension value
        N: N dimension value
    """
    # Sort by M for proper line plotting
    data = data.sort_values('M')
    
    M_values = data['M'].values
    actual_norm = data['actual_normalized'].values
    model_norm = data['model_normalized'].values
    
    # Plot both normalized values on the same axis
    color_actual = '#1f77b4'  # Blue
    color_model = '#d62728'   # Red
    
    line1 = ax.plot(M_values, actual_norm, '-', color=color_actual, linewidth=1.5, 
                    label='Actual Duration (normalized)', alpha=0.8)
    line2 = ax.plot(M_values, model_norm, '--', color=color_model, linewidth=1.5, 
                    label='Model Output (normalized)', alpha=0.8, dashes=(5, 3))
    
    # Formatting
    ax.set_xlabel('M', fontweight='bold')
    ax.set_ylabel('Normalized Latency', fontweight='bold')
    ax.set_title(f'K={K}, N={N}', fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Set y-axis range [0, 1.05]
    ax.set_ylim(0, 1.05)
    
    # Set x-axis limits to show full range
    ax.set_xlim(M_values.min() - 10, M_values.max() + 10)
    
    # Legend
    ax.legend(loc='best', framealpha=0.9)
    
    return ax

def main():
    # Get script directory
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'merged_verification_results_with_model.csv'
    output_path = script_dir / 'gemm_performance_normalized.png'
    
    print(f"Loading data from: {csv_path}")
    df = load_and_process_data(csv_path)
    
    print(f"Data loaded: {len(df)} rows")
    print(f"M range: {df['M'].min()} to {df['M'].max()}")
    print(f"K values: {sorted(df['K'].unique())}")
    print(f"N values: {sorted(df['N'].unique())}")
    print("\nNormalizing data within each (K, N) group:")
    
    # Define (K, N) combinations for subplots
    kn_combinations = [
        (128, 128),
        (128, 1024),
        (1024, 128),
        (1024, 1024)
    ]
    
    # Normalize data for each group
    normalized_data = {}
    for K, N in kn_combinations:
        normalized_data[(K, N)] = normalize_within_group(df, K, N)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('GEMM Performance: Normalized Actual Duration vs Model Output', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Create each subplot
    print("\nGenerating subplots:")
    for idx, (K, N) in enumerate(kn_combinations):
        ax = axes_flat[idx]
        subset = normalized_data[(K, N)]
        
        if len(subset) == 0:
            print(f"  Warning: No data for K={K}, N={N}")
            ax.text(0.5, 0.5, f'No data for K={K}, N={N}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        print(f"  Plotting K={K}, N={N}: {len(subset)} points")
        create_subplot(ax, subset, K, N)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    print(f"\nSaving figure to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved successfully!")
    
    # Also save as PDF for publication quality
    pdf_path = script_dir / 'gemm_performance_normalized.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"PDF version saved to: {pdf_path}")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics (Normalized Values):")
    print("="*60)
    for K, N in kn_combinations:
        subset = normalized_data[(K, N)]
        if len(subset) > 0:
            print(f"\nK={K}, N={N}:")
            print(f"  Actual Duration (normalized): min={subset['actual_normalized'].min():.4f}, "
                  f"max={subset['actual_normalized'].max():.4f}, "
                  f"mean={subset['actual_normalized'].mean():.4f}")
            print(f"  Model Output (normalized):    min={subset['model_normalized'].min():.4f}, "
                  f"max={subset['model_normalized'].max():.4f}, "
                  f"mean={subset['model_normalized'].mean():.4f}")

if __name__ == '__main__':
    main()
