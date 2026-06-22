#!/usr/bin/env python3
"""
Generate 2x2 subplot visualization of GEMM performance results.
Each subplot shows actual_duration and model_output vs M for different (K, N) combinations.
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

def create_subplot(ax, data, K, N, subplot_position):
    """
    Create a single subplot with dual y-axes.
    
    Args:
        ax: matplotlib axis object
        data: filtered dataframe for this (K, N) combination
        K: K dimension value
        N: N dimension value
        subplot_position: string like '(a)', '(b)', etc.
    """
    # Sort by M for proper line plotting
    data = data.sort_values('M')
    
    M_values = data['M'].values
    actual = data['actual_duration'].values
    model = data['model_output'].values
    
    # Left y-axis: actual_duration (blue)
    color_actual = '#1f77b4'  # Blue
    ax.set_xlabel('M', fontweight='bold')
    ax.set_ylabel('Actual Duration (μs)', color=color_actual, fontweight='bold')
    line1 = ax.plot(M_values, actual, '-', color=color_actual, linewidth=1.5, 
                    label='Actual Duration', alpha=0.8)
    ax.tick_params(axis='y', labelcolor=color_actual)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Right y-axis: model_output (red)
    ax2 = ax.twinx()
    color_model = '#d62728'  # Red
    ax2.set_ylabel('Model Output', color=color_model, fontweight='bold')
    line2 = ax2.plot(M_values, model, '--', color=color_model, linewidth=1.5, 
                     label='Model Output', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color_model)
    
    # Title
    ax.set_title(f'K={K}, N={N}', fontweight='bold', pad=10)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', framealpha=0.9)
    
    # Set x-axis limits to show full range
    ax.set_xlim(M_values.min() - 10, M_values.max() + 10)
    
    return ax, ax2

def main():
    # Get script directory
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'merged_verification_results_with_model2.csv'
    output_path = script_dir / 'gemm_performance_comparison2.png'
    
    print(f"Loading data from: {csv_path}")
    df = load_and_process_data(csv_path)
    
    print(f"Data loaded: {len(df)} rows")
    print(f"M range: {df['M'].min()} to {df['M'].max()}")
    print(f"K values: {sorted(df['K'].unique())}")
    print(f"N values: {sorted(df['N'].unique())}")
    
    # Define (K, N) combinations for subplots
    kn_combinations = [
        (128, 128),
        (128, 1024),
        (1024, 128),
        (1024, 1024)
    ]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('GEMM Performance: Actual Duration vs Model Output', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Create each subplot
    for idx, (K, N) in enumerate(kn_combinations):
        ax = axes_flat[idx]
        
        # Filter data for this (K, N) combination
        mask = (df['K'] == K) & (df['N'] == N)
        subset = df[mask].copy()
        
        if len(subset) == 0:
            print(f"Warning: No data for K={K}, N={N}")
            ax.text(0.5, 0.5, f'No data for K={K}, N={N}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        print(f"Plotting K={K}, N={N}: {len(subset)} points")
        create_subplot(ax, subset, K, N, f'({chr(97+idx)})')
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    print(f"Saving figure to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved successfully!")
    
    # Also save as PDF for publication quality
    # pdf_path = script_dir / 'gemm_performance_comparison.pdf'
    # plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    # print(f"PDF version saved to: {pdf_path}")
    
    plt.close()

if __name__ == '__main__':
    main()
