#!/usr/bin/env python3
"""
Visualization script for SCALE-Sim TPU latency verification results.

Generates:
1. Estimated vs. Measured TPU Latency (Scatter Plot)
2. Relative Error vs. GEMM Size
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from pathlib import Path

def parse_input_shapes(shape_str):
    """Parse input shapes string to extract M, N, K dimensions."""
    try:
        shapes = ast.literal_eval(shape_str)
        # For matmul: [(M, K), (K, N)]
        M, K1 = shapes[0]
        K2, N = shapes[1]
        assert K1 == K2, f"K dimensions don't match: {K1} != {K2}"
        return M, N, K1
    except Exception as e:
        print(f"Error parsing shape {shape_str}: {e}")
        return None, None, None

def calculate_flops(M, N, K):
    """Calculate total FLOPs for a GEMM operation."""
    # GEMM: C = A @ B where A is MxK, B is KxN
    # FLOPs = 2 * M * N * K (multiply-add operations)
    return 2 * M * N * K

def load_and_process_data(csv_path):
    """Load CSV and extract relevant metrics."""
    df = pd.read_csv(csv_path)
    
    # Parse shapes and calculate FLOPS
    dimensions = df['Input_Shapes'].apply(parse_input_shapes)
    df['M'] = dimensions.apply(lambda x: x[0])
    df['N'] = dimensions.apply(lambda x: x[1])
    df['K'] = dimensions.apply(lambda x: x[2])
    
    # Remove rows where parsing failed
    df = df.dropna(subset=['M', 'N', 'K'])
    
    # Calculate FLOPs
    df['FLOPs'] = df.apply(lambda row: calculate_flops(row['M'], row['N'], row['K']), axis=1)
    
    # Calculate problem size (M*N*K)
    df['Problem_Size'] = df['M'] * df['N'] * df['K']
    
    return df

def plot_estimated_vs_measured(df, output_path):
    """
    Primary plot: Estimated vs. Measured TPU Latency
    Shows prediction accuracy and bias.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data
    measured = df['Actual_Duration_us']
    estimated = df['Predicted_Latency_us']
    
    # Create scatter plot
    scatter = ax.scatter(measured, estimated, alpha=0.6, s=50, c='steelblue', edgecolors='black', linewidth=0.5)
    
    # Add y=x reference line (perfect prediction)
    min_val = min(measured.min(), estimated.min())
    max_val = max(measured.max(), estimated.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)', alpha=0.7)
    
    # Add ±10% error bands
    x_range = np.linspace(min_val, max_val, 100)
    ax.fill_between(x_range, x_range * 0.9, x_range * 1.1, alpha=0.2, color='green', label='±10% Error Band')
    
    # Calculate statistics
    mape = np.mean(np.abs(df['Error_Percentage']))
    median_error = np.median(df['Error_Percentage'])
    rmse = np.sqrt(np.mean((estimated - measured) ** 2))
    
    # Add statistics text box
    stats_text = f'MAPE: {mape:.2f}%\nMedian Error: {median_error:.2f}%\nRMSE: {rmse:.2f} μs\nN: {len(df)}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Labels and formatting
    ax.set_xlabel('Measured TPU Latency (μs)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Estimated Latency (SCALE-Sim TPU) (μs)', fontsize=12, fontweight='bold')
    ax.set_title('SCALE-Sim TPU Latency Prediction Accuracy', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Equal aspect ratio for better visualization
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_error_vs_size(df, output_path):
    """
    Secondary plot: Relative Error vs. GEMM Size
    Shows where the model struggles and robustness across sizes.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Extract data
    problem_size = df['Problem_Size']
    error_pct = df['Error_Percentage']
    flops = df['FLOPs']
    
    # Color by FLOPs magnitude for additional insight
    scatter = ax.scatter(problem_size, error_pct, c=np.log10(flops), 
                        cmap='viridis', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log₁₀(FLOPs)', fontsize=11, fontweight='bold')
    
    # Add reference lines
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction', alpha=0.7)
    ax.axhline(y=10, color='orange', linestyle=':', linewidth=1.5, label='±10% Error', alpha=0.6)
    ax.axhline(y=-10, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    
    # Highlight problematic regions
    high_error_threshold = 30
    high_error_points = df[np.abs(df['Error_Percentage']) > high_error_threshold]
    if len(high_error_points) > 0:
        ax.scatter(high_error_points['Problem_Size'], high_error_points['Error_Percentage'],
                  s=100, facecolors='none', edgecolors='red', linewidths=2, 
                  label=f'High Error (>±{high_error_threshold}%)')
    
    # Labels and formatting
    ax.set_xlabel('GEMM Problem Size (M × N × K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Error vs. GEMM Problem Size', fontsize=14, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Add statistics by size category
    small_size = problem_size.quantile(0.33)
    large_size = problem_size.quantile(0.67)
    
    small_error = df[problem_size <= small_size]['Error_Percentage'].abs().mean()
    medium_error = df[(problem_size > small_size) & (problem_size <= large_size)]['Error_Percentage'].abs().mean()
    large_error = df[problem_size > large_size]['Error_Percentage'].abs().mean()
    
    stats_text = f'Mean Absolute Error by Size:\n'
    stats_text += f'Small: {small_error:.2f}%\n'
    stats_text += f'Medium: {medium_error:.2f}%\n'
    stats_text += f'Large: {large_error:.2f}%'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_error_distribution(df, output_path):
    """
    Bonus plot: Error distribution histogram
    Shows the distribution of prediction errors.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of errors
    errors = df['Error_Percentage']
    ax1.hist(errors, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax1.axvline(x=errors.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.2f}%')
    ax1.axvline(x=errors.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {errors.median():.2f}%')
    ax1.set_xlabel('Relative Error (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot by size category
    problem_size = df['Problem_Size']
    small_size = problem_size.quantile(0.33)
    large_size = problem_size.quantile(0.67)
    
    df['Size_Category'] = pd.cut(problem_size, 
                                  bins=[0, small_size, large_size, float('inf')],
                                  labels=['Small', 'Medium', 'Large'])
    
    df.boxplot(column='Error_Percentage', by='Size_Category', ax=ax2)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Problem Size Category', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Relative Error (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Error Distribution by Problem Size', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.suptitle('')  # Remove automatic title
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def generate_summary_report(df, output_path):
    """Generate a text summary of the verification results."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SCALE-Sim TPU Latency Verification Summary\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total GEMM kernels evaluated: {len(df)}\n")
        f.write(f"Mean Absolute Percentage Error (MAPE): {np.mean(np.abs(df['Error_Percentage'])):.2f}%\n")
        f.write(f"Median Error: {np.median(df['Error_Percentage']):.2f}%\n")
        f.write(f"Standard Deviation: {np.std(df['Error_Percentage']):.2f}%\n")
        f.write(f"Min Error: {df['Error_Percentage'].min():.2f}%\n")
        f.write(f"Max Error: {df['Error_Percentage'].max():.2f}%\n\n")
        
        # Accuracy metrics
        within_10 = (np.abs(df['Error_Percentage']) <= 10).sum()
        within_20 = (np.abs(df['Error_Percentage']) <= 20).sum()
        within_30 = (np.abs(df['Error_Percentage']) <= 30).sum()
        
        f.write("Accuracy Breakdown:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Within ±10% error: {within_10}/{len(df)} ({100*within_10/len(df):.1f}%)\n")
        f.write(f"Within ±20% error: {within_20}/{len(df)} ({100*within_20/len(df):.1f}%)\n")
        f.write(f"Within ±30% error: {within_30}/{len(df)} ({100*within_30/len(df):.1f}%)\n\n")
        
        # Bias analysis
        overestimations = (df['Error_Percentage'] > 0).sum()
        underestimations = (df['Error_Percentage'] < 0).sum()
        
        f.write("Bias Analysis:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Overestimations: {overestimations}/{len(df)} ({100*overestimations/len(df):.1f}%)\n")
        f.write(f"Underestimations: {underestimations}/{len(df)} ({100*underestimations/len(df):.1f}%)\n")
        f.write(f"Mean signed error: {df['Error_Percentage'].mean():.2f}%\n\n")
        
        # Size-based analysis
        problem_size = df['Problem_Size']
        small_size = problem_size.quantile(0.33)
        large_size = problem_size.quantile(0.67)
        
        small_df = df[problem_size <= small_size]
        medium_df = df[(problem_size > small_size) & (problem_size <= large_size)]
        large_df = df[problem_size > large_size]
        
        f.write("Error by Problem Size:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Small (≤{small_size:.0f}): MAPE = {np.mean(np.abs(small_df['Error_Percentage'])):.2f}%\n")
        f.write(f"Medium ({small_size:.0f}-{large_size:.0f}): MAPE = {np.mean(np.abs(medium_df['Error_Percentage'])):.2f}%\n")
        f.write(f"Large (≥{large_size:.0f}): MAPE = {np.mean(np.abs(large_df['Error_Percentage'])):.2f}%\n\n")
        
        # Worst cases
        f.write("Top 5 Worst Predictions (by absolute error):\n")
        f.write("-" * 40 + "\n")
        worst_cases = df.nlargest(5, 'Error_Percentage', keep='all')[['Kernel_Name', 'Input_Shapes', 
                                                                        'Predicted_Latency_us', 'Actual_Duration_us', 
                                                                        'Error_Percentage']]
        for idx, row in worst_cases.iterrows():
            f.write(f"{row['Kernel_Name']}: {row['Input_Shapes']}\n")
            f.write(f"  Predicted: {row['Predicted_Latency_us']:.2f} μs, Actual: {row['Actual_Duration_us']:.2f} μs\n")
            f.write(f"  Error: {row['Error_Percentage']:.2f}%\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"✓ Saved: {output_path}")

def main():
    # Setup paths
    script_dir = Path(__file__).parent
    csv_path = script_dir / "merged_verification_results.csv"
    
    # Output directory
    output_dir = script_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("SCALE-Sim TPU Latency Verification Visualization")
    print("=" * 80)
    print(f"\nLoading data from: {csv_path}")
    
    # Load and process data
    df = load_and_process_data(csv_path)
    print(f"✓ Loaded {len(df)} GEMM kernels")
    
    # Generate plots
    print("\nGenerating plots...")
    print("-" * 40)
    
    # Primary plot
    plot_estimated_vs_measured(df, output_dir / "estimated_vs_measured.png")
    
    # Secondary plot
    plot_error_vs_size(df, output_dir / "error_vs_size.png")
    
    # Bonus plot
    plot_error_distribution(df, output_dir / "error_distribution.png")
    
    # Generate summary report
    print("\nGenerating summary report...")
    print("-" * 40)
    generate_summary_report(df, output_dir / "verification_summary.txt")
    
    print("\n" + "=" * 80)
    print("✓ All visualizations complete!")
    print(f"✓ Output directory: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()

