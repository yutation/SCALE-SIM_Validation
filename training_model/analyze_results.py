"""
Analyze and visualize pipeline results.

This script provides additional analysis of the comparison data.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def analyze_comparison(csv_path: str):
    """Analyze comparison CSV and print detailed statistics."""
    df = pd.read_csv(csv_path)
    
    print("="*70)
    print("DETAILED RESULTS ANALYSIS")
    print("="*70)
    print(f"Comparison file: {csv_path}")
    print(f"Total test cases: {len(df)}")
    print()
    
    # Overall statistics
    print("="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    print(f"Mean Absolute Error (MAE):           {df['absolute_error'].mean():.6f}")
    print(f"Std Dev of Absolute Error:           {df['absolute_error'].std():.6f}")
    print(f"Mean Absolute Percentage Error:      {df['relative_error_pct'].mean():.2f}%")
    print(f"Median Absolute Percentage Error:    {df['relative_error_pct'].median():.2f}%")
    print(f"Std Dev of Relative Error:           {df['relative_error_pct'].std():.2f}%")
    print()
    
    # Percentiles
    print("="*70)
    print("ERROR PERCENTILES")
    print("="*70)
    print("Absolute Error:")
    for p in [25, 50, 75, 90, 95, 99]:
        val = np.percentile(df['absolute_error'], p)
        print(f"  {p:2d}th percentile: {val:.6f}")
    print()
    print("Relative Error (%):")
    for p in [25, 50, 75, 90, 95, 99]:
        val = np.percentile(df['relative_error_pct'], p)
        print(f"  {p:2d}th percentile: {val:.2f}%")
    print()
    
    # Size-based analysis
    print("="*70)
    print("ANALYSIS BY SIZE")
    print("="*70)
    
    # Create size bins
    df['size_bin'] = pd.cut(df['size'], bins=[0, 100, 1000, 10000, 100000, float('inf')],
                            labels=['<100', '100-1K', '1K-10K', '10K-100K', '>100K'])
    
    for bin_name in df['size_bin'].cat.categories:
        bin_df = df[df['size_bin'] == bin_name]
        if len(bin_df) > 0:
            print(f"\nSize: {bin_name} ({len(bin_df)} samples)")
            print(f"  MAE:        {bin_df['absolute_error'].mean():.6f}")
            print(f"  MAPE:       {bin_df['relative_error_pct'].mean():.2f}%")
            print(f"  Median APE: {bin_df['relative_error_pct'].median():.2f}%")
    print()
    
    # Best and worst predictions
    print("="*70)
    print("BEST PREDICTIONS (Lowest Relative Error)")
    print("="*70)
    best = df.nsmallest(5, 'relative_error_pct')
    for _, row in best.iterrows():
        print(f"Shape ({row['dim_0']}, {row['dim_1']}, {row['dim_2']}):")
        print(f"  Actual: {row['actual_latency']:.6f}, Predicted: {row['predicted_latency']:.6f}")
        print(f"  Error: {row['absolute_error']:.6f} ({row['relative_error_pct']:.2f}%)")
    print()
    
    print("="*70)
    print("WORST PREDICTIONS (Highest Relative Error)")
    print("="*70)
    worst = df.nlargest(5, 'relative_error_pct')
    for _, row in worst.iterrows():
        print(f"Shape ({row['dim_0']}, {row['dim_1']}, {row['dim_2']}):")
        print(f"  Actual: {row['actual_latency']:.6f}, Predicted: {row['predicted_latency']:.6f}")
        print(f"  Error: {row['absolute_error']:.6f} ({row['relative_error_pct']:.2f}%)")
    print()
    
    # Accuracy buckets
    print("="*70)
    print("ACCURACY DISTRIBUTION")
    print("="*70)
    buckets = [
        (0, 1, "< 1%"),
        (1, 5, "1-5%"),
        (5, 10, "5-10%"),
        (10, 20, "10-20%"),
        (20, float('inf'), "> 20%")
    ]
    
    for low, high, label in buckets:
        count = len(df[(df['relative_error_pct'] >= low) & (df['relative_error_pct'] < high)])
        pct = count / len(df) * 100
        print(f"{label:>10}: {count:3d} cases ({pct:5.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze pipeline comparison results"
    )
    parser.add_argument(
        "comparison_csv",
        type=str,
        help="Path to comparison CSV file"
    )
    
    args = parser.parse_args()
    
    if not Path(args.comparison_csv).exists():
        print(f"Error: File not found: {args.comparison_csv}")
        return
    
    analyze_comparison(args.comparison_csv)


if __name__ == "__main__":
    main()







