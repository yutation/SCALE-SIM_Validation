#!/usr/bin/env python3
"""
Detect significant increases/decreases (anomalies) in actual_duration for each (K, N) combination.
Identifies sudden jumps or drops in the performance measurements.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

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

def detect_anomalies(data, K, N, threshold_percentile=98):
    """
    Detect anomalies (sudden jumps/drops) in actual_duration.
    
    Args:
        data: DataFrame for a specific (K, N) combination
        K: K dimension value
        N: N dimension value
        threshold_percentile: Percentile threshold for detecting anomalies
    
    Returns:
        DataFrame with detected anomalies
    """
    # Sort by M
    data = data.sort_values('M').reset_index(drop=True)
    
    # Calculate differences between consecutive measurements
    data['duration_diff'] = data['actual_duration'].diff()
    data['duration_pct_change'] = data['actual_duration'].pct_change() * 100
    
    # Calculate absolute differences for jump detection
    data['abs_diff'] = data['duration_diff'].abs()
    data['abs_pct_change'] = data['duration_pct_change'].abs()
    
    # Define thresholds based on statistics (less sensitive)
    # Method 1: Absolute difference threshold (98th percentile instead of 95th)
    diff_threshold = np.percentile(data['abs_diff'].dropna(), threshold_percentile)
    
    # Method 2: Percentage change threshold (98th percentile)
    pct_threshold = np.percentile(data['abs_pct_change'].dropna(), threshold_percentile)
    
    # Method 3: Z-score based detection (more robust, higher threshold)
    mean_diff = data['abs_diff'].mean()
    std_diff = data['abs_diff'].std()
    z_threshold = 3.5  # Standard deviations (increased from 2.5 to 3.5)
    
    # Detect anomalies using stricter criteria (AND instead of OR for some conditions)
    # Only flag if it meets BOTH percentile thresholds OR exceeds z-score significantly
    anomalies = data[
        ((data['abs_diff'] > diff_threshold) & (data['abs_pct_change'] > pct_threshold)) |
        (data['abs_diff'] > mean_diff + z_threshold * std_diff)
    ].copy()
    
    # Add context: previous and current values
    anomalies['prev_M'] = anomalies['M'] - 2  # Step is 2
    anomalies['prev_duration'] = data.loc[anomalies.index - 1, 'actual_duration'].values if len(anomalies) > 0 else []
    
    # Classify as increase or decrease
    anomalies['change_type'] = anomalies['duration_diff'].apply(
        lambda x: 'INCREASE' if x > 0 else 'DECREASE'
    )
    
    return anomalies, diff_threshold, pct_threshold

def print_anomaly_report(anomalies, K, N, diff_threshold, pct_threshold):
    """Print a formatted report of detected anomalies."""
    print(f"\n{'='*80}")
    print(f"K={K}, N={N}")
    print(f"{'='*80}")
    print(f"Thresholds: Absolute diff > {diff_threshold:.4f} μs, "
          f"Percentage change > {pct_threshold:.2f}%")
    print(f"Detected {len(anomalies)} anomalies:\n")
    
    if len(anomalies) == 0:
        print("  No significant anomalies detected.")
        return
    
    for idx, row in anomalies.iterrows():
        print(f"  M = {int(row['M']):4d}  [{row['change_type']}]")
        print(f"    Previous (M={int(row['prev_M']):4d}): {row['prev_duration']:.4f} μs")
        print(f"    Current  (M={int(row['M']):4d}): {row['actual_duration']:.4f} μs")
        print(f"    Absolute change: {row['duration_diff']:+.4f} μs")
        print(f"    Percentage change: {row['duration_pct_change']:+.2f}%")
        print()

def save_anomalies_to_csv(all_anomalies, output_path):
    """Save all detected anomalies to a CSV file."""
    if len(all_anomalies) == 0:
        print("\nNo anomalies to save.")
        return
    
    # Combine all anomalies
    combined = pd.concat(all_anomalies, ignore_index=True)
    
    # Select relevant columns
    output_df = combined[[
        'K', 'N', 'M', 'prev_M', 
        'prev_duration', 'actual_duration', 
        'duration_diff', 'duration_pct_change',
        'change_type'
    ]].copy()
    
    # Rename for clarity
    output_df.columns = [
        'K', 'N', 'M_anomaly', 'M_previous',
        'Duration_Previous_us', 'Duration_Current_us',
        'Absolute_Change_us', 'Percentage_Change',
        'Change_Type'
    ]
    
    # Sort by K, N, M
    output_df = output_df.sort_values(['K', 'N', 'M_anomaly'])
    
    # Save to CSV
    output_df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\nAnomalies saved to: {output_path}")
    print(f"Total anomalies detected: {len(output_df)}")

def main():
    # Get script directory
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'merged_verification_results_with_model.csv'
    output_path = script_dir / 'detected_anomalies.csv'
    
    print("="*80)
    print("ANOMALY DETECTION IN GEMM PERFORMANCE DATA")
    print("="*80)
    print(f"\nLoading data from: {csv_path}")
    
    df = load_and_process_data(csv_path)
    
    print(f"Data loaded: {len(df)} rows")
    print(f"M range: {df['M'].min()} to {df['M'].max()}")
    print(f"K values: {sorted(df['K'].unique())}")
    print(f"N values: {sorted(df['N'].unique())}")
    
    # Define (K, N) combinations
    kn_combinations = [
        (128, 128),
        (128, 1024),
        (1024, 128),
        (1024, 1024)
    ]
    
    all_anomalies = []
    
    # Detect anomalies for each (K, N) combination
    for K, N in kn_combinations:
        # Filter data
        mask = (df['K'] == K) & (df['N'] == N)
        subset = df[mask].copy()
        
        if len(subset) == 0:
            print(f"\nWarning: No data for K={K}, N={N}")
            continue
        
        # Detect anomalies
        anomalies, diff_threshold, pct_threshold = detect_anomalies(subset, K, N)
        
        if len(anomalies) > 0:
            anomalies['K'] = K
            anomalies['N'] = N
            all_anomalies.append(anomalies)
        
        # Print report
        print_anomaly_report(anomalies, K, N, diff_threshold, pct_threshold)
    
    # Save all anomalies to CSV
    save_anomalies_to_csv(all_anomalies, output_path)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for K, N in kn_combinations:
        mask = (df['K'] == K) & (df['N'] == N)
        subset = df[mask].copy()
        if len(subset) > 0:
            subset_sorted = subset.sort_values('M')
            subset_sorted['diff'] = subset_sorted['actual_duration'].diff().abs()
            max_jump_idx = subset_sorted['diff'].idxmax()
            max_jump_row = subset_sorted.loc[max_jump_idx]
            
            print(f"\nK={K}, N={N}:")
            print(f"  Duration range: {subset['actual_duration'].min():.4f} - "
                  f"{subset['actual_duration'].max():.4f} μs")
            print(f"  Largest absolute jump: {subset_sorted['diff'].max():.4f} μs at M={int(max_jump_row['M'])}")

if __name__ == '__main__':
    main()
