#!/usr/bin/env python3
"""
Test script to verify the M, N, K extraction worked correctly.
"""

import pandas as pd
import os

def main():
    # Test the extracted file
    input_file = "validation/data_matmul_linear/kernel_report_updated_merged_02.csv"
    output_file = "validation/data_matmul_linear/kernel_report_with_mnk.csv"
    
    if not os.path.exists(output_file):
        print(f"Error: Output file {output_file} not found. Run extract_mnk.py first.")
        return
    
    print("=== Testing M, N, K Extraction ===")
    
    # Read the original and processed files
    df_original = pd.read_csv(input_file)
    df_processed = pd.read_csv(output_file)
    
    print(f"Original file shape: {df_original.shape}")
    print(f"Processed file shape: {df_processed.shape}")
    print(f"New columns added: {df_processed.shape[1] - df_original.shape[1]}")
    
    # Check that M, N, K columns exist
    required_cols = ['M', 'N', 'K']
    missing_cols = [col for col in required_cols if col not in df_processed.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return
    
    print(f"✓ All required columns found: {required_cols}")
    
    # Verify some extractions manually
    print("\n=== Manual Verification ===")
    
    test_cases = [
        ("matmul_128x128x128", 128, 128, 128),
        ("matmul_192x256x384", 192, 256, 384),
        ("matmul_512x512x512", 512, 512, 512),
        ("matmul_32x64x96", 32, 64, 96)
    ]
    
    for kernel_name, expected_M, expected_N, expected_K in test_cases:
        # Find the row in processed data
        row = df_processed[df_processed['kernel_name'].str.strip() == kernel_name]
        
        if len(row) == 0:
            print(f"❌ Could not find kernel: {kernel_name}")
            continue
        
        actual_M = row['M'].iloc[0]
        actual_N = row['N'].iloc[0]
        actual_K = row['K'].iloc[0]
        
        if actual_M == expected_M and actual_N == expected_N and actual_K == expected_K:
            print(f"✓ {kernel_name} -> M={actual_M}, N={actual_N}, K={actual_K}")
        else:
            print(f"❌ {kernel_name} -> Expected: M={expected_M}, N={expected_N}, K={expected_K}, Got: M={actual_M}, N={actual_N}, K={actual_K}")
    
    # Show statistics
    print("\n=== Statistics ===")
    print(f"Total rows: {len(df_processed)}")
    print(f"Rows with valid M, N, K: {len(df_processed.dropna(subset=['M', 'N', 'K']))}")
    print(f"Rows with missing M, N, K: {len(df_processed) - len(df_processed.dropna(subset=['M', 'N', 'K']))}")
    
    print(f"\nM range: {df_processed['M'].min()} to {df_processed['M'].max()}")
    print(f"N range: {df_processed['N'].min()} to {df_processed['N'].max()}")
    print(f"K range: {df_processed['K'].min()} to {df_processed['K'].max()}")
    
    # Show unique values
    print(f"\nUnique M values: {sorted(df_processed['M'].dropna().unique())}")
    print(f"Unique N values: {sorted(df_processed['N'].dropna().unique())}")
    print(f"Unique K values: {sorted(df_processed['K'].dropna().unique())}")
    
    # Check column order
    print("\n=== Column Order ===")
    kernel_name_idx = df_processed.columns.get_loc('kernel_name')
    m_idx = df_processed.columns.get_loc('M')
    n_idx = df_processed.columns.get_loc('N')
    k_idx = df_processed.columns.get_loc('K')
    
    print(f"kernel_name position: {kernel_name_idx}")
    print(f"M position: {m_idx}")
    print(f"N position: {n_idx}")
    print(f"K position: {k_idx}")
    
    if m_idx == kernel_name_idx + 1 and n_idx == kernel_name_idx + 2 and k_idx == kernel_name_idx + 3:
        print("✓ M, N, K columns are correctly positioned after kernel_name")
    else:
        print("❌ M, N, K columns are not in the expected position")
    
    # Show sample data
    print("\n=== Sample Data ===")
    sample_cols = ['kernel_name', 'M', 'N', 'K', 'total_cycles']
    print(df_processed[sample_cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()



