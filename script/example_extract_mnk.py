#!/usr/bin/env python3
"""
Example script showing how to use extract_mnk.py with different options.

This script demonstrates various ways to extract M, N, K values from kernel names.
"""

import pandas as pd
import os
from extract_mnk import process_csv_file

def main():
    print("=== M, N, K Extraction Examples ===\n")
    
    # Example 1: Basic usage
    print("Example 1: Basic extraction")
    print("python extract_mnk.py input.csv output.csv")
    print()
    
    # Example 2: Custom kernel column name
    print("Example 2: Custom kernel column name")
    print("python extract_mnk.py data.csv output.csv --kernel-column operation_name")
    print()
    
    # Example 3: Using the function directly
    print("Example 3: Using the function directly in Python code")
    
    input_file = "validation/data_matmul_linear/kernel_report_updated_merged_02.csv"
    output_file = "validation/data_matmul_linear/kernel_report_with_mnk_direct.csv"
    
    if os.path.exists(input_file):
        print(f"Processing {input_file}...")
        success = process_csv_file(input_file, output_file)
        
        if success:
            print(f"✓ Successfully created {output_file}")
            
            # Show some analysis
            df = pd.read_csv(output_file)
            print(f"\nData shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Show some interesting patterns
            print("\n=== Matrix Size Analysis ===")
            
            # Count by M size
            m_counts = df['M'].value_counts().sort_index()
            print("Count by M dimension:")
            for m, count in m_counts.items():
                print(f"  M={m}: {count} kernels")
            
            # Count by N size
            n_counts = df['N'].value_counts().sort_index()
            print("\nCount by N dimension:")
            for n, count in n_counts.items():
                print(f"  N={n}: {count} kernels")
            
            # Count by K size
            k_counts = df['K'].value_counts().sort_index()
            print("\nCount by K dimension:")
            for k, count in k_counts.items():
                print(f"  K={k}: {count} kernels")
            
            # Show some performance analysis
            print("\n=== Performance Analysis ===")
            
            # Average cycles by matrix size
            avg_cycles_by_size = df.groupby(['M', 'N', 'K'])['total_cycles'].mean().sort_values(ascending=False)
            print("Top 10 most expensive matrix multiplications (by average cycles):")
            for (m, n, k), cycles in avg_cycles_by_size.head(10).items():
                print(f"  {m}x{n}x{k}: {cycles:.1f} cycles")
            
            # Find square matrices
            square_matrices = df[(df['M'] == df['N']) & (df['N'] == df['K'])]
            print(f"\nSquare matrices (M=N=K): {len(square_matrices)} kernels")
            
            if len(square_matrices) > 0:
                print("Square matrix sizes:")
                for size in sorted(square_matrices['M'].unique()):
                    print(f"  {size}x{size}x{size}")
            
        else:
            print("❌ Failed to process file")
    else:
        print(f"❌ Input file {input_file} not found")
    
    print("\n=== Usage Tips ===")
    print("1. The script automatically handles column name cleaning (strips whitespace)")
    print("2. M, N, K columns are inserted right after the kernel_name column")
    print("3. The script works with any CSV file that has kernel names in 'matmul_MxNxK' format")
    print("4. You can customize the kernel column name using --kernel-column option")
    print("5. The script provides detailed statistics about the extraction process")

if __name__ == "__main__":
    main()



