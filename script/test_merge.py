#!/usr/bin/env python3
"""
Test script to verify the merged CSV files and show differences between merge strategies.
"""

import pandas as pd
import os

def main():
    # Test the merged files
    merged_files = [
        "merged_kernel_reports_improved.csv",
        "merged_kernel_reports_concat.csv", 
        "merged_kernel_reports_join.csv",
        "merged_kernel_reports_all.csv"
    ]
    
    print("=== Testing Merged Files ===")
    
    for file_path in merged_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"\n{file_path}:")
            print(f"  - Shape: {df.shape}")
            print(f"  - Columns: {len(df.columns)}")
            print(f"  - First few column names: {list(df.columns[:5])}")
            
            if 'kernel_name' in df.columns:
                unique_kernels = df['kernel_name'].nunique()
                total_kernels = len(df)
                print(f"  - Unique kernel names: {unique_kernels}")
                print(f"  - Total rows: {total_kernels}")
                
                if unique_kernels != total_kernels:
                    print(f"  - Duplicates: {total_kernels - unique_kernels}")
        else:
            print(f"\n{file_path}: File not found")
    
    # Show differences between merge strategies
    print("\n=== Merge Strategy Comparison ===")
    
    if os.path.exists("merged_kernel_reports_concat.csv") and os.path.exists("merged_kernel_reports_join.csv"):
        concat_df = pd.read_csv("merged_kernel_reports_concat.csv")
        join_df = pd.read_csv("merged_kernel_reports_join.csv")
        
        print(f"Concatenation strategy:")
        print(f"  - Shape: {concat_df.shape}")
        print(f"  - Columns: {len(concat_df.columns)}")
        
        print(f"\nJoin strategy:")
        print(f"  - Shape: {join_df.shape}")
        print(f"  - Columns: {len(join_df.columns)}")
        
        print(f"\nDifference:")
        print(f"  - Rows: {join_df.shape[0] - concat_df.shape[0]}")
        print(f"  - Columns: {join_df.shape[1] - concat_df.shape[1]}")
        
        if join_df.shape[1] > concat_df.shape[1]:
            extra_cols = set(join_df.columns) - set(concat_df.columns)
            print(f"  - Extra columns in join: {sorted(extra_cols)}")
    
    # Show sample data
    print("\n=== Sample Data ===")
    if os.path.exists("merged_kernel_reports_improved.csv"):
        df = pd.read_csv("merged_kernel_reports_improved.csv")
        print("First 3 rows of merged data:")
        print(df.head(3).to_string())
        
        print(f"\nLast 3 rows of merged data:")
        print(df.tail(3).to_string())

if __name__ == "__main__":
    main()



