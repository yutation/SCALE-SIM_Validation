#!/usr/bin/env python3
"""
Example script showing how to merge the kernel report CSV files.

This script demonstrates different ways to merge the kernel report CSV files.
"""

import pandas as pd
from merge_csv_files_improved import merge_csv_files
import os

def main():
    # Define the input files
    file1 = "validation/data_matmul_linear/kernel_report_updated.csv"
    file2 = "validation/data_matmul_linear/kernel_report_updated_2.csv"
    
    # Check if files exist
    if not os.path.exists(file1):
        print(f"Error: {file1} not found")
        return
    if not os.path.exists(file2):
        print(f"Error: {file2} not found")
        return
    
    print("=== Example 1: Simple concatenation ===")
    print("This will combine all rows from both files, removing duplicates.")
    success = merge_csv_files(
        input_files=[file1, file2],
        output_file="merged_kernel_reports_concat.csv",
        merge_strategy='concat',
        drop_duplicates=True,
        sort_by='kernel_name'
    )
    
    if success:
        print("✓ Successfully created merged_kernel_reports_concat.csv")
    print()
    
    print("=== Example 2: Merge by kernel_name ===")
    print("This will merge files based on the kernel_name column (database-style join).")
    success = merge_csv_files(
        input_files=[file1, file2],
        output_file="merged_kernel_reports_join.csv",
        merge_strategy='merge',
        key_column='kernel_name',
        drop_duplicates=True,
        sort_by='kernel_name'
    )
    
    if success:
        print("✓ Successfully created merged_kernel_reports_join.csv")
    print()
    
    print("=== Example 3: Merge without dropping duplicates ===")
    print("This will keep all rows, including duplicates.")
    success = merge_csv_files(
        input_files=[file1, file2],
        output_file="merged_kernel_reports_all.csv",
        merge_strategy='concat',
        drop_duplicates=False,
        sort_by='kernel_name'
    )
    
    if success:
        print("✓ Successfully created merged_kernel_reports_all.csv")
    print()
    
    # Show some statistics
    print("=== File Statistics ===")
    for file_path in [file1, file2]:
        df = pd.read_csv(file_path)
        # Clean column names for statistics
        df.columns = df.columns.str.strip()
        print(f"{file_path}:")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {len(df.columns)}")
        print(f"  - Unique kernel names: {df['kernel_name'].nunique()}")
        print()

if __name__ == "__main__":
    main()
