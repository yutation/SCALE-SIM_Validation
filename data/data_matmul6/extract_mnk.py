#!/usr/bin/env python3
"""
Extract M, N, K values from kernel_name and create three new columns.

This script takes a CSV file with kernel_name column containing patterns like 'matmul_MxNxK'
and extracts the M, N, K values into separate columns.

Usage:
    python extract_mnk.py input.csv output.csv
    python extract_mnk.py kernel_report_updated_merged_02.csv kernel_report_with_mnk.csv
"""

import pandas as pd
import argparse
import re
import sys
import os


def extract_mnk_from_kernel_name(kernel_name):
    """
    Extract M, N, K values from kernel_name like 'matmul_MxNxK'.
    
    Args:
        kernel_name (str): Kernel name string
        
    Returns:
        tuple: (M, N, K) as integers, or (None, None, None) if pattern not found
    """
    if pd.isna(kernel_name):
        return None, None, None
    
    kernel_name = str(kernel_name).strip()
    
    # Pattern to match matmul_MxNxK
    pattern = r'matmul_(\d+)x(\d+)x(\d+)'
    match = re.match(pattern, kernel_name)
    
    if match:
        M = int(match.group(1))
        N = int(match.group(2))
        K = int(match.group(3))
        return M, N, K
    else:
        print(f"Warning: Could not extract M, N, K from kernel_name: '{kernel_name}'")
        return None, None, None


def process_csv_file(input_file, output_file, kernel_name_column='kernel_name'):
    """
    Process CSV file to extract M, N, K values and create new columns.
    
    Args:
        input_file (str): Input CSV file path
        output_file (str): Output CSV file path
        kernel_name_column (str): Name of the column containing kernel names
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    try:
        # Read the CSV file
        print(f"Reading {input_file}...")
        df = pd.read_csv(input_file)
        
        print(f"Original shape: {df.shape}")
        print(f"Original columns: {list(df.columns)}")
        
        # Clean column names (strip whitespace) first
        df.columns = df.columns.str.strip()
        print(f"Cleaned columns: {list(df.columns)}")
        
        # Check if kernel_name column exists (after cleaning)
        if kernel_name_column not in df.columns:
            print(f"Error: Column '{kernel_name_column}' not found in CSV file.")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Extract M, N, K values
        print("Extracting M, N, K values from kernel_name...")
        mnk_results = df[kernel_name_column].apply(extract_mnk_from_kernel_name)
        
        # Convert results to separate columns
        M_values = [result[0] for result in mnk_results]
        N_values = [result[1] for result in mnk_results]
        K_values = [result[2] for result in mnk_results]
        
        # Count successful extractions
        successful_extractions = sum(1 for m, n, k in mnk_results if m is not None and n is not None and k is not None)
        total_rows = len(df)
        
        print(f"Successfully extracted M, N, K from {successful_extractions}/{total_rows} rows")
        
        # Create new dataframe with M, N, K columns inserted after kernel_name
        print("Creating new dataframe with M, N, K columns...")
        
        # Get the position of kernel_name column
        kernel_name_idx = df.columns.get_loc(kernel_name_column)
        
        # Create new columns list
        new_columns = list(df.columns[:kernel_name_idx + 1])  # Include kernel_name
        new_columns.extend(['M', 'N', 'K'])  # Add M, N, K columns
        new_columns.extend(list(df.columns[kernel_name_idx + 1:]))  # Add remaining columns
        
        # Create new dataframe
        new_df = df.copy()
        new_df['M'] = M_values
        new_df['N'] = N_values
        new_df['K'] = K_values
        
        # Reorder columns to put M, N, K after kernel_name
        new_df = new_df[new_columns]
        
        print(f"New shape: {new_df.shape}")
        print(f"New columns: {list(new_df.columns)}")
        
        # Show some statistics
        print("\nM, N, K Statistics:")
        print(f"M range: {new_df['M'].min()} to {new_df['M'].max()}")
        print(f"N range: {new_df['N'].min()} to {new_df['N'].max()}")
        print(f"K range: {new_df['K'].min()} to {new_df['K'].max()}")
        
        # Show unique M, N, K values
        print(f"Unique M values: {sorted(new_df['M'].dropna().unique())}")
        print(f"Unique N values: {sorted(new_df['N'].dropna().unique())}")
        print(f"Unique K values: {sorted(new_df['K'].dropna().unique())}")
        
        # Save to output file
        print(f"\nSaving to {output_file}...")
        new_df.to_csv(output_file, index=False)
        print(f"Successfully created {output_file}")
        
        # Show sample of the result
        print("\nSample of processed data:")
        sample_cols = [kernel_name_column, 'M', 'N', 'K', 'total_cycles']
        available_cols = [col for col in sample_cols if col in new_df.columns]
        print(new_df[available_cols].head(5).to_string())
        
        return True
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract M, N, K values from kernel_name and create three new columns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_mnk.py input.csv output.csv
  python extract_mnk.py kernel_report_updated_merged_02.csv kernel_report_with_mnk.csv
  python extract_mnk.py data.csv output.csv --kernel-column operation_name
        """
    )
    
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('output_file', help='Output CSV file path')
    parser.add_argument('--kernel-column', default='kernel_name',
                       help='Name of the column containing kernel names (default: kernel_name)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)
    
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Kernel column: {args.kernel_column}")
    print()
    
    # Process the file
    success = process_csv_file(args.input_file, args.output_file, args.kernel_column)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
