#!/usr/bin/env python3
"""
Improved CSV File Merger

This script merges multiple CSV files with the same structure.
It handles column name inconsistencies and provides various merge options.

Usage:
    python merge_csv_files_improved.py file1.csv file2.csv file3.csv -o output.csv
    python merge_csv_files_improved.py *.csv -o merged_output.csv
"""

import argparse
import pandas as pd
import sys
import os
from pathlib import Path


def clean_column_names(df):
    """Clean column names by stripping whitespace and standardizing them."""
    df.columns = df.columns.str.strip()
    return df


def merge_csv_files(input_files, output_file, merge_strategy='concat', key_column=None, 
                   drop_duplicates=True, sort_by=None, ascending=True, clean_columns=True):
    """
    Merge multiple CSV files into a single output file.
    
    Args:
        input_files (list): List of input CSV file paths
        output_file (str): Output CSV file path
        merge_strategy (str): 'concat' for simple concatenation, 'merge' for database-style merge
        key_column (str): Column to use as key for merge strategy (required if merge_strategy='merge')
        drop_duplicates (bool): Whether to drop duplicate rows
        sort_by (str): Column to sort by after merging
        ascending (bool): Sort order (True for ascending, False for descending)
        clean_columns (bool): Whether to clean column names (strip whitespace)
    """
    
    if not input_files:
        print("Error: No input files provided.")
        return False
    
    # Check if all input files exist
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' does not exist.")
            return False
    
    try:
        # Read all CSV files
        dataframes = []
        for file_path in input_files:
            print(f"Reading {file_path}...")
            df = pd.read_csv(file_path)
            
            # Clean column names if requested
            if clean_columns:
                df = clean_column_names(df)
                print(f"  - Cleaned columns: {list(df.columns)}")
            
            print(f"  - Shape: {df.shape}")
            print(f"  - Original columns: {list(df.columns)}")
            dataframes.append(df)
        
        # Verify all dataframes have the same columns
        if len(dataframes) > 1:
            first_columns = set(dataframes[0].columns)
            for i, df in enumerate(dataframes[1:], 1):
                current_columns = set(df.columns)
                if first_columns != current_columns:
                    print(f"Warning: Column mismatch between files:")
                    print(f"  File 1 columns: {sorted(first_columns)}")
                    print(f"  File {i+1} columns: {sorted(current_columns)}")
                    print(f"  Missing in file {i+1}: {first_columns - current_columns}")
                    print(f"  Extra in file {i+1}: {current_columns - first_columns}")
                    
                    # Try to fix by using only common columns
                    common_columns = first_columns.intersection(current_columns)
                    if len(common_columns) > 0:
                        print(f"  Using common columns: {sorted(common_columns)}")
                        dataframes[0] = dataframes[0][list(common_columns)]
                        dataframes[i] = dataframes[i][list(common_columns)]
                    else:
                        print("Error: No common columns found between files.")
                        return False
        
        # Merge dataframes based on strategy
        if merge_strategy == 'concat':
            print("Using concatenation strategy...")
            merged_df = pd.concat(dataframes, ignore_index=True)
        elif merge_strategy == 'merge':
            if not key_column:
                print("Error: key_column is required for merge strategy.")
                return False
            if key_column not in dataframes[0].columns:
                print(f"Error: Key column '{key_column}' not found in CSV files.")
                return False
            
            print(f"Using merge strategy with key column '{key_column}'...")
            merged_df = dataframes[0]
            for df in dataframes[1:]:
                merged_df = pd.merge(merged_df, df, on=key_column, how='outer')
        else:
            print(f"Error: Unknown merge strategy '{merge_strategy}'.")
            return False
        
        print(f"Merged dataframe shape: {merged_df.shape}")
        
        # Drop duplicates if requested
        if drop_duplicates:
            print("Dropping duplicates...")
            initial_rows = len(merged_df)
            merged_df = merged_df.drop_duplicates()
            final_rows = len(merged_df)
            print(f"  - Removed {initial_rows - final_rows} duplicate rows")
        
        # Sort if requested
        if sort_by:
            if sort_by in merged_df.columns:
                print(f"Sorting by '{sort_by}'...")
                merged_df = merged_df.sort_values(by=sort_by, ascending=ascending)
            else:
                print(f"Warning: Sort column '{sort_by}' not found in merged data.")
        
        # Save to output file
        print(f"Saving to {output_file}...")
        merged_df.to_csv(output_file, index=False)
        print(f"Successfully merged {len(input_files)} files into {output_file}")
        print(f"Final shape: {merged_df.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error during merge: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple CSV files with the same structure (improved version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_csv_files_improved.py file1.csv file2.csv file3.csv -o output.csv
  python merge_csv_files_improved.py *.csv -o merged_output.csv
  python merge_csv_files_improved.py file1.csv file2.csv -o output.csv --merge-strategy merge --key-column kernel_name
  python merge_csv_files_improved.py *.csv -o output.csv --sort-by kernel_name --no-drop-duplicates
        """
    )
    
    parser.add_argument('input_files', nargs='+', help='Input CSV files to merge')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file path')
    parser.add_argument('--merge-strategy', choices=['concat', 'merge'], default='concat',
                       help='Merge strategy: concat (simple concatenation) or merge (database-style merge)')
    parser.add_argument('--key-column', help='Column to use as key for merge strategy')
    parser.add_argument('--no-drop-duplicates', action='store_true',
                       help='Do not drop duplicate rows')
    parser.add_argument('--sort-by', help='Column to sort by after merging')
    parser.add_argument('--descending', action='store_true',
                       help='Sort in descending order (default is ascending)')
    parser.add_argument('--no-clean-columns', action='store_true',
                       help='Do not clean column names (strip whitespace)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.merge_strategy == 'merge' and not args.key_column:
        print("Error: --key-column is required when using merge strategy.")
        sys.exit(1)
    
    # Expand wildcards in input files
    expanded_files = []
    for pattern in args.input_files:
        if '*' in pattern or '?' in pattern:
            import glob
            expanded_files.extend(glob.glob(pattern))
        else:
            expanded_files.append(pattern)
    
    if not expanded_files:
        print("Error: No files found matching the provided patterns.")
        sys.exit(1)
    
    # Remove duplicates from file list
    expanded_files = list(set(expanded_files))
    
    print(f"Input files: {expanded_files}")
    print(f"Output file: {args.output}")
    print(f"Merge strategy: {args.merge_strategy}")
    if args.key_column:
        print(f"Key column: {args.key_column}")
    print(f"Drop duplicates: {not args.no_drop_duplicates}")
    print(f"Clean columns: {not args.no_clean_columns}")
    if args.sort_by:
        print(f"Sort by: {args.sort_by} ({'descending' if args.descending else 'ascending'})")
    print()
    
    # Perform the merge
    success = merge_csv_files(
        input_files=expanded_files,
        output_file=args.output,
        merge_strategy=args.merge_strategy,
        key_column=args.key_column,
        drop_duplicates=not args.no_drop_duplicates,
        sort_by=args.sort_by,
        ascending=not args.descending,
        clean_columns=not args.no_clean_columns
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
