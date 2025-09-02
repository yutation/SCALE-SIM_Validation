#!/usr/bin/env python3
"""
Script to extract specific columns from kernel report CSV file.
Keeps only: kernel_name, M, N, K, total_cycles, and fusion-related columns.
"""

import pandas as pd
import sys
import os

def extract_fusion_columns(input_file, output_file=None):
    """
    Extract only the specified columns from the CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
    """
    
    # Define the columns to keep
    columns_to_keep = [
        'kernel_name',
        'M', 
        'N', 
        'K', 
        'total_cycles',
        'fusion_avg',
        'fusion_min', 
        'fusion_max',
        'fusion_stddev'
    ]
    
    try:
        # Read the CSV file
        print(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)
        
        # Check if required columns exist
        missing_columns = [col for col in columns_to_keep if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Extract only the specified columns
        df_filtered = df[columns_to_keep]
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_fusion_only.csv"
        
        # Save to output file
        df_filtered.to_csv(output_file, index=False)
        print(f"Extracted {len(df_filtered)} rows with {len(columns_to_keep)} columns")
        print(f"Output saved to: {output_file}")
        
        # Display first few rows
        print("\nFirst 5 rows of extracted data:")
        print(df_filtered.head())
        
        # Display column information
        print(f"\nColumns kept: {columns_to_keep}")
        print(f"Total rows: {len(df_filtered)}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")

def main():
    """Main function to handle command line arguments and execute the script."""
    
    # Default input file
    input_file = "kernel_report_updated6_mnk.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found in current directory.")
        print("Available files:")
        for file in os.listdir('.'):
            if file.endswith('.csv'):
                print(f"  - {file}")
        return
    
    # Extract fusion columns
    extract_fusion_columns(input_file)

if __name__ == "__main__":
    main()


