#!/usr/bin/env python3
"""
Script to split kernel_statistics_filtered.csv into multiple CSV files based on function name.
Each function (like sigmoid, elu, relu, etc.) will get its own CSV file.
"""

import pandas as pd
import os
import re
from pathlib import Path

def extract_function_name(kernel_name):
    """
    Extract function name from kernel name.
    
    Args:
        kernel_name (str): Kernel name like "sigmoid_(16,16,32)" or "elu_(24,16,16)"
    
    Returns:
        str: Function name (e.g., "sigmoid", "elu", "relu")
    """
    try:
        # Remove quotes if present
        clean_name = kernel_name.strip('"')
        
        # Extract function name before the underscore and parentheses
        # Pattern: function_name_(shape)
        match = re.match(r'^([^_]+)_', clean_name)
        if match:
            return match.group(1)
        else:
            # Fallback: try to find function name before parentheses
            match = re.match(r'^([^(]+)', clean_name)
            if match:
                return match.group(1).rstrip('_')
            else:
                return "unknown"
    except Exception as e:
        print(f"Warning: Could not parse kernel name '{kernel_name}': {e}")
        return "unknown"

def split_csv_by_function(input_file, output_dir="function_splits"):
    """
    Split the kernel statistics CSV by function name.
    
    Args:
        input_file (str): Path to input CSV file
        output_dir (str): Directory to save split CSV files
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Read the CSV file
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Data shape: {df.shape}")
    print(f"Total kernels: {len(df)}")
    
    # Extract function names
    print("Extracting function names from kernel names...")
    df['function_name'] = df['kernel_name'].apply(extract_function_name)
    
    # Get unique function names
    unique_functions = df['function_name'].unique()
    print(f"Found {len(unique_functions)} unique functions: {sorted(unique_functions)}")
    
    # Count kernels per function
    function_counts = df['function_name'].value_counts()
    print(f"\nKernels per function:")
    for func, count in function_counts.items():
        print(f"  {func}: {count} kernels")
    
    # Split and save by function
    print(f"\nSplitting data by function and saving to {output_dir}/...")
    
    summary_data = []
    
    for function_name in sorted(unique_functions):
        # Filter data for this function
        function_df = df[df['function_name'] == function_name].copy()
        
        # Drop the temporary function_name column for the output
        function_df = function_df.drop('function_name', axis=1)
        
        # Sort by average duration (descending)
        function_df = function_df.sort_values('avg_duration_us', ascending=False)
        
        # Create output filename
        output_file = os.path.join(output_dir, f"{function_name}_kernel_stats.csv")
        
        # Save to CSV
        function_df.to_csv(output_file, index=False)
        
        # Collect summary info
        summary_data.append({
            'function_name': function_name,
            'kernel_count': len(function_df),
            'avg_duration_mean': function_df['avg_duration_us'].mean(),
            'avg_duration_std': function_df['avg_duration_us'].std(),
            'avg_duration_min': function_df['avg_duration_us'].min(),
            'avg_duration_max': function_df['avg_duration_us'].max(),
            'output_file': output_file
        })
        
        print(f"  {function_name}: {len(function_df)} kernels -> {output_file}")
    
    # Create summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('avg_duration_mean', ascending=False)
    summary_file = os.path.join(output_dir, "function_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Display summary statistics
    print(f"\nFunction Summary (sorted by mean average duration):")
    print("=" * 80)
    for _, row in summary_df.iterrows():
        print(f"Function: {row['function_name']}")
        print(f"  Kernels: {row['kernel_count']}")
        print(f"  Avg Duration: {row['avg_duration_mean']:.3f} ± {row['avg_duration_std']:.3f} us")
        print(f"  Range: {row['avg_duration_min']:.3f} - {row['avg_duration_max']:.3f} us")
        print(f"  File: {row['output_file']}")
        print()
    
    return summary_df

def main():
    # File paths
    input_file = "kernel_statistics_filtered.csv"
    output_dir = "function_splits"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        return
    
    try:
        # Split the data by function
        summary = split_csv_by_function(input_file, output_dir)
        
        print(f"\nSplit complete!")
        print(f"Created {len(summary)} function-specific CSV files in '{output_dir}/' directory")
        
        # Show some examples of split files
        print(f"\nExample of generated files:")
        for file in sorted(os.listdir(output_dir))[:5]:
            file_path = os.path.join(output_dir, file)
            if file.endswith('.csv') and file != 'function_summary.csv':
                df_sample = pd.read_csv(file_path)
                print(f"  {file}: {len(df_sample)} kernels")
        
        if len(os.listdir(output_dir)) > 6:
            print("  ... and more")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


