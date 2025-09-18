#!/usr/bin/env python3
"""
Script to merge function-specific CSV files into one CSV with columns:
- kernel_shape: extracted from kernel_name (e.g., "(16,16,32)")
- tuple_product: product of shape dimensions
- function1_avg_duration_us, function2_avg_duration_us, etc.

This creates a wide-format table where each row represents a unique shape
and columns show the average duration for each activation function.
"""

import pandas as pd
import os
import re
from pathlib import Path

def extract_kernel_shape(kernel_name):
    """
    Extract shape tuple from kernel name.
    
    Args:
        kernel_name (str): Kernel name like "sigmoid_(16,16,32)" or "elu_(128,)"
    
    Returns:
        str: Shape string like "(16,16,32)" or "(128,)"
    """
    try:
        # Remove quotes if present
        clean_name = kernel_name.strip('"')
        
        # Extract content between parentheses
        match = re.search(r'\((.*?)\)', clean_name)
        if match:
            return f"({match.group(1)})"
        else:
            return "unknown"
    except Exception as e:
        print(f"Warning: Could not parse kernel name '{kernel_name}': {e}")
        return "unknown"

def merge_function_csvs(function_splits_dir="function_splits", output_file="merged_functions_by_shape.csv"):
    """
    Merge function-specific CSV files into one comprehensive CSV.
    
    Args:
        function_splits_dir (str): Directory containing function-specific CSV files
        output_file (str): Output CSV file name
    """
    
    # Check if directory exists
    if not os.path.exists(function_splits_dir):
        print(f"Error: Directory '{function_splits_dir}' not found!")
        return
    
    # Get all function CSV files (exclude summary)
    csv_files = [f for f in os.listdir(function_splits_dir) 
                 if f.endswith('_kernel_stats.csv')]
    
    if not csv_files:
        print(f"Error: No function CSV files found in '{function_splits_dir}'!")
        return
    
    print(f"Found {len(csv_files)} function CSV files:")
    for file in sorted(csv_files):
        print(f"  {file}")
    
    # Dictionary to store data from all functions
    all_data = {}
    function_names = []
    
    # Read each function CSV
    for csv_file in sorted(csv_files):
        # Extract function name from filename
        function_name = csv_file.replace('_kernel_stats.csv', '')
        function_names.append(function_name)
        
        print(f"\nProcessing {function_name}...")
        
        # Read the CSV
        file_path = os.path.join(function_splits_dir, csv_file)
        df = pd.read_csv(file_path)
        
        print(f"  Found {len(df)} kernels")
        
        # Extract kernel shapes and store avg_duration_us
        for _, row in df.iterrows():
            kernel_shape = extract_kernel_shape(row['kernel_name'])
            tuple_product = row['tuple_product']
            avg_duration = row['avg_duration_us']
            
            # Initialize shape entry if not exists
            if kernel_shape not in all_data:
                all_data[kernel_shape] = {
                    'kernel_shape': kernel_shape,
                    'tuple_product': tuple_product
                }
            
            # Add this function's avg_duration
            all_data[kernel_shape][f'{function_name}_avg_duration_us'] = avg_duration
    
    # Convert to DataFrame
    print(f"\nCreating merged DataFrame...")
    merged_df = pd.DataFrame(list(all_data.values()))
    
    # Sort by tuple_product
    merged_df = merged_df.sort_values('tuple_product')
    
    # Reorder columns: kernel_shape, tuple_product, then all function columns
    function_columns = [f'{func}_avg_duration_us' for func in sorted(function_names)]
    column_order = ['kernel_shape', 'tuple_product'] + function_columns
    merged_df = merged_df[column_order]
    
    # Save to CSV
    print(f"Saving merged data to {output_file}...")
    merged_df.to_csv(output_file, index=False)
    
    # Display summary
    print(f"\nMerge complete!")
    print(f"Merged data shape: {merged_df.shape}")
    print(f"Unique kernel shapes: {len(merged_df)}")
    print(f"Functions included: {len(function_names)}")
    print(f"Functions: {', '.join(sorted(function_names))}")
    
    # Display first few rows
    print(f"\nFirst 5 rows of merged data:")
    print("=" * 120)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(merged_df.head().to_string(index=False))
    
    # Check for any missing data
    print(f"\nMissing data check:")
    missing_counts = merged_df.isnull().sum()
    if missing_counts.sum() > 0:
        print("Missing values found:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"  {col}: {count} missing values")
    else:
        print("No missing values found - all shapes have data for all functions!")
    
    # Show some statistics
    print(f"\nSummary statistics for tuple_product:")
    print(f"  Min: {merged_df['tuple_product'].min()}")
    print(f"  Max: {merged_df['tuple_product'].max()}")
    print(f"  Mean: {merged_df['tuple_product'].mean():.1f}")
    print(f"  Unique values: {merged_df['tuple_product'].nunique()}")
    
    # Show average duration ranges for each function
    print(f"\nAverage duration ranges by function (microseconds):")
    for func in sorted(function_names):
        col_name = f'{func}_avg_duration_us'
        if col_name in merged_df.columns:
            min_val = merged_df[col_name].min()
            max_val = merged_df[col_name].max()
            mean_val = merged_df[col_name].mean()
            print(f"  {func:12}: {min_val:.3f} - {max_val:.3f} (mean: {mean_val:.3f})")
    
    return merged_df

def main():
    # Set paths
    function_splits_dir = "function_splits"
    output_file = "merged_functions_by_shape.csv"
    
    # Change to the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"Working directory: {os.getcwd()}")
    
    try:
        # Merge the function CSVs
        merged_data = merge_function_csvs(function_splits_dir, output_file)
        
        if merged_data is not None:
            print(f"\nSuccess! Merged data saved to: {output_file}")
            
            # Create a sample analysis
            print(f"\nSample analysis - Top 5 shapes by tuple_product:")
            top_shapes = merged_data.nlargest(5, 'tuple_product')
            print(top_shapes[['kernel_shape', 'tuple_product']].to_string(index=False))
            
            print(f"\nSample analysis - Fastest function for largest shape:")
            largest_shape = merged_data.loc[merged_data['tuple_product'].idxmax()]
            print(f"Shape: {largest_shape['kernel_shape']} (product: {largest_shape['tuple_product']})")
            
            # Find fastest function for this shape
            function_cols = [col for col in merged_data.columns if col.endswith('_avg_duration_us')]
            shape_durations = {col.replace('_avg_duration_us', ''): largest_shape[col] 
                              for col in function_cols}
            fastest_func = min(shape_durations, key=shape_durations.get)
            fastest_time = shape_durations[fastest_func]
            
            print(f"Fastest function: {fastest_func} ({fastest_time:.3f} us)")
            
            print(f"\nAll functions for this shape:")
            for func, duration in sorted(shape_durations.items(), key=lambda x: x[1]):
                print(f"  {func:12}: {duration:.3f} us")
        
    except Exception as e:
        print(f"Error during merge: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


