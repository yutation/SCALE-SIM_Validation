#!/usr/bin/env python3
"""
Script to analyze kernel event statistics from filtered_events.csv
Generates a CSV with kernel name, event duration statistics (avg, min, max, stddev)
Only includes sub events where event_name is not "copy-start" or "copy-done"
"""

import pandas as pd
import numpy as np
import os
import re

def extract_shape_dimensions(kernel_name):
    """
    Extract shape dimensions from kernel name
    
    Args:
        kernel_name (str): Kernel name like "add_(8, 16, 32)" or "add_(128,)"
    
    Returns:
        tuple: (dimensions_list, product) where dimensions_list contains individual dims
               and product is their multiplication, or ([], 0) if parsing fails
    """
    try:
        # Remove quotes and extract content between parentheses
        clean_name = kernel_name.strip('"')
        
        # Find content between parentheses - improved regex to handle single values
        match = re.search(r'\((.*?)\)', clean_name)
        if not match:
            return ([], 0)
        
        # Extract the tuple string and split by comma
        tuple_str = match.group(1).strip()
        
        # Handle single value case (e.g., "128," -> "128")
        if tuple_str.endswith(','):
            tuple_str = tuple_str[:-1].strip()
        
        # Split by comma and convert to integers
        if ',' in tuple_str:
            values = [int(x.strip()) for x in tuple_str.split(',')]
        else:
            # Single value case
            values = [int(tuple_str)]
        
        # Calculate product
        product = 1
        for value in values:
            product *= value
            
        return (values, product)
        
    except (ValueError, AttributeError) as e:
        print(f"Warning: Could not parse kernel name '{kernel_name}': {e}")
        return ([], 0)

def analyze_kernel_stats(input_file, output_file):
    """
    Analyze kernel event statistics and generate summary CSV
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
    """
    
    # Read the CSV file
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Display basic info about the data
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique kernels: {df['kernel_name'].nunique()}")
    
    # Filter data: only sub events where event_name is not "copy-start" or "copy-done"
    print("Filtering data: only sub events, excluding copy-start and copy-done...")
    filtered_df = df[
        (df['event_type'] == 'sub') & 
        (~df['event_name'].isin(['copy-start', 'copy-done']))
    ]
    
    print(f"After filtering: {len(filtered_df)} events (from {len(df)} total)")
    print(f"Event names in filtered data: {filtered_df['event_name'].unique()}")
    
    # Group by kernel name and calculate statistics
    print("Calculating statistics for each kernel...")
    
    # Convert duration to numeric, handling any potential string values
    filtered_df['dur(us)'] = pd.to_numeric(filtered_df['dur(us)'], errors='coerce')
    
    # Remove any rows with NaN durations
    df_clean = filtered_df.dropna(subset=['dur(us)'])
    
    # Group by kernel name and calculate statistics
    kernel_stats = df_clean.groupby('kernel_name')['dur(us)'].agg([
        ('avg_duration', 'mean'),
        ('min_duration', 'min'),
        ('max_duration', 'max'),
        ('stddev_duration', 'std'),
        ('count', 'count')
    ]).round(6)
    
    # Reset index to make kernel_name a column
    kernel_stats = kernel_stats.reset_index()
    
    # Rename columns for clarity
    kernel_stats.columns = ['kernel_name', 'avg_duration_us', 'min_duration_us', 
                           'max_duration_us', 'stddev_duration_us', 'event_count']
    
    # Extract shape dimensions and add dimension columns
    print("Extracting shape dimensions from kernel names...")
    
    # Extract dimensions for all kernel names
    dimensions_data = kernel_stats['kernel_name'].apply(extract_shape_dimensions)
    
    # Separate dimensions and products
    dimensions_list = [dims for dims, _ in dimensions_data]
    products_list = [product for _, product in dimensions_data]
    
    # Find the maximum number of dimensions across all kernels
    max_dims = max(len(dims) for dims in dimensions_list) if dimensions_list else 0
    
    # Create dimension columns
    for i in range(max_dims):
        col_name = f'dim_{i}'
        kernel_stats[col_name] = [dims[i] if i < len(dims) else None for dims in dimensions_list]
    
    # Add tuple product column
    kernel_stats['tuple_product'] = products_list
    
    # Reorder columns to put dimensions right after kernel_name
    base_cols = ['kernel_name']
    dim_cols = [col for col in kernel_stats.columns if col.startswith('dim_')]
    stat_cols = ['avg_duration_us', 'min_duration_us', 'max_duration_us', 'stddev_duration_us', 'event_count']
    product_cols = ['tuple_product']
    
    # Reorder the dataframe columns
    new_column_order = base_cols + dim_cols + stat_cols + product_cols
    kernel_stats = kernel_stats[new_column_order]
    
    # Sort by average duration (descending)
    kernel_stats = kernel_stats.sort_values('avg_duration_us', ascending=False)
    
    # Save to CSV
    print(f"Saving results to {output_file}...")
    kernel_stats.to_csv(output_file, index=False)
    
    # Display summary
    print(f"\nAnalysis complete!")
    print(f"Processed {len(df_clean)} filtered events from {len(kernel_stats)} kernels")
    print(f"Results saved to: {output_file}")
    
    # Display top 10 kernels by average duration
    print(f"\nTop 10 kernels by average duration:")
    print(kernel_stats.head(10).to_string(index=False))
    
    # Display some examples of shape dimensions
    print(f"\nExamples of shape dimensions:")
    # Get dimension columns dynamically
    dim_cols = [col for col in kernel_stats.columns if col.startswith('dim_')]
    display_cols = ['kernel_name'] + dim_cols + ['tuple_product']
    sample_kernels = kernel_stats.head(5)[display_cols]
    
    for _, row in sample_kernels.iterrows():
        dims_str = ', '.join([str(int(row[col])) for col in dim_cols if pd.notna(row[col])])
        print(f"  {row['kernel_name']} -> Dims: ({dims_str}) -> Product: {row['tuple_product']}")
    
    return kernel_stats

def main():
    # File paths
    input_file = "filtered_events.csv"
    output_file = "kernel_statistics_filtered.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        return
    
    try:
        # Analyze the data
        stats = analyze_kernel_stats(input_file, output_file)
        
        # Additional analysis: event type breakdown
        print(f"\nEvent type breakdown (original data):")
        df = pd.read_csv(input_file)
        event_type_counts = df['event_type'].value_counts()
        print(event_type_counts)
        
        print(f"\nEvent name breakdown (filtered sub events):")
        filtered_df = df[
            (df['event_type'] == 'sub') & 
            (~df['event_name'].isin(['copy-start', 'copy-done']))
        ]
        event_name_counts = filtered_df['event_name'].value_counts()
        print(event_name_counts)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
