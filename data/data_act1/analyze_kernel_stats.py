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

def extract_tuple_product(kernel_name):
    """
    Extract tuple values from kernel name and calculate their product
    
    Args:
        kernel_name (str): Kernel name like "add_(8, 16, 32)" or "add_(128,)"
    
    Returns:
        int: Product of tuple values, or 0 if parsing fails
    """
    try:
        # Remove quotes and extract content between parentheses
        clean_name = kernel_name.strip('"')
        
        # Find content between parentheses - improved regex to handle single values
        match = re.search(r'\((.*?)\)', clean_name)
        if not match:
            return 0
        
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
            
        return product
        
    except (ValueError, AttributeError) as e:
        print(f"Warning: Could not parse kernel name '{kernel_name}': {e}")
        return 0

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
    
    # Add tuple product column
    print("Calculating tuple products for kernel names...")
    kernel_stats['tuple_product'] = kernel_stats['kernel_name'].apply(extract_tuple_product)
    
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
    
    # Display some examples of tuple products
    print(f"\nExamples of tuple products:")
    sample_kernels = kernel_stats.head(5)[['kernel_name', 'tuple_product']]
    for _, row in sample_kernels.iterrows():
        print(f"  {row['kernel_name']} -> Product: {row['tuple_product']}")
    
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
