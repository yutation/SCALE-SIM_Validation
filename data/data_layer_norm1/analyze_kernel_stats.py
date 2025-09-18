#!/usr/bin/env python3
"""
Script to analyze kernel statistics from filtered_events.csv
Extracts kernel dimensions and computes statistics for main events.
"""

import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path

def parse_kernel_dimensions(kernel_name):
    """
    Extract dimensions from kernel name like 'layer_norm_(1,128,512)'
    Returns: (batch_size, seq_len, hidden_dim)
    """
    match = re.search(r'layer_norm_\((\d+),(\d+),(\d+)\)', kernel_name)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    else:
        # Return None values if pattern doesn't match
        return None, None, None

def analyze_kernel_stats(input_file='filtered_events.csv', output_file='kernel_analysis.csv'):
    """
    Analyze kernel statistics from the filtered events CSV file.
    """
    try:
        # Read the CSV file
        print(f"Reading {input_file}...")
        df = pd.read_csv(input_file)
        
        # Filter for main events only (as requested)
        main_events = df[df['event_type'] == 'main'].copy()
        
        if main_events.empty:
            print("No main events found in the data!")
            return
        
        print(f"Found {len(main_events)} main events")
        
        # Extract dimensions from kernel names
        dimensions = main_events['kernel_name'].apply(parse_kernel_dimensions)
        main_events['batch_size'] = [d[0] for d in dimensions]
        main_events['seq_len'] = [d[1] for d in dimensions]
        main_events['hidden_dim'] = [d[2] for d in dimensions]
        
        # Group by kernel configuration and compute statistics
        grouped = main_events.groupby(['kernel_name', 'batch_size', 'seq_len', 'hidden_dim'])
        
        stats_list = []
        
        for (kernel_name, batch_size, seq_len, hidden_dim), group in grouped:
            durations = group['dur(us)']
            
            stats = {
                'kernel_name': kernel_name,
                'batch_size': batch_size,
                'seq_len': seq_len,
                'hidden_dim': hidden_dim,
                'avg_duration_us': durations.mean(),
                'min_duration_us': durations.min(),
                'max_duration_us': durations.max(),
                'stddev_duration_us': durations.std()
            }
            stats_list.append(stats)
        
        # Create output DataFrame
        output_df = pd.DataFrame(stats_list)
        
        # Sort by dimensions for better readability
        output_df = output_df.sort_values(['batch_size', 'seq_len', 'hidden_dim'])
        
        # Save to CSV
        output_df.to_csv(output_file, index=False)
        print(f"Analysis complete! Results saved to {output_file}")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"- Total unique kernel configurations: {len(output_df)}")
        print(f"- Batch sizes: {sorted(output_df['batch_size'].unique())}")
        print(f"- Sequence lengths: {sorted(output_df['seq_len'].unique())}")
        print(f"- Hidden dimensions: {sorted(output_df['hidden_dim'].unique())}")
        
        # Show first few rows
        print(f"\nFirst 5 results:")
        print(output_df.head().to_string(index=False))
        
        return output_df
        
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set up paths
    script_dir = Path(__file__).parent
    input_file = script_dir / "filtered_events.csv"
    output_file = script_dir / "kernel_analysis.csv"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        sys.exit(1)
    
    # Run analysis
    analyze_kernel_stats(str(input_file), str(output_file))
