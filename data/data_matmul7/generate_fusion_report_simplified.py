#!/usr/bin/env python3
"""
Generate fusion statistics report from filtered_events.csv - Simplified version

This script creates exactly the columns requested:
- kernel_name
- dim_m, dim_n, dim_k (values for each dimension in the name)  
- fusion_avg, fusion_min, fusion_max, fusion_stddev
"""

import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


def parse_kernel_dimensions(kernel_name):
    """Extract M, N, K dimensions from kernel name."""
    match = re.match(r'matmul_(\d+)x(\d+)x(\d+)', kernel_name)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None


def process_fusion_events(input_file):
    """Process the filtered_events.csv file to extract fusion statistics."""
    fusion_data = defaultdict(list)
    
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            kernel_name = row['kernel_name']
            event_type = row['event_type']
            event_name = row['event_name']
            duration = float(row['dur(us)'])
            
            # Only process fusion events
            if event_type == 'sub' and event_name == 'fusion':
                fusion_data[kernel_name].append(duration)
    
    return fusion_data


def calculate_statistics(fusion_data):
    """Calculate statistics for each kernel's fusion times."""
    results = []
    
    for kernel_name, times in fusion_data.items():
        if not times:  # Skip if no fusion events
            continue
            
        dim_m, dim_n, dim_k = parse_kernel_dimensions(kernel_name)
        
        if dim_m is None:  # Skip if couldn't parse dimensions
            print(f"Warning: Could not parse dimensions from {kernel_name}")
            continue
        
        stats = {
            'kernel_name': kernel_name,
            'dim_m': dim_m,
            'dim_n': dim_n,
            'dim_k': dim_k,
            'fusion_avg': statistics.mean(times),
            'fusion_min': min(times),
            'fusion_max': max(times),
            'fusion_stddev': statistics.stdev(times) if len(times) > 1 else 0.0
        }
        
        results.append(stats)
    
    return results


def write_report(results, output_file):
    """Write the fusion statistics report to a CSV file."""
    fieldnames = [
        'kernel_name', 'dim_m', 'dim_n', 'dim_k',
        'fusion_avg', 'fusion_min', 'fusion_max', 'fusion_stddev'
    ]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Sort by kernel name for consistent output
        sorted_results = sorted(results, key=lambda x: x['kernel_name'])
        writer.writerows(sorted_results)


def main():
    """Main function to process fusion events and generate report."""
    # File paths
    input_file = 'filtered_events.csv'
    output_file = 'fusion_report.csv'
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    print(f"Processing {input_file}...")
    
    # Process the data
    fusion_data = process_fusion_events(input_file)
    print(f"Found {len(fusion_data)} unique kernels")
    
    # Calculate statistics
    results = calculate_statistics(fusion_data)
    print(f"Generated statistics for {len(results)} kernels")
    
    # Write the report
    write_report(results, output_file)
    print(f"Report written to {output_file}")


if __name__ == '__main__':
    main()

