#!/usr/bin/env python3
"""
Generate fusion statistics report from filtered_events.csv

This script processes the filtered_events.csv file to create a report with:
- kernel_name: The full kernel name (e.g., matmul_1024x1024x1024)
- dim_m, dim_n, dim_k: Individual matrix dimensions extracted from kernel name
- fusion_avg: Average fusion time
- fusion_min: Minimum fusion time
- fusion_max: Maximum fusion time
- fusion_stddev: Standard deviation of fusion times
"""

import csv
import re
import statistics
import sys
import glob
from collections import defaultdict
from pathlib import Path


def parse_kernel_dimensions(kernel_name):
    """
    Extract M, N, K dimensions from kernel name.
    
    Args:
        kernel_name (str): Kernel name in format 'matmul_MxNxK'
        
    Returns:
        tuple: (M, N, K) dimensions as integers, or (None, None, None) if parsing fails
    """
    match = re.match(r'matmul_(\d+)x(\d+)x(\d+)', kernel_name)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None


def process_fusion_events(input_file):
    """
    Process the filtered_events.csv file to extract fusion statistics.
    
    Args:
        input_file (str): Path to the input CSV file
        
    Returns:
        dict: Dictionary with kernel names as keys and fusion statistics as values
    """
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
    """
    Calculate statistics for each kernel's fusion times.
    
    Args:
        fusion_data (dict): Dictionary with kernel names and their fusion times
        
    Returns:
        list: List of dictionaries with kernel statistics
    """
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
            'fusion_stddev': statistics.stdev(times) if len(times) > 1 else 0.0,
            'sample_count': len(times)
        }
        
        results.append(stats)
    
    return results


def write_report(results, output_file):
    """
    Write the fusion statistics report to a CSV file.
    
    Args:
        results (list): List of kernel statistics dictionaries
        output_file (str): Path to the output CSV file
    """
    fieldnames = [
        'kernel_name', 'dim_m', 'dim_n', 'dim_k',
        'fusion_avg', 'fusion_min', 'fusion_max', 'fusion_stddev',
        'sample_count'
    ]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Sort by kernel name for consistent output
        sorted_results = sorted(results, key=lambda x: x['kernel_name'])
        writer.writerows(sorted_results)


def generate_output_filename(input_file):
    """
    Generate output filename based on input filename.
    
    Args:
        input_file (str): Path to input CSV file
        
    Returns:
        str: Output filename
    """
    input_path = Path(input_file)
    # Extract the base name without extension
    base_name = input_path.stem
    
    # If the filename contains 'filtered_events', replace it with 'fusion_statistics_report'
    if 'filtered_events' in base_name:
        output_name = base_name.replace('filtered_events', 'fusion_statistics_report')
    else:
        # Otherwise, prepend 'fusion_statistics_report_' to the base name
        output_name = f'fusion_statistics_report_{base_name}'
    
    return f'{output_name}.csv'


def process_single_file(input_file):
    """
    Process a single input file and generate its report.
    
    Args:
        input_file (str): Path to the input CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found!")
        return False
    
    # Generate output filename
    output_file = generate_output_filename(input_file)
    
    print(f"\n{'='*60}")
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
    
    # Print summary statistics
    if results:
        total_samples = sum(r['sample_count'] for r in results)
        avg_fusion_time = statistics.mean([r['fusion_avg'] for r in results])
        print(f"\nSummary:")
        print(f"  Total fusion events processed: {total_samples}")
        print(f"  Average fusion time across all kernels: {avg_fusion_time:.3f} μs")
        print(f"  Kernel dimension ranges:")
        print(f"    M: {min(r['dim_m'] for r in results)} - {max(r['dim_m'] for r in results)}")
        print(f"    N: {min(r['dim_n'] for r in results)} - {max(r['dim_n'] for r in results)}")
        print(f"    K: {min(r['dim_k'] for r in results)} - {max(r['dim_k'] for r in results)}")
    
    return True


def main():
    """Main function to process fusion events and generate report."""
    # Check for command-line arguments
    if len(sys.argv) > 1:
        # Use provided file paths
        input_files = sys.argv[1:]
    else:
        # Default: find all filtered_events*.csv files in current directory
        input_files = sorted(glob.glob('filtered_events*.csv'))
        
        if not input_files:
            print("Error: No filtered_events*.csv files found in current directory!")
            print("Usage: python generate_fusion_report.py [input_file1.csv input_file2.csv ...]")
            return
    
    print(f"Found {len(input_files)} file(s) to process")
    
    # Process each file
    success_count = 0
    for input_file in input_files:
        if process_single_file(input_file):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete: {success_count}/{len(input_files)} file(s) successfully processed")


if __name__ == '__main__':
    main()

