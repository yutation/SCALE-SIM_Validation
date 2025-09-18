#!/usr/bin/env python3
"""
Enhanced script to generate a CSV report combining filtered events and compute report data.
Each row represents a kernel with statistics for different event types.

Features:
- Robust event type matching
- Better error handling
- Detailed logging
- Configurable output format
- Preserves original kernel order from filtered events
- Formatted float values (6 decimal places)
"""

import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_filtered_events(file_path):
    """Load and process filtered events CSV with error handling."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} events from {file_path}")
        
        # Validate required columns
        required_columns = ['kernel_name', 'event_name', 'dur(us)']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Group by kernel_name and event_name to calculate statistics
        # Use OrderedDict to preserve the order of kernels as they appear in the file
        kernel_stats = OrderedDict()
        kernel_order = []
        
        for _, row in df.iterrows():
            kernel_name = row['kernel_name']
            event_name = row['event_name']
            duration = row['dur(us)']
            
            # Track the order of kernels as they first appear
            if kernel_name not in kernel_order:
                kernel_order.append(kernel_name)
                kernel_stats[kernel_name] = defaultdict(list)
            
            kernel_stats[kernel_name][event_name].append(duration)
        
        logger.info(f"Processed {len(kernel_stats)} unique kernels")
        logger.info(f"Kernel order preserved from filtered events file")
        return kernel_stats, kernel_order
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading filtered events: {e}")
        raise

def load_compute_report(file_path):
    """Load compute report CSV and extract total cycles per layer."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} layers from {file_path}")
        
        # Validate required columns
        if 'LayerID' not in df.columns or ' Total Cycles' not in df.columns:
            raise ValueError("Missing required columns: LayerID or Total Cycles")
        
        # Create a mapping from LayerID to Total Cycles
        layer_cycles = {}
        for _, row in df.iterrows():
            layer_id = row['LayerID']
            total_cycles = row[' Total Cycles']
            layer_cycles[layer_id] = total_cycles
        
        logger.info(f"Processed {len(layer_cycles)} layer cycles")
        return layer_cycles
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading compute report: {e}")
        raise

def calculate_statistics(values):
    """Calculate avg, min, max, stddev for a list of values."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    
    values = np.array(values)
    return np.mean(values), np.min(values), np.max(values), np.std(values)

def format_float(value):
    """Format float value to 6 decimal places."""
    return round(value, 6)

def match_event_type(event_name, target_type):
    """Enhanced event type matching with multiple patterns."""
    event_name_lower = event_name.lower()
    
    # Define matching patterns for each event type
    patterns = {
        'main': ['main', 'jit_validation_matrix_multiply'],
        'fusion': ['fusion'],
        'copy-start': ['copy-start', 'copy_start'],
        'copy-done': ['copy-done', 'copy_done']
    }
    
    if target_type not in patterns:
        return False
    
    return any(pattern in event_name_lower for pattern in patterns[target_type])

def generate_kernel_report(filtered_events_file, compute_report_file, output_file):
    """Generate the combined kernel report CSV with enhanced processing."""
    
    logger.info("Starting kernel report generation...")
    
    # Load data
    kernel_stats, kernel_order = load_filtered_events(filtered_events_file)
    layer_cycles = load_compute_report(compute_report_file)
    
    # Prepare output data
    output_rows = []
    
    # Process kernels in the order they appear in the filtered events file
    logger.info(f"Processing {len(kernel_order)} kernels in original order")
    
    for i, kernel_name in enumerate(kernel_order):
        stats = kernel_stats[kernel_name]
        
        # Get total cycles (assuming LayerID corresponds to kernel order)
        total_cycles = layer_cycles.get(i, 0)
        
        # Calculate statistics for each event type
        event_types = ['main', 'fusion', 'copy-start', 'copy-done']
        row_data = {
            'kernel_name': kernel_name,
            'total_cycles': total_cycles
        }
        
        for event_type in event_types:
            # Find events that match the event type
            durations = []
            for event_name, duration_list in stats.items():
                if match_event_type(event_name, event_type):
                    durations.extend(duration_list)
            
            # Calculate statistics
            avg, min_val, max_val, stddev = calculate_statistics(durations)
            
            # Format float values to 6 decimal places
            row_data[f'{event_type}_avg'] = format_float(avg)
            row_data[f'{event_type}_min'] = format_float(min_val)
            row_data[f'{event_type}_max'] = format_float(max_val)
            row_data[f'{event_type}_stddev'] = format_float(stddev)
            
            # Log statistics for debugging
            if durations:
                logger.debug(f"{kernel_name} - {event_type}: {len(durations)} events, "
                           f"avg={format_float(avg)}, min={format_float(min_val)}, "
                           f"max={format_float(max_val)}, std={format_float(stddev)}")
        
        output_rows.append(row_data)
    
    # Create DataFrame and save to CSV
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_file, index=False)
    
    logger.info(f"Generated kernel report: {output_file}")
    logger.info(f"Total kernels processed: {len(output_rows)}")
    
    # Print summary statistics
    print_summary_statistics(output_df)
    
    return output_df

def print_summary_statistics(df):
    """Print summary statistics for the generated report."""
    print("\n" + "="*60)
    print("KERNEL REPORT SUMMARY")
    print("="*60)
    print(f"Total kernels: {len(df)}")
    print(f"Total cycles range: {df['total_cycles'].min()} - {df['total_cycles'].max()}")
    print(f"First kernel: {df.iloc[0]['kernel_name']}")
    print(f"Last kernel: {df.iloc[-1]['kernel_name']}")
    
    event_types = ['main', 'fusion', 'copy-start', 'copy-done']
    for event_type in event_types:
        avg_col = f'{event_type}_avg'
        if avg_col in df.columns:
            non_zero_count = (df[avg_col] > 0).sum()
            print(f"{event_type} events: {non_zero_count} kernels have non-zero values")
    
    print("="*60)

def main():
    """Main function with command line argument support."""
    # File paths
    filtered_events_file = "filtered_events_repeat20.csv"
    compute_report_file = "COMPUTE_REPORT.csv"
    output_file = "kernel_report_enhanced.csv"
    
    # Check if input files exist
    for file_path in [filtered_events_file, compute_report_file]:
        if not Path(file_path).exists():
            logger.error(f"Input file not found: {file_path}")
            sys.exit(1)
    
    try:
        # Generate the report
        result_df = generate_kernel_report(filtered_events_file, compute_report_file, output_file)
        
        # Display first few rows
        print("\nFirst few rows of the generated report:")
        print(result_df.head().to_string())
        
        logger.info("Kernel report generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error generating kernel report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
