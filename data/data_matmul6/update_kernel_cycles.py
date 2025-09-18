#!/usr/bin/env python3
"""
Script to update kernel report total cycles based on matrix dimensions.
If M > N in matmul_MxNxK, replace total_cycles with value from COMPUTE_REPORT_is.csv.
"""

import pandas as pd
import re
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_kernel_name(kernel_name):
    """Parse kernel name to extract M, N, K dimensions."""
    # Pattern to match matmul_MxNxK format
    pattern = r'matmul_(\d+)x(\d+)x(\d+)'
    match = re.match(pattern, kernel_name)
    
    if match:
        M, N, K = map(int, match.groups())
        return M, N, K
    else:
        logger.warning(f"Could not parse kernel name: {kernel_name}")
        return None, None, None

def load_compute_report_is(file_path):
    """Load COMPUTE_REPORT_is.csv and extract total cycles per layer."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} layers from {file_path}")
        
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

def update_kernel_cycles(kernel_report_file, compute_report_is_file, output_file):
    """Update kernel report with new total cycles based on matrix dimensions."""
    
    logger.info("Starting kernel cycles update...")
    
    # Load the kernel report
    try:
        df = pd.read_csv(kernel_report_file)
        logger.info(f"Loaded {len(df)} kernels from {kernel_report_file}")
    except Exception as e:
        logger.error(f"Error loading kernel report: {e}")
        raise
    
    # Load the compute report IS
    layer_cycles = load_compute_report_is(compute_report_is_file)
    
    # Track statistics
    updated_count = 0
    skipped_count = 0
    
    # Process each kernel
    for i, row in df.iterrows():
        kernel_name = row['kernel_name']
        M, N, K = parse_kernel_name(kernel_name)
        
        if M is not None and N is not None:
            # Check if M > N
            if M > N:
                # Get the corresponding cycle value from compute report IS
                new_cycles = layer_cycles.get(i, row['total_cycles'])
                df.at[i, 'total_cycles'] = new_cycles
                updated_count += 1
                logger.debug(f"Updated {kernel_name}: M={M}, N={N}, M>N=True, new_cycles={new_cycles}")
            else:
                skipped_count += 1
                logger.debug(f"Skipped {kernel_name}: M={M}, N={N}, M>N=False")
        else:
            skipped_count += 1
            logger.warning(f"Could not parse dimensions for {kernel_name}")
    
    # Save the updated report
    df.to_csv(output_file, index=False)
    
    logger.info(f"Updated kernel report: {output_file}")
    logger.info(f"Total kernels processed: {len(df)}")
    logger.info(f"Kernels updated (M>N): {updated_count}")
    logger.info(f"Kernels skipped: {skipped_count}")
    
    return df

def print_summary_statistics(df):
    """Print summary statistics for the updated report."""
    print("\n" + "="*60)
    print("KERNEL CYCLES UPDATE SUMMARY")
    print("="*60)
    print(f"Total kernels: {len(df)}")
    print(f"Total cycles range: {df['total_cycles'].min()} - {df['total_cycles'].max()}")
    print(f"First kernel: {df.iloc[0]['kernel_name']}")
    print(f"Last kernel: {df.iloc[-1]['kernel_name']}")
    
    # Count different cycle values
    cycle_counts = df['total_cycles'].value_counts()
    print(f"\nUnique cycle values: {len(cycle_counts)}")
    for cycles, count in cycle_counts.head(10).items():
        print(f"  {cycles}: {count} kernels")
    
    print("="*60)

def main():
    """Main function."""
    # File paths
    kernel_report_file = "kernel_report_enhanced.csv"
    compute_report_is_file = "COMPUTE_REPORT_is.csv"
    output_file = "kernel_report_updated.csv"
    
    # Check if input files exist
    for file_path in [kernel_report_file, compute_report_is_file]:
        if not Path(file_path).exists():
            logger.error(f"Input file not found: {file_path}")
            return
    
    try:
        # Update the kernel cycles
        result_df = update_kernel_cycles(kernel_report_file, compute_report_is_file, output_file)
        
        # Print summary statistics
        print_summary_statistics(result_df)
        
        # Display first few rows
        print("\nFirst few rows of the updated report:")
        print(result_df.head().to_string())
        
        logger.info("Kernel cycles update completed successfully!")
        
    except Exception as e:
        logger.error(f"Error updating kernel cycles: {e}")

if __name__ == "__main__":
    main()
