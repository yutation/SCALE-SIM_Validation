#!/usr/bin/env python3
"""
Script to analyze kernel updates and show which kernels were updated vs skipped.
"""

import pandas as pd
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_kernel_name(kernel_name):
    """Parse kernel name to extract M, N, K dimensions."""
    pattern = r'matmul_(\d+)x(\d+)x(\d+)'
    match = re.match(pattern, kernel_name)
    
    if match:
        M, N, K = map(int, match.groups())
        return M, N, K
    else:
        return None, None, None

def analyze_kernel_updates(original_file, updated_file):
    """Analyze which kernels were updated and which were skipped."""
    
    # Load both files
    df_original = pd.read_csv(original_file)
    df_updated = pd.read_csv(updated_file)
    
    print("="*80)
    print("KERNEL UPDATE ANALYSIS")
    print("="*80)
    
    updated_kernels = []
    skipped_kernels = []
    
    for i, (orig_row, updated_row) in enumerate(zip(df_original.iterrows(), df_updated.iterrows())):
        kernel_name = orig_row[1]['kernel_name']
        original_cycles = orig_row[1]['total_cycles']
        updated_cycles = updated_row[1]['total_cycles']
        
        M, N, K = parse_kernel_name(kernel_name)
        
        if M is not None and N is not None:
            if M > N:
                if original_cycles != updated_cycles:
                    updated_kernels.append({
                        'kernel': kernel_name,
                        'M': M, 'N': N, 'K': K,
                        'original_cycles': original_cycles,
                        'updated_cycles': updated_cycles,
                        'layer_id': i
                    })
                else:
                    skipped_kernels.append({
                        'kernel': kernel_name,
                        'M': M, 'N': N, 'K': K,
                        'cycles': original_cycles,
                        'reason': 'M>N but cycles unchanged',
                        'layer_id': i
                    })
            else:
                skipped_kernels.append({
                    'kernel': kernel_name,
                    'M': M, 'N': N, 'K': K,
                    'cycles': original_cycles,
                    'reason': 'M≤N (no update needed)',
                    'layer_id': i
                })
    
    # Print summary
    print(f"Total kernels analyzed: {len(df_original)}")
    print(f"Kernels updated (M>N): {len(updated_kernels)}")
    print(f"Kernels skipped: {len(skipped_kernels)}")
    print()
    
    # Show some examples of updated kernels
    if updated_kernels:
        print("EXAMPLES OF UPDATED KERNELS (M>N):")
        print("-" * 80)
        for i, kernel in enumerate(updated_kernels[:10]):  # Show first 10
            print(f"{i+1:2d}. {kernel['kernel']:20s} | M={kernel['M']:3d}, N={kernel['N']:3d}, K={kernel['K']:3d} | "
                  f"{kernel['original_cycles']:6.1f} → {kernel['updated_cycles']:6.1f} | LayerID={kernel['layer_id']}")
        
        if len(updated_kernels) > 10:
            print(f"... and {len(updated_kernels) - 10} more updated kernels")
        print()
    
    # Show some examples of skipped kernels
    if skipped_kernels:
        print("EXAMPLES OF SKIPPED KERNELS:")
        print("-" * 80)
        skipped_by_reason = {}
        for kernel in skipped_kernels:
            reason = kernel['reason']
            if reason not in skipped_by_reason:
                skipped_by_reason[reason] = []
            skipped_by_reason[reason].append(kernel)
        
        for reason, kernels in skipped_by_reason.items():
            print(f"\n{reason}:")
            for i, kernel in enumerate(kernels[:5]):  # Show first 5 of each type
                print(f"  {i+1:2d}. {kernel['kernel']:20s} | M={kernel['M']:3d}, N={kernel['N']:3d}, K={kernel['K']:3d} | "
                      f"cycles={kernel['cycles']:6.1f}")
            if len(kernels) > 5:
                print(f"  ... and {len(kernels) - 5} more")
    
    print("="*80)

def main():
    """Main function."""
    original_file = "kernel_report_enhanced.csv"
    updated_file = "kernel_report_updated.csv"
    
    try:
        analyze_kernel_updates(original_file, updated_file)
    except Exception as e:
        logger.error(f"Error analyzing updates: {e}")

if __name__ == "__main__":
    main()
