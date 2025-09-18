#!/usr/bin/env python3
"""
Script to filter kernel report CSV file by removing rows where K > 128.
The kernel names follow the pattern: matmul_MxNxK
"""

import csv
import re
import sys

def extract_k_value(kernel_name):
    """Extract the K value from kernel name like 'matmul_MxNxK'"""
    match = re.search(r'matmul_\d+x\d+x(\d+)', kernel_name)
    if match:
        return int(match.group(1))
    return None

def filter_csv(input_file, output_file):
    """Filter CSV file to keep only rows where K <= 128"""
    rows_kept = 0
    rows_removed = 0
    
    with open(input_file, 'r', newline='') as infile, \
         open(output_file, 'w', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write header row
        header = next(reader)
        writer.writerow(header)
        
        # Process data rows
        for row_num, row in enumerate(reader, start=2):  # Start at 2 since we already read header
            if len(row) > 0:
                kernel_name = row[0]
                k_value = extract_k_value(kernel_name)
                
                if k_value is None:
                    print(f"Warning: Could not parse K value from kernel name '{kernel_name}' at row {row_num}")
                    # Keep rows we can't parse (safer to keep than remove)
                    writer.writerow(row)
                    rows_kept += 1
                elif k_value <= 128:
                    writer.writerow(row)
                    rows_kept += 1
                else:
                    print(f"Removing row {row_num}: {kernel_name} (K={k_value} > 128)")
                    rows_removed += 1
            else:
                # Keep empty rows
                writer.writerow(row)
                rows_kept += 1
    
    print(f"\nFiltering complete:")
    print(f"Rows kept: {rows_kept}")
    print(f"Rows removed: {rows_removed}")
    print(f"Output saved to: {output_file}")

def main():
    input_file = "kernel_report_updated.csv"
    output_file = "kernel_report_filtered.csv"
    
    print(f"Filtering {input_file} to remove rows where K > 128...")
    print(f"Output will be saved to {output_file}")
    
    try:
        filter_csv(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        print("Make sure you're running this script from the same directory as the CSV file.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
