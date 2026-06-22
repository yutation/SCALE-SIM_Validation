#!/usr/bin/env python3
"""
Script to merge two CSV files with the same structure.
"""

import pandas as pd
import os

# Define file paths
file1 = "merged_verification_results.csv"
file2 = "merged_verification_results2.csv"
output_file = "merged_combined.csv"

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct full paths
file1_path = os.path.join(script_dir, file1)
file2_path = os.path.join(script_dir, file2)
output_path = os.path.join(script_dir, output_file)

# Read both CSV files
print(f"Reading {file1}...")
df1 = pd.read_csv(file1_path)
print(f"  - Rows: {len(df1)}")

print(f"Reading {file2}...")
df2 = pd.read_csv(file2_path)
print(f"  - Rows: {len(df2)}")

# Merge the dataframes (concatenate vertically)
print("\nMerging dataframes...")
merged_df = pd.concat([df1, df2], ignore_index=True)

print(f"  - Total rows after merge: {len(merged_df)}")
print(f"  - Columns: {list(merged_df.columns)}")

# Save the merged dataframe
print(f"\nSaving to {output_file}...")
merged_df.to_csv(output_path, index=False)

print(f"✓ Successfully merged files!")
print(f"  Output: {output_path}")
