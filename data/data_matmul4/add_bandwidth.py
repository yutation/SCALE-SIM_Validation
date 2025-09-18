#!/usr/bin/env python3
import pandas as pd
import numpy as np

def add_bandwidth_column(csv_file):
    """
    Add a bandwidth column to the CSV file using raw_bytes_accessed and dur(us).
    Bandwidth is calculated in GB/s.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Calculate bandwidth in GB/s
    # Formula: (raw_bytes_accessed / (dur_us * 1e-6)) / 1e9
    # This converts bytes per microsecond to GB per second
    df['bandwidth_gbps'] = (df['raw_bytes_accessed'] / (df['dur(us)'] * 1e-6)) / 1e9
    
    # Round to 3 decimal places for readability
    df['bandwidth_gbps'] = df['bandwidth_gbps'].round(3)
    
    # Reorder columns to put bandwidth after dur(us)
    columns = list(df.columns)
    dur_index = columns.index('dur(us)')
    columns.insert(dur_index + 1, columns.pop(columns.index('bandwidth_gbps')))
    df = df[columns]
    
    # Write back to the same file
    df.to_csv(csv_file, index=False)
    
    print(f"Added bandwidth_gbps column to {csv_file}")
    print(f"Bandwidth range: {df['bandwidth_gbps'].min():.3f} - {df['bandwidth_gbps'].max():.3f} GB/s")
    
    return df

if __name__ == "__main__":
    csv_file = "filtered_events_copy_done3.csv"
    df = add_bandwidth_column(csv_file)
    
    # Display first few rows to verify
    print("\nFirst few rows of updated CSV:")
    print(df.head())
