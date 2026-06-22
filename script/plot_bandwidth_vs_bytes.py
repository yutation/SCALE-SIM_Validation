#!/usr/bin/env python3
"""
Script to generate Bandwidth vs. Raw byte access (MB) figure from merged_copy_events.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_process_data(csv_file):
    """Load CSV data and convert bytes to MB"""
    df = pd.read_csv(csv_file)
    
    # Convert raw_bytes_accessed to MB
    df['raw_bytes_mb'] = df['raw_bytes_accessed'] / (1024 * 1024)
    
    # Extract matrix dimensions from kernel_name for better labeling
    df['matrix_size'] = df['kernel_name'].str.extract(r'matmul_(\d+x\d+x\d+)')
    
    return df

def create_bandwidth_vs_bytes_plot(df, output_file='bandwidth_vs_bytes.png'):
    """Create the main plot"""
    plt.figure(figsize=(12, 8))
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    
    # Create scatter plot with all data points in single color
    plt.scatter(df['raw_bytes_mb'], 
               df['bandwidth_gbps'],
               alpha=0.7, 
               s=60,
               color='#1f77b4')
    
    # Customize plot
    plt.xlabel('Raw Bytes Accessed (MB)', fontsize=14, fontweight='bold')
    plt.ylabel('Bandwidth (GB/s)', fontsize=14, fontweight='bold')
    plt.title('Memory Bandwidth vs. Raw Bytes Accessed\nMatrix Multiplication Copy Events', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Total samples: {len(df)}\n'
    stats_text += f'Bandwidth range: {df["bandwidth_gbps"].min():.1f} - {df["bandwidth_gbps"].max():.1f} GB/s\n'
    stats_text += f'Bytes range: {df["raw_bytes_mb"].min():.1f} - {df["raw_bytes_mb"].max():.1f} MB'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    
    return plt

def create_additional_analysis(df):
    """Create additional analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Additional Analysis of Matrix Multiplication Copy Events', fontsize=16, fontweight='bold')
    
    # Plot 1: Bandwidth distribution by dataset
    df.boxplot(column='bandwidth_gbps', by='dataset', ax=axes[0,0])
    axes[0,0].set_title('Bandwidth Distribution by Dataset')
    axes[0,0].set_ylabel('Bandwidth (GB/s)')
    axes[0,0].set_xlabel('Dataset')
    
    # Plot 2: Duration vs Matrix size
    matrix_sizes = df['matrix_size'].str.split('x').str[0].astype(int)
    axes[0,1].scatter(matrix_sizes, df['dur(us)'], alpha=0.6)
    axes[0,1].set_xlabel('Matrix Dimension (N)')
    axes[0,1].set_ylabel('Duration (μs)')
    axes[0,1].set_title('Duration vs Matrix Dimension')
    
    # Plot 3: Bandwidth efficiency (bandwidth / bytes)
    df['bandwidth_efficiency'] = df['bandwidth_gbps'] / df['raw_bytes_mb']
    axes[1,0].scatter(df['raw_bytes_mb'], df['bandwidth_efficiency'], alpha=0.6)
    axes[1,0].set_xlabel('Raw Bytes Accessed (MB)')
    axes[1,0].set_ylabel('Bandwidth Efficiency (GB/s/MB)')
    axes[1,0].set_title('Bandwidth Efficiency vs Data Size')
    
    # Plot 4: Duration vs Bandwidth
    axes[1,1].scatter(df['dur(us)'], df['bandwidth_gbps'], alpha=0.6)
    axes[1,1].set_xlabel('Duration (μs)')
    axes[1,1].set_ylabel('Bandwidth (GB/s)')
    axes[1,1].set_title('Duration vs Bandwidth')
    
    plt.tight_layout()
    plt.savefig('additional_analysis.png', dpi=300, bbox_inches='tight')
    print("Additional analysis plots saved as additional_analysis.png")

def main():
    """Main function"""
    csv_file = 'validation/data_matmul4/merged_copy_events.csv'
    
    try:
        # Load and process data
        print("Loading data...")
        df = load_and_process_data(csv_file)
        print(f"Loaded {len(df)} data points")
        
        # Display basic statistics
        print("\nData Summary:")
        print(f"Total samples: {len(df)}")
        print(f"Bandwidth range: {df['bandwidth_gbps'].min():.2f} - {df['bandwidth_gbps'].max():.2f} GB/s")
        print(f"Bytes range: {df['raw_bytes_mb'].min():.2f} - {df['raw_bytes_mb'].max():.2f} MB")
        
        # Create main plot
        print("\nCreating main plot...")
        plt = create_bandwidth_vs_bytes_plot(df)
        
        # Create additional analysis
        print("\nCreating additional analysis plots...")
        create_additional_analysis(df)
        
        # Show the main plot
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        print("Please make sure the CSV file exists in the specified path.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
