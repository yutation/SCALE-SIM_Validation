#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def merge_datasets():
    """
    Merge the three CSV datasets into one combined dataset.
    """
    # Read the three CSV files
    df1 = pd.read_csv("filtered_events_copy_done.csv")
    df2 = pd.read_csv("filtered_events_copy_done2.csv")
    df3 = pd.read_csv("filtered_events_copy_done3.csv")
    
    # Add a dataset identifier column
    df1['dataset'] = 'small'
    df2['dataset'] = 'medium'
    df3['dataset'] = 'large'
    
    # Combine all datasets
    combined_df = pd.concat([df1, df2, df3], ignore_index=True)
    
    # Save merged dataset
    combined_df.to_csv("merged_copy_events.csv", index=False)
    print(f"Merged dataset saved with {len(combined_df)} rows")
    print(f"Dataset sizes: Small={len(df1)}, Medium={len(df2)}, Large={len(df3)}")
    
    return combined_df

def visualize_bandwidth_vs_bytes(df):
    """
    Create visualizations showing the relationship between raw_bytes_accessed and bandwidth.
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Relationship between Raw Bytes Accessed and Bandwidth', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot: raw_bytes_accessed vs bandwidth
    ax1 = axes[0, 0]
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        ax1.scatter(subset['raw_bytes_accessed'], subset['bandwidth_gbps'], 
                   label=dataset.capitalize(), alpha=0.7, s=50)
    
    ax1.set_xlabel('Raw Bytes Accessed')
    ax1.set_ylabel('Bandwidth (GB/s)')
    ax1.set_title('Bandwidth vs Raw Bytes Accessed')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Log-log plot for better visualization of relationship
    ax2 = axes[0, 1]
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        ax2.scatter(subset['raw_bytes_accessed'], subset['bandwidth_gbps'], 
                   label=dataset.capitalize(), alpha=0.7, s=50)
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Raw Bytes Accessed (log scale)')
    ax2.set_ylabel('Bandwidth (GB/s, log scale)')
    ax2.set_title('Log-Log Plot: Bandwidth vs Raw Bytes Accessed')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot by dataset
    ax3 = axes[1, 0]
    df.boxplot(column='bandwidth_gbps', by='dataset', ax=ax3)
    ax3.set_title('Bandwidth Distribution by Dataset')
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Bandwidth (GB/s)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Histogram of bandwidth
    ax4 = axes[1, 1]
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        ax4.hist(subset['bandwidth_gbps'], bins=15, alpha=0.6, label=dataset.capitalize())
    
    ax4.set_xlabel('Bandwidth (GB/s)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Bandwidth Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bandwidth_vs_bytes_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total samples: {len(df)}")
    print(f"Bandwidth range: {df['bandwidth_gbps'].min():.3f} - {df['bandwidth_gbps'].max():.3f} GB/s")
    print(f"Raw bytes range: {df['raw_bytes_accessed'].min():,} - {df['raw_bytes_accessed'].max():,} bytes")
    
    print("\n=== Statistics by Dataset ===")
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        print(f"\n{dataset.capitalize()} Dataset:")
        print(f"  Count: {len(subset)}")
        print(f"  Bandwidth: {subset['bandwidth_gbps'].mean():.3f} ± {subset['bandwidth_gbps'].std():.3f} GB/s")
        print(f"  Raw bytes: {subset['raw_bytes_accessed'].mean():,.0f} ± {subset['raw_bytes_accessed'].std():,.0f} bytes")

def analyze_correlation(df):
    """
    Analyze correlation between raw_bytes_accessed and bandwidth.
    """
    correlation = df['raw_bytes_accessed'].corr(df['bandwidth_gbps'])
    print(f"\n=== Correlation Analysis ===")
    print(f"Correlation between raw_bytes_accessed and bandwidth: {correlation:.4f}")
    
    # Calculate correlation by dataset
    print("\nCorrelation by dataset:")
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        corr = subset['raw_bytes_accessed'].corr(subset['bandwidth_gbps'])
        print(f"  {dataset.capitalize()}: {corr:.4f}")

if __name__ == "__main__":
    # Merge datasets
    print("Merging datasets...")
    combined_df = merge_datasets()
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_bandwidth_vs_bytes(combined_df)
    
    # Analyze correlations
    analyze_correlation(combined_df)
    
    print(f"\nAnalysis complete! Check 'merged_copy_events.csv' for the merged data")
    print(f"and 'bandwidth_vs_bytes_analysis.png' for the visualizations.")
