#!/usr/bin/env python3
"""
Simple script to generate Scatter plot with correlation and Violin plot.
Based on the comprehensive correlation analysis script.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("Blues_d")

def load_data(file_path):
    """Load the kernel report data."""
    df = pd.read_csv(file_path)
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()
    print(f"Loaded {len(df)} kernels from {file_path}")
    print(f"Columns: {list(df.columns)}")
    return df

def create_simple_plots(df):
    """Create only Scatter plot with correlation and Violin plot."""
    print("Creating Scatter plot with correlation and Violin plot...")
    
    # Set up the plotting area - 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Correlation Analysis: Total Cycles vs Fusion Average', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot with correlation
    ax1.scatter(df['total_cycles'], df['fusion_avg'], alpha=0.6, s=30, color='steelblue')
    ax1.set_xlabel('Total Cycles')
    ax1.set_ylabel('Fusion Average (μs)')
    ax1.set_title('Scatter Plot: Total Cycles vs Fusion Average')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['total_cycles'], df['fusion_avg'], 1)
    p = np.poly1d(z)
    ax1.plot(df['total_cycles'], p(df['total_cycles']), color='steelblue', linestyle='--', alpha=0.8, linewidth=2)
    
    # Add correlation coefficient as text
    corr_coef = df['total_cycles'].corr(df['fusion_avg'])
    ax1.text(0.05, 0.95, f'Correlation: {corr_coef:.4f}', 
              transform=ax1.transAxes, fontsize=12, 
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Violin plot
    sns.violinplot(data=df, x='total_cycles', y='fusion_avg', ax=ax2)
    ax2.set_title('Violin Plot: Fusion Average by Total Cycles')
    ax2.set_xlabel('Total Cycles')
    ax2.set_ylabel('Fusion Average (μs)')
    
    plt.tight_layout()
    plt.savefig('simple_correlation_plots.png', dpi=300, bbox_inches='tight')
    print("Saved simple_correlation_plots.png")
    
    # Display the plots
    plt.show()

def main():
    """Main function to run the simple plot generation."""
    file_path = "kernel_report_updated.csv"
    
    try:
        # Load data
        df = load_data(file_path)
        
        # Create only the two plots
        create_simple_plots(df)
        
        print("\n" + "="*60)
        print("PLOT GENERATION COMPLETE")
        print("="*60)
        print("Generated file: simple_correlation_plots.png")
        
    except Exception as e:
        print(f"Error during plot generation: {e}")

if __name__ == "__main__":
    main()
