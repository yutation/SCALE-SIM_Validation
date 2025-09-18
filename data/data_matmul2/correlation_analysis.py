#!/usr/bin/env python3
"""
Comprehensive correlation analysis between total_cycles and fusion_avg.
Provides statistical information and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_data(file_path):
    """Load the kernel report data."""
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} kernels from {file_path}")
    return df

def basic_statistics(df):
    """Calculate basic statistics for total_cycles and fusion_avg."""
    print("="*60)
    print("BASIC STATISTICS")
    print("="*60)
    
    # Total cycles statistics
    print("TOTAL CYCLES:")
    print(f"  Mean: {df['total_cycles'].mean():.2f}")
    print(f"  Median: {df['total_cycles'].median():.2f}")
    print(f"  Std Dev: {df['total_cycles'].std():.2f}")
    print(f"  Min: {df['total_cycles'].min():.2f}")
    print(f"  Max: {df['total_cycles'].max():.2f}")
    print(f"  Range: {df['total_cycles'].max() - df['total_cycles'].min():.2f}")
    
    print("\nFUSION_AVG:")
    print(f"  Mean: {df['fusion_avg'].mean():.6f}")
    print(f"  Median: {df['fusion_avg'].median():.6f}")
    print(f"  Std Dev: {df['fusion_avg'].std():.6f}")
    print(f"  Min: {df['fusion_avg'].min():.6f}")
    print(f"  Max: {df['fusion_avg'].max():.6f}")
    print(f"  Range: {df['fusion_avg'].max() - df['fusion_avg'].min():.6f}")
    
    # Unique values
    print(f"\nUnique total_cycles values: {df['total_cycles'].nunique()}")
    print(f"Unique fusion_avg values: {df['fusion_avg'].nunique()}")

def correlation_analysis(df):
    """Perform comprehensive correlation analysis."""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Pearson correlation
    pearson_corr, pearson_p = stats.pearsonr(df['total_cycles'], df['fusion_avg'])
    print(f"Pearson Correlation: {pearson_corr:.6f}")
    print(f"Pearson p-value: {pearson_p:.6f}")
    
    # Spearman correlation
    spearman_corr, spearman_p = stats.spearmanr(df['total_cycles'], df['fusion_avg'])
    print(f"Spearman Correlation: {spearman_corr:.6f}")
    print(f"Spearman p-value: {spearman_p:.6f}")
    
    # Kendall correlation
    kendall_corr, kendall_p = stats.kendalltau(df['total_cycles'], df['fusion_avg'])
    print(f"Kendall Correlation: {kendall_corr:.6f}")
    print(f"Kendall p-value: {kendall_p:.6f}")
    
    # Interpretation
    print(f"\nINTERPRETATION:")
    if abs(pearson_corr) < 0.1:
        strength = "negligible"
    elif abs(pearson_corr) < 0.3:
        strength = "weak"
    elif abs(pearson_corr) < 0.5:
        strength = "moderate"
    elif abs(pearson_corr) < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    
    direction = "positive" if pearson_corr > 0 else "negative"
    print(f"  The correlation is {strength} and {direction}")
    
    if pearson_p < 0.001:
        significance = "highly significant"
    elif pearson_p < 0.01:
        significance = "very significant"
    elif pearson_p < 0.05:
        significance = "significant"
    else:
        significance = "not significant"
    
    print(f"  The correlation is {significance} (p < {pearson_p:.6f})")

def group_analysis(df):
    """Analyze correlation by total_cycles groups."""
    print("\n" + "="*60)
    print("GROUP ANALYSIS BY TOTAL CYCLES")
    print("="*60)
    
    # Group by total_cycles
    grouped = df.groupby('total_cycles')['fusion_avg'].agg(['count', 'mean', 'std', 'min', 'max'])
    grouped = grouped.sort_index()
    
    print("Statistics by total_cycles value:")
    print(grouped.to_string())
    
    # Correlation within each group
    print(f"\nCorrelation analysis within each total_cycles group:")
    for cycles in sorted(df['total_cycles'].unique()):
        subset = df[df['total_cycles'] == cycles]
        if len(subset) > 1:  # Need at least 2 points for correlation
            corr, p_val = stats.pearsonr(subset['total_cycles'], subset['fusion_avg'])
            print(f"  total_cycles={cycles}: n={len(subset)}, fusion_avg_mean={subset['fusion_avg'].mean():.6f}")

def create_visualizations(df):
    """Create comprehensive visualizations."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Set up the plotting area
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Correlation Analysis: Total Cycles vs Fusion Average', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot
    axes[0, 0].scatter(df['total_cycles'], df['fusion_avg'], alpha=0.6, s=30)
    axes[0, 0].set_xlabel('Total Cycles')
    axes[0, 0].set_ylabel('Fusion Average (μs)')
    axes[0, 0].set_title('Scatter Plot: Total Cycles vs Fusion Average')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['total_cycles'], df['fusion_avg'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(df['total_cycles'], p(df['total_cycles']), "r--", alpha=0.8, linewidth=2)
    
    # 2. Box plot by total_cycles
    df.boxplot(column='fusion_avg', by='total_cycles', ax=axes[0, 1])
    axes[0, 1].set_title('Box Plot: Fusion Average by Total Cycles')
    axes[0, 1].set_xlabel('Total Cycles')
    axes[0, 1].set_ylabel('Fusion Average (μs)')
    
    # 3. Histogram of fusion_avg
    axes[0, 2].hist(df['fusion_avg'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 2].set_xlabel('Fusion Average (μs)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distribution of Fusion Average')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Violin plot
    sns.violinplot(data=df, x='total_cycles', y='fusion_avg', ax=axes[1, 0])
    axes[1, 0].set_title('Violin Plot: Fusion Average by Total Cycles')
    axes[1, 0].set_xlabel('Total Cycles')
    axes[1, 0].set_ylabel('Fusion Average (μs)')
    
    # 5. Heatmap of correlation matrix
    corr_matrix = df[['total_cycles', 'fusion_avg']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=axes[1, 1], cbar_kws={'shrink': 0.8})
    axes[1, 1].set_title('Correlation Heatmap')
    
    # 6. Joint plot (scatter + histograms)
    joint_ax = axes[1, 2]
    joint_ax.scatter(df['total_cycles'], df['fusion_avg'], alpha=0.6, s=20)
    joint_ax.set_xlabel('Total Cycles')
    joint_ax.set_ylabel('Fusion Average (μs)')
    joint_ax.set_title('Joint Distribution')
    joint_ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient as text
    corr_coef = df['total_cycles'].corr(df['fusion_avg'])
    joint_ax.text(0.05, 0.95, f'Correlation: {corr_coef:.4f}', 
                  transform=joint_ax.transAxes, fontsize=12, 
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved correlation_analysis.png")
    
    # Create additional detailed plots
    create_detailed_plots(df)

def create_detailed_plots(df):
    """Create additional detailed plots."""
    
    # Plot 1: Detailed scatter with kernel names for outliers
    plt.figure(figsize=(12, 8))
    plt.scatter(df['total_cycles'], df['fusion_avg'], alpha=0.6, s=30)
    
    # Find outliers (points beyond 2 standard deviations)
    z_scores = np.abs(stats.zscore(df['fusion_avg']))
    outliers = df[z_scores > 2]
    
    if len(outliers) > 0:
        plt.scatter(outliers['total_cycles'], outliers['fusion_avg'], 
                   color='red', s=50, alpha=0.8, label='Outliers')
        
        # Annotate some outliers
        for idx, row in outliers.head(5).iterrows():
            plt.annotate(row['kernel_name'], 
                        (row['total_cycles'], row['fusion_avg']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
    
    plt.xlabel('Total Cycles')
    plt.ylabel('Fusion Average (μs)')
    plt.title('Scatter Plot with Outliers Highlighted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('correlation_with_outliers.png', dpi=300, bbox_inches='tight')
    print("Saved correlation_with_outliers.png")
    
    # Plot 2: Distribution comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distribution of fusion_avg by total_cycles
    for cycles in sorted(df['total_cycles'].unique()):
        subset = df[df['total_cycles'] == cycles]
        ax1.hist(subset['fusion_avg'], alpha=0.6, label=f'Cycles={cycles}', bins=15)
    
    ax1.set_xlabel('Fusion Average (μs)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Fusion Average by Total Cycles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot with individual points
    sns.boxplot(data=df, x='total_cycles', y='fusion_avg', ax=ax2)
    sns.stripplot(data=df, x='total_cycles', y='fusion_avg', 
                  color='red', alpha=0.5, size=3, ax=ax2)
    ax2.set_title('Box Plot with Individual Points')
    ax2.set_xlabel('Total Cycles')
    ax2.set_ylabel('Fusion Average (μs)')
    
    plt.tight_layout()
    plt.savefig('distribution_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved distribution_analysis.png")

def summary_report(df):
    """Generate a comprehensive summary report."""
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    # Calculate key metrics
    total_kernels = len(df)
    unique_cycles = df['total_cycles'].nunique()
    corr_coef = df['total_cycles'].corr(df['fusion_avg'])
    
    # Find extremes
    min_fusion = df.loc[df['fusion_avg'].idxmin()]
    max_fusion = df.loc[df['fusion_avg'].idxmax()]
    
    print(f"Total kernels analyzed: {total_kernels}")
    print(f"Unique total_cycles values: {unique_cycles}")
    print(f"Overall correlation coefficient: {corr_coef:.6f}")
    print(f"\nLowest fusion_avg: {min_fusion['kernel_name']} ({min_fusion['fusion_avg']:.6f} μs)")
    print(f"Highest fusion_avg: {max_fusion['kernel_name']} ({max_fusion['fusion_avg']:.6f} μs)")
    
    # Summary by total_cycles
    print(f"\nSummary by total_cycles:")
    summary = df.groupby('total_cycles')['fusion_avg'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(summary.to_string())

def main():
    """Main function to run the complete analysis."""
    file_path = "kernel_report_updated.csv"
    
    try:
        # Load data
        df = load_data(file_path)
        
        # Run analyses
        basic_statistics(df)
        correlation_analysis(df)
        group_analysis(df)
        summary_report(df)
        
        # Create visualizations
        create_visualizations(df)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Generated files:")
        print("- correlation_analysis.png")
        print("- correlation_with_outliers.png") 
        print("- distribution_analysis.png")
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
