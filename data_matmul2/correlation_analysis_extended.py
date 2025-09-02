#!/usr/bin/env python3
"""
Extended correlation analysis between total_cycles and fusion metrics.
Includes fusion_avg, fusion_min, fusion_max, and fusion_stddev.
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
    """Calculate basic statistics for all fusion metrics."""
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
    
    # Fusion metrics statistics
    fusion_metrics = ['fusion_avg', 'fusion_min', 'fusion_max', 'fusion_stddev']
    
    for metric in fusion_metrics:
        print(f"\n{metric.upper().replace('_', ' ')}:")
        print(f"  Mean: {df[metric].mean():.6f}")
        print(f"  Median: {df[metric].median():.6f}")
        print(f"  Std Dev: {df[metric].std():.6f}")
        print(f"  Min: {df[metric].min():.6f}")
        print(f"  Max: {df[metric].max():.6f}")
        print(f"  Range: {df[metric].max() - df[metric].min():.6f}")
    
    # Unique values
    print(f"\nUnique total_cycles values: {df['total_cycles'].nunique()}")
    for metric in fusion_metrics:
        print(f"Unique {metric} values: {df[metric].nunique()}")

def correlation_analysis(df):
    """Perform comprehensive correlation analysis for all fusion metrics."""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    fusion_metrics = ['fusion_avg', 'fusion_min', 'fusion_max', 'fusion_stddev']
    
    for metric in fusion_metrics:
        print(f"\n{metric.upper().replace('_', ' ')} vs TOTAL_CYCLES:")
        print("-" * 50)
        
        # Pearson correlation
        pearson_corr, pearson_p = stats.pearsonr(df['total_cycles'], df[metric])
        print(f"  Pearson Correlation: {pearson_corr:.6f}")
        print(f"  Pearson p-value: {pearson_p:.6f}")
        
        # Spearman correlation
        spearman_corr, spearman_p = stats.spearmanr(df['total_cycles'], df[metric])
        print(f"  Spearman Correlation: {spearman_corr:.6f}")
        print(f"  Spearman p-value: {spearman_p:.6f}")
        
        # Kendall correlation
        kendall_corr, kendall_p = stats.kendalltau(df['total_cycles'], df[metric])
        print(f"  Kendall Correlation: {kendall_corr:.6f}")
        print(f"  Kendall p-value: {kendall_p:.6f}")
        
        # Interpretation
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
        print(f"  Interpretation: {strength} {direction} correlation")
        
        if pearson_p < 0.001:
            significance = "highly significant"
        elif pearson_p < 0.01:
            significance = "very significant"
        elif pearson_p < 0.05:
            significance = "significant"
        else:
            significance = "not significant"
        
        print(f"  Significance: {significance} (p < {pearson_p:.6f})")

def correlation_matrix_analysis(df):
    """Analyze correlation matrix between all variables."""
    print("\n" + "="*60)
    print("CORRELATION MATRIX ANALYSIS")
    print("="*60)
    
    # Select relevant columns
    columns = ['total_cycles', 'fusion_avg', 'fusion_min', 'fusion_max', 'fusion_stddev']
    corr_matrix = df[columns].corr()
    
    print("Correlation Matrix:")
    print(corr_matrix.round(4))
    
    # Find strongest correlations
    print(f"\nStrongest correlations with total_cycles:")
    total_cycles_corr = corr_matrix['total_cycles'].drop('total_cycles')
    sorted_corr = total_cycles_corr.abs().sort_values(ascending=False)
    
    for metric, corr_value in sorted_corr.items():
        actual_corr = total_cycles_corr[metric]
        print(f"  {metric}: {actual_corr:.4f}")

def group_analysis(df):
    """Analyze all fusion metrics by total_cycles groups."""
    print("\n" + "="*60)
    print("GROUP ANALYSIS BY TOTAL CYCLES")
    print("="*60)
    
    fusion_metrics = ['fusion_avg', 'fusion_min', 'fusion_max', 'fusion_stddev']
    
    # Group by total_cycles and calculate statistics for each fusion metric
    grouped_stats = {}
    for metric in fusion_metrics:
        grouped_stats[metric] = df.groupby('total_cycles')[metric].agg(['count', 'mean', 'std', 'min', 'max'])
    
    # Print statistics for each metric
    for metric in fusion_metrics:
        print(f"\n{metric.upper().replace('_', ' ')} by Total Cycles:")
        print(grouped_stats[metric].round(6))

def create_comprehensive_visualizations(df):
    """Create comprehensive visualizations for all fusion metrics."""
    print("\n" + "="*60)
    print("CREATING COMPREHENSIVE VISUALIZATIONS")
    print("="*60)
    
    fusion_metrics = ['fusion_avg', 'fusion_min', 'fusion_max', 'fusion_stddev']
    
    # Create a large figure with subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Correlation Analysis: Total Cycles vs Fusion Metrics', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(fusion_metrics):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Scatter plot
        ax.scatter(df['total_cycles'], df[metric], alpha=0.6, s=30)
        ax.set_xlabel('Total Cycles')
        ax.set_ylabel(f'{metric.replace("_", " ").title()} (μs)')
        ax.set_title(f'{metric.replace("_", " ").title()} vs Total Cycles')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['total_cycles'], df[metric], 1)
        p = np.poly1d(z)
        ax.plot(df['total_cycles'], p(df['total_cycles']), "r--", alpha=0.8, linewidth=2)
        
        # Add correlation coefficient
        corr_coef = df['total_cycles'].corr(df[metric])
        ax.text(0.05, 0.95, f'Correlation: {corr_coef:.4f}', 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('comprehensive_correlation_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved comprehensive_correlation_analysis.png")
    
    # Create correlation heatmap
    create_correlation_heatmap(df)
    
    # Create detailed plots for each metric
    create_detailed_plots(df)

def create_correlation_heatmap(df):
    """Create correlation heatmap for all variables."""
    plt.figure(figsize=(10, 8))
    
    columns = ['total_cycles', 'fusion_avg', 'fusion_min', 'fusion_max', 'fusion_stddev']
    corr_matrix = df[columns].corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, cbar_kws={'shrink': 0.8}, fmt='.3f')
    plt.title('Correlation Heatmap: Total Cycles vs Fusion Metrics')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved correlation_heatmap.png")

def create_detailed_plots(df):
    """Create detailed plots for each fusion metric."""
    fusion_metrics = ['fusion_avg', 'fusion_min', 'fusion_max', 'fusion_stddev']
    
    for metric in fusion_metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot with outliers
        ax1.scatter(df['total_cycles'], df[metric], alpha=0.6, s=30)
        
        # Find outliers (points beyond 2 standard deviations)
        z_scores = np.abs(stats.zscore(df[metric]))
        outliers = df[z_scores > 2]
        
        if len(outliers) > 0:
            ax1.scatter(outliers['total_cycles'], outliers[metric], 
                       color='red', s=50, alpha=0.8, label='Outliers')
            ax1.legend()
        
        ax1.set_xlabel('Total Cycles')
        ax1.set_ylabel(f'{metric.replace("_", " ").title()} (μs)')
        ax1.set_title(f'{metric.replace("_", " ").title()} vs Total Cycles (with Outliers)')
        ax1.grid(True, alpha=0.3)
        
        # Box plot by total_cycles
        df.boxplot(column=metric, by='total_cycles', ax=ax2)
        ax2.set_title(f'{metric.replace("_", " ").title()} by Total Cycles')
        ax2.set_xlabel('Total Cycles')
        ax2.set_ylabel(f'{metric.replace("_", " ").title()} (μs)')
        
        plt.suptitle(f'Detailed Analysis: {metric.replace("_", " ").title()}')
        plt.tight_layout()
        plt.savefig(f'detailed_{metric}_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved detailed_{metric}_analysis.png")

def summary_report(df):
    """Generate a comprehensive summary report for all metrics."""
    print("\n" + "="*60)
    print("COMPREHENSIVE SUMMARY REPORT")
    print("="*60)
    
    fusion_metrics = ['fusion_avg', 'fusion_min', 'fusion_max', 'fusion_stddev']
    
    # Calculate key metrics
    total_kernels = len(df)
    unique_cycles = df['total_cycles'].nunique()
    
    print(f"Total kernels analyzed: {total_kernels}")
    print(f"Unique total_cycles values: {unique_cycles}")
    
    # Find extremes for each metric
    for metric in fusion_metrics:
        min_val = df.loc[df[metric].idxmin()]
        max_val = df.loc[df[metric].idxmax()]
        corr_coef = df['total_cycles'].corr(df[metric])
        
        print(f"\n{metric.upper().replace('_', ' ')}:")
        print(f"  Correlation with total_cycles: {corr_coef:.6f}")
        print(f"  Lowest: {min_val['kernel_name']} ({min_val[metric]:.6f} μs)")
        print(f"  Highest: {max_val['kernel_name']} ({max_val[metric]:.6f} μs)")
    
    # Summary by total_cycles for all metrics
    print(f"\nSummary by total_cycles for all fusion metrics:")
    for metric in fusion_metrics:
        print(f"\n{metric.upper().replace('_', ' ')}:")
        summary = df.groupby('total_cycles')[metric].agg(['count', 'mean', 'std', 'min', 'max'])
        print(summary.round(6).to_string())

def main():
    """Main function to run the complete extended analysis."""
    file_path = "kernel_report_updated.csv"
    
    try:
        # Load data
        df = load_data(file_path)
        
        # Run analyses
        basic_statistics(df)
        correlation_analysis(df)
        correlation_matrix_analysis(df)
        group_analysis(df)
        summary_report(df)
        
        # Create visualizations
        create_comprehensive_visualizations(df)
        
        print("\n" + "="*60)
        print("EXTENDED ANALYSIS COMPLETE")
        print("="*60)
        print("Generated files:")
        print("- comprehensive_correlation_analysis.png")
        print("- correlation_heatmap.png")
        print("- detailed_fusion_avg_analysis.png")
        print("- detailed_fusion_min_analysis.png")
        print("- detailed_fusion_max_analysis.png")
        print("- detailed_fusion_stddev_analysis.png")
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
