#!/usr/bin/env python3
"""
Script to analyze correlations between different activation functions' durations.
This helps identify which functions have similar performance patterns across different kernel shapes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os

def load_merged_data(file_path="merged_functions_by_shape.csv"):
    """Load the merged functions data."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found!")
        return None
    
    df = pd.read_csv(file_path)
    print(f"Loaded data with shape: {df.shape}")
    return df

def extract_function_columns(df):
    """Extract only the function duration columns."""
    function_cols = [col for col in df.columns if col.endswith('_avg_duration_us')]
    function_names = [col.replace('_avg_duration_us', '') for col in function_cols]
    
    print(f"Found {len(function_cols)} function columns:")
    for name in sorted(function_names):
        print(f"  {name}")
    
    return df[function_cols], function_names

def calculate_correlations(df_functions, method='pearson'):
    """Calculate correlation matrix for function durations."""
    if method == 'pearson':
        corr_matrix = df_functions.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = df_functions.corr(method='spearman')
    else:
        raise ValueError("Method must be 'pearson' or 'spearman'")
    
    # Clean up column names for display
    clean_names = [col.replace('_avg_duration_us', '') for col in corr_matrix.columns]
    corr_matrix.columns = clean_names
    corr_matrix.index = clean_names
    
    return corr_matrix

def plot_correlation_heatmap(corr_matrix, method='Pearson', save_path=None):
    """Create a correlation heatmap."""
    plt.figure(figsize=(12, 10))
    
    # Create mask for upper triangle (optional - shows full matrix)
    # mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                annot=True,           # Show correlation values
                cmap='RdBu_r',        # Red-Blue color scheme (red=high, blue=low)
                center=0,             # Center colormap at 0
                square=True,          # Square cells
                fmt='.3f',            # 3 decimal places
                cbar_kws={'label': f'{method} Correlation Coefficient'})
    
    plt.title(f'{method} Correlation Matrix\nActivation Function Durations', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Activation Functions', fontsize=12)
    plt.ylabel('Activation Functions', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    plt.show()

def find_highest_correlations(corr_matrix, top_n=10):
    """Find the highest correlations (excluding self-correlations)."""
    # Get upper triangle of correlation matrix (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    correlations = []
    
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            if mask[i, j]:
                correlations.append({
                    'function_1': corr_matrix.index[i],
                    'function_2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    # Sort by absolute correlation value
    correlations_df = pd.DataFrame(correlations)
    correlations_df['abs_correlation'] = correlations_df['correlation'].abs()
    correlations_df = correlations_df.sort_values('abs_correlation', ascending=False)
    
    return correlations_df.head(top_n)

def find_lowest_correlations(corr_matrix, top_n=10):
    """Find the lowest correlations (most different functions)."""
    # Get upper triangle of correlation matrix (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    correlations = []
    
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            if mask[i, j]:
                correlations.append({
                    'function_1': corr_matrix.index[i],
                    'function_2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    # Sort by correlation value (lowest first)
    correlations_df = pd.DataFrame(correlations)
    correlations_df = correlations_df.sort_values('correlation', ascending=True)
    
    return correlations_df.head(top_n)

def analyze_correlation_by_size(df, size_ranges=None):
    """Analyze correlations for different tensor size ranges."""
    if size_ranges is None:
        # Define size ranges based on tuple_product
        size_ranges = [
            (0, 1000, "Small (≤1K)"),
            (1000, 4000, "Medium (1K-4K)"),
            (4000, 8200, "Large (4K-8K)")
        ]
    
    results = {}
    
    for min_size, max_size, label in size_ranges:
        subset = df[(df['tuple_product'] >= min_size) & (df['tuple_product'] < max_size)]
        if len(subset) > 1:  # Need at least 2 samples for correlation
            function_cols = [col for col in subset.columns if col.endswith('_avg_duration_us')]
            corr_matrix = subset[function_cols].corr()
            # Clean up names
            clean_names = [col.replace('_avg_duration_us', '') for col in corr_matrix.columns]
            corr_matrix.columns = clean_names
            corr_matrix.index = clean_names
            results[label] = corr_matrix
            print(f"{label}: {len(subset)} samples")
    
    return results

def statistical_significance_test(df_functions):
    """Test statistical significance of correlations."""
    function_cols = df_functions.columns
    n_functions = len(function_cols)
    
    print(f"\nStatistical Significance Tests (n={len(df_functions)}):")
    print("=" * 60)
    print(f"{'Function Pair':<25} {'Pearson r':<12} {'p-value':<12} {'Significant'}")
    print("-" * 60)
    
    significant_pairs = []
    
    for i in range(n_functions):
        for j in range(i+1, n_functions):
            func1 = function_cols[i].replace('_avg_duration_us', '')
            func2 = function_cols[j].replace('_avg_duration_us', '')
            
            r, p_value = pearsonr(df_functions.iloc[:, i], df_functions.iloc[:, j])
            is_significant = p_value < 0.05
            
            print(f"{func1}-{func2:<20} {r:>8.3f}    {p_value:>8.6f}    {'Yes' if is_significant else 'No'}")
            
            if is_significant:
                significant_pairs.append((func1, func2, r, p_value))
    
    return significant_pairs

def create_pairwise_scatter_plots(df, function_names, save_dir="correlation_plots"):
    """Create scatter plots for highly correlated function pairs."""
    os.makedirs(save_dir, exist_ok=True)
    
    function_cols = [f"{name}_avg_duration_us" for name in function_names]
    df_functions = df[function_cols]
    
    # Calculate correlations to find interesting pairs
    corr_matrix = df_functions.corr()
    
    # Find some interesting pairs (high correlation)
    high_corr_pairs = []
    for i in range(len(function_names)):
        for j in range(i+1, len(function_names)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:  # High correlation threshold
                high_corr_pairs.append((function_names[i], function_names[j], corr_val))
    
    # Sort by absolute correlation
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print(f"\nCreating scatter plots for {min(5, len(high_corr_pairs))} highest correlated pairs:")
    
    for i, (func1, func2, corr_val) in enumerate(high_corr_pairs[:5]):
        plt.figure(figsize=(8, 6))
        
        x_col = f"{func1}_avg_duration_us"
        y_col = f"{func2}_avg_duration_us"
        
        plt.scatter(df[x_col], df[y_col], alpha=0.6, s=30)
        plt.xlabel(f"{func1.title()} Average Duration (μs)")
        plt.ylabel(f"{func2.title()} Average Duration (μs)")
        plt.title(f"{func1.title()} vs {func2.title()}\nCorrelation: {corr_val:.3f}")
        
        # Add trend line
        z = np.polyfit(df[x_col], df[y_col], 1)
        p = np.poly1d(z)
        plt.plot(df[x_col].sort_values(), p(df[x_col].sort_values()), "r--", alpha=0.8)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"scatter_{func1}_vs_{func2}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  {func1} vs {func2}: r={corr_val:.3f} -> {save_path}")
        plt.close()  # Close to save memory

def main():
    # Set working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("Activation Function Duration Correlation Analysis")
    print("=" * 55)
    
    # Load data
    df = load_merged_data()
    if df is None:
        return
    
    # Extract function columns
    df_functions, function_names = extract_function_columns(df)
    
    print(f"\nBasic Statistics:")
    print(f"Number of kernel shapes: {len(df)}")
    print(f"Tuple product range: {df['tuple_product'].min()} - {df['tuple_product'].max()}")
    
    # Calculate Pearson correlations
    print(f"\n1. Pearson Correlation Analysis")
    print("-" * 40)
    pearson_corr = calculate_correlations(df_functions, method='pearson')
    
    # Display correlation matrix
    print(f"\nPearson Correlation Matrix:")
    print(pearson_corr.round(3))
    
    # Find highest correlations
    print(f"\nTop 10 Highest Correlations:")
    highest_corr = find_highest_correlations(pearson_corr, top_n=10)
    for _, row in highest_corr.iterrows():
        print(f"  {row['function_1']} ↔ {row['function_2']}: {row['correlation']:.3f}")
    
    # Find lowest correlations
    print(f"\nTop 10 Lowest Correlations (Most Different):")
    lowest_corr = find_lowest_correlations(pearson_corr, top_n=10)
    for _, row in lowest_corr.iterrows():
        print(f"  {row['function_1']} ↔ {row['function_2']}: {row['correlation']:.3f}")
    
    # Statistical significance
    significant_pairs = statistical_significance_test(df_functions)
    print(f"\nFound {len(significant_pairs)} statistically significant correlations (p < 0.05)")
    
    # Spearman correlation (rank-based)
    print(f"\n2. Spearman Correlation Analysis (Rank-based)")
    print("-" * 50)
    spearman_corr = calculate_correlations(df_functions, method='spearman')
    print(f"\nSpearman Correlation Matrix:")
    print(spearman_corr.round(3))
    
    # Analyze by size
    print(f"\n3. Correlation by Tensor Size")
    print("-" * 35)
    size_correlations = analyze_correlation_by_size(df)
    
    # Create visualizations
    print(f"\n4. Creating Visualizations")
    print("-" * 30)
    
    # Plot heatmaps
    plot_correlation_heatmap(pearson_corr, method='Pearson', 
                           save_path="pearson_correlation_heatmap.png")
    
    plot_correlation_heatmap(spearman_corr, method='Spearman', 
                           save_path="spearman_correlation_heatmap.png")
    
    # Create scatter plots for highly correlated pairs
    create_pairwise_scatter_plots(df, function_names)
    
    # Summary insights
    print(f"\n5. Key Insights")
    print("-" * 20)
    
    # Overall correlation summary
    upper_triangle = pearson_corr.where(np.triu(np.ones(pearson_corr.shape), k=1).astype(bool))
    mean_correlation = upper_triangle.stack().mean()
    max_correlation = upper_triangle.stack().max()
    min_correlation = upper_triangle.stack().min()
    
    print(f"Average correlation between functions: {mean_correlation:.3f}")
    print(f"Highest correlation: {max_correlation:.3f}")
    print(f"Lowest correlation: {min_correlation:.3f}")
    
    # Function similarity groups
    high_threshold = 0.9
    similar_pairs = highest_corr[highest_corr['correlation'] > high_threshold]
    if len(similar_pairs) > 0:
        print(f"\nHighly similar functions (r > {high_threshold}):")
        for _, row in similar_pairs.iterrows():
            print(f"  {row['function_1']} ≈ {row['function_2']} (r={row['correlation']:.3f})")
    else:
        print(f"\nNo function pairs with correlation > {high_threshold}")
    
    print(f"\nAnalysis complete! Check generated plots and correlation matrices.")

if __name__ == "__main__":
    main()


