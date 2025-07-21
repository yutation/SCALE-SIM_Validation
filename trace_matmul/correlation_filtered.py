import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def calculate_filtered_correlation():
    """
    Calculate correlation between main event duration and scale sim total cycles
    Filtering out cycles > 4000 to reduce outlier influence
    """
    
    # Read the combined results CSV
    try:
        df = pd.read_csv('combined_results.csv')
        print("Successfully loaded combined_results.csv")
        print(f"Original dataset shape: {df.shape}")
        
        # Filter data to only include cycles < 4000
        df_filtered = df[df['scale_sim_total_cycles'] < 4000]
        print(f"Filtered dataset shape (cycles < 4000): {df_filtered.shape}")
        print(f"Removed {len(df) - len(df_filtered)} data points with cycles >= 4000")
        
        # Check for missing values
        print(f"\nMissing values in filtered data:")
        print(df_filtered.isnull().sum())
        
        # Remove any rows with missing values
        df_clean = df_filtered.dropna()
        print(f"\nAfter removing missing values: {len(df_clean)} rows")
        
        # Extract the two variables for correlation analysis
        main_duration = df_clean['main_event_duration']
        scale_cycles = df_clean['scale_sim_total_cycles']
        
        print(f"\n=== Filtered Correlation Analysis (cycles < 4000) ===")
        
        # Calculate Pearson correlation
        pearson_corr, pearson_p = stats.pearsonr(main_duration, scale_cycles)
        print(f"Pearson Correlation: {pearson_corr:.6f}")
        print(f"Pearson p-value: {pearson_p:.6f}")
        
        # Calculate Spearman correlation
        spearman_corr, spearman_p = stats.spearmanr(main_duration, scale_cycles)
        print(f"Spearman Correlation: {spearman_corr:.6f}")
        print(f"Spearman p-value: {spearman_p:.6f}")
        
        # Calculate Kendall correlation
        kendall_corr, kendall_p = stats.kendalltau(main_duration, scale_cycles)
        print(f"Kendall Correlation: {kendall_corr:.6f}")
        print(f"Kendall p-value: {kendall_p:.6f}")
        
        # Basic statistics
        print(f"\n=== Basic Statistics (Filtered) ===")
        print(f"Main Event Duration:")
        print(f"  Mean: {main_duration.mean():.6f} us")
        print(f"  Std: {main_duration.std():.6f} us")
        print(f"  Min: {main_duration.min():.6f} us")
        print(f"  Max: {main_duration.max():.6f} us")
        
        print(f"\nScale Sim Total Cycles:")
        print(f"  Mean: {scale_cycles.mean():.2f} cycles")
        print(f"  Std: {scale_cycles.std():.2f} cycles")
        print(f"  Min: {scale_cycles.min():.0f} cycles")
        print(f"  Max: {scale_cycles.max():.0f} cycles")
        
        # Create correlation matrix
        correlation_matrix = df_clean[['main_event_duration', 'scale_sim_total_cycles']].corr()
        print(f"\n=== Correlation Matrix (Filtered) ===")
        print(correlation_matrix)
        
        # Save detailed results to CSV
        results = {
            'correlation_type': ['Pearson', 'Spearman', 'Kendall'],
            'correlation_coefficient': [pearson_corr, spearman_corr, kendall_corr],
            'p_value': [pearson_p, spearman_p, kendall_p],
            'interpretation': [
                'Linear correlation',
                'Monotonic correlation (rank-based)',
                'Monotonic correlation (concordant pairs)'
            ]
        }
        
        results_df = pd.DataFrame(results)
        results_df.to_csv('correlation_results_filtered.csv', index=False)
        print(f"\nDetailed correlation results saved to 'correlation_results_filtered.csv'")
        
        # Create scatter plot with filtered data
        plt.figure(figsize=(12, 8))
        
        # Plot filtered data points
        plt.scatter(scale_cycles, main_duration, alpha=0.7, s=60, color='blue', 
                   label=f'Filtered data (cycles < 4000, n={len(df_clean)})')
        
        # Add trend line for filtered data
        z = np.polyfit(scale_cycles, main_duration, 1)
        p = np.poly1d(z)
        plt.plot(scale_cycles, p(scale_cycles), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend line (r = {pearson_corr:.4f})')
        
        # Add excluded data points in different color
        excluded_data = df[df['scale_sim_total_cycles'] >= 4000]
        if len(excluded_data) > 0:
            plt.scatter(excluded_data['scale_sim_total_cycles'], excluded_data['main_event_duration'], 
                       alpha=0.3, s=40, color='red', marker='x', 
                       label=f'Excluded data (cycles >= 4000, n={len(excluded_data)})')
        
        plt.xlabel('Scale Sim Total Cycles')
        plt.ylabel('Main Event Duration (us)')
        plt.title(f'Correlation: Main Event Duration vs Scale Sim Total Cycles\nFiltered Analysis (cycles < 4000)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add text box with correlation info
        textstr = f'Pearson r = {pearson_corr:.4f}\nR² = {pearson_corr**2:.4f}\np < 0.001'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig('correlation_plot_filtered.png', dpi=300, bbox_inches='tight')
        print(f"Filtered scatter plot saved as 'correlation_plot_filtered.png'")
        
        # Create comparison plot showing before and after filtering
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Original data
        ax1.scatter(df['scale_sim_total_cycles'], df['main_event_duration'], alpha=0.6, s=50, color='blue')
        ax1.set_xlabel('Scale Sim Total Cycles')
        ax1.set_ylabel('Main Event Duration (us)')
        ax1.set_title('Original Data (All Points)')
        ax1.grid(True, alpha=0.3)
        
        # Filtered data
        ax2.scatter(scale_cycles, main_duration, alpha=0.7, s=60, color='green')
        ax2.set_xlabel('Scale Sim Total Cycles')
        ax2.set_ylabel('Main Event Duration (us)')
        ax2.set_title('Filtered Data (cycles < 4000)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('correlation_comparison_filtered.png', dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved as 'correlation_comparison_filtered.png'")
        
        return {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'kendall': kendall_corr,
            'r_squared': pearson_corr ** 2,
            'data_filtered': df_clean,
            'data_original': df,
            'excluded_count': len(df) - len(df_clean)
        }
        
    except FileNotFoundError:
        print("Error: combined_results.csv not found!")
        print("Please run combine_data_enhanced.py first to generate the combined results.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def interpret_correlation(correlation_value):
    """
    Interpret correlation strength based on absolute value
    """
    abs_corr = abs(correlation_value)
    if abs_corr >= 0.9:
        return "Very strong"
    elif abs_corr >= 0.7:
        return "Strong"
    elif abs_corr >= 0.5:
        return "Moderate"
    elif abs_corr >= 0.3:
        return "Weak"
    else:
        return "Very weak"

if __name__ == "__main__":
    results = calculate_filtered_correlation()
    
    if results:
        print(f"\n=== Interpretation (Filtered Data) ===")
        print(f"Pearson correlation ({results['pearson']:.4f}): {interpret_correlation(results['pearson'])} linear relationship")
        print(f"Spearman correlation ({results['spearman']:.4f}): {interpret_correlation(results['spearman'])} monotonic relationship")
        print(f"Kendall correlation ({results['kendall']:.4f}): {interpret_correlation(results['kendall'])} monotonic relationship")
        
        # Additional analysis
        df_filtered = results['data_filtered']
        print(f"\n=== Additional Analysis ===")
        print(f"Number of data points (filtered): {len(df_filtered)}")
        print(f"Number of excluded points: {results['excluded_count']}")
        print(f"Filtered data range: {df_filtered['main_event_duration'].min():.3f} - {df_filtered['main_event_duration'].max():.3f} us")
        print(f"Filtered cycles range: {df_filtered['scale_sim_total_cycles'].min():.0f} - {df_filtered['scale_sim_total_cycles'].max():.0f} cycles")
        
        # Calculate R-squared
        r_squared = results['r_squared']
        print(f"R-squared (coefficient of determination): {r_squared:.4f}")
        print(f"This means {r_squared*100:.2f}% of the variance in main event duration can be explained by scale sim cycles")
        
        # Summary
        print(f"\n=== Summary ===")
        print(f"By filtering out data points with cycles >= 4000, we removed {results['excluded_count']} outliers.")
        print(f"The correlation analysis on the remaining {len(df_filtered)} data points shows a")
        print(f"{interpret_correlation(results['pearson'])} relationship between main event duration and scale sim cycles.")
        print(f"With R-squared = {r_squared:.4f}, {r_squared*100:.1f}% of the variance is explained.") 