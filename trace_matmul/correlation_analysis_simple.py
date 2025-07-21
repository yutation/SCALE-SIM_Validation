import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def calculate_correlation():
    """
    Calculate correlation between main event duration and scale sim total cycles
    """
    
    # Read the combined results CSV
    try:
        df = pd.read_csv('combined_results.csv')
        print("Successfully loaded combined_results.csv")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for missing values
        print(f"\nMissing values:")
        print(df.isnull().sum())
        
        # Remove any rows with missing values
        df_clean = df.dropna()
        print(f"\nAfter removing missing values: {len(df_clean)} rows")
        
        # Extract the two variables for correlation analysis
        main_duration = df_clean['main_event_duration']
        scale_cycles = df_clean['scale_sim_total_cycles']
        
        print(f"\n=== Correlation Analysis ===")
        
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
        print(f"\n=== Basic Statistics ===")
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
        print(f"\n=== Correlation Matrix ===")
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
        results_df.to_csv('correlation_results.csv', index=False)
        print(f"\nDetailed correlation results saved to 'correlation_results.csv'")
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(scale_cycles, main_duration, alpha=0.6, s=50, color='blue')
        plt.xlabel('Scale Sim Total Cycles')
        plt.ylabel('Main Event Duration (us)')
        plt.title(f'Correlation: Main Event Duration vs Scale Sim Total Cycles\nPearson r = {pearson_corr:.4f}')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(scale_cycles, main_duration, 1)
        p = np.poly1d(z)
        plt.plot(scale_cycles, p(scale_cycles), "r--", alpha=0.8, linewidth=2, label='Trend line')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('correlation_plot.png', dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved as 'correlation_plot.png'")
        
        # Create simple correlation visualization
        plt.figure(figsize=(8, 6))
        correlation_values = [pearson_corr, spearman_corr, kendall_corr]
        correlation_names = ['Pearson', 'Spearman', 'Kendall']
        colors = ['red', 'blue', 'green']
        
        bars = plt.bar(correlation_names, correlation_values, color=colors, alpha=0.7)
        plt.ylabel('Correlation Coefficient')
        plt.title('Correlation Coefficients Comparison')
        plt.ylim(-1, 1)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, correlation_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('correlation_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Correlation comparison chart saved as 'correlation_comparison.png'")
        
        return {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'kendall': kendall_corr,
            'data': df_clean
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
    results = calculate_correlation()
    
    if results:
        print(f"\n=== Interpretation ===")
        print(f"Pearson correlation ({results['pearson']:.4f}): {interpret_correlation(results['pearson'])} linear relationship")
        print(f"Spearman correlation ({results['spearman']:.4f}): {interpret_correlation(results['spearman'])} monotonic relationship")
        print(f"Kendall correlation ({results['kendall']:.4f}): {interpret_correlation(results['kendall'])} monotonic relationship")
        
        # Additional analysis
        df = results['data']
        print(f"\n=== Additional Analysis ===")
        print(f"Number of data points: {len(df)}")
        print(f"Data range: {df['main_event_duration'].min():.3f} - {df['main_event_duration'].max():.3f} us")
        print(f"Cycles range: {df['scale_sim_total_cycles'].min():.0f} - {df['scale_sim_total_cycles'].max():.0f} cycles")
        
        # Calculate R-squared
        r_squared = results['pearson'] ** 2
        print(f"R-squared (coefficient of determination): {r_squared:.4f}")
        print(f"This means {r_squared*100:.2f}% of the variance in main event duration can be explained by scale sim cycles") 