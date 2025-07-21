import pandas as pd
import numpy as np

def calculate_correlation():
    """
    Calculate correlation between main event duration and scale sim total cycles
    using only pandas and numpy
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
        
        # Calculate Pearson correlation using pandas
        pearson_corr = main_duration.corr(scale_cycles, method='pearson')
        print(f"Pearson Correlation: {pearson_corr:.6f}")
        
        # Calculate Spearman correlation using pandas
        spearman_corr = main_duration.corr(scale_cycles, method='spearman')
        print(f"Spearman Correlation: {spearman_corr:.6f}")
        
        # Calculate Kendall correlation using pandas
        kendall_corr = main_duration.corr(scale_cycles, method='kendall')
        print(f"Kendall Correlation: {kendall_corr:.6f}")
        
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
            'interpretation': [
                'Linear correlation',
                'Monotonic correlation (rank-based)',
                'Monotonic correlation (concordant pairs)'
            ]
        }
        
        results_df = pd.DataFrame(results)
        results_df.to_csv('correlation_results.csv', index=False)
        print(f"\nDetailed correlation results saved to 'correlation_results.csv'")
        
        # Calculate additional statistics
        print(f"\n=== Additional Statistics ===")
        
        # Calculate R-squared
        r_squared = pearson_corr ** 2
        print(f"R-squared (coefficient of determination): {r_squared:.4f}")
        print(f"This means {r_squared*100:.2f}% of the variance in main event duration can be explained by scale sim cycles")
        
        # Calculate covariance
        covariance = np.cov(main_duration, scale_cycles)[0, 1]
        print(f"Covariance: {covariance:.6f}")
        
        # Calculate correlation ratio
        correlation_ratio = abs(pearson_corr)
        print(f"Correlation ratio (absolute): {correlation_ratio:.4f}")
        
        return {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'kendall': kendall_corr,
            'r_squared': r_squared,
            'covariance': covariance,
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

def calculate_manual_correlation(x, y):
    """
    Calculate Pearson correlation manually for verification
    """
    n = len(x)
    if n != len(y):
        return None
    
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate numerator and denominators
    numerator = np.sum((x - x_mean) * (y - y_mean))
    x_denominator = np.sqrt(np.sum((x - x_mean) ** 2))
    y_denominator = np.sqrt(np.sum((y - y_mean) ** 2))
    
    # Calculate correlation
    if x_denominator == 0 or y_denominator == 0:
        return 0
    
    correlation = numerator / (x_denominator * y_denominator)
    return correlation

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
        
        # Manual verification
        manual_corr = calculate_manual_correlation(df['main_event_duration'], df['scale_sim_total_cycles'])
        print(f"Manual Pearson correlation verification: {manual_corr:.6f}")
        
        # Summary
        print(f"\n=== Summary ===")
        print(f"The correlation analysis shows a {interpret_correlation(results['pearson'])} relationship")
        print(f"between main event duration and scale sim total cycles.")
        print(f"With R-squared = {results['r_squared']:.4f}, we can say that")
        print(f"{results['r_squared']*100:.1f}% of the variance in main event duration")
        print(f"is explained by the scale sim total cycles.") 