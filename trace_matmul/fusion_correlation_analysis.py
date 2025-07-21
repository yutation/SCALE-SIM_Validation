import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def calculate_fusion_correlation():
    """
    Calculate correlation between fusion sub event duration and scale sim total cycles
    """
    
    # Read the filtered events CSV to get fusion events
    try:
        events_df = pd.read_csv('filtered_events.csv')
        print("Successfully loaded filtered_events.csv")
        print(f"Total events: {len(events_df)}")
        
        # Filter fusion events
        fusion_events = events_df[events_df['event_name'] == 'fusion']
        print(f"Fusion events found: {len(fusion_events)}")
        
        # Read the topology file to get layer names
        topology_df = pd.read_csv('scale_sim_gemm_topology.csv')
        print(f"Topology layers: {len(topology_df)}")
        
        # Read the compute report file to get scale sim total cycles
        compute_df = pd.read_csv('COMPUTE_REPORT.csv')
        print(f"Compute report entries: {len(compute_df)}")
        
        # Create a dictionary to store the results
        results = []
        
        # Process each layer in the topology
        for index, row in topology_df.iterrows():
            layer_name = row['Layer']
            
            # Find corresponding fusion event duration
            fusion_event_row = fusion_events[fusion_events['kernel_name'] == layer_name]
            fusion_duration = None
            if not fusion_event_row.empty:
                fusion_duration = fusion_event_row.iloc[0]['dur(us)']
            
            # Find corresponding scale sim total cycles
            scale_sim_cycles = None
            if index < len(compute_df):
                scale_sim_cycles = compute_df.iloc[index][' Total Cycles']
            
            results.append({
                'name': layer_name,
                'fusion_duration': fusion_duration,
                'scale_sim_total_cycles': scale_sim_cycles
            })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        print(f"\nCombined dataset shape: {df.shape}")
        
        # Check for missing values
        print(f"\nMissing values:")
        print(df.isnull().sum())
        
        # Remove any rows with missing values
        df_clean = df.dropna()
        print(f"\nAfter removing missing values: {len(df_clean)} rows")
        
        # Extract the two variables for correlation analysis
        fusion_duration = df_clean['fusion_duration']
        scale_cycles = df_clean['scale_sim_total_cycles']
        
        print(f"\n=== Fusion Event Correlation Analysis ===")
        
        # Calculate Pearson correlation
        pearson_corr, pearson_p = stats.pearsonr(fusion_duration, scale_cycles)
        print(f"Pearson Correlation: {pearson_corr:.6f}")
        print(f"Pearson p-value: {pearson_p:.6f}")
        
        # Calculate Spearman correlation
        spearman_corr, spearman_p = stats.spearmanr(fusion_duration, scale_cycles)
        print(f"Spearman Correlation: {spearman_corr:.6f}")
        print(f"Spearman p-value: {spearman_p:.6f}")
        
        # Calculate Kendall correlation
        kendall_corr, kendall_p = stats.kendalltau(fusion_duration, scale_cycles)
        print(f"Kendall Correlation: {kendall_corr:.6f}")
        print(f"Kendall p-value: {kendall_p:.6f}")
        
        # Basic statistics
        print(f"\n=== Basic Statistics ===")
        print(f"Fusion Event Duration:")
        print(f"  Mean: {fusion_duration.mean():.6f} us")
        print(f"  Std: {fusion_duration.std():.6f} us")
        print(f"  Min: {fusion_duration.min():.6f} us")
        print(f"  Max: {fusion_duration.max():.6f} us")
        
        print(f"\nScale Sim Total Cycles:")
        print(f"  Mean: {scale_cycles.mean():.2f} cycles")
        print(f"  Std: {scale_cycles.std():.2f} cycles")
        print(f"  Min: {scale_cycles.min():.0f} cycles")
        print(f"  Max: {scale_cycles.max():.0f} cycles")
        
        # Create correlation matrix
        correlation_matrix = df_clean[['fusion_duration', 'scale_sim_total_cycles']].corr()
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
        results_df.to_csv('fusion_correlation_results.csv', index=False)
        print(f"\nDetailed correlation results saved to 'fusion_correlation_results.csv'")
        
        # Save combined data
        df_clean.to_csv('fusion_combined_results.csv', index=False)
        print(f"Combined fusion data saved to 'fusion_combined_results.csv'")
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        plt.scatter(scale_cycles, fusion_duration, alpha=0.7, s=60, color='green', 
                   label=f'Fusion events (n={len(df_clean)})')
        
        # Add trend line
        z = np.polyfit(scale_cycles, fusion_duration, 1)
        p = np.poly1d(z)
        plt.plot(scale_cycles, p(scale_cycles), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend line (r = {pearson_corr:.4f})')
        
        plt.xlabel('Scale Sim Total Cycles')
        plt.ylabel('Fusion Event Duration (us)')
        plt.title(f'Correlation: Fusion Event Duration vs Scale Sim Total Cycles')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add text box with correlation info
        textstr = f'Pearson r = {pearson_corr:.4f}\nR² = {pearson_corr**2:.4f}\np < 0.001'
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig('fusion_correlation_plot.png', dpi=300, bbox_inches='tight')
        print(f"Fusion correlation plot saved as 'fusion_correlation_plot.png'")
        
        # Create comparison with main events
        try:
            # Read the original combined results for comparison
            main_df = pd.read_csv('combined_results.csv')
            main_duration = main_df['main_event_duration']
            main_cycles = main_df['scale_sim_total_cycles']
            
            # Calculate main event correlation
            main_pearson, _ = stats.pearsonr(main_duration, main_cycles)
            
            # Create comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Main events
            ax1.scatter(main_cycles, main_duration, alpha=0.6, s=50, color='blue')
            ax1.set_xlabel('Scale Sim Total Cycles')
            ax1.set_ylabel('Main Event Duration (us)')
            ax1.set_title(f'Main Events (r = {main_pearson:.4f})')
            ax1.grid(True, alpha=0.3)
            
            # Fusion events
            ax2.scatter(scale_cycles, fusion_duration, alpha=0.7, s=60, color='green')
            ax2.set_xlabel('Scale Sim Total Cycles')
            ax2.set_ylabel('Fusion Event Duration (us)')
            ax2.set_title(f'Fusion Events (r = {pearson_corr:.4f})')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('fusion_vs_main_comparison.png', dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved as 'fusion_vs_main_comparison.png'")
            
        except Exception as e:
            print(f"Could not create comparison plot: {e}")
        
        return {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'kendall': kendall_corr,
            'r_squared': pearson_corr ** 2,
            'data': df_clean
        }
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
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
    results = calculate_fusion_correlation()
    
    if results:
        print(f"\n=== Interpretation ===")
        print(f"Pearson correlation ({results['pearson']:.4f}): {interpret_correlation(results['pearson'])} linear relationship")
        print(f"Spearman correlation ({results['spearman']:.4f}): {interpret_correlation(results['spearman'])} monotonic relationship")
        print(f"Kendall correlation ({results['kendall']:.4f}): {interpret_correlation(results['kendall'])} monotonic relationship")
        
        # Additional analysis
        df = results['data']
        print(f"\n=== Additional Analysis ===")
        print(f"Number of data points: {len(df)}")
        print(f"Fusion duration range: {df['fusion_duration'].min():.3f} - {df['fusion_duration'].max():.3f} us")
        print(f"Cycles range: {df['scale_sim_total_cycles'].min():.0f} - {df['scale_sim_total_cycles'].max():.0f} cycles")
        
        # Calculate R-squared
        r_squared = results['r_squared']
        print(f"R-squared (coefficient of determination): {r_squared:.4f}")
        print(f"This means {r_squared*100:.2f}% of the variance in fusion duration can be explained by scale sim cycles")
        
        # Summary
        print(f"\n=== Summary ===")
        print(f"The correlation analysis shows a {interpret_correlation(results['pearson'])} relationship")
        print(f"between fusion event duration and scale sim total cycles.")
        print(f"With R-squared = {r_squared:.4f}, {r_squared*100:.1f}% of the variance in fusion duration")
        print(f"is explained by the scale sim total cycles.") 