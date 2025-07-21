import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_cycle_stats_comparison():
    """
    Create a comprehensive figure showing duration statistics vs cycle values
    """
    
    # Read the analysis data
    try:
        analysis_df = pd.read_csv('cycle_configuration_analysis.csv')
        print("Successfully loaded cycle_configuration_analysis.csv")
        
        # Group by cycles and calculate statistics
        cycle_stats = []
        
        for cycles, group in analysis_df.groupby('cycles'):
            if len(group) > 1:  # Only include cycles with multiple configurations
                stats = {
                    'cycles': cycles,
                    'num_configs': len(group),
                    
                    # Main duration stats
                    'main_mean': group['main_duration'].mean(),
                    'main_std': group['main_duration'].std(),
                    'main_range': group['main_duration'].max() - group['main_duration'].min(),
                    'main_cv': group['main_duration'].std() / group['main_duration'].mean() * 100,
                    
                    # Fusion duration stats
                    'fusion_mean': group['fusion_duration'].mean(),
                    'fusion_std': group['fusion_duration'].std(),
                    'fusion_range': group['fusion_duration'].max() - group['fusion_duration'].min(),
                    'fusion_cv': group['fusion_duration'].std() / group['fusion_duration'].mean() * 100,
                    
                    # Main-Fusion difference stats
                    'diff_mean': group['main_minus_fusion'].mean(),
                    'diff_std': group['main_minus_fusion'].std(),
                    'diff_range': group['main_minus_fusion'].max() - group['main_minus_fusion'].min(),
                    'diff_cv': group['main_minus_fusion'].std() / abs(group['main_minus_fusion'].mean()) * 100 if group['main_minus_fusion'].mean() != 0 else 0
                }
                cycle_stats.append(stats)
        
        stats_df = pd.DataFrame(cycle_stats)
        stats_df = stats_df.sort_values('cycles')
        
        print(f"Generated statistics for {len(stats_df)} cycle groups")
        
        # Create the comprehensive figure
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Duration Statistics vs Scale Sim Cycles\n(For configurations with same cycle count)', fontsize=16, fontweight='bold')
        
        # 1. Mean durations vs cycles
        axes[0, 0].plot(stats_df['cycles'], stats_df['main_mean'], 'o-', label='Main Duration', linewidth=2, markersize=8)
        axes[0, 0].plot(stats_df['cycles'], stats_df['fusion_mean'], 's-', label='Fusion Duration', linewidth=2, markersize=8)
        axes[0, 0].plot(stats_df['cycles'], stats_df['diff_mean'], '^-', label='Main-Fusion Diff', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Scale Sim Cycles')
        axes[0, 0].set_ylabel('Mean Duration (μs)')
        axes[0, 0].set_title('Mean Duration vs Cycles')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Standard deviation vs cycles
        axes[0, 1].plot(stats_df['cycles'], stats_df['main_std'], 'o-', label='Main Duration', linewidth=2, markersize=8)
        axes[0, 1].plot(stats_df['cycles'], stats_df['fusion_std'], 's-', label='Fusion Duration', linewidth=2, markersize=8)
        axes[0, 1].plot(stats_df['cycles'], stats_df['diff_std'], '^-', label='Main-Fusion Diff', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Scale Sim Cycles')
        axes[0, 1].set_ylabel('Standard Deviation (μs)')
        axes[0, 1].set_title('Standard Deviation vs Cycles')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Range vs cycles
        axes[0, 2].plot(stats_df['cycles'], stats_df['main_range'], 'o-', label='Main Duration', linewidth=2, markersize=8)
        axes[0, 2].plot(stats_df['cycles'], stats_df['fusion_range'], 's-', label='Fusion Duration', linewidth=2, markersize=8)
        axes[0, 2].plot(stats_df['cycles'], stats_df['diff_range'], '^-', label='Main-Fusion Diff', linewidth=2, markersize=8)
        axes[0, 2].set_xlabel('Scale Sim Cycles')
        axes[0, 2].set_ylabel('Range (μs)')
        axes[0, 2].set_title('Range vs Cycles')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Coefficient of Variation vs cycles
        axes[1, 0].plot(stats_df['cycles'], stats_df['main_cv'], 'o-', label='Main Duration', linewidth=2, markersize=8)
        axes[1, 0].plot(stats_df['cycles'], stats_df['fusion_cv'], 's-', label='Fusion Duration', linewidth=2, markersize=8)
        axes[1, 0].plot(stats_df['cycles'], stats_df['diff_cv'], '^-', label='Main-Fusion Diff', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Scale Sim Cycles')
        axes[1, 0].set_ylabel('Coefficient of Variation (%)')
        axes[1, 0].set_title('CV vs Cycles')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Number of configurations vs cycles
        axes[1, 1].bar(stats_df['cycles'], stats_df['num_configs'], alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_xlabel('Scale Sim Cycles')
        axes[1, 1].set_ylabel('Number of Configurations')
        axes[1, 1].set_title('Configurations per Cycle Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (cycles, count) in enumerate(zip(stats_df['cycles'], stats_df['num_configs'])):
            axes[1, 1].text(cycles, count + 0.1, str(int(count)), ha='center', va='bottom', fontweight='bold')
        
        # 6. Main vs Fusion mean scatter
        axes[1, 2].scatter(stats_df['main_mean'], stats_df['fusion_mean'], s=100, alpha=0.7, c=stats_df['cycles'], cmap='viridis')
        axes[1, 2].set_xlabel('Mean Main Duration (μs)')
        axes[1, 2].set_ylabel('Mean Fusion Duration (μs)')
        axes[1, 2].set_title('Mean Main vs Fusion Duration')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add colorbar
        scatter = axes[1, 2].scatter(stats_df['main_mean'], stats_df['fusion_mean'], s=100, alpha=0.7, c=stats_df['cycles'], cmap='viridis')
        cbar = plt.colorbar(scatter, ax=axes[1, 2])
        cbar.set_label('Scale Sim Cycles')
        
        # 7. Main duration distribution by cycle group
        cycle_groups = analysis_df.groupby('cycles')
        for cycles in stats_df['cycles']:
            group_data = cycle_groups.get_group(cycles)
            axes[2, 0].hist(group_data['main_duration'], alpha=0.6, label=f'{cycles} cycles', bins=10)
        axes[2, 0].set_xlabel('Main Duration (μs)')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].set_title('Main Duration Distribution by Cycle Group')
        axes[2, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Fusion duration distribution by cycle group
        for cycles in stats_df['cycles']:
            group_data = cycle_groups.get_group(cycles)
            axes[2, 1].hist(group_data['fusion_duration'], alpha=0.6, label=f'{cycles} cycles', bins=10)
        axes[2, 1].set_xlabel('Fusion Duration (μs)')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].set_title('Fusion Duration Distribution by Cycle Group')
        axes[2, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2, 1].grid(True, alpha=0.3)
        
        # 9. Main-Fusion difference distribution by cycle group
        for cycles in stats_df['cycles']:
            group_data = cycle_groups.get_group(cycles)
            axes[2, 2].hist(group_data['main_minus_fusion'], alpha=0.6, label=f'{cycles} cycles', bins=10)
        axes[2, 2].set_xlabel('Main-Fusion Difference (μs)')
        axes[2, 2].set_ylabel('Frequency')
        axes[2, 2].set_title('Main-Fusion Difference Distribution by Cycle Group')
        axes[2, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cycle_stats_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Cycle statistics comparison figure saved as 'cycle_stats_comparison.png'")
        
        # Save the statistics data
        stats_df.to_csv('cycle_statistics_summary.csv', index=False)
        print(f"Cycle statistics summary saved as 'cycle_statistics_summary.csv'")
        
        # Print summary statistics
        print(f"\n=== Summary Statistics ===")
        print(f"Total cycle groups analyzed: {len(stats_df)}")
        print(f"Total configurations: {stats_df['num_configs'].sum()}")
        print(f"\nMain Duration Statistics:")
        print(f"  Overall Mean: {stats_df['main_mean'].mean():.3f} μs")
        print(f"  Overall Std: {stats_df['main_std'].mean():.3f} μs")
        print(f"  Overall CV: {stats_df['main_cv'].mean():.1f}%")
        print(f"\nFusion Duration Statistics:")
        print(f"  Overall Mean: {stats_df['fusion_mean'].mean():.3f} μs")
        print(f"  Overall Std: {stats_df['fusion_std'].mean():.3f} μs")
        print(f"  Overall CV: {stats_df['fusion_cv'].mean():.1f}%")
        print(f"\nMain-Fusion Difference Statistics:")
        print(f"  Overall Mean: {stats_df['diff_mean'].mean():.3f} μs")
        print(f"  Overall Std: {stats_df['diff_std'].mean():.3f} μs")
        print(f"  Overall CV: {stats_df['diff_cv'].mean():.1f}%")
        
        return stats_df
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    stats_df = create_cycle_stats_comparison() 