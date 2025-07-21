import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_cycle_configurations():
    """
    Analyze configurations and durations for the same scale sim cycle values
    """
    
    # Read the combined results CSV
    try:
        df = pd.read_csv('combined_results.csv')
        print("Successfully loaded combined_results.csv")
        print(f"Dataset shape: {df.shape}")
        
        # Read the topology file to get matrix dimensions
        topology_df = pd.read_csv('scale_sim_gemm_topology.csv')
        
        # Read the filtered events to get fusion durations
        events_df = pd.read_csv('filtered_events.csv')
        
        # Merge the data to get matrix dimensions
        df_with_dims = pd.merge(df, topology_df, left_on='name', right_on='Layer')
        
        # Get fusion durations for each layer
        fusion_data = events_df[(events_df['event_type'] == 'sub') & (events_df['event_name'] == 'fusion')]
        fusion_by_layer = fusion_data.groupby('kernel_name')['dur(us)'].first().reset_index()
        fusion_by_layer.columns = ['name', 'fusion_duration']
        
        # Merge fusion data
        df_with_fusion = pd.merge(df_with_dims, fusion_by_layer, on='name', how='left')
        
        # Calculate main - fusion difference
        df_with_fusion['main_minus_fusion'] = df_with_fusion['main_event_duration'] - df_with_fusion['fusion_duration']
        
        # Group by scale sim cycles
        cycle_groups = df_with_fusion.groupby('scale_sim_total_cycles')
        
        print(f"\n=== Analysis by Scale Sim Cycles ===")
        print(f"Number of unique cycle values: {len(cycle_groups)}")
        
        # Create detailed analysis
        cycle_analysis = []
        
        for cycles, group in cycle_groups:
            if len(group) > 1:  # Only show cycles with multiple configurations
                print(f"\nCycles: {cycles}")
                print(f"Number of configurations: {len(group)}")
                
                for _, row in group.iterrows():
                    config_info = {
                        'cycles': cycles,
                        'name': row['name'],
                        'M': row['M'],
                        'N': row['N'],
                        'K': row['K'],
                        'main_duration': row['main_event_duration'],
                        'fusion_duration': row['fusion_duration'],
                        'main_minus_fusion': row['main_minus_fusion'],
                        'config_str': f"{row['M']}x{row['N']}x{row['K']}"
                    }
                    cycle_analysis.append(config_info)
                    
                    print(f"  {row['name']}: {row['M']}x{row['N']}x{row['K']} -> Main: {row['main_event_duration']:.3f} us, Fusion: {row['fusion_duration']:.3f} us, Diff: {row['main_minus_fusion']:.3f} us")
        
        # Create DataFrame for analysis
        analysis_df = pd.DataFrame(cycle_analysis)
        
        # Save detailed results
        analysis_df.to_csv('cycle_configuration_analysis.csv', index=False)
        print(f"\nDetailed analysis saved to 'cycle_configuration_analysis.csv'")
        
        # Create visualizations
        create_cycle_visualizations(analysis_df)
        
        return analysis_df
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def create_cycle_visualizations(analysis_df):
    """
    Create visualizations for cycle configuration analysis
    """
    
    # 1. Scatter plot: Cycles vs Duration with different colors for same cycles
    plt.figure(figsize=(14, 8))
    
    # Get unique cycle values
    unique_cycles = analysis_df['cycles'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_cycles)))
    
    for i, cycles in enumerate(unique_cycles):
        cycle_data = analysis_df[analysis_df['cycles'] == cycles]
        if len(cycle_data) > 1:
            plt.scatter(cycle_data['cycles'], cycle_data['main_duration'], 
                       c=[colors[i]], s=100, alpha=0.7, 
                       label=f'{cycles} cycles ({len(cycle_data)} configs)')
        else:
            plt.scatter(cycle_data['cycles'], cycle_data['main_duration'], 
                       c='gray', s=50, alpha=0.5, label=f'{cycles} cycles')
    
    plt.xlabel('Scale Sim Total Cycles')
    plt.ylabel('Main Event Duration (us)')
    plt.title('Configurations with Same Scale Sim Cycles\n(Colors indicate cycle values with multiple configurations)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cycle_configuration_scatter.png', dpi=300, bbox_inches='tight')
    print(f"Cycle configuration scatter plot saved as 'cycle_configuration_scatter.png'")
    
    # 2. Bar chart showing cycle values with multiple configurations
    cycle_counts = analysis_df['cycles'].value_counts()
    multiple_config_cycles = cycle_counts[cycle_counts > 1]
    
    if len(multiple_config_cycles) > 0:
        plt.figure(figsize=(12, 6))
        bars = plt.bar(multiple_config_cycles.index, multiple_config_cycles.values, 
                      color='skyblue', alpha=0.7)
        plt.xlabel('Scale Sim Total Cycles')
        plt.ylabel('Number of Configurations')
        plt.title('Cycle Values with Multiple Matrix Configurations')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, multiple_config_cycles.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('cycle_multi_config_counts.png', dpi=300, bbox_inches='tight')
        print(f"Cycle multi-config counts saved as 'cycle_multi_config_counts.png'")
    
    # 3. Detailed table for cycles with multiple configurations
    create_detailed_table(analysis_df)
    
    # 4. Duration variation analysis
    create_duration_variation_analysis(analysis_df)
    
    # 5. Fusion vs Main analysis
    create_fusion_main_analysis(analysis_df)

def create_detailed_table(analysis_df):
    """
    Create a detailed table showing configurations for each cycle value
    """
    
    # Group by cycles and create detailed tables
    cycle_groups = analysis_df.groupby('cycles')
    
    with open('cycle_configuration_details.txt', 'w') as f:
        f.write("=== Detailed Cycle Configuration Analysis ===\n\n")
        
        for cycles, group in cycle_groups:
            if len(group) > 1:
                f.write(f"Cycles: {cycles} ({len(group)} configurations)\n")
                f.write("-" * 80 + "\n")
                
                # Sort by main duration
                sorted_group = group.sort_values('main_duration')
                
                for _, row in sorted_group.iterrows():
                    f.write(f"  {row['name']:15} | {row['config_str']:10} | Main: {row['main_duration']:8.3f} us | Fusion: {row['fusion_duration']:8.3f} us | Diff: {row['main_minus_fusion']:8.3f} us\n")
                
                # Calculate statistics for main, fusion, and difference
                main_mean = sorted_group['main_duration'].mean()
                main_std = sorted_group['main_duration'].std()
                main_range = sorted_group['main_duration'].max() - sorted_group['main_duration'].min()
                
                fusion_mean = sorted_group['fusion_duration'].mean()
                fusion_std = sorted_group['fusion_duration'].std()
                fusion_range = sorted_group['fusion_duration'].max() - sorted_group['fusion_duration'].min()
                
                diff_mean = sorted_group['main_minus_fusion'].mean()
                diff_std = sorted_group['main_minus_fusion'].std()
                diff_range = sorted_group['main_minus_fusion'].max() - sorted_group['main_minus_fusion'].min()
                
                f.write(f"\n  Main Duration Stats: Mean={main_mean:.3f} us, Std={main_std:.3f} us, Range={main_range:.3f} us\n")
                f.write(f"  Fusion Duration Stats: Mean={fusion_mean:.3f} us, Std={fusion_std:.3f} us, Range={fusion_range:.3f} us\n")
                f.write(f"  Main-Fusion Diff Stats: Mean={diff_mean:.3f} us, Std={diff_std:.3f} us, Range={diff_range:.3f} us\n")
                f.write("\n")
    
    print(f"Detailed cycle configuration table saved as 'cycle_configuration_details.txt'")

def create_duration_variation_analysis(analysis_df):
    """
    Analyze duration variation for same cycle values
    """
    
    # Group by cycles and calculate variation statistics
    cycle_stats = []
    
    for cycles, group in analysis_df.groupby('cycles'):
        if len(group) > 1:
            stats = {
                'cycles': cycles,
                'num_configs': len(group),
                'mean_main_duration': group['main_duration'].mean(),
                'std_main_duration': group['main_duration'].std(),
                'mean_fusion_duration': group['fusion_duration'].mean(),
                'std_fusion_duration': group['fusion_duration'].std(),
                'mean_diff': group['main_minus_fusion'].mean(),
                'std_diff': group['main_minus_fusion'].std(),
                'cv_main': group['main_duration'].std() / group['main_duration'].mean() * 100,
                'cv_fusion': group['fusion_duration'].std() / group['fusion_duration'].mean() * 100,
                'cv_diff': group['main_minus_fusion'].std() / abs(group['main_minus_fusion'].mean()) * 100 if group['main_minus_fusion'].mean() != 0 else 0
            }
            cycle_stats.append(stats)
    
    if cycle_stats:
        stats_df = pd.DataFrame(cycle_stats)
        stats_df.to_csv('cycle_duration_variation.csv', index=False)
        print(f"Duration variation analysis saved as 'cycle_duration_variation.csv'")
        
        # Create visualization of duration variation
        plt.figure(figsize=(15, 10))
        
        # Plot coefficient of variation for main vs fusion
        plt.subplot(2, 3, 1)
        plt.scatter(stats_df['cv_main'], stats_df['cv_fusion'], alpha=0.7, s=100)
        plt.xlabel('Main Duration CV (%)')
        plt.ylabel('Fusion Duration CV (%)')
        plt.title('Main vs Fusion Duration Variation')
        plt.grid(True, alpha=0.3)
        
        # Plot mean main vs fusion duration
        plt.subplot(2, 3, 2)
        plt.scatter(stats_df['mean_main_duration'], stats_df['mean_fusion_duration'], alpha=0.7, s=100)
        plt.xlabel('Mean Main Duration (us)')
        plt.ylabel('Mean Fusion Duration (us)')
        plt.title('Mean Main vs Fusion Duration')
        plt.grid(True, alpha=0.3)
        
        # Plot main-fusion difference vs cycles
        plt.subplot(2, 3, 3)
        plt.scatter(stats_df['cycles'], stats_df['mean_diff'], alpha=0.7, s=100)
        plt.xlabel('Scale Sim Cycles')
        plt.ylabel('Mean Main-Fusion Difference (us)')
        plt.title('Main-Fusion Difference vs Cycles')
        plt.grid(True, alpha=0.3)
        
        # Plot CV comparison
        plt.subplot(2, 3, 4)
        x = np.arange(len(stats_df))
        width = 0.25
        plt.bar(x - width, stats_df['cv_main'], width, label='Main CV', alpha=0.7)
        plt.bar(x, stats_df['cv_fusion'], width, label='Fusion CV', alpha=0.7)
        plt.bar(x + width, stats_df['cv_diff'], width, label='Diff CV', alpha=0.7)
        plt.xlabel('Cycle Groups')
        plt.ylabel('Coefficient of Variation (%)')
        plt.title('CV Comparison: Main vs Fusion vs Difference')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot mean duration comparison
        plt.subplot(2, 3, 5)
        plt.bar(x - width, stats_df['mean_main_duration'], width, label='Main Mean', alpha=0.7)
        plt.bar(x, stats_df['mean_fusion_duration'], width, label='Fusion Mean', alpha=0.7)
        plt.bar(x + width, stats_df['mean_diff'], width, label='Mean Diff', alpha=0.7)
        plt.xlabel('Cycle Groups')
        plt.ylabel('Duration (us)')
        plt.title('Mean Duration Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot std comparison
        plt.subplot(2, 3, 6)
        plt.bar(x - width, stats_df['std_main_duration'], width, label='Main Std', alpha=0.7)
        plt.bar(x, stats_df['std_fusion_duration'], width, label='Fusion Std', alpha=0.7)
        plt.bar(x + width, stats_df['std_diff'], width, label='Diff Std', alpha=0.7)
        plt.xlabel('Cycle Groups')
        plt.ylabel('Standard Deviation (us)')
        plt.title('Standard Deviation Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cycle_duration_variation_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Duration variation analysis plots saved as 'cycle_duration_variation_analysis.png'")

def create_fusion_main_analysis(analysis_df):
    """
    Create specific analysis comparing fusion and main events
    """
    
    # Create scatter plot comparing main vs fusion durations
    plt.figure(figsize=(12, 8))
    
    # Color by cycle value
    unique_cycles = analysis_df['cycles'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_cycles)))
    
    for i, cycles in enumerate(unique_cycles):
        cycle_data = analysis_df[analysis_df['cycles'] == cycles]
        plt.scatter(cycle_data['main_duration'], cycle_data['fusion_duration'], 
                   c=[colors[i]], s=100, alpha=0.7, 
                   label=f'{cycles} cycles')
    
    # Add diagonal line for reference
    min_val = min(analysis_df['main_duration'].min(), analysis_df['fusion_duration'].min())
    max_val = max(analysis_df['main_duration'].max(), analysis_df['fusion_duration'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
    
    plt.xlabel('Main Event Duration (us)')
    plt.ylabel('Fusion Event Duration (us)')
    plt.title('Main vs Fusion Event Durations\n(Colors indicate cycle values)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('main_vs_fusion_scatter.png', dpi=300, bbox_inches='tight')
    print(f"Main vs Fusion scatter plot saved as 'main_vs_fusion_scatter.png'")
    
    # Create histogram of main-fusion differences
    plt.figure(figsize=(10, 6))
    plt.hist(analysis_df['main_minus_fusion'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Main Duration - Fusion Duration (us)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Main-Fusion Duration Differences')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('main_fusion_diff_histogram.png', dpi=300, bbox_inches='tight')
    print(f"Main-Fusion difference histogram saved as 'main_fusion_diff_histogram.png'")

def find_specific_cycle_analysis(target_cycles):
    """
    Find detailed analysis for specific cycle values
    """
    
    try:
        analysis_df = pd.read_csv('cycle_configuration_analysis.csv')
        
        if target_cycles in analysis_df['cycles'].values:
            cycle_data = analysis_df[analysis_df['cycles'] == target_cycles]
            print(f"\n=== Analysis for {target_cycles} cycles ===")
            print(f"Number of configurations: {len(cycle_data)}")
            print("\nConfigurations:")
            
            for _, row in cycle_data.iterrows():
                print(f"  {row['name']}: {row['config_str']} -> Main: {row['main_duration']:.3f} us, Fusion: {row['fusion_duration']:.3f} us, Diff: {row['main_minus_fusion']:.3f} us")
            
            if len(cycle_data) > 1:
                main_stats = cycle_data['main_duration']
                fusion_stats = cycle_data['fusion_duration']
                diff_stats = cycle_data['main_minus_fusion']
                
                print(f"\nMain Duration Statistics:")
                print(f"  Mean: {main_stats.mean():.3f} us")
                print(f"  Std: {main_stats.std():.3f} us")
                print(f"  Range: {main_stats.max() - main_stats.min():.3f} us")
                print(f"  CV: {main_stats.std() / main_stats.mean() * 100:.1f}%")
                
                print(f"\nFusion Duration Statistics:")
                print(f"  Mean: {fusion_stats.mean():.3f} us")
                print(f"  Std: {fusion_stats.std():.3f} us")
                print(f"  Range: {fusion_stats.max() - fusion_stats.min():.3f} us")
                print(f"  CV: {fusion_stats.std() / fusion_stats.mean() * 100:.1f}%")
                
                print(f"\nMain-Fusion Difference Statistics:")
                print(f"  Mean: {diff_stats.mean():.3f} us")
                print(f"  Std: {diff_stats.std():.3f} us")
                print(f"  Range: {diff_stats.max() - diff_stats.min():.3f} us")
        else:
            print(f"No configurations found for {target_cycles} cycles")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    analysis_df = analyze_cycle_configurations()
    
    if analysis_df is not None:
        print(f"\n=== Summary ===")
        print(f"Total configurations analyzed: {len(analysis_df)}")
        print(f"Unique cycle values: {analysis_df['cycles'].nunique()}")
        
        # Find cycles with multiple configurations
        cycle_counts = analysis_df['cycles'].value_counts()
        multi_config_cycles = cycle_counts[cycle_counts > 1]
        
        if len(multi_config_cycles) > 0:
            print(f"Cycle values with multiple configurations: {len(multi_config_cycles)}")
            print("Top cycle values with multiple configs:")
            for cycles, count in multi_config_cycles.head().items():
                print(f"  {cycles} cycles: {count} configurations")
        else:
            print("No cycle values found with multiple configurations")
        
        # Example: Analyze specific cycle values
        print(f"\n=== Example Analysis ===")
        example_cycles = [383, 1535, 6143]
        for cycles in example_cycles:
            find_specific_cycle_analysis(cycles) 