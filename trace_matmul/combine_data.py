import pandas as pd
import csv

def combine_data():
    """
    Combine data from three CSV files:
    1. scale_sim_gemm_topology.csv - contains layer names and dimensions
    2. filtered_events.csv - contains main event durations
    3. COMPUTE_REPORT.csv - contains scale sim total cycles
    """
    
    # Read the topology file to get layer names
    topology_df = pd.read_csv('scale_sim_gemm_topology.csv')
    
    # Read the filtered events file to get main event durations
    events_df = pd.read_csv('filtered_events.csv')
    
    # Read the compute report file to get scale sim total cycles
    compute_df = pd.read_csv('COMPUTE_REPORT.csv')
    
    # Filter main events from events_df
    main_events = events_df[events_df['event_type'] == 'main']
    
    # Create a dictionary to store the results
    results = []
    
    # Process each layer in the topology
    for index, row in topology_df.iterrows():
        layer_name = row['Layer']
        
        # Find corresponding main event duration
        main_event_row = main_events[main_events['kernel_name'] == layer_name]
        main_duration = None
        if not main_event_row.empty:
            main_duration = main_event_row.iloc[0]['dur(us)']
        
        # Find corresponding scale sim total cycles
        # The LayerID in compute report corresponds to the index in topology
        scale_sim_cycles = None
        if index < len(compute_df):
            scale_sim_cycles = compute_df.iloc[index][' Total Cycles']
        
        results.append({
            'name': layer_name,
            'main_event_duration': main_duration,
            'scale_sim_total_cycles': scale_sim_cycles
        })
    
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    
    # Write to CSV
    output_df.to_csv('combined_results.csv', index=False)
    
    print(f"Generated combined_results.csv with {len(results)} entries")
    print("Columns: name, main_event_duration, scale_sim_total_cycles")
    
    # Display first few rows for verification
    print("\nFirst 5 rows of the output:")
    print(output_df.head())

if __name__ == "__main__":
    combine_data() 