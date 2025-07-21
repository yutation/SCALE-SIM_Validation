import pandas as pd
import csv
import os

def combine_data():
    """
    Combine data from three CSV files:
    1. scale_sim_gemm_topology.csv - contains layer names and dimensions
    2. filtered_events.csv - contains main event durations
    3. COMPUTE_REPORT.csv - contains scale sim total cycles
    
    Output: CSV with name, main_event_duration, scale_sim_total_cycles
    """
    
    # Check if all required files exist
    required_files = ['scale_sim_gemm_topology.csv', 'filtered_events.csv', 'COMPUTE_REPORT.csv']
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: Required file '{file}' not found!")
            return
    
    try:
        # Read the topology file to get layer names
        print("Reading scale_sim_gemm_topology.csv...")
        topology_df = pd.read_csv('scale_sim_gemm_topology.csv')
        print(f"Found {len(topology_df)} layers in topology file")
        
        # Read the filtered events file to get main event durations
        print("Reading filtered_events.csv...")
        events_df = pd.read_csv('filtered_events.csv')
        print(f"Found {len(events_df)} events in filtered events file")
        
        # Read the compute report file to get scale sim total cycles
        print("Reading COMPUTE_REPORT.csv...")
        compute_df = pd.read_csv('COMPUTE_REPORT.csv')
        print(f"Found {len(compute_df)} compute report entries")
        
        # Filter main events from events_df
        main_events = events_df[events_df['event_type'] == 'main']
        print(f"Found {len(main_events)} main events")
        
        # Create a dictionary to store the results
        results = []
        matched_count = 0
        missing_duration_count = 0
        missing_cycles_count = 0
        
        # Process each layer in the topology
        for index, row in topology_df.iterrows():
            layer_name = row['Layer']
            
            # Find corresponding main event duration
            main_event_row = main_events[main_events['kernel_name'] == layer_name]
            main_duration = None
            if not main_event_row.empty:
                main_duration = main_event_row.iloc[0]['dur(us)']
                matched_count += 1
            else:
                missing_duration_count += 1
            
            # Find corresponding scale sim total cycles
            # The LayerID in compute report corresponds to the index in topology
            scale_sim_cycles = None
            if index < len(compute_df):
                scale_sim_cycles = compute_df.iloc[index][' Total Cycles']
            else:
                missing_cycles_count += 1
            
            results.append({
                'name': layer_name,
                'main_event_duration': main_duration,
                'scale_sim_total_cycles': scale_sim_cycles
            })
        
        # Create output DataFrame
        output_df = pd.DataFrame(results)
        
        # Write to CSV
        output_filename = 'combined_results.csv'
        output_df.to_csv(output_filename, index=False)
        
        # Print summary statistics
        print(f"\n=== Summary ===")
        print(f"Total layers processed: {len(results)}")
        print(f"Layers with matched main event duration: {matched_count}")
        print(f"Layers missing main event duration: {missing_duration_count}")
        print(f"Layers missing scale sim cycles: {missing_cycles_count}")
        print(f"Output file: {output_filename}")
        print(f"Columns: name, main_event_duration, scale_sim_total_cycles")
        
        # Display first few rows for verification
        print("\n=== First 10 rows of the output ===")
        print(output_df.head(10).to_string(index=False))
        
        # Display some statistics
        print(f"\n=== Statistics ===")
        print(f"Main event duration range: {output_df['main_event_duration'].min():.6f} - {output_df['main_event_duration'].max():.6f} us")
        print(f"Scale sim cycles range: {output_df['scale_sim_total_cycles'].min():.0f} - {output_df['scale_sim_total_cycles'].max():.0f} cycles")
        
        return output_df
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return None

if __name__ == "__main__":
    combine_data() 