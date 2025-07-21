# Data Combination Script

This directory contains scripts to combine data from three different CSV files related to matrix multiplication performance analysis.

## Files

### Input Files
- `scale_sim_gemm_topology.csv`: Contains layer names and dimensions (M, N, K)
- `filtered_events.csv`: Contains event timing data including main event durations
- `COMPUTE_REPORT.csv`: Contains SCALE-Sim total cycles for each layer

### Scripts
- `combine_data.py`: Basic script to combine the three CSV files
- `combine_data_enhanced.py`: Enhanced version with error handling and detailed statistics

### Output Files
- `combined_results.csv`: Combined data with columns: name, main_event_duration, scale_sim_total_cycles

## Usage

```bash
python combine_data_enhanced.py
```

## Output Format

The generated `combined_results.csv` contains:

| Column | Description |
|--------|-------------|
| name | Layer name (e.g., matmul_2x2x2) |
| main_event_duration | Main event duration in microseconds |
| scale_sim_total_cycles | SCALE-Sim total cycles for the layer |

## Data Summary

- **Total layers processed**: 125
- **Main event duration range**: 1.07 - 7.97 microseconds
- **Scale sim cycles range**: 383 - 14,303 cycles
- **All layers successfully matched**: 125/125

## Example Output

```
name,main_event_duration,scale_sim_total_cycles
matmul_2x2x2,1.075803,383.0
matmul_2x2x8,1.074285,383.0
matmul_2x2x32,1.069911,383.0
...
matmul_512x512x512,7.972768,14303.0
```

## Notes

- The script matches layers by name between the topology and events files
- Scale sim cycles are matched by index position in the compute report
- All 125 layers have complete data (no missing values)
- The data covers matrix multiplication operations with varying dimensions from 2x2x2 to 512x512x512 