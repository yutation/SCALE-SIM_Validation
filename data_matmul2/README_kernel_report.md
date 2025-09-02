# Kernel Report Generation

This directory contains Python scripts to generate CSV reports combining data from filtered events and compute reports.

## Files

- `generate_kernel_report.py` - Basic script for generating kernel reports
- `generate_kernel_report_enhanced.py` - Enhanced version with better error handling and logging
- `update_kernel_cycles.py` - Script to update total cycles based on matrix dimensions
- `analyze_updates.py` - Script to analyze which kernels were updated vs skipped
- `kernel_report.csv` - Generated report from basic script
- `kernel_report_enhanced.csv` - Generated report from enhanced script
- `kernel_report_updated.csv` - Updated report with conditional cycle replacement

## Input Files

- `filtered_events_repeat20.csv` - Contains kernel events with timing data
- `COMPUTE_REPORT.csv` - Contains layer information with total cycles
- `COMPUTE_REPORT_is.csv` - Contains alternative layer information for conditional updates

## Output Format

The generated CSV contains one row per kernel with the following columns:

- `kernel_name` - Name of the kernel
- `total_cycles` - Total cycles from compute report
- `main_avg`, `main_min`, `main_max`, `main_stddev` - Statistics for main events
- `fusion_avg`, `fusion_min`, `fusion_max`, `fusion_stddev` - Statistics for fusion events
- `copy-start_avg`, `copy-start_min`, `copy-start_max`, `copy-start_stddev` - Statistics for copy-start events
- `copy-done_avg`, `copy-done_min`, `copy-done_max`, `copy-done_stddev` - Statistics for copy-done events

**Note**: All float values are formatted to 6 decimal places for consistency and readability.

## Usage

### Basic Script
```bash
python generate_kernel_report.py
```

### Enhanced Script
```bash
python generate_kernel_report_enhanced.py
```

### Update Kernel Cycles
```bash
python update_kernel_cycles.py
```

### Analyze Updates
```bash
python analyze_updates.py
```

## Features

### Basic Script Features
- Combines data from both CSV files
- Calculates statistics (avg, min, max, stddev) for each event type
- Generates one row per kernel
- **Preserves original kernel order from filtered events file**
- **Formatted float values (6 decimal places)**

### Enhanced Script Features
- Robust error handling and validation
- Detailed logging and progress tracking
- Enhanced event type matching
- Summary statistics output
- File existence validation
- Better documentation
- **Preserves original kernel order from filtered events file**
- **Formatted float values (6 decimal places)**

### Kernel Cycle Update Features
- **Conditional cycle replacement**: If M > N in `matmul_MxNxK`, replaces total_cycles with value from `COMPUTE_REPORT_is.csv`
- **Matrix dimension parsing**: Extracts M, N, K values from kernel names
- **Layer ID mapping**: Maps kernel order to LayerID for cycle replacement
- **Detailed logging**: Shows which kernels were updated vs skipped
- **Analysis tools**: Separate script to analyze update results

## Event Types

The scripts process the following event types:
- **main**: Main computation events (e.g., `jit_validation_matrix_multiply`)
- **fusion**: Fusion operation events
- **copy-start**: Memory copy start events
- **copy-done**: Memory copy completion events

## Statistics Calculated

For each event type, the following statistics are calculated:
- **Average**: Mean duration in microseconds (6 decimal places)
- **Minimum**: Minimum duration in microseconds (6 decimal places)
- **Maximum**: Maximum duration in microseconds (6 decimal places)
- **Standard Deviation**: Standard deviation of durations in microseconds (6 decimal places)

## Example Output

```
kernel_name,total_cycles,main_avg,main_min,main_max,main_stddev,fusion_avg,fusion_min,fusion_max,fusion_stddev,copy-start_avg,copy-start_min,copy-start_max,copy-start_stddev,copy-done_avg,copy-done_min,copy-done_max,copy-done_stddev
matmul_32x32x32,413.0,1.230036,1.188661,1.594375,0.101972,1.093562,1.020000,1.280090,0.059128,0.008571,0.008571,0.008572,0.000000,0.076719,0.054375,0.481518,0.092869
```

## Dependencies

- pandas
- numpy
- pathlib (for enhanced script)
- logging (for enhanced script)
- re (for kernel name parsing)

## Important Notes

- **Kernel Order**: The scripts preserve the exact order of kernels as they appear in the filtered events file. This is crucial for maintaining the relationship between kernel execution order and layer IDs in the compute report.
- **Float Formatting**: All float values are rounded to 6 decimal places for consistency and readability.
- **Conditional Updates**: The update script only modifies kernels where M > N in the matrix dimensions.
- The scripts assume that LayerID in the compute report corresponds to kernel execution order
- Event type matching is case-insensitive
- Missing events result in zero values for all statistics
- The enhanced script provides better debugging information and error handling

## Kernel Order Preservation

The scripts now correctly preserve the kernel order from the filtered events file:
1. Kernels are processed in the order they first appear in the filtered events file
2. This ensures the LayerID mapping from the compute report aligns correctly with kernel execution order
3. The output CSV maintains this original order for accurate analysis

## Float Formatting

All float values in the output are formatted to 6 decimal places:
- Provides consistent precision across all statistics
- Improves readability of the CSV output
- Reduces file size by eliminating excessive decimal places
- Maintains sufficient precision for timing analysis

## Conditional Cycle Updates

The `update_kernel_cycles.py` script implements conditional cycle replacement:

### Logic
- Parses kernel names in format `matmul_MxNxK`
- Extracts M, N, K dimensions
- If M > N, replaces `total_cycles` with corresponding value from `COMPUTE_REPORT_is.csv`
- If M ≤ N, keeps original `total_cycles` value

### Example
- `matmul_48x32x64`: M=48, N=32, K=64 → M>N (48>32) → **UPDATED**
- `matmul_32x32x64`: M=32, N=32, K=64 → M≤N (32≤32) → **SKIPPED**
- `matmul_64x32x32`: M=64, N=32, K=32 → M>N (64>32) → **UPDATED**

### Statistics
- **147 kernels updated** (M>N condition met)
- **196 kernels skipped** (M≤N or parsing issues)
- **7 unique cycle values** in final output (413.0, 429.0, 445.0, 461.0, 477.0, 493.0, 509.0)
