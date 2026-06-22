# Matrix Multiplication Copy Events Analysis

This script analyzes the `merged_copy_events.csv` file and generates visualizations showing the relationship between memory bandwidth and data access patterns.

## Files Generated

1. **`bandwidth_vs_bytes.png`** - Main scatter plot showing Bandwidth vs. Raw Bytes Accessed (MB)
2. **`additional_analysis.png`** - Four additional analysis plots:
   - Bandwidth distribution by dataset
   - Duration vs Matrix dimension
   - Bandwidth efficiency vs Data size
   - Duration vs Bandwidth

## Usage

### Prerequisites
Install required packages:
```bash
pip install -r requirements.txt
```

### Run the script
```bash
python plot_bandwidth_vs_bytes.py
```

## Data Analysis

The script processes matrix multiplication copy events with three datasets:
- **Small**: 27 samples (512x512x512 to 1024x1024x1024)
- **Medium**: 27 samples (1024x1024x1024 to 2048x2048x2048)  
- **Large**: 27 samples (2048x2048x2048 to 4096x4096x4096)

### Key Insights
- **Bandwidth Range**: 403.79 - 1002.48 GB/s
- **Data Size Range**: 0.50 - 32.00 MB
- **Total Samples**: 81 data points

## Plot Features

- **Color-coded by dataset** for easy identification
- **Trend line** showing overall relationship
- **Statistics box** with key metrics
- **Grid lines** for better readability
- **High-resolution output** (300 DPI)

## Customization

You can modify the script to:
- Change plot colors and styles
- Adjust figure sizes
- Add more analysis plots
- Export in different formats (PDF, SVG, etc.)
- Filter data by specific criteria








