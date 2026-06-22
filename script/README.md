# CSV File Merger

This tool merges multiple CSV files with the same structure. It's particularly useful for merging kernel report CSV files like the ones in your validation directory.

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The main script `merge_csv_files.py` provides a flexible command-line interface:

#### Basic Usage
```bash
# Merge two specific files
python merge_csv_files.py file1.csv file2.csv -o merged_output.csv

# Merge all CSV files in current directory
python merge_csv_files.py *.csv -o merged_output.csv

# Merge specific files with wildcards
python merge_csv_files.py validation/data_matmul_linear/*.csv -o merged_output.csv
```

#### Advanced Options
```bash
# Merge using database-style join on kernel_name column
python merge_csv_files.py file1.csv file2.csv -o output.csv --merge-strategy merge --key-column kernel_name

# Sort by specific column
python merge_csv_files.py *.csv -o output.csv --sort-by kernel_name

# Keep duplicates (don't drop them)
python merge_csv_files.py *.csv -o output.csv --no-drop-duplicates

# Sort in descending order
python merge_csv_files.py *.csv -o output.csv --sort-by total_cycles --descending
```

### Python API

You can also use the merger as a Python module:

```python
from merge_csv_files import merge_csv_files

# Simple concatenation
success = merge_csv_files(
    input_files=['file1.csv', 'file2.csv'],
    output_file='merged.csv',
    merge_strategy='concat',
    drop_duplicates=True,
    sort_by='kernel_name'
)

# Database-style merge
success = merge_csv_files(
    input_files=['file1.csv', 'file2.csv'],
    output_file='merged.csv',
    merge_strategy='merge',
    key_column='kernel_name',
    drop_duplicates=True
)
```

## Merge Strategies

### 1. Concatenation (`concat`)
- Simply combines all rows from all files
- Useful when files contain different data (no overlapping rows)
- Automatically removes duplicate rows if `drop_duplicates=True`

### 2. Database-style Merge (`merge`)
- Merges files based on a key column (like SQL JOIN)
- Useful when files contain related data with common keys
- Requires specifying `--key-column`

## Example with Your Files

For your specific kernel report files:

```bash
# Simple merge (recommended for your case)
python merge_csv_files.py validation/data_matmul_linear/kernel_report_updated.csv validation/data_matmul_linear/kernel_report_updated_2.csv -o merged_kernel_reports.csv

# Or run the example script
python example_merge.py
```

## Command Line Options

- `input_files`: One or more CSV files to merge (supports wildcards)
- `-o, --output`: Output file path (required)
- `--merge-strategy`: Choose between 'concat' or 'merge' (default: concat)
- `--key-column`: Column to use for merge strategy (required for merge strategy)
- `--no-drop-duplicates`: Keep duplicate rows
- `--sort-by`: Column to sort by after merging
- `--descending`: Sort in descending order (default is ascending)

## Features

- ✅ Handles multiple input files
- ✅ Supports wildcard file patterns
- ✅ Two merge strategies (concatenation and database-style merge)
- ✅ Automatic duplicate removal
- ✅ Sorting options
- ✅ Detailed progress reporting
- ✅ Error handling and validation
- ✅ Works with your kernel report CSV format

## Example Output

When you run the merger, you'll see output like:

```
Input files: ['file1.csv', 'file2.csv']
Output file: merged_output.csv
Merge strategy: concat
Drop duplicates: True
Sort by: kernel_name (ascending)

Reading file1.csv...
  - Shape: (344, 18)
  - Columns: ['kernel_name', 'total_cycles', 'main_avg', ...]
Reading file2.csv...
  - Shape: (344, 18)
  - Columns: ['kernel_name', 'total_cycles', 'main_avg', ...]
Using concatenation strategy...
Merged dataframe shape: (688, 18)
Dropping duplicates...
  - Removed 0 duplicate rows
Sorting by 'kernel_name'...
Saving to merged_output.csv...
Successfully merged 2 files into merged_output.csv
Final shape: (688, 18)
```



