# Data MatMul Linear Analysis

This folder contains the data and analysis for linear regression between `total_cycles` and `fusion_avg` from kernel performance data.

## Files

- **`kernel_report_updated.csv`** - Original dataset with kernel performance metrics
- **`kernel_report_updated_2.csv`** - Additional dataset with kernel performance metrics
- **`linear_regression_analysis.py`** - Comprehensive analysis script with visualizations
- **`simple_linear_analysis.py`** - Simple script for basic linear function calculation
- **`linear_regression_analysis_no_intercept.py`** - Comprehensive analysis with intercept=0
- **`simple_linear_analysis_no_intercept.py`** - Simple analysis with intercept=0
- **`requirements.txt`** - Required Python packages
- **`README.md`** - This documentation file

## Output Files (Generated)

The scripts generate output files with names that correspond to the input file:

### With Intercept (Standard Linear Regression)
- **`kernel_report_updated_linear_function_results.txt`** - Simple analysis results
- **`kernel_report_updated_linear_regression_results.txt`** - Comprehensive analysis results
- **`kernel_report_updated_linear_regression_analysis.png`** - Visualization plots

### No Intercept (Regression Through Origin)
- **`kernel_report_updated_2_linear_function_no_intercept_results.txt`** - Simple analysis results
- **`kernel_report_updated_2_linear_regression_no_intercept_results.txt`** - Comprehensive analysis results
- **`kernel_report_updated_2_linear_regression_no_intercept_analysis.png`** - Visualization plots

## Linear Function Results

### With Intercept (Standard)
**Linear Function:** `fusion_avg = 0.002762 × total_cycles + (-0.059902)`

**Key Statistics:**
- **Slope:** 0.002762
- **Intercept:** -0.059902
- **R-squared:** 0.789323 (78.93% of variance explained)
- **Correlation:** 0.888 (strong positive correlation)

### No Intercept (Through Origin)
**Linear Function:** `fusion_avg = 0.000478 × total_cycles`

**Key Statistics:**
- **Slope:** 0.000478
- **Intercept:** 0.000000 (forced to 0)
- **R-squared:** 0.579841 (57.98% of variance explained)
- **Correlation:** 0.761 (moderate positive correlation)

## Usage

### Standard Analysis (With Intercept)

#### Simple Analysis
```bash
python simple_linear_analysis.py
```
Generates: `kernel_report_updated_linear_function_results.txt`

#### Comprehensive Analysis with Visualizations
```bash
python linear_regression_analysis.py
```
Generates: 
- `kernel_report_updated_linear_regression_results.txt`
- `kernel_report_updated_linear_regression_analysis.png`

### No Intercept Analysis (Through Origin)

#### Simple Analysis
```bash
python simple_linear_analysis_no_intercept.py
```
Generates: `kernel_report_updated_2_linear_function_no_intercept_results.txt`

#### Comprehensive Analysis with Visualizations
```bash
python linear_regression_analysis_no_intercept.py
```
Generates: 
- `kernel_report_updated_2_linear_regression_no_intercept_results.txt`
- `kernel_report_updated_2_linear_regression_no_intercept_analysis.png`

## File Naming Convention

The output files follow this naming pattern:
- Input: `filename.csv`
- Simple results: `filename_linear_function_results.txt`
- Comprehensive results: `filename_linear_regression_results.txt`
- Visualization: `filename_linear_regression_analysis.png`
- No-intercept simple results: `filename_linear_function_no_intercept_results.txt`
- No-intercept comprehensive results: `filename_linear_regression_no_intercept_results.txt`
- No-intercept visualization: `filename_linear_regression_no_intercept_analysis.png`

This makes it easy to track which results belong to which input file and analysis type.

## Comparison of Models

| Model Type | Linear Function | R-squared | Intercept | Use Case |
|------------|----------------|-----------|-----------|----------|
| With Intercept | `fusion_avg = 0.002762 × total_cycles + (-0.059902)` | 0.789 | -0.060 | Standard regression |
| No Intercept | `fusion_avg = 0.000478 × total_cycles` | 0.580 | 0.000 | Regression through origin |

## Interpretation

### With Intercept Model:
- For every 1 unit increase in `total_cycles`, `fusion_avg` increases by 0.002762 units
- The relationship is strong (R² > 0.7)
- There's a strong positive linear relationship between the variables

### No Intercept Model:
- For every 1 unit increase in `total_cycles`, `fusion_avg` increases by 0.000478 units
- The relationship is moderate (R² > 0.5)
- The regression line passes through the origin (0,0)
- Useful when you know the relationship should start from zero

## Example Predictions

### With Intercept:
- `total_cycles = 400` → `fusion_avg = 1.045`
- `total_cycles = 450` → `fusion_avg = 1.183`
- `total_cycles = 500` → `fusion_avg = 1.321`

### No Intercept:
- `total_cycles = 400` → `fusion_avg = 0.191`
- `total_cycles = 450` → `fusion_avg = 0.215`
- `total_cycles = 500` → `fusion_avg = 0.239`

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- matplotlib
- scikit-learn
- seaborn
- scipy
