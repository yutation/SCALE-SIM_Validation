# Correlation Analysis Summary

## Overview
This analysis examines the correlation between main event duration and scale sim total cycles for matrix multiplication operations.

## Dataset
- **Total samples**: 125 matrix multiplication operations
- **Data range**: Main event duration (1.07 - 7.97 μs), Scale sim cycles (383 - 14,303 cycles)
- **No missing values**: Complete dataset with all 125 entries

## Correlation Results

### Pearson Correlation
- **Coefficient**: 0.8984
- **P-value**: 8.79e-46 (highly significant)
- **Interpretation**: Strong linear relationship
- **R-squared**: 0.8071 (80.71% of variance explained)

### Spearman Correlation
- **Coefficient**: 0.8272
- **P-value**: 1.41e-32 (highly significant)
- **Interpretation**: Strong monotonic relationship

### Kendall Correlation
- **Coefficient**: 0.6719
- **P-value**: 2.08e-26 (highly significant)
- **Interpretation**: Moderate monotonic relationship

## Statistical Summary

### Main Event Duration
- **Mean**: 1.854 μs
- **Standard Deviation**: 0.807 μs
- **Range**: 1.070 - 7.973 μs

### Scale Sim Total Cycles
- **Mean**: 1,326.10 cycles
- **Standard Deviation**: 1,777.92 cycles
- **Range**: 383 - 14,303 cycles

## Key Findings

1. **Strong Positive Correlation**: There is a very strong positive correlation (r = 0.8984) between main event duration and scale sim total cycles.

2. **High Predictive Power**: The scale sim total cycles can explain 80.71% of the variance in main event duration.

3. **Consistent Across Methods**: All three correlation methods (Pearson, Spearman, Kendall) show significant positive relationships, indicating robustness of the finding.

4. **Highly Significant**: All p-values are extremely small (< 1e-26), indicating the correlations are statistically significant.

## Interpretation

The strong positive correlation suggests that:
- As the computational complexity (measured by scale sim cycles) increases, the actual execution time (main event duration) also increases
- The SCALE-Sim simulator is effectively modeling the relationship between computational workload and execution time
- There is a predictable relationship between the theoretical cycles and actual performance

## Generated Files

- `correlation_results.csv`: Detailed correlation coefficients and p-values
- `correlation_plot.png`: Scatter plot with trend line
- `correlation_heatmap.png`: Correlation matrix visualization
- `correlation_comparison.png`: Bar chart comparing different correlation methods

## Conclusion

The analysis reveals a very strong and statistically significant positive correlation between main event duration and scale sim total cycles. This suggests that the SCALE-Sim simulator is accurately modeling the relationship between computational complexity and execution time for matrix multiplication operations. 