# Fusion Event Correlation Analysis Summary

## Overview
This analysis examines the correlation between fusion sub event duration and scale sim total cycles for matrix multiplication operations.

## Dataset
- **Total fusion events**: 125 (one per matrix multiplication operation)
- **Data range**: Fusion duration (0.594 - 6.900 μs), Scale sim cycles (383 - 14,303 cycles)
- **No missing values**: Complete dataset with all 125 fusion events matched

## Correlation Results

### Pearson Correlation
- **Coefficient**: 0.8398
- **P-value**: 1.99e-34 (highly significant)
- **Interpretation**: Strong linear relationship
- **R-squared**: 0.7053 (70.53% of variance explained)

### Spearman Correlation
- **Coefficient**: 0.7969
- **P-value**: 1.07e-28 (highly significant)
- **Interpretation**: Strong monotonic relationship

### Kendall Correlation
- **Coefficient**: 0.6306
- **P-value**: 1.85e-23 (highly significant)
- **Interpretation**: Moderate monotonic relationship

## Statistical Summary

### Fusion Event Duration
- **Mean**: 1.260 μs
- **Standard Deviation**: 0.696 μs
- **Range**: 0.594 - 6.900 μs

### Scale Sim Total Cycles
- **Mean**: 1,326.10 cycles
- **Standard Deviation**: 1,777.92 cycles
- **Range**: 383 - 14,303 cycles

## Comparison: Fusion vs Main Events

| Metric | Main Events | Fusion Events |
|--------|-------------|---------------|
| **Pearson Correlation** | 0.8984 | 0.8398 |
| **Spearman Correlation** | 0.8272 | 0.7969 |
| **Kendall Correlation** | 0.6719 | 0.6306 |
| **R-squared** | 0.8071 | 0.7053 |
| **Mean Duration** | 1.854 μs | 1.260 μs |
| **Duration Range** | 1.070 - 7.973 μs | 0.594 - 6.900 μs |

## Key Findings

1. **Strong Correlation**: There is a very strong positive correlation (r = 0.8398) between fusion event duration and scale sim total cycles.

2. **High Predictive Power**: The scale sim total cycles can explain 70.53% of the variance in fusion event duration.

3. **Consistent with Main Events**: The fusion event correlation is slightly lower than main event correlation but still very strong, indicating both events follow similar patterns.

4. **Highly Significant**: All p-values are extremely small (< 1e-23), indicating the correlations are statistically significant.

5. **Fusion Events are Shorter**: On average, fusion events (1.260 μs) are shorter than main events (1.854 μs), but both show similar correlation patterns.

## Interpretation

The strong positive correlation for fusion events suggests that:
- Fusion operations scale with computational complexity, similar to main events
- The SCALE-Sim simulator effectively models the relationship between computational workload and fusion operation time
- Fusion events represent a significant portion of the total execution time and follow predictable patterns
- The correlation is robust and consistent across different correlation methods

## Generated Files

- `fusion_correlation_results.csv`: Detailed correlation coefficients and p-values
- `fusion_combined_results.csv`: Combined fusion event data with scale sim cycles
- `fusion_correlation_plot.png`: Scatter plot of fusion duration vs scale sim cycles
- `fusion_vs_main_comparison.png`: Side-by-side comparison of fusion vs main events
- `fusion_correlation_analysis.py`: Script for fusion correlation analysis

## Conclusion

The fusion event correlation analysis reveals a strong positive relationship between fusion event duration and scale sim total cycles, similar to but slightly weaker than the main event correlation. This suggests that fusion operations are an important component of matrix multiplication performance and are well-modeled by the SCALE-Sim simulator. The correlation indicates that fusion operations scale predictably with computational complexity, making them a reliable indicator of overall performance characteristics. 