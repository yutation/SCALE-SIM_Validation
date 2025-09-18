🔍 Key Correlation Findings
📊 Overall Correlation Strength
Average correlation: 0.982 (very high!)
Range: 0.975 - 0.986 (very tight range)
All correlations are statistically significant (p < 0.001)
🥇 Most Highly Correlated Function Pairs (r > 0.984):
leaky_relu ↔ selu: 0.986 (highest)
selu ↔ sigmoid: 0.985
elu ↔ leaky_relu: 0.985
sigmoid ↔ tanh: 0.985
elu ↔ selu: 0.984
🥉 Least Correlated Function Pairs (still very high):
binary ↔ relu: 0.975 (lowest, but still very strong!)
binary ↔ tanh: 0.976
binary ↔ linear: 0.977
📈 What This Means
Function Similarity Groups:
Similar Group 1: leaky_relu, selu, elu (r > 0.984)
Similar Group 2: sigmoid, tanh (r = 0.985)
Somewhat Distinct: binary (lowest correlations with others)
Performance Implications:
Highly Predictable: If one function is slow/fast for a shape, others will be too
Consistent Ranking: Function performance ranking stays relatively consistent across shapes
binary is Most Unique: Has the most distinct performance characteristics
Shape Dependency: All functions scale similarly with tensor size/complexity
📊 Generated Visualizations:
pearson_correlation_heatmap.png - Overall correlation matrix
spearman_correlation_heatmap.png - Rank-based correlations
Scatter plots (in correlation_plots/):
leaky_vs_selu.png
selu_vs_sigmoid.png
elu_vs_leaky.png
sigmoid_vs_tanh.png
elu_vs_selu.png
🎯 Practical Insights:
Hardware Optimization: Since functions are highly correlated, optimizations for one function likely benefit others
Benchmarking: You can use a subset of functions for performance testing
Binary Functions Stand Out: binary_step has the most unique performance profile
Function Clusters: Consider grouping similar functions for analysis
The extremely high correlations (0.975-0.986) suggest that kernel shape and tensor size are the primary drivers of performance, rather than the specific activation function implementation. This indicates consistent computational patterns across different activation functions! 