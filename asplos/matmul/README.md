# SCALE-Sim TPU Latency Verification Results

This directory contains verification results comparing SCALE-Sim's predicted TPU latencies against actual measured TPU latencies for matrix multiplication (GEMM) operations.

## Files

- `merged_verification_results.csv` - Raw verification data with 100 GEMM kernels
- `plot_verification_results.py` - Visualization script
- `plots/` - Generated visualizations and summary

## Quick Summary

**Overall Performance:**
- **MAPE (Mean Absolute Percentage Error):** 19.69%
- **Median Error:** -1.54%
- **Predictions within ±20% error:** 77/100 (77%)

**Key Findings:**
- ✅ Good accuracy for **small GEMMs** (MAPE: 8.19%)
- ⚠️ Moderate accuracy for **medium GEMMs** (MAPE: 22.71%)
- ⚠️ Lower accuracy for **large GEMMs** (MAPE: 28.07%)
- 📊 Slight underestimation bias (mean signed error: -5.89%)
- 🎯 40% of predictions within ±10% error
- 🎯 77% of predictions within ±20% error

## Visualizations

### 1️⃣ Primary Plot: Estimated vs. Measured TPU Latency

**File:** `plots/estimated_vs_measured.png`

**What it shows:**
- Scatter plot comparing predicted vs. actual latencies
- Red dashed line (y=x) represents perfect prediction
- Green shaded region shows ±10% error band
- Statistics box includes MAPE, median error, RMSE, and sample count

**Key Insights:**
- Most points cluster near the y=x line, indicating good overall accuracy
- Some outliers show significant underestimation for large GEMMs
- Model shows slight tendency to underestimate (points above y=x line)

### 2️⃣ Secondary Plot: Relative Error vs. GEMM Size

**File:** `plots/error_vs_size.png`

**What it shows:**
- X-axis: GEMM problem size (M × N × K) on log scale
- Y-axis: Relative error percentage
- Color indicates FLOPs magnitude
- Reference lines at 0% (perfect) and ±10% error
- Red circles highlight high-error cases (>±30%)

**Key Insights:**
- Error increases with problem size (log scale reveals pattern)
- Small GEMMs have tighter error distribution
- Large GEMMs show more variability and underestimation
- Most problematic predictions occur at larger problem sizes

### 3️⃣ Bonus Plot: Error Distribution

**File:** `plots/error_distribution.png`

**What it shows:**
- Left: Histogram of prediction errors
- Right: Box plot by problem size category (Small/Medium/Large)

**Key Insights:**
- Error distribution is roughly centered near zero
- Some extreme outliers (especially one at ~293% error)
- Error variance increases with problem size
- Median error remains close to zero across all size categories

## Running the Visualization

```bash
cd /home/Owner/work/SCALE-Sim/validation/asplos/matmul
python plot_verification_results.py
```

This will regenerate all plots in the `plots/` directory.

## Interpretation

### ✅ What Works Well
1. **Small GEMMs:** Excellent accuracy (8.19% MAPE)
2. **Overall bias:** Minimal systematic bias (-5.89% mean error)
3. **Majority accuracy:** 77% of predictions within ±20%

### ⚠️ Areas for Improvement
1. **Large GEMMs:** Higher error rates (28.07% MAPE)
2. **Outliers:** A few extreme cases (e.g., matmul_linear_62 at 293.5% error)
3. **Scaling:** Model accuracy degrades with problem size

### 🔍 Recommended Next Steps
1. Investigate why matmul_linear_62 [(131, 1024), (1024, 867)] has 293% error
2. Analyze if large GEMM underestimation is due to:
   - Memory hierarchy effects not captured
   - Parallelization efficiency at scale
   - Cache behavior for large matrices
3. Consider size-dependent calibration factors
4. Validate against more large-scale GEMMs

## Data Format

The CSV contains the following columns:
- `Operation_Type` - Type of operation (matmul)
- `Operation` - Specific operation (linear)
- `Input_Shapes` - Matrix dimensions [(M, K), (K, N)]
- `Predicted_Latency_us` - SCALE-Sim prediction (microseconds)
- `Kernel_Name` - Unique kernel identifier
- `Actual_Duration_us` - Measured TPU latency (microseconds)
- `Error_Percentage` - Relative error: 100 × (Predicted - Actual) / Actual

## Citation

If you use these results, please cite the SCALE-Sim paper and mention this validation study.




