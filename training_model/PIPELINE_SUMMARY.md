# Pipeline Test Results Summary

## Test Execution

**Date:** February 2, 2026  
**Training Dataset:** `model/add_dataset_20260202_171650.csv` (1001 samples)  
**Test Shapes:** 50 shapes  
**Random Seed:** 100  
**Operation:** elementwise add (FP16)  

## Pipeline Overview

The pipeline successfully completed all 6 steps:

1. ✅ **Model Training** - Trained HistGradientBoostingRegressor on 803 samples
2. ✅ **Shape Generation** - Generated 50 diverse test shapes
3. ✅ **TPU Profiling** - Profiled all 50 shapes on TPUv4
4. ✅ **Latency Prediction** - Generated predictions using trained model
5. ✅ **Comparison** - Compared predictions vs actual measurements
6. ✅ **Reporting** - Generated detailed analysis and metrics

## Model Performance

### Training Metrics
- **Training Samples:** 803
- **Validation Samples:** 197
- **Validation MAE:** 0.401064
- **Validation MAPE:** 6.63%

### Test Set Performance
- **Test Cases:** 50 shapes
- **Mean Absolute Error (MAE):** 0.213347
- **Mean Absolute Percentage Error:** 5.77%
- **Median Absolute Percentage Error:** 2.57%
- **R² Score:** 0.8945 (89.45% variance explained)

## Detailed Results

### Error Distribution

| Percentile | Absolute Error | Relative Error |
|------------|----------------|----------------|
| 25th       | 0.014309       | 1.55%          |
| 50th       | 0.023025       | 2.57%          |
| 75th       | 0.047294       | 3.99%          |
| 90th       | 0.287371       | 10.37%         |
| 95th       | 1.096237       | 16.56%         |
| 99th       | 3.337411       | 58.56%         |

### Accuracy Breakdown

| Error Range | Count | Percentage |
|-------------|-------|------------|
| < 1%        | 8     | 16.0%      |
| 1-5%        | 32    | 64.0%      |
| 5-10%       | 4     | 8.0%       |
| 10-20%      | 4     | 8.0%       |
| > 20%       | 2     | 4.0%       |

**Key Insight:** 80% of predictions have < 5% error!

### Performance by Tensor Size

| Size Range | Samples | MAE      | MAPE   | Median APE |
|------------|---------|----------|--------|------------|
| < 100      | 20      | 0.017770 | 2.02%  | 2.07%      |
| 100-1K     | 8       | 0.017736 | 1.97%  | 1.94%      |
| 1K-10K     | 16      | 0.066133 | 5.59%  | 3.70%      |
| 10K-100K   | 1       | 0.073166 | 4.03%  | 4.03%      |
| > 100K     | 5       | 1.807753 | 27.84% | 15.07%     |

**Observation:** Model performs best on small to medium tensors (< 10K elements)

### Best Predictions (Top 5)

1. **Shape (2, 2, 2):** 0.13% error
2. **Shape (4, 18, 7):** 0.13% error
3. **Shape (23, 2, 1):** 0.68% error
4. **Shape (16, 1, 1):** 0.68% error
5. **Shape (4515, 1, 1):** 0.73% error

### Worst Predictions (Top 5)

1. **Shape (3, 33349, 1):** 94.43% error (very large 2D tensor)
2. **Shape (14, 266, 2):** 21.22% error
3. **Shape (30, 17476, 1):** 17.78% error (large 2D tensor)
4. **Shape (453390, 1, 1):** 15.07% error (very large 1D tensor)
5. **Shape (8, 266, 4):** 11.19% error

**Pattern:** Large outlier shapes (> 100K elements) are harder to predict accurately.

## Output Files

All results saved to `pipeline_test_results/`:

| File | Size | Description |
|------|------|-------------|
| `add_model_20260202_174819.pkl` | 1.3 MB | Trained model (pickle) |
| `comparison_20260202_174819.csv` | 4.3 KB | Predictions vs actual with errors |
| `summary_20260202_174819.txt` | 806 B | Quick summary report |
| `test_profiling_20260202_174819.csv` | 1.6 KB | Raw TPU profiling results |
| `test_metadata_20260202_174819.csv` | 317 B | Profiling metadata |

## Key Findings

### ✅ Strengths
1. **High Accuracy:** Median error of only 2.57%
2. **Consistent:** 80% of predictions within 5% error
3. **Fast Training:** Model trains in seconds
4. **Good Generalization:** R² = 0.8945 on unseen data
5. **Small-Medium Tensors:** Excellent accuracy for most common sizes

### ⚠️ Limitations
1. **Large Tensors:** Higher errors for tensors > 100K elements
2. **Outliers:** Some extreme shapes show 10-20%+ errors
3. **Coverage:** Model trained on up to 1M elements; may extrapolate poorly beyond

### 💡 Recommendations
1. **For Production Use:**
   - Use confidently for tensors < 10K elements
   - Add uncertainty bounds for large tensors
   - Retrain with more large tensor samples if needed

2. **Model Improvements:**
   - Collect more training data for large shapes (> 100K elements)
   - Consider separate models for different size ranges
   - Add shape-specific features (aspect ratio, dimensionality)

3. **Next Steps:**
   - Profile additional operations (mul, matmul, etc.)
   - Extend to other data types (FP32, BF16)
   - Create ensemble models for better accuracy

## Usage Examples

### Run Full Pipeline
```bash
python pipeline.py model/add_dataset_20260202_171650.csv \
    --output-dir ./my_results \
    --n-test-shapes 100 \
    --seed 42
```

### Analyze Results
```bash
python analyze_results.py pipeline_test_results/comparison_20260202_174819.csv
```

### Load Model and Predict
```python
from model import model_manager as mm

# Load model
model_data = mm.load_model("pipeline_test_results/add_model_20260202_174819.pkl")

# Predict for new shapes
shapes = [(128, 128, 1), (256, 256, 1), (512, 512, 1)]
predictions = mm.predict_latency(model_data["model"], shapes)
print(predictions)
```

## Conclusion

The pipeline demonstrates **strong performance** with a median prediction error of only **2.57%**. The model is production-ready for small to medium tensor sizes and provides a solid foundation for TPU latency prediction. The automated workflow successfully trains models, validates predictions, and generates comprehensive comparison reports.

**Status:** ✅ **PIPELINE VALIDATED AND READY FOR USE**







