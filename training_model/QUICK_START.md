# Quick Start Guide

## 🚀 Running the Pipeline in 3 Steps

### Step 1: Prepare Your Data
Make sure you have a profiling dataset CSV with these columns:
- `operation_name`, `input_dim_0`, `input_dim_1`, `input_dim_2`, `size`, `computation_mean`

### Step 2: Run the Pipeline
```bash
python pipeline.py <your_training_csv> --output-dir ./results --n-test-shapes 50
```

### Step 3: View Results
```bash
cat results/summary_*.txt
python analyze_results.py results/comparison_*.csv
```

## 📋 Common Commands

### Basic Pipeline Run
```bash
python pipeline.py model/add_dataset_20260202_171650.csv \
    --output-dir ./my_results \
    --n-test-shapes 50 \
    --seed 42
```

### Larger Test Set
```bash
python pipeline.py model/add_dataset_20260202_171650.csv \
    --output-dir ./large_test \
    --n-test-shapes 200 \
    --max-numel 2097152 \
    --seed 42
```

### Skip Profiling (Use Existing Data)
```bash
python pipeline.py model/add_dataset_20260202_171650.csv \
    --output-dir ./results \
    --skip-profiling \
    --test-profiling-csv path/to/existing_test.csv
```

### Use Shell Script
```bash
./example_pipeline.sh
```

## 📊 Understanding the Output

### Files Generated
- **`add_model_*.pkl`** - Trained model (save this!)
- **`comparison_*.csv`** - Main results with errors
- **`summary_*.txt`** - Quick overview
- **`test_profiling_*.csv`** - Raw measurements

### Key Metrics
- **MAE**: Mean Absolute Error (lower is better)
- **MAPE**: Mean Absolute Percentage Error (lower is better)
- **R²**: Coefficient of determination (higher is better, max 1.0)
- **Median APE**: Median error percentage (more robust than mean)

## 🔍 Analyzing Results

### View Summary
```bash
cat results/summary_*.txt
```

### Detailed Analysis
```bash
python analyze_results.py results/comparison_*.csv
```

### View Comparison Data
```bash
head -20 results/comparison_*.csv
```

### Check Best/Worst Predictions
```bash
# Sort by error to see worst predictions
sort -t',' -k9 -rn results/comparison_*.csv | head -10
```

## 💻 Using the Model Programmatically

### Load and Predict
```python
from model import model_manager as mm

# Load trained model
model_data = mm.load_model("results/add_model_20260202_174819.pkl")

# Predict latencies
shapes = [(100, 100, 1), (256, 256, 1), (64, 64, 64)]
predictions = mm.predict_latency(model_data["model"], shapes)

for shape, pred in zip(shapes, predictions):
    print(f"Shape {shape}: {pred:.6f} ms")
```

### Train New Model
```python
from model import model_manager as mm

model = mm.train_and_save(
    csv_path="data/my_profiling_data.csv",
    save_path="models/my_model.pkl",
    op_name="add",
    seed=42
)
```

## 🛠️ Troubleshooting

### Problem: Import Errors
**Solution:** Make sure you're in the project root:
```bash
cd /home/Owner/work/SCALE-Sim/training_model
python pipeline.py ...
```

### Problem: TPU Not Found
**Solution:** Check JAX can see TPUs:
```python
import jax
print(jax.devices())
```

### Problem: Out of Memory
**Solution:** Reduce test set size or max_numel:
```bash
python pipeline.py data.csv --n-test-shapes 25 --max-numel 262144
```

## 📖 Documentation

- **`PIPELINE_README.md`** - Full documentation
- **`PIPELINE_SUMMARY.md`** - Latest test results
- **`analyze_results.py`** - Result analysis tool

## ✅ Expected Results

With the provided test dataset, you should see:
- **Median Error:** ~2.5%
- **80% of predictions:** < 5% error
- **R² Score:** > 0.89

## 🎯 Next Steps

1. ✅ Verify pipeline works with test data
2. ✅ Train on your own profiling data
3. ✅ Use trained model for predictions
4. ✅ Extend to other operations (mul, matmul, etc.)
5. ✅ Optimize model hyperparameters if needed

## 💡 Tips

- Start with small test sets (50 shapes) to iterate quickly
- Use different seeds to test model robustness
- Check the comparison CSV to understand error patterns
- Large tensors (> 100K elements) may need more training data
- The model trains in seconds, so experiment freely!

---

**Need help?** Check `PIPELINE_README.md` for detailed documentation.





