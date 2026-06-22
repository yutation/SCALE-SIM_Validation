# Latency Prediction Pipeline

Complete pipeline for training latency prediction models and validating them on TPU.

## Overview

This pipeline orchestrates the following workflow:

1. **Train** a model on existing profiling data
2. **Generate** new test shapes
3. **Profile** those shapes on TPU
4. **Predict** latencies using the trained model
5. **Compare** predictions vs actual measurements
6. **Report** results with detailed error analysis

## Files

- **`pipeline.py`** - Main pipeline script
- **`model/model_manager.py`** - Model training, saving, loading, and inference utilities
- **`model/train.py`** - Original training code (unchanged)
- **`tpu_profiling/`** - TPU profiling modules

## Quick Start

### Basic Usage

```bash
# Run complete pipeline with existing training data
python pipeline.py model/add_dataset_20260202_170757.csv \
    --output-dir ./results \
    --n-test-shapes 100 \
    --seed 42
```

### Advanced Usage

```bash
# Custom configuration
python pipeline.py path/to/training_data.csv \
    --output-dir ./my_results \
    --n-test-shapes 200 \
    --max-numel 2097152 \
    --seed 123 \
    --op-name add \
    --dtype float16
```

### Skip Profiling (Use Existing Test Data)

If you already have test profiling data:

```bash
python pipeline.py model/add_dataset_20260202_170757.csv \
    --output-dir ./results \
    --skip-profiling \
    --test-profiling-csv path/to/existing_test_profiling.csv
```

## Arguments

- **`training_csv`** (required): Path to training dataset CSV
- **`--output-dir, -o`**: Output directory for results (default: `./pipeline_results`)
- **`--n-test-shapes, -n`**: Number of test shapes to profile (default: `100`)
- **`--max-numel, -m`**: Maximum elements per test tensor (default: `1048576`)
- **`--seed, -s`**: Random seed for reproducibility (default: `42`)
- **`--op-name`**: Operation name (default: `add`)
- **`--dtype`**: Data type - `float16`, `float32`, or `bfloat16` (default: `float16`)
- **`--skip-profiling`**: Skip TPU profiling and use existing data
- **`--test-profiling-csv`**: Path to existing test profiling CSV (required with `--skip-profiling`)

## Output Files

The pipeline generates the following in the output directory:

1. **`{op_name}_model_{timestamp}.pkl`** - Trained model (pickle format)
2. **`test_profiling_{timestamp}.csv`** - Actual TPU profiling results
3. **`comparison_{timestamp}.csv`** - Predictions vs actual with errors
4. **`summary_{timestamp}.txt`** - Detailed performance report
5. **`test_metadata_{timestamp}.csv`** - Profiling metadata

## Comparison CSV Format

The comparison CSV contains:

| Column | Description |
|--------|-------------|
| `shape_idx` | Test shape index |
| `dim_0`, `dim_1`, `dim_2` | Shape dimensions |
| `size` | Total number of elements |
| `actual_latency` | Measured latency from TPU |
| `predicted_latency` | Model prediction |
| `absolute_error` | Absolute difference |
| `relative_error_pct` | Percentage error |

## Using the Model Manager Directly

You can also use `model_manager.py` independently:

### Train and Save a Model

```python
from model import model_manager as mm

model = mm.train_and_save(
    csv_path="data/training.csv",
    save_path="models/my_model.pkl",
    op_name="add",
    seed=42
)
```

### Load and Predict

```python
from model import model_manager as mm

# Load model
model_data = mm.load_model("models/my_model.pkl")

# Predict for new shapes
test_shapes = [(100, 100, 1), (256, 256, 1), (512, 512, 1)]
predictions = mm.predict_latency(model_data["model"], test_shapes)

print("Predictions:", predictions)
```

### Quick Prediction

```python
from model import model_manager as mm

predictions = mm.predict_from_saved_model(
    model_path="models/my_model.pkl",
    shapes=[(100, 1, 1), (1000, 1, 1)]
)
```

## Example Workflow

```bash
# 1. Profile shapes to create training dataset (if you don't have one)
cd tpu_profiling
python profile_dataset.py --n-shapes 1000 --output-dir ./add_dataset

# 2. Run the full pipeline
cd ..
python pipeline.py \
    tpu_profiling/add_dataset/add_dataset_*.csv \
    --output-dir ./pipeline_results \
    --n-test-shapes 100 \
    --seed 42

# 3. Check results
cat pipeline_results/summary_*.txt
```

## Performance Metrics

The pipeline reports:

- **MAE** (Mean Absolute Error): Average absolute difference
- **MAPE** (Mean Absolute Percentage Error): Average relative error
- **Median APE**: Median relative error (more robust to outliers)
- **Max Errors**: Worst-case errors
- **R² Score**: Coefficient of determination
- **Error Percentiles**: Distribution analysis (50th, 75th, 90th, 95th, 99th)

## Tips

1. **Training Data Size**: Use at least 500-1000 diverse shapes for training
2. **Test Set**: Keep test shapes separate from training (pipeline uses different seed)
3. **Max Numel**: Adjust based on your hardware memory constraints
4. **Seed**: Use consistent seeds for reproducible experiments
5. **Error Analysis**: Check the comparison CSV to identify problematic shape patterns

## Troubleshooting

### Import Errors

If you get import errors, ensure you're running from the project root:

```bash
cd /home/Owner/work/SCALE-Sim/training_model
python pipeline.py ...
```

### TPU Not Available

If TPU profiling fails, check JAX TPU setup:

```python
import jax
print(jax.devices())  # Should show TPU devices
```

### Memory Issues

For large test sets, reduce `n_test_shapes` or `max_numel`:

```bash
python pipeline.py data.csv --n-test-shapes 50 --max-numel 524288
```

## Requirements

See `tpu_profiling/requirements.txt` for dependencies:

- JAX (with TPU support)
- NumPy
- Pandas
- scikit-learn

Install with:

```bash
pip install -r tpu_profiling/requirements.txt
pip install scikit-learn
```







