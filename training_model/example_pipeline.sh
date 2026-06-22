#!/bin/bash
# Example script to run the full pipeline

set -e  # Exit on error

echo "========================================"
echo "Latency Prediction Pipeline Example"
echo "========================================"

# Configuration
TRAINING_CSV="model/add_dataset_20260202_170757.csv"
OUTPUT_DIR="./pipeline_results_$(date +%Y%m%d_%H%M%S)"
N_TEST_SHAPES=100
MAX_NUMEL=1048576  # 1M elements
SEED=42
OP_NAME="add"
DTYPE="float16"

echo ""
echo "Configuration:"
echo "  Training CSV:  $TRAINING_CSV"
echo "  Output Dir:    $OUTPUT_DIR"
echo "  Test Shapes:   $N_TEST_SHAPES"
echo "  Max Numel:     $MAX_NUMEL"
echo "  Seed:          $SEED"
echo "  Operation:     $OP_NAME"
echo "  Data Type:     $DTYPE"
echo ""

# Check if training CSV exists
if [ ! -f "$TRAINING_CSV" ]; then
    echo "ERROR: Training CSV not found: $TRAINING_CSV"
    echo ""
    echo "You can create one by running:"
    echo "  cd tpu_profiling"
    echo "  python profile_dataset.py --n-shapes 1000 --output-dir ../model"
    exit 1
fi

echo "Starting pipeline..."
echo ""

# Run the pipeline
python pipeline.py "$TRAINING_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --n-test-shapes "$N_TEST_SHAPES" \
    --max-numel "$MAX_NUMEL" \
    --seed "$SEED" \
    --op-name "$OP_NAME" \
    --dtype "$DTYPE"

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files generated:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "View summary:"
echo "  cat $OUTPUT_DIR/summary_*.txt"
echo ""
echo "View comparison data:"
echo "  head -20 $OUTPUT_DIR/comparison_*.csv"
echo ""







