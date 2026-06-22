"""
Full pipeline for 2D ReLU operations: Train model -> Profile new shapes -> Compare predictions vs actual.

This script orchestrates the complete workflow for 2D tensor shapes:
1. Train a model on existing profiling dataset (2D shapes)
2. Generate new test shapes (2D)
3. Profile those shapes on TPU
4. Predict latencies using the trained model
5. Compare predictions vs actual measurements
6. Generate comparison report and CSV
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# Add tpu_profiling to path
tpu_profiling_dir = Path(__file__).parent / "tpu_profiling"
sys.path.insert(0, str(tpu_profiling_dir))

# Add model to path
model_dir = Path(__file__).parent / "model"
sys.path.insert(0, str(model_dir))

import jax.numpy as jnp
from model import model_manager_2d as mm2d


# ============================================================================
# CONFIGURATION - Edit these variables to customize the pipeline
# ============================================================================

TRAINING_CSV = "./model/relu_dataset_2d.csv"
OUTPUT_DIR = "./pipeline_results_relu_2d"
N_TEST_SHAPES = 100
MAX_NUMEL = 16 * 1024 * 1024  # 16M elements max
SEED = 1999
OP_NAME = "relu"
DTYPE = "float16"  # Options: "float16", "float32", "bfloat16"

# Set to True to skip profiling and use existing data
SKIP_PROFILING = False
TEST_PROFILING_CSV = None  # Path to existing profiling CSV (if SKIP_PROFILING=True)

# ============================================================================


def generate_2d_shapes(
    n_shapes: int,
    max_numel: int,
    seed: int,
    boundary_frac: float = 0.30,
    perturb_range: int = 32,
) -> list:
    """
    Generate 2D shapes for testing.
    
    Args:
        n_shapes: Number of shapes to generate
        max_numel: Maximum number of elements (d0 * d1)
        seed: Random seed
        boundary_frac: Fraction of shapes at power-of-2 boundaries
        perturb_range: Range for perturbation around boundaries
        
    Returns:
        List of (d0, d1) tuples
    """
    rng = np.random.RandomState(seed)
    shapes = []
    
    # Common sizes for 2D tensors
    common_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    
    n_boundary = int(n_shapes * boundary_frac)
    n_random = n_shapes - n_boundary
    
    # Generate boundary shapes (power-of-2 or near power-of-2)
    for _ in range(n_boundary):
        # Pick common sizes that satisfy the constraint
        valid_sizes = [s for s in common_sizes if s <= max_numel]
        d0 = rng.choice(valid_sizes)
        
        # Choose d1 such that d0 * d1 <= max_numel
        max_d1 = max_numel // d0
        valid_d1_sizes = [s for s in common_sizes if s <= max_d1]
        
        if valid_d1_sizes:
            d1 = rng.choice(valid_d1_sizes)
        else:
            d1 = max_d1
        
        # Optionally perturb
        if rng.rand() < 0.5:
            d0 = max(1, d0 + rng.randint(-perturb_range, perturb_range + 1))
            d1 = max(1, d1 + rng.randint(-perturb_range, perturb_range + 1))
            
            # Ensure constraint is still satisfied
            if d0 * d1 > max_numel:
                d1 = max_numel // d0
        
        shapes.append((int(d0), int(d1)))
    
    # Generate random shapes
    for _ in range(n_random):
        # Random sampling with constraint
        d0 = rng.randint(1, int(np.sqrt(max_numel)) + 1)
        max_d1 = max_numel // d0
        d1 = rng.randint(1, max_d1 + 1)
        
        shapes.append((int(d0), int(d1)))
    
    # Remove duplicates
    shapes = list(set(shapes))
    
    # If we lost shapes due to deduplication, generate more
    while len(shapes) < n_shapes:
        d0 = rng.randint(1, int(np.sqrt(max_numel)) + 1)
        max_d1 = max_numel // d0
        d1 = rng.randint(1, max_d1 + 1)
        shape = (int(d0), int(d1))
        if shape not in shapes:
            shapes.append(shape)
    
    return shapes[:n_shapes]


def run_pipeline(
    training_csv: str,
    output_dir: str,
    n_test_shapes: int = 100,
    max_numel: int = 1024 * 1024,
    seed: int = 42,
    op_name: str = "relu",
    dtype_str: str = "float16",
    skip_profiling: bool = False,
    test_profiling_csv: str = None,
):
    """
    Run the complete train-profile-compare pipeline for 2D shapes.
    
    Args:
        training_csv: Path to CSV with training data
        output_dir: Directory for all outputs
        n_test_shapes: Number of test shapes to profile
        max_numel: Maximum number of elements per test tensor
        seed: Random seed
        op_name: Operation name (e.g., "relu")
        dtype_str: Data type ("float16", "float32", "bfloat16")
        skip_profiling: If True, use existing test_profiling_csv instead of profiling
        test_profiling_csv: Path to existing test profiling results (if skip_profiling=True)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"LATENCY PREDICTION PIPELINE (2D) - {op_name.upper()}")
    print("="*70)
    print(f"Training CSV: {training_csv}")
    print(f"Output dir:   {output_dir}")
    print(f"Test shapes:  {n_test_shapes}")
    print(f"Max numel:    {max_numel:,}")
    print(f"Seed:         {seed}")
    print("="*70 + "\n")
    
    # ========================================
    # STEP 1: Train model on existing dataset
    # ========================================
    print("\n" + "="*70)
    print("STEP 1: TRAINING 2D MODEL")
    print("="*70)
    
    model_path = output_dir / f"{op_name}_model_2d_{timestamp}.pkl"
    model = mm2d.train_and_save(
        training_csv,
        str(model_path),
        op_name=op_name,
        seed=seed
    )
    
    # ========================================
    # STEP 2: Generate test shapes
    # ========================================
    print("\n" + "="*70)
    print("STEP 2: GENERATING 2D TEST SHAPES")
    print("="*70)
    
    # Use different seed for test shapes to avoid overlap with training
    test_seed = seed + 1000
    
    print(f"Generating {n_test_shapes} 2D test shapes (seed={test_seed})...")
    test_shapes = generate_2d_shapes(
        n_shapes=n_test_shapes,
        max_numel=max_numel,
        seed=test_seed,
        boundary_frac=0.30,
        perturb_range=32,
    )
    
    print(f"Generated {len(test_shapes)} unique 2D test shapes")
    
    # Statistics
    numels = [d0 * d1 for d0, d1 in test_shapes]
    print(f"  Min numel:    {min(numels):,}")
    print(f"  Max numel:    {max(numels):,}")
    print(f"  Median numel: {sorted(numels)[len(numels)//2]:,}")
    
    # ========================================
    # STEP 3: Profile test shapes on TPU
    # ========================================
    if not skip_profiling:
        print("\n" + "="*70)
        print("STEP 3: PROFILING 2D TEST SHAPES ON TPU")
        print("="*70)
        
        test_profiling_dir = output_dir / f"test_profiling_2d_{timestamp}"
        
        # Map dtype
        dtype_map = {
            "float16": jnp.float16,
            "float32": jnp.float32,
            "bfloat16": jnp.bfloat16,
        }
        dtype = dtype_map[dtype_str]
        
        # Run profiling using the existing profile_dataset module
        print(f"Profiling {len(test_shapes)} 2D shapes on TPU...")
        
        # We need to manually create the profiling using the lower-level functions
        import tpu_profiling.profiling_manager as pm
        import tpu_profiling.jax_kernel_functions as jkf
        
        # Create profiling configuration
        test_profiling_csv_path = output_dir / f"test_profiling_2d_{timestamp}.csv"
        metadata_file = output_dir / f"test_metadata_2d_{timestamp}.csv"
        
        pm_configuration = {
            "storage_file": str(test_profiling_csv_path),
            "storage_metadata_file": str(metadata_file),
            "append_to_metadata_file": False,
            "hardware_config": "TPUv4",
            "operator_name": op_name,
            "common_operator_dimensions": None,
            "data_precision": dtype_str.upper(),
            "profiler_iterations": 5,
            "random_seed": test_seed,
            "repo_version": "v0.0.1",
            "comment": f"2D test_shapes n={n_test_shapes} max_numel={max_numel} seed={test_seed}",
        }
        
        # Create profiling manager
        manager = pm.ProfilingManagerSimpleElementwise(
            f"{op_name}_test_2d",
            str(test_profiling_dir),
            pm_configuration
        )
        
        # Add profilers for each test shape
        print(f"Adding {len(test_shapes)} profilers...")
        for i, (d0, d1) in enumerate(test_shapes):
            # 2D shapes
            shape = (d0, d1)
            
            # ReLU is a unary operation (only one input)
            kernel_wrapper = jkf.KernelWarpper(
                op_name,
                [(shape, dtype)]
            )
            
            shape_str = f"{d0}x{d1}"
            profiler_name = f"{op_name}_{shape_str}_{i:05d}"
            
            manager.add_profiler(profiler_name, kernel_wrapper)
            
            if (i + 1) % 50 == 0:
                print(f"  Added {i + 1}/{len(test_shapes)} profilers")
        
        print(f"Added {len(test_shapes)} profilers")
        
        # Run profiling
        print("\nRunning profiling on TPU...")
        manager.profile_and_post_process_all_profilers()
        
        # Write results
        print("Writing profiling results...")
        manager.write_results()
        
        print(f"Test profiling complete: {test_profiling_csv_path}")
    else:
        print("\n" + "="*70)
        print("STEP 3: SKIPPING PROFILING (using existing data)")
        print("="*70)
        test_profiling_csv_path = Path(test_profiling_csv)
        print(f"Using existing profiling data: {test_profiling_csv_path}")
    
    # ========================================
    # STEP 4: Predict latencies using model
    # ========================================
    print("\n" + "="*70)
    print("STEP 4: PREDICTING LATENCIES")
    print("="*70)
    
    print(f"Predicting latencies for {len(test_shapes)} 2D shapes...")
    predictions = mm2d.predict_latency(model, test_shapes)
    print(f"Generated {len(predictions)} predictions")
    
    # ========================================
    # STEP 5: Compare predictions vs actual
    # ========================================
    print("\n" + "="*70)
    print("STEP 5: COMPARING PREDICTIONS VS ACTUAL")
    print("="*70)
    
    # Load actual profiling results
    actual_df = pd.read_csv(test_profiling_csv_path)
    actual_df.columns = actual_df.columns.str.strip()
    
    # Prepare comparison dataframe
    comparison_data = []
    
    for i, (d0, d1) in enumerate(test_shapes):
        # Find matching row in actual results
        mask = (
            (actual_df["input_dim_0"] == d0) &
            (actual_df["input_dim_1"] == d1)
        )
        
        if mask.sum() == 0:
            print(f"Warning: No match found for shape ({d0}, {d1})")
            continue
        
        actual_row = actual_df[mask].iloc[0]
        actual_latency = float(actual_row["computation_mean"])
        predicted_latency = float(predictions[i])
        
        # Calculate errors
        absolute_error = abs(predicted_latency - actual_latency)
        relative_error = absolute_error / max(actual_latency, 1e-9) * 100
        
        comparison_data.append({
            "shape_idx": i,
            "dim_0": d0,
            "dim_1": d1,
            "size": d0 * d1,
            "actual_latency": actual_latency,
            "predicted_latency": predicted_latency,
            "absolute_error": absolute_error,
            "relative_error_pct": relative_error,
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison CSV
    comparison_csv = output_dir / f"comparison_2d_{timestamp}.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"Comparison saved to: {comparison_csv}")
    
    # ========================================
    # STEP 6: Generate summary report
    # ========================================
    print("\n" + "="*70)
    print("STEP 6: SUMMARY REPORT")
    print("="*70)
    
    actual = comparison_df["actual_latency"].values
    predicted = comparison_df["predicted_latency"].values
    abs_errors = comparison_df["absolute_error"].values
    rel_errors = comparison_df["relative_error_pct"].values
    
    mae = np.mean(abs_errors)
    mape = np.mean(rel_errors)
    median_ape = np.median(rel_errors)
    max_error = np.max(abs_errors)
    max_rel_error = np.max(rel_errors)
    
    # R² score
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"\nPrediction Performance (2D {op_name.upper()}):")
    print(f"  Number of test cases: {len(comparison_df)}")
    print(f"  Mean Absolute Error (MAE):     {mae:.6f}")
    print(f"  Mean Abs Percentage Error:     {mape:.2f}%")
    print(f"  Median Abs Percentage Error:   {median_ape:.2f}%")
    print(f"  Max Absolute Error:            {max_error:.6f}")
    print(f"  Max Relative Error:            {max_rel_error:.2f}%")
    print(f"  R² Score:                      {r2:.4f}")
    
    # Percentile analysis
    print(f"\nError Distribution (Relative %):")
    for p in [50, 75, 90, 95, 99]:
        val = np.percentile(rel_errors, p)
        print(f"  {p}th percentile: {val:.2f}%")
    
    # Save summary report
    summary_path = output_dir / f"summary_2d_{timestamp}.txt"
    with open(summary_path, "w") as f:
        f.write("="*70 + "\n")
        f.write(f"LATENCY PREDICTION PIPELINE SUMMARY (2D) - {op_name.upper()}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Training CSV: {training_csv}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test shapes: {n_test_shapes}\n")
        f.write(f"Seed: {seed}\n\n")
        f.write("Prediction Performance:\n")
        f.write(f"  Number of test cases: {len(comparison_df)}\n")
        f.write(f"  Mean Absolute Error (MAE):     {mae:.6f}\n")
        f.write(f"  Mean Abs Percentage Error:     {mape:.2f}%\n")
        f.write(f"  Median Abs Percentage Error:   {median_ape:.2f}%\n")
        f.write(f"  Max Absolute Error:            {max_error:.6f}\n")
        f.write(f"  Max Relative Error:            {max_rel_error:.2f}%\n")
        f.write(f"  R² Score:                      {r2:.4f}\n\n")
        f.write("Error Distribution (Relative %):\n")
        for p in [50, 75, 90, 95, 99]:
            val = np.percentile(rel_errors, p)
            f.write(f"  {p}th percentile: {val:.2f}%\n")
    
    print(f"\nSummary report saved to: {summary_path}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  Model:        {model_path}")
    print(f"  Profiling:    {test_profiling_csv_path}")
    print(f"  Comparison:   {comparison_csv}")
    print(f"  Summary:      {summary_path}")
    print()
    
    return {
        "model": model,
        "model_path": model_path,
        "test_shapes": test_shapes,
        "comparison_df": comparison_df,
        "metrics": {
            "mae": mae,
            "mape": mape,
            "median_ape": median_ape,
            "r2": r2,
        }
    }


def main():
    """Run the pipeline with configuration from variables at top of file."""
    
    if SKIP_PROFILING and not TEST_PROFILING_CSV:
        raise ValueError("TEST_PROFILING_CSV must be set when SKIP_PROFILING is True")
    
    run_pipeline(
        training_csv=TRAINING_CSV,
        output_dir=OUTPUT_DIR,
        n_test_shapes=N_TEST_SHAPES,
        max_numel=MAX_NUMEL,
        seed=SEED,
        op_name=OP_NAME,
        dtype_str=DTYPE,
        skip_profiling=SKIP_PROFILING,
        test_profiling_csv=TEST_PROFILING_CSV,
    )


if __name__ == "__main__":
    main()

