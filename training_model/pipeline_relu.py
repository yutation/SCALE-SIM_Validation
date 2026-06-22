"""
Full pipeline: Train model -> Profile new shapes -> Compare predictions vs actual.

This script orchestrates the complete workflow:
1. Train a model on existing profiling dataset
2. Generate new test shapes
3. Profile those shapes on TPU
4. Predict latencies using the trained model
5. Compare predictions vs actual measurements
6. Generate comparison report and CSV
"""

import argparse
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
from tpu_profiling import dataset_generation as dg
from model import model_manager as mm


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
    Run the complete train-profile-compare pipeline.
    
    Args:
        training_csv: Path to CSV with training data
        output_dir: Directory for all outputs
        n_test_shapes: Number of test shapes to profile
        max_numel: Maximum number of elements per test tensor
        seed: Random seed
        op_name: Operation name (e.g., "add")
        dtype_str: Data type ("float16", "float32", "bfloat16")
        skip_profiling: If True, use existing test_profiling_csv instead of profiling
        test_profiling_csv: Path to existing test profiling results (if skip_profiling=True)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"LATENCY PREDICTION PIPELINE - {op_name.upper()}")
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
    print("STEP 1: TRAINING MODEL")
    print("="*70)
    
    model_path = output_dir / f"{op_name}_model_{timestamp}.pkl"
    model = mm.train_and_save(
        training_csv,
        str(model_path),
        op_name=op_name,
        seed=seed
    )
    
    # ========================================
    # STEP 2: Generate test shapes
    # ========================================
    print("\n" + "="*70)
    print("STEP 2: GENERATING TEST SHAPES")
    print("="*70)
    
    # Use different seed for test shapes to avoid overlap with training
    test_seed = seed + 1000
    
    print(f"Generating {n_test_shapes} test shapes (seed={test_seed})...")
    test_shapes = dg.generate_shapes_simple_2d(
        n_shapes=n_test_shapes,
        max_numel=max_numel,
        dim_probs=(0.3, 0.7),  # 30% 1D, 70% 2D
        seed=test_seed,
        ensure_unique=True,
    )
    
    print(f"Generated {len(test_shapes)} unique test shapes")
    
    # Statistics
    numels = [d0 * d1 * d2 for d0, d1, d2 in test_shapes]
    print(f"  Min numel:    {min(numels):,}")
    print(f"  Max numel:    {max(numels):,}")
    print(f"  Median numel: {sorted(numels)[len(numels)//2]:,}")
    
    # Count dimensionalities
    n_1d = sum(1 for s in test_shapes if s[1] == 1 and s[2] == 1)
    n_2d = sum(1 for s in test_shapes if s[1] > 1)
    print(f"  1D shapes: {n_1d}, 2D shapes: {n_2d}")
    
    # ========================================
    # STEP 3: Profile test shapes on TPU
    # ========================================
    if not skip_profiling:
        print("\n" + "="*70)
        print("STEP 3: PROFILING TEST SHAPES ON TPU")
        print("="*70)
        
        test_profiling_dir = output_dir / f"test_profiling_{timestamp}"
        
        # Map dtype
        dtype_map = {
            "float16": jnp.float16,
            "float32": jnp.float32,
            "bfloat16": jnp.bfloat16,
        }
        dtype = dtype_map[dtype_str]
        
        # Run profiling using the existing profile_dataset module
        print(f"Profiling {len(test_shapes)} shapes on TPU...")
        
        # We need to manually create the profiling using the lower-level functions
        # since profile_elementwise_add_dataset generates its own shapes
        import tpu_profiling.profiling_manager as pm
        import tpu_profiling.jax_kernel_functions as jkf
        
        # Create profiling configuration
        test_profiling_csv_path = output_dir / f"test_profiling_{timestamp}.csv"
        metadata_file = output_dir / f"test_metadata_{timestamp}.csv"
        
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
            "comment": f"test_shapes n={n_test_shapes} max_numel={max_numel} seed={test_seed}",
        }
        
        # Create profiling manager
        manager = pm.ProfilingManagerSimpleElementwise(
            f"{op_name}_test",
            str(test_profiling_dir),
            pm_configuration
        )
        
        # Add profilers for each test shape
        print(f"Adding {len(test_shapes)} profilers...")
        for i, (d0, d1, d2) in enumerate(test_shapes):
            # Determine effective shape
            if d2 == 1 and d1 == 1:
                shape = (d0,)
            elif d2 == 1:
                shape = (d0, d1)
            else:
                shape = (d0, d1, d2)
            
            # Unary operations (like relu) only need one input
            if op_name in ["relu"]:
                kernel_wrapper = jkf.KernelWarpper(
                    op_name,
                    [(shape, dtype)]
                )
            else:
                # Binary operations (like add, mul) need two inputs
                kernel_wrapper = jkf.KernelWarpper(
                    op_name,
                    [(shape, dtype), (shape, dtype)]
                )
            
            shape_str = "x".join(str(d) for d in shape)
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
    
    print(f"Predicting latencies for {len(test_shapes)} shapes...")
    predictions = mm.predict_latency(model, test_shapes)
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
    
    for i, (d0, d1, d2) in enumerate(test_shapes):
        # Find matching row in actual results
        mask = (
            (actual_df["input_dim_0"] == d0) &
            (actual_df["input_dim_1"] == d1) &
            (actual_df["input_dim_2"] == d2)
        )
        
        if mask.sum() == 0:
            print(f"Warning: No match found for shape ({d0}, {d1}, {d2})")
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
            "dim_2": d2,
            "size": d0 * d1 * d2,
            "actual_latency": actual_latency,
            "predicted_latency": predicted_latency,
            "absolute_error": absolute_error,
            "relative_error_pct": relative_error,
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison CSV
    comparison_csv = output_dir / f"comparison_{timestamp}.csv"
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
    
    print(f"\nPrediction Performance:")
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
    summary_path = output_dir / f"summary_{timestamp}.txt"
    with open(summary_path, "w") as f:
        f.write("="*70 + "\n")
        f.write(f"LATENCY PREDICTION PIPELINE SUMMARY - {op_name.upper()}\n")
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
    parser = argparse.ArgumentParser(
        description="Full pipeline: Train model -> Profile shapes -> Compare predictions"
    )
    parser.add_argument(
        "training_csv",
        type=str,
        help="Path to training dataset CSV"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./pipeline_results",
        help="Output directory for all results (default: ./pipeline_results)"
    )
    parser.add_argument(
        "--n-test-shapes", "-n",
        type=int,
        default=100,
        help="Number of test shapes to profile (default: 100)"
    )
    parser.add_argument(
        "--max-numel", "-m",
        type=int,
        default=1024*1024,
        help="Maximum number of elements per test tensor (default: 1048576)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--op-name",
        type=str,
        default="relu",
        help="Operation name (default: relu)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32", "bfloat16"],
        default="float16",
        help="Data type for profiling (default: float16)"
    )
    parser.add_argument(
        "--skip-profiling",
        action="store_true",
        help="Skip profiling step and use existing test profiling CSV"
    )
    parser.add_argument(
        "--test-profiling-csv",
        type=str,
        help="Path to existing test profiling CSV (required if --skip-profiling)"
    )
    
    args = parser.parse_args()
    
    if args.skip_profiling and not args.test_profiling_csv:
        parser.error("--test-profiling-csv is required when --skip-profiling is used")
    
    run_pipeline(
        training_csv=args.training_csv,
        output_dir=args.output_dir,
        n_test_shapes=args.n_test_shapes,
        max_numel=args.max_numel,
        seed=args.seed,
        op_name=args.op_name,
        dtype_str=args.dtype,
        skip_profiling=args.skip_profiling,
        test_profiling_csv=args.test_profiling_csv,
    )


if __name__ == "__main__":
    main()

