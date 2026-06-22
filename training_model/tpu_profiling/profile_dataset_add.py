"""
Profile elementwise add operation across a generated dataset of shapes.

This script generates a diverse set of tensor shapes using the dataset_generation
module and profiles the elementwise add operation on TPU for each shape.
"""

import argparse
import os
from datetime import datetime

import jax.numpy as jnp

import profiling_manager as pm
import jax_kernel_functions as jkf
from dataset_generation import generate_shapes_simple_2d


def create_profiling_configuration(
    output_dir: str,
    n_shapes: int,
    max_numel: int,
    seed: int,
    comment: str = "",
) -> dict:
    """Create the profiling manager configuration dictionary."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    storage_file = os.path.join(output_dir, f"add_dataset_{timestamp}.csv")
    metadata_file = os.path.join(output_dir, f"add_metadata_{timestamp}.csv")
    
    return {
        "storage_file": storage_file,
        "storage_metadata_file": metadata_file,
        "append_to_metadata_file": False,
        "hardware_config": "TPUv4",
        "operator_name": "add",
        "common_operator_dimensions": None,
        "data_precision": "FP16",
        "profiler_iterations": 5,
        "random_seed": seed,
        "repo_version": "v0.0.1",
        "comment": f"elementwise_add_dataset n={n_shapes} max_numel={max_numel} seed={seed} {comment}",
    }


def profile_elementwise_add_dataset(
    output_dir: str = "./add_profiling_results",
    n_shapes: int = 2000,
    max_numel: int = 16 * 1024 * 1024,
    seed: int = 42,
    dim_probs: tuple = (0.3, 0.7),
    boundary_frac: float = 0.30,
    perturb_range: int = 32,
    dtype=jnp.float16,
    dry_run: bool = False,
    comment: str = "",
):
    """
    Generate shapes and profile elementwise add operation.
    
    Args:
        output_dir: Directory to store profiling results.
        n_shapes: Number of shapes to generate and profile.
        max_numel: Maximum number of elements per tensor.
        seed: Random seed for shape generation.
        dim_probs: Probability distribution for 1D, 2D shapes (2-tuple).
        boundary_frac: (Deprecated, kept for compatibility).
        perturb_range: (Deprecated, kept for compatibility).
        dtype: JAX dtype for the tensors.
        dry_run: If True, only generate shapes without profiling.
        comment: Optional comment for metadata.
    """
    # Generate shapes using the new simple 2D generation
    print(f"Generating {n_shapes} shapes with max_numel={max_numel}, seed={seed}...")
    shapes = generate_shapes_simple_2d(
        n_shapes=n_shapes,
        max_numel=max_numel,
        dim_probs=dim_probs,
        seed=seed,
        ensure_unique=True,
    )
    print(f"Generated {len(shapes)} unique shapes")
    
    # Print some statistics
    numels = [d0 * d1 * d2 for d0, d1, d2 in shapes]
    print(f"  Min numel: {min(numels):,}")
    print(f"  Max numel: {max(numels):,}")
    print(f"  Median numel: {sorted(numels)[len(numels)//2]:,}")
    
    # Count dimensionalities
    n_1d = sum(1 for s in shapes if s[1] == 1 and s[2] == 1)
    n_2d = sum(1 for s in shapes if s[2] == 1 and s[1] != 1)
    n_3d = sum(1 for s in shapes if s[2] != 1)
    print(f"  1D shapes: {n_1d}, 2D shapes: {n_2d}, 3D shapes: {n_3d}")
    
    if dry_run:
        print("\nDry run mode - skipping profiling")
        print("First 10 shapes:", shapes[:10])
        return shapes
    
    # Create output directory (only when not dry run)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create profiling configuration
    pm_configuration = create_profiling_configuration(
        output_dir=output_dir,
        n_shapes=n_shapes,
        max_numel=max_numel,
        seed=seed,
        comment=comment,
    )
    
    print(f"\nResults will be saved to: {pm_configuration['storage_file']}")
    print(f"Metadata will be saved to: {pm_configuration['storage_metadata_file']}")
    
    # Create profiling manager
    manager = pm.ProfilingManagerSimpleElementwise(
        "add_dataset", 
        output_dir, 
        pm_configuration
    )
    
    # Add profilers for each shape
    print(f"\nAdding {len(shapes)} profilers...")
    for i, (d0, d1, d2) in enumerate(shapes):
        # Determine effective shape (remove trailing 1s for cleaner naming)
        if d2 == 1 and d1 == 1:
            shape = (d0,)
        elif d2 == 1:
            shape = (d0, d1)
        else:
            shape = (d0, d1, d2)
        
        # Create kernel wrapper for elementwise add (both inputs same shape)
        kernel_wrapper = jkf.KernelWarpper(
            "add",
            [(shape, dtype), (shape, dtype)]
        )
        
        # Create profiler name
        shape_str = "x".join(str(d) for d in shape)
        profiler_name = f"add_{shape_str}_{i:05d}"
        
        manager.add_profiler(profiler_name, kernel_wrapper)
        
        if (i + 1) % 100 == 0:
            print(f"  Added {i + 1}/{len(shapes)} profilers")
    
    print(f"Added {len(shapes)} profilers")
    

    parse_result_only = False
    # Run profiling
    if not parse_result_only:
        print("\nStarting profiling...")
        manager.profile_and_post_process_all_profilers()
    else:
        print("\nParsing results only...")
        manager.post_process_all_profilers()

    
    # Write results
    print("\nWriting results...")
    manager.write_results()
    
    print(f"\nProfiling complete!")
    print(f"  Results: {pm_configuration['storage_file']}")
    print(f"  Metadata: {pm_configuration['storage_metadata_file']}")
    
    return shapes


def main():
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    default_output_dir = f"./add_dataset/add_profiling_results_{timestamp}"
    default_n_shapes = 2000
    default_max_numel = 16*1024*1024
    default_seed = 0
    default_dim_probs = (0.3, 0.7)
    default_boundary_frac = 0.30
    default_perturb_range = 32
    default_dtype = "float16"
    default_dry_run = False
    default_comment = ""
    
    parser = argparse.ArgumentParser(
        description="Profile elementwise add operation across a dataset of shapes"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=default_output_dir,
        help=f"Directory to store profiling results (default: {default_output_dir})"
    )
    parser.add_argument(
        "--n-shapes", "-n",
        type=int,
        default=default_n_shapes,
        help=f"Number of shapes to generate and profile (default: {default_n_shapes})"
    )
    parser.add_argument(
        "--max-numel", "-m",
        type=int,
        default=default_max_numel,
        help=f"Maximum number of elements per tensor (default: {default_max_numel})"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=default_seed,
        help=f"Random seed for shape generation (default: {default_seed})"
    )
    parser.add_argument(
        "--dim-probs",
        type=float,
        nargs=2,
        default=list(default_dim_probs),
        help=f"Probability distribution for 1D, 2D shapes (default: {default_dim_probs})"
    )
    parser.add_argument(
        "--boundary-frac",
        type=float,
        default=default_boundary_frac,
        help=f"(Deprecated) Fraction of shapes focused on boundary conditions (default: {default_boundary_frac})"
    )
    parser.add_argument(
        "--perturb-range",
        type=int,
        default=default_perturb_range,
        help=f"(Deprecated) Perturbation range for boundary-focused shapes (default: {default_perturb_range})"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32", "bfloat16"],
        default=default_dtype,
        help=f"Data type for tensors (default: {default_dtype})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=default_dry_run,
        help="Only generate shapes without profiling"
    )
    parser.add_argument(
        "--comment", "-c",
        type=str,
        default=default_comment,
        help="Optional comment for metadata"
    )
    
    args = parser.parse_args()
    
    # Map dtype string to jax dtype
    dtype_map = {
        "float16": jnp.float16,
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    profile_elementwise_add_dataset(
        output_dir=args.output_dir,
        n_shapes=args.n_shapes,
        max_numel=args.max_numel,
        seed=args.seed,
        dim_probs=tuple(args.dim_probs),
        boundary_frac=args.boundary_frac,
        perturb_range=args.perturb_range,
        dtype=dtype,
        dry_run=args.dry_run,
        comment=args.comment,
    )


if __name__ == "__main__":
    main()

