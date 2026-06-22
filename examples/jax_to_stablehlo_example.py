"""
Example: Converting JAX models to StableHLO

JAX has native support for exporting to StableHLO via jax.jit and the export API.
"""

import jax
import jax.numpy as jnp
from jax import export


def simple_gemm_mul(x, w, y):
    """
    Simple model with GEMM and element-wise multiplication
    
    Args:
        x: Input tensor [128, 256]
        w: Weight tensor [256, 512]
        y: Element-wise multiplier [128, 512]
    
    Returns:
        Output tensor [128, 512]
    """
    # GEMM: Matrix multiplication
    z = jnp.matmul(x, w)  # [128, 256] @ [256, 512] -> [128, 512]
    
    # Element-wise multiplication
    output = z * y  # [128, 512] * [128, 512] -> [128, 512]
    
    return output


def main():
    # Create example inputs with bfloat16 dtype
    x = jnp.ones((128, 256), dtype=jnp.bfloat16)
    w = jnp.ones((256, 512), dtype=jnp.bfloat16)
    y = jnp.ones((128, 512), dtype=jnp.bfloat16)
    
    # Method 1: Using jax.export (Recommended for JAX 0.4.1+)
    print("=" * 60)
    print("Method 1: Using jax.export API")
    print("=" * 60)
    
    # Export the function to StableHLO
    exported = export.export(jax.jit(simple_gemm_mul))(x, w, y)
    
    # Get the StableHLO MLIR module
    stablehlo_mlir = exported.mlir_module()
    
    # Save to file
    output_file = "jax_simple_gemm_mul.mlir"
    with open(output_file, "w") as f:
        f.write(str(stablehlo_mlir))
    
    print(f"✓ Exported StableHLO to: {output_file}")
    print(f"✓ Function signature: {exported.fun_name}")
    print(f"✓ Input shapes: {exported.in_avals}")
    print(f"✓ Output shape: {exported.out_avals}")
    print()
    
    # Method 2: Using lower() API (Alternative approach)
    print("=" * 60)
    print("Method 2: Using jax.jit().lower() API")
    print("=" * 60)
    
    # JIT compile and lower to StableHLO
    lowered = jax.jit(simple_gemm_mul).lower(x, w, y)
    
    # Get the StableHLO HLO module
    stablehlo_text = lowered.compiler_ir(dialect="stablehlo")
    
    # Save to file
    output_file_2 = "jax_simple_gemm_mul_lowered.mlir"
    with open(output_file_2, "w") as f:
        f.write(str(stablehlo_text))
    
    print(f"✓ Exported StableHLO to: {output_file_2}")
    print()
    
    # Print a preview of the StableHLO IR
    print("=" * 60)
    print("StableHLO IR Preview (first 50 lines):")
    print("=" * 60)
    lines = str(stablehlo_text).split('\n')
    for line in lines[:50]:
        print(line)
    
    if len(lines) > 50:
        print(f"... ({len(lines) - 50} more lines)")


if __name__ == "__main__":
    main()





