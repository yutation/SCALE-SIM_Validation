# Framework to StableHLO Conversion Examples

This directory contains examples showing how to convert models from popular ML frameworks to StableHLO format.

## Files

- `jax_to_stablehlo_example.py` - JAX to StableHLO conversion
- `pytorch_to_stablehlo_example.py` - PyTorch to StableHLO conversion

## Requirements

### For JAX Example

```bash
pip install jax jaxlib
```

For GPU support:
```bash
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### For PyTorch Example

```bash
pip install torch
```

Optional (for direct StableHLO conversion):
```bash
pip install torch-mlir -f https://github.com/llvm/torch-mlir/releases
```

## Usage

### JAX to StableHLO

```bash
cd examples
python jax_to_stablehlo_example.py
```

**Output files:**
- `jax_simple_gemm_mul.mlir` - StableHLO MLIR file (via export API)
- `jax_simple_gemm_mul_lowered.mlir` - StableHLO MLIR file (via lower API)

### PyTorch to StableHLO

```bash
cd examples
python pytorch_to_stablehlo_example.py
```

**Output files:**
- `pytorch_simple_gemm_mul.mlir` - StableHLO MLIR file (if torch-mlir is installed)
- `pytorch_simple_gemm_mul.onnx` - ONNX intermediate format
- `pytorch_simple_gemm_mul.pt` - TorchScript format

## Conversion Paths

### JAX → StableHLO (Direct, Recommended)

JAX has native StableHLO support:

1. **Using `jax.export` API (Recommended)**
   ```python
   from jax.experimental import export
   exported = export.export(jax.jit(func))(inputs)
   stablehlo = exported.mlir_module()
   ```

2. **Using `lower()` API**
   ```python
   lowered = jax.jit(func).lower(inputs)
   stablehlo = lowered.compiler_ir(dialect="stablehlo")
   ```

### PyTorch → StableHLO (Multiple Paths)

PyTorch requires intermediate conversions:

1. **PyTorch → torch-mlir → StableHLO (Most Direct)**
   - Requires: `torch-mlir` package
   - Best for direct conversion
   - May have compatibility issues with some PyTorch ops

2. **PyTorch → ONNX → StableHLO**
   - Requires: ONNX export + onnx-mlir or iree-compiler
   - More mature toolchain
   - Better op coverage

3. **PyTorch → TorchScript → torch-mlir → StableHLO**
   - Hybrid approach
   - Good for complex models

## Model Structure

Both examples implement the same computation:

```
Input: x [128, 256], w [256, 512], y [128, 512]

1. GEMM: z = matmul(x, w)  # [128, 512]
2. Element-wise multiply: output = z * y  # [128, 512]

Output: [128, 512]
```

Data type: **bfloat16** (bf16)

## Viewing StableHLO Output

To view the generated StableHLO MLIR files:

```bash
# View the file
cat jax_simple_gemm_mul.mlir

# Or use mlir-opt if installed (from LLVM/MLIR)
mlir-opt jax_simple_gemm_mul.mlir
```

## Common Issues

### JAX
- **Issue**: `jax.export` not found
- **Solution**: Upgrade JAX to version 0.4.1 or later

### PyTorch
- **Issue**: `torch-mlir` not installed or not compatible
- **Solution**: Use ONNX intermediate format or check torch-mlir compatibility

- **Issue**: BFloat16 not supported on CPU
- **Solution**: Use float32 instead, or enable BFloat16 on compatible hardware

## Next Steps

After generating StableHLO files, you can:

1. **Run in SCALE-Sim**: Use the generated `.mlir` files as input topologies
2. **Optimize**: Use MLIR passes to optimize the StableHLO
3. **Convert to other formats**: Use StableHLO as a portable IR

## References

- [JAX Export Documentation](https://jax.readthedocs.io/en/latest/export.html)
- [torch-mlir Project](https://github.com/llvm/torch-mlir)
- [StableHLO Specification](https://github.com/openxla/stablehlo)
- [ONNX-MLIR](https://github.com/onnx/onnx-mlir)





