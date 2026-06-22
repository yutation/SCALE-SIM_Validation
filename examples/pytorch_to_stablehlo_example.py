"""
Example: Converting PyTorch models to StableHLO

PyTorch can be converted to StableHLO via:
1. PyTorch -> ONNX -> StableHLO (using torch-mlir or onnx-mlir)
2. PyTorch -> TorchScript -> StableHLO (using torch-mlir)
3. PyTorch -> Torch-MLIR -> StableHLO (direct conversion)

This example shows the torch-mlir approach which is the most direct.
"""

import torch
import torch.nn as nn

# Note: torch-mlir needs to be installed separately
# pip install torch-mlir -f https://github.com/llvm/torch-mlir/releases
try:
    from torch_mlir import torchscript
except ImportError:
    print("Warning: torch-mlir not installed. Install with:")
    print("pip install torch-mlir -f https://github.com/llvm/torch-mlir/releases")
    torchscript = None


class SimpleGEMMMul(nn.Module):
    """
    Simple model with GEMM and element-wise multiplication
    """
    def __init__(self):
        super(SimpleGEMMMul, self).__init__()
        # No learnable parameters for this simple example
        
    def forward(self, x, w, y):
        """
        Args:
            x: Input tensor [128, 256]
            w: Weight tensor [256, 512]
            y: Element-wise multiplier [128, 512]
        
        Returns:
            Output tensor [128, 512]
        """
        # GEMM: Matrix multiplication
        z = torch.matmul(x, w)  # [128, 256] @ [256, 512] -> [128, 512]
        
        # Element-wise multiplication
        output = z * y  # [128, 512] * [128, 512] -> [128, 512]
        
        return output


def main():
    print("=" * 60)
    print("PyTorch to StableHLO Conversion Example")
    print("=" * 60)
    print()
    
    # Create the model
    model = SimpleGEMMMul()
    model.eval()
    
    # Create example inputs with bfloat16 dtype
    x = torch.ones(128, 256, dtype=torch.bfloat16)
    w = torch.ones(256, 512, dtype=torch.bfloat16)
    y = torch.ones(128, 512, dtype=torch.bfloat16)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shapes: x={x.shape}, w={w.shape}, y={y.shape}")
    print(f"Data type: {x.dtype}")
    print()
    
    # Method 1: PyTorch -> TorchScript -> StableHLO (via torch-mlir)
    if torchscript is not None:
        print("=" * 60)
        print("Method 1: Using torch-mlir (TorchScript path)")
        print("=" * 60)
        
        try:
            # Trace the model
            traced_model = torch.jit.trace(model, (x, w, y))
            
            # Convert to torch-mlir
            mlir_module = torchscript.compile(
                traced_model,
                (x, w, y),
                output_type=torchscript.OutputType.STABLEHLO
            )
            
            # Save to file
            output_file = "pytorch_simple_gemm_mul.mlir"
            with open(output_file, "w") as f:
                f.write(str(mlir_module))
            
            print(f"✓ Exported StableHLO to: {output_file}")
            print()
            
            # Print a preview
            print("StableHLO IR Preview (first 50 lines):")
            print("-" * 60)
            lines = str(mlir_module).split('\n')
            for line in lines[:50]:
                print(line)
            
            if len(lines) > 50:
                print(f"... ({len(lines) - 50} more lines)")
                
        except Exception as e:
            print(f"✗ Error during conversion: {e}")
            print("This might be due to torch-mlir version or compatibility issues.")
    
    else:
        print("torch-mlir is not installed. Showing alternative approaches:")
        print()
    
    # Method 2: PyTorch -> ONNX (as intermediate step)
    print()
    print("=" * 60)
    print("Method 2: PyTorch -> ONNX (intermediate representation)")
    print("=" * 60)
    
    try:
        # Export to ONNX
        onnx_file = "pytorch_simple_gemm_mul.onnx"
        torch.onnx.export(
            model,
            (x, w, y),
            onnx_file,
            input_names=['x', 'w', 'y'],
            output_names=['output'],
            opset_version=17,
            do_constant_folding=True,
        )
        
        print(f"✓ Exported to ONNX: {onnx_file}")
        print("  To convert ONNX to StableHLO, you can use:")
        print("  - onnx-mlir: https://github.com/onnx/onnx-mlir")
        print("  - Or iree-compiler: https://github.com/openxla/iree")
        print()
        
    except Exception as e:
        print(f"✗ Error during ONNX export: {e}")
    
    # Method 3: Show TorchScript IR (intermediate representation)
    print()
    print("=" * 60)
    print("Method 3: TorchScript IR (intermediate representation)")
    print("=" * 60)
    
    try:
        traced_model = torch.jit.trace(model, (x, w, y))
        
        # Save TorchScript
        torchscript_file = "pytorch_simple_gemm_mul.pt"
        torch.jit.save(traced_model, torchscript_file)
        print(f"✓ Saved TorchScript to: {torchscript_file}")
        
        # Print TorchScript graph
        print("\nTorchScript Graph:")
        print("-" * 60)
        print(traced_model.graph)
        
    except Exception as e:
        print(f"✗ Error during TorchScript tracing: {e}")
    
    print()
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    print("✓ JAX: Native StableHLO export via jax.export")
    print("✓ PyTorch: Requires torch-mlir or ONNX as intermediate step")
    print("✓ Both frameworks support the same GEMM + element-wise mul pattern")


if __name__ == "__main__":
    main()





