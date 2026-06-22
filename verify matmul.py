#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Unified Model Verification Script

This script runs verification for all kernel types in a single execution,
saving all results to one directory and one merged CSV file.
"""

import os
import sys
sys.path.append('.')
import operation_classification as oc
from model_verification import ModelVerification

def run_unified_verification():
    """Run verification for all kernel types in a single unified test."""
    
    print("=" * 60)
    print("UNIFIED MODEL VERIFICATION")
    print("=" * 60)
    
    # Create single verification directory
    verification_dir = "./unified_verification_results"
    os.makedirs(verification_dir, exist_ok=True)
    
    # Initialize single model verification instance
    model_verifier = ModelVerification(profile_dir=verification_dir)
    
    print("Adding all kernel configurations...")
    
    # 1. ADD operations (1D and 2D)
    print("  - Adding ADD operations...")
    add_1d_shapes = [(1024,), (2048,), (4096,), (8192,), (16384,), (32768,)]
    add_2d_shapes = [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]
    
    for shape in add_1d_shapes + add_2d_shapes:
        model_verifier.add_verification_config(
            operation_type=oc.OperationType.ELEMENTWISE,
            operation=oc.OperationElementwise.ADD,
            shapes=[shape],
            operation_params={}
        )
    
    # 2. RELU activation operations
    print("  - Adding RELU operations...")
    relu_shapes = [
        (512,), (1024,), (2048,), (4096,), (8192,),
        (64, 64), (128, 128), (256, 256), (512, 512)
    ]
    
    for shape in relu_shapes:
        model_verifier.add_verification_config(
            operation_type=oc.OperationType.ACTIVATION,
            operation=oc.OperationActivation.RELU,
            shapes=[shape],
            operation_params={}
        )
    
    # 3. Matrix multiplication operations (more realistic dimensions)
    print("  - Adding MATMUL operations...")
    matmul_configs = [
        (96, 144, 192), (192, 288, 384), (384, 576, 768), (768, 1152, 1536),
        (224, 299, 196), (299, 384, 512), (384, 768, 1024),
        (1024, 768, 512), (768, 1536, 1024), (512, 2048, 768)
    ]
    
    for M, N, K in matmul_configs:
        model_verifier.add_verification_config(
            operation_type=oc.OperationType.MATMUL,
            operation=oc.OperationMatmul.LINEAR,
            shapes=[(M, K), (K, N)],
            operation_params={'M': M, 'N': N, 'K': K}
        )
    
    # 4. Layer normalization operations (3D shapes: N, L, H)
    print("  - Adding Layer Normalization operations...")
    layernorm_shapes = [
        (2, 128, 512), (4, 256, 1024), (6, 384, 1536), (8, 512, 2048),
        (2, 512, 1024), (4, 768, 2048), (6, 1024, 3072)
    ]
    
    for shape in layernorm_shapes:
        model_verifier.add_verification_config(
            operation_type=oc.OperationType.NORMALIZATION,
            operation=oc.OperationNormalization.LAYER_NORM,
            shapes=[shape],
            operation_params={'axis': (2,)}  # Normalize over hidden dimension
        )
    
    total_configs = len(model_verifier.prediction_manager.config_list)
    print(f"\nTotal configurations: {total_configs}")
    print("  - ADD: 11 configurations")
    print("  - RELU: 9 configurations") 
    print("  - MATMUL: 9 configurations")
    print("  - LAYER_NORM: 7 configurations")
    print("=" * 60)
    
    # Run unified verification
    print("\nRunning unified verification (this will take a while)...")
    try:
        results = model_verifier.verify()
        
        print(f"\n🎉 Unified verification completed successfully!")
        print(f"📁 All results saved to: {verification_dir}/")
        print(f"📊 Main results file: {verification_dir}/merged_verification_results.csv")
        print(f"📈 Profiling data: {verification_dir}/filtered_events_avg_fusion.csv")
        
        # Enhanced analysis by operation type
        print("\n" + "=" * 60)
        print("DETAILED ANALYSIS BY OPERATION TYPE")
        print("=" * 60)
        
        # Group results by operation type
        add_results = results[results['Operation_Type'] == 'elementwise']
        relu_results = results[results['Operation_Type'] == 'activation']  
        matmul_results = results[results['Operation_Type'] == 'matmul']
        layernorm_results = results[results['Operation_Type'] == 'normalization']
        
        operation_groups = [
            ("ADD", add_results),
            ("RELU", relu_results), 
            ("MATMUL", matmul_results),
            ("LAYER_NORM", layernorm_results)
        ]
        
        for name, group_results in operation_groups:
            if len(group_results) > 0:
                mape = group_results['Error_Percentage'].abs().mean()
                rmse = ((group_results['Predicted_Latency_us'] - group_results['Actual_Duration_us']) ** 2).mean() ** 0.5
                min_error = group_results['Error_Percentage'].abs().min()
                max_error = group_results['Error_Percentage'].abs().max()
                
                print(f"{name:12} | Tests: {len(group_results):2d} | MAPE: {mape:6.2f}% | RMSE: {rmse:7.2f} μs | Range: {min_error:.1f}%-{max_error:.1f}%")
        
        # Overall statistics
        print("\n" + "=" * 60)
        print("OVERALL STATISTICS")
        print("=" * 60)
        overall_mape = results['Error_Percentage'].abs().mean()
        overall_rmse = ((results['Predicted_Latency_us'] - results['Actual_Duration_us']) ** 2).mean() ** 0.5
        
        print(f"Total test cases: {len(results)}")
        print(f"Overall MAPE: {overall_mape:.2f}%")
        print(f"Overall RMSE: {overall_rmse:.2f} μs")
        
        # Best and worst predictions
        best_idx = results['Error_Percentage'].abs().idxmin()
        worst_idx = results['Error_Percentage'].abs().idxmax()
        
        print(f"\n🏆 Best prediction:")
        print(f"   {results.loc[best_idx, 'Operation_Type']} - {results.loc[best_idx, 'Operation']}")
        print(f"   Shape: {results.loc[best_idx, 'Input_Shapes']}")
        print(f"   Error: {results.loc[best_idx, 'Error_Percentage']:.2f}%")
        
        print(f"\n⚠️  Worst prediction:")
        print(f"   {results.loc[worst_idx, 'Operation_Type']} - {results.loc[worst_idx, 'Operation']}")
        print(f"   Shape: {results.loc[worst_idx, 'Input_Shapes']}")
        print(f"   Error: {results.loc[worst_idx, 'Error_Percentage']:.2f}%")
        
        return results
        
    except Exception as e:
        import traceback
        print(f"❌ Error during unified verification: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return None



def matmul_unified_verification():
    """Run unified verification for matrix multiplication operations."""
    
    print("=" * 60)
    print("MATMUL UNIFIED VERIFICATION")
    print("=" * 60)
    
    verification_dir = "./matmul_unified_verification_results"
    os.makedirs(verification_dir, exist_ok=True)
    
    model_verifier = ModelVerification(profile_dir=verification_dir)
    
    print("Adding matrix multiplication configurations...")


    for M in range(31, 128, 33):
        for N in range(32, 128, 31):
            for K in range(33, 128, 32):
                model_verifier.add_verification_config(
                    operation_type=oc.OperationType.MATMUL,
                    operation=oc.OperationMatmul.LINEAR,
                    shapes=[(M, K), (K, N)],
                    operation_params={'M': M, 'N': N, 'K': K}
                )
    
    for M in range(129, 1024, 201):
        for N in range(131, 1024, 202):
            for K in range(133, 1024, 203):
                model_verifier.add_verification_config(
                    operation_type=oc.OperationType.MATMUL,
                    operation=oc.OperationMatmul.LINEAR,
                    shapes=[(M, K), (K, N)],
                    operation_params={'M': M, 'N': N, 'K': K}
                )
    return model_verifier.verify()

def large_matmul_unified_verification():
    """Run unified verification for matrix multiplication operations."""
    
    print("=" * 60)
    print("MATMUL UNIFIED VERIFICATION")
    print("=" * 60)
    
    verification_dir = "./large_matmul_unified_verification_results2"
    os.makedirs(verification_dir, exist_ok=True)
    
    model_verifier = ModelVerification(profile_dir=verification_dir)
    
    print("Adding matrix multiplication configurations...")

    
    for M in range(1024, 4096, 765):
        for N in range(1024, 4096, 977):
            for K in range(1024, 4096, 651):
                model_verifier.add_verification_config(
                    operation_type=oc.OperationType.MATMUL,
                    operation=oc.OperationMatmul.LINEAR,
                    shapes=[(M, K), (K, N)],
                    operation_params={'M': M, 'N': N, 'K': K}
                )
    return model_verifier.verify()


def pooling_unified_verification():
    """Run unified verification for pooling operations."""
    
    print("=" * 60)
    print("POOLING UNIFIED VERIFICATION")
    print("=" * 60)
    
    verification_dir = "./pooling_unified_verification_results"
    os.makedirs(verification_dir, exist_ok=True)
    
    model_verifier = ModelVerification(profile_dir=verification_dir)
    
    print("Adding pooling configurations...")
    
    # Max pooling configurations with different shapes (non-power-of-2 values)
    max_pooling_shapes = [
        (1, 3, 224, 224), (2, 64, 112, 112), (4, 128, 56, 56),
        (8, 256, 28, 28), (1, 512, 14, 14), (2, 1024, 7, 7),
        (1, 96, 48, 48), (4, 192, 24, 24), (8, 384, 12, 12),
        (1, 768, 6, 6), (2, 144, 36, 36), (4, 288, 18, 18)
    ]
    
    for shape in max_pooling_shapes:
        model_verifier.add_verification_config(
            operation_type=oc.OperationType.POOLING,
            operation=oc.OperationPooling.MAX_POOLING,
            shapes=[shape],
            operation_params={'window_shape': (2, 2), 'strides': (2, 2), 'padding': 'VALID'}
        )
    
    # Average pooling configurations (varied dimensions)
    avg_pooling_shapes = [
        (1, 3, 150, 150), (2, 96, 75, 75), (4, 192, 38, 38),
        (8, 384, 19, 19), (1, 576, 10, 10), (2, 720, 5, 5),
        (4, 432, 12, 12), (8, 864, 6, 6)
    ]
    
    for shape in avg_pooling_shapes:
        model_verifier.add_verification_config(
            operation_type=oc.OperationType.POOLING,
            operation=oc.OperationPooling.AVG_POOLING,
            shapes=[shape],
            operation_params={'window_shape': (2, 2), 'strides': (2, 2), 'padding': 'VALID'}
        )
    
    total_configs = len(model_verifier.prediction_manager.config_list)
    print(f"\nTotal pooling configurations: {total_configs}")
    print(f"  - MAX_POOLING: {len(max_pooling_shapes)} configurations")
    print(f"  - AVG_POOLING: {len(avg_pooling_shapes)} configurations")
    print("=" * 60)
    
    return model_verifier.verify()


def elementwise_unified_verification():
    """Run unified verification for elementwise operations."""
    
    print("=" * 60)
    print("ELEMENTWISE UNIFIED VERIFICATION")
    print("=" * 60)
    
    verification_dir = "./elementwise_unified_verification_results"
    os.makedirs(verification_dir, exist_ok=True)
    
    model_verifier = ModelVerification(profile_dir=verification_dir)
    
    print("Adding elementwise configurations...")
    
    # 1D shapes for elementwise operations (varied sizes)
    shapes_1d = [
        (300,), (789,), (1500,), (2345,), (5670,), (12800,), (25600,), (50000,),
        (768,), (1536,), (3072,), (6144,)
    ]
    
    # 2D shapes for elementwise operations (non-square and irregular dimensions)
    shapes_2d = [
        (96, 128), (144, 192), (240, 320), (384, 512), (576, 768),
        (48, 224), (96, 448), (192, 896), (768, 1024), (432, 576),
        (150, 200), (300, 400), (600, 800)
    ]
    
    # All elementwise operations
    elementwise_ops = [
        oc.OperationElementwise.ADD,
        oc.OperationElementwise.SUBTRACT,
        oc.OperationElementwise.MULTIPLY,
        oc.OperationElementwise.DIVIDE
    ]
    
    for operation in elementwise_ops:
        # Add 1D configurations
        for shape in shapes_1d:
            model_verifier.add_verification_config(
                operation_type=oc.OperationType.ELEMENTWISE,
                operation=operation,
                shapes=[shape],
                operation_params={}
            )
        
        # Add 2D configurations
        for shape in shapes_2d:
            model_verifier.add_verification_config(
                operation_type=oc.OperationType.ELEMENTWISE,
                operation=operation,
                shapes=[shape],
                operation_params={}
            )
    
    total_configs = len(model_verifier.prediction_manager.config_list)
    print(f"\nTotal elementwise configurations: {total_configs}")
    print(f"  - Per operation: {len(shapes_1d) + len(shapes_2d)} configurations")
    print(f"  - Operations: {[op.value for op in elementwise_ops]}")
    print("=" * 60)
    
    return model_verifier.verify()

def elementwise_unified_verification_3d():
    """Run unified verification for elementwise operations."""
    
    print("=" * 60)
    print("ELEMENTWISE UNIFIED VERIFICATION 3D")
    print("=" * 60)
    
    verification_dir = "./elementwise_unified_verification_3d_results"
    os.makedirs(verification_dir, exist_ok=True)
    
    model_verifier = ModelVerification(profile_dir=verification_dir)
    
    # 3D shapes for elementwise operations (non-square and irregular dimensions)
    shapes_3d = [
        (96, 128, 128), (144, 192, 192), (240, 320, 320), (384, 512, 512),
        (576, 768, 768), (48, 224, 224), (96, 448, 448), (192, 896, 896),
        (768, 1024, 1024), (432, 576, 576), (150, 200, 200), (300, 400, 400)
    ]
    print("Adding elementwise configurations...")

    for shape in shapes_3d:
        model_verifier.add_verification_config(
            operation_type=oc.OperationType.ELEMENTWISE,
            operation=oc.OperationElementwise.ADD,
            shapes=[shape],
            operation_params={}
        )
    
    total_configs = len(model_verifier.prediction_manager.config_list)
    print(f"\nTotal elementwise configurations: {total_configs}")
    print(f"  - Per operation: {len(shapes_3d)} configurations")
    print("=" * 60)
    
    return model_verifier.verify()


def normalization_unified_verification():
    """Run unified verification for normalization operations."""
    
    print("=" * 60)
    print("NORMALIZATION UNIFIED VERIFICATION")
    print("=" * 60)
    
    verification_dir = "./unified/normalization_unified_verification_results2"
    os.makedirs(verification_dir, exist_ok=True)
    
    model_verifier = ModelVerification(profile_dir=verification_dir)
    
    print("Adding normalization configurations...")
    
    # Layer normalization shapes (3D: batch, sequence, hidden) - varied dimensions
    layernorm_shapes = [
        (1, 197, 768), (2, 345, 512), (4, 577, 1024), (8, 224, 1536),
        (1, 299, 2048), (2, 486, 768), (4, 672, 1280), (8, 150, 3072),
        (16, 77, 4096), (32, 49, 2560), (12, 196, 1792), (6, 384, 1408)
    ]
    
    for shape in layernorm_shapes:
        model_verifier.add_verification_config(
            operation_type=oc.OperationType.NORMALIZATION,
            operation=oc.OperationNormalization.LAYER_NORM,
            shapes=[shape],
            operation_params={'axis': (2,)}  # Normalize over hidden dimension
        )
    
    # RMS normalization shapes (varied dimensions)
    rmsnorm_shapes = [
        (1, 197, 768), (2, 299, 1280), (4, 577, 1792), (8, 345, 2560),
        (1, 672, 1536), (2, 486, 2304), (4, 224, 3584), (12, 150, 4352)
    ]
    
    for shape in rmsnorm_shapes:
        model_verifier.add_verification_config(
            operation_type=oc.OperationType.NORMALIZATION,
            operation=oc.OperationNormalization.RMS_NORM,
            shapes=[shape],
            operation_params={'axis': (2,)}
        )
    
    # Batch normalization shapes (4D: batch, channels, height, width) - realistic CNN dimensions
    batchnorm_shapes = [
        (1, 96, 224, 224), (2, 144, 112, 112), (4, 288, 56, 56),
        (8, 576, 28, 28), (16, 192, 64, 64), (32, 384, 32, 32),
        (12, 480, 16, 16), (24, 720, 8, 8), (6, 240, 48, 48)
    ]
    
    for shape in batchnorm_shapes:
        model_verifier.add_verification_config(
            operation_type=oc.OperationType.NORMALIZATION,
            operation=oc.OperationNormalization.BATCH_NORM,
            shapes=[shape],
            operation_params={'axis': 1}  # Normalize over channel dimension
        )
    
    total_configs = len(model_verifier.prediction_manager.config_list)
    print(f"\nTotal normalization configurations: {total_configs}")
    print(f"  - LAYER_NORM: {len(layernorm_shapes)} configurations")
    print(f"  - RMS_NORM: {len(rmsnorm_shapes)} configurations")
    print(f"  - BATCH_NORM: {len(batchnorm_shapes)} configurations")
    print("=" * 60)
    
    return model_verifier.verify()


def activation_unified_verification():
    """Run unified verification for activation operations."""
    
    print("=" * 60)
    print("ACTIVATION UNIFIED VERIFICATION")
    print("=" * 60)
    
    verification_dir = "./activation_unified_verification_results"
    os.makedirs(verification_dir, exist_ok=True)
    
    model_verifier = ModelVerification(profile_dir=verification_dir)
    
    print("Adding activation configurations...")
    
    # 1D shapes for activations (varied sizes)
    shapes_1d = [
        (768,), (1536,), (2304,), (3072,), (4608,), (6912,), (9216,),
        (300,), (900,), (1800,), (5400,), (12150,)
    ]
    
    # 2D shapes for activations (realistic neural network dimensions)
    shapes_2d = [
        (96, 144), (192, 288), (384, 576), (768, 1152),
        (48, 196), (96, 392), (192, 784), (384, 1568),
        (224, 224), (299, 299), (150, 600), (300, 1200)
    ]
    
    # All activation operations
    activation_ops = [
        oc.OperationActivation.RELU,
        oc.OperationActivation.SIGMOID,
        oc.OperationActivation.TANH,
        oc.OperationActivation.LEAKY_RELU,
        oc.OperationActivation.ELU,
        oc.OperationActivation.SELU,
        oc.OperationActivation.LINEAR,
        oc.OperationActivation.BINARY
    ]
    
    for operation in activation_ops:
        # Add 1D configurations
        for shape in shapes_1d:
            model_verifier.add_verification_config(
                operation_type=oc.OperationType.ACTIVATION,
                operation=operation,
                shapes=[shape],
                operation_params={}
            )
        
        # Add 2D configurations
        for shape in shapes_2d:
            model_verifier.add_verification_config(
                operation_type=oc.OperationType.ACTIVATION,
                operation=operation,
                shapes=[shape],
                operation_params={}
            )
    
    total_configs = len(model_verifier.prediction_manager.config_list)
    print(f"\nTotal activation configurations: {total_configs}")
    print(f"  - Per operation: {len(shapes_1d) + len(shapes_2d)} configurations")
    print(f"  - Operations: {[op.value for op in activation_ops]}")
    print("=" * 60)
    
    return model_verifier.verify()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--matmul":
            matmul_unified_verification()
        elif arg == "--large-matmul":
            large_matmul_unified_verification()
        elif arg == "--pooling":
            pooling_unified_verification()
        elif arg == "--elementwise":
            elementwise_unified_verification()
        elif arg == "--normalization":
            normalization_unified_verification()
        elif arg == "--activation":
            activation_unified_verification()
        elif arg == "--elementwise-3d":
            elementwise_unified_verification_3d()
        elif arg == "--help":
            print("Available verification options:")
            print("  --matmul          Run matrix multiplication verification")
            print("  --large-matmul    Run large matrix multiplication verification")
            print("  --pooling         Run pooling operations verification")
            print("  --elementwise     Run elementwise operations verification")
            print("  --normalization   Run normalization operations verification")
            print("  --activation      Run activation operations verification")
            print("  (no args)         Run unified verification for all operation types")
        else:
            print(f"Unknown argument: {arg}")
            print("Use --help to see available options")
    else:
        run_unified_verification()
