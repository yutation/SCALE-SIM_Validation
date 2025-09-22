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
    
    # 3. Matrix multiplication operations
    print("  - Adding MATMUL operations...")
    matmul_configs = [
        (64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512),
        (64, 128, 256), (128, 256, 512), (256, 512, 1024),
        (1024, 512, 256), (512, 1024, 512)
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

def run_quick_unified_test():
    """Run a smaller unified test for quick verification."""
    
    print("=" * 50)
    print("QUICK UNIFIED VERIFICATION")
    print("=" * 50)
    
    verification_dir = "./quick_unified_results"
    os.makedirs(verification_dir, exist_ok=True)
    
    model_verifier = ModelVerification(profile_dir=verification_dir)
    
    # Add a mix of different operation types
    configs = [
        # ADD operations
        (oc.OperationType.ELEMENTWISE, oc.OperationElementwise.ADD, [(1024,)], {}),
        (oc.OperationType.ELEMENTWISE, oc.OperationElementwise.ADD, [(128, 128)], {}),
        
        # RELU activations
        (oc.OperationType.ACTIVATION, oc.OperationActivation.RELU, [(2048,)], {}),
        (oc.OperationType.ACTIVATION, oc.OperationActivation.RELU, [(256, 256)], {}),
        
        # Matrix multiplications
        (oc.OperationType.MATMUL, oc.OperationMatmul.LINEAR, [(128, 256), (256, 512)], 
         {'M': 128, 'N': 512, 'K': 256}),
        (oc.OperationType.MATMUL, oc.OperationMatmul.LINEAR, [(64, 128), (128, 256)], 
         {'M': 64, 'N': 256, 'K': 128}),
        
        # Layer normalization (3D shape: N, L, H)
        (oc.OperationType.NORMALIZATION, oc.OperationNormalization.LAYER_NORM, [(4, 256, 1024)], 
         {'axis': (2,)}),
    ]
    
    print("Adding mixed kernel configurations...")
    for op_type, operation, shapes, params in configs:
        model_verifier.add_verification_config(op_type, operation, shapes, params)
    
    print(f"Running quick unified verification with {len(configs)} test cases...")
    
    try:
        results = model_verifier.verify()
        print(f"\n✅ Quick unified verification completed!")
        print(f"Results saved to: {verification_dir}/merged_verification_results.csv")
        return results
    except Exception as e:
        import traceback
        print(f"❌ Error during quick unified verification: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_unified_test()
    else:
        run_unified_verification()
