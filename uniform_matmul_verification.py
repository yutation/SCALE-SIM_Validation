#!/usr/bin/env python3
"""
Uniform Matrix Multiplication Verification Script

This script runs verification for 100 matrix multiplication configurations
where M * N * K (total operations) is uniformly distributed across the range.
Individual dimensions M, N, K are constrained to [32, 4096].
"""

import os
import sys
import random
import math
sys.path.append('.')
import operation_classification as oc
from model_verification import ModelVerification


def generate_uniform_matmul_configs(num_configs=100, min_dim=32, max_dim=4096):
    """
    Generate matmul configurations where M*N*K is uniformly distributed.
    
    Strategy:
    1. Calculate min and max possible M*N*K values
    2. Uniformly sample M*N*K values across this range
    3. For each target M*N*K, randomly decompose into M, N, K dimensions
       while keeping each dimension in [min_dim, max_dim]
    """
    
    # Calculate the range of possible M*N*K values
    min_ops = min_dim ** 3  # 32^3 = 32,768
    max_ops = max_dim ** 3  # 4096^3 = 68,719,476,736
    
    print(f"Operation count range: {min_ops:,} to {max_ops:,}")
    print(f"Using log-uniform distribution for better coverage\n")
    
    configs = []
    random.seed(42)
    
    for i in range(num_configs):
        # Use log-uniform distribution for better coverage across orders of magnitude
        log_min = math.log10(min_ops)
        log_max = math.log10(max_ops)
        target_ops = 10 ** random.uniform(log_min, log_max)
        
        # Try to decompose target_ops into M, N, K
        # Strategy: randomly choose M and N, then calculate K
        max_attempts = 100
        for attempt in range(max_attempts):
            # Randomly choose M and N
            M = random.randint(min_dim, max_dim)
            N = random.randint(min_dim, max_dim)
            
            # Calculate required K
            K_target = target_ops / (M * N)
            
            # If K is in valid range, use it
            if min_dim <= K_target <= max_dim:
                K = int(K_target)
                actual_ops = M * N * K
                configs.append((M, N, K, actual_ops))
                break
            
            # If we're on last attempt, just clamp K to valid range
            if attempt == max_attempts - 1:
                K = max(min_dim, min(max_dim, int(K_target)))
                actual_ops = M * N * K
                configs.append((M, N, K, actual_ops))
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_configs} configurations...")
    
    # Sort by operation count for analysis
    configs.sort(key=lambda x: x[3])
    
    return configs


def uniform_matmul_verification():
    """Run unified verification for 100 matrix multiplication operations with uniform M*N*K."""
    
    print("=" * 60)
    print("UNIFORM MATMUL VERIFICATION - 100 Configurations")
    print("M * N * K uniformly distributed")
    print("=" * 60)
    
    verification_dir = "./uniform_matmul_verification_results2"
    os.makedirs(verification_dir, exist_ok=True)
    
    model_verifier = ModelVerification(profile_dir=verification_dir)
    
    print("\nGenerating 100 matrix multiplication configurations...")
    print("Dimension constraints: M, N, K ∈ [32, 4096]")
    
    # Generate configurations
    configs = generate_uniform_matmul_configs(num_configs=100)
    
    print(f"\n✓ Generated {len(configs)} configurations")
    
    # Add configurations to verifier
    for M, N, K, ops in configs:
        model_verifier.add_verification_config(
            operation_type=oc.OperationType.MATMUL,
            operation=oc.OperationMatmul.LINEAR,
            shapes=[(M, K), (K, N)],
            operation_params={'M': M, 'N': N, 'K': K}
        )
    
    # Show distribution statistics
    print("\n" + "=" * 60)
    print("CONFIGURATION STATISTICS")
    print("=" * 60)
    
    ops_values = [ops for _, _, _, ops in configs]
    
    print(f"M*N*K range:")
    print(f"  Minimum: {min(ops_values):,} operations")
    print(f"  Maximum: {max(ops_values):,} operations")
    print(f"  Median:  {sorted(ops_values)[len(ops_values)//2]:,} operations")
    
    # Show distribution across orders of magnitude
    print(f"\nDistribution by operation count:")
    ranges = [
        (1e3, 1e6, "1K - 1M"),
        (1e6, 1e9, "1M - 1B"),
        (1e9, 1e12, "1B - 1T"),
        (1e12, 1e15, "1T - 1Q"),
    ]
    
    for low, high, label in ranges:
        count = sum(1 for ops in ops_values if low <= ops < high)
        if count > 0:
            print(f"  {label:12}: {count:3d} configs ({count/len(configs)*100:.1f}%)")
    
    # Show sample configurations
    print(f"\nSample configurations (evenly spaced by M*N*K):")
    indices = [0, len(configs)//4, len(configs)//2, 3*len(configs)//4, len(configs)-1]
    for idx in indices:
        M, N, K, ops = configs[idx]
        print(f"  M={M:4d}, N={N:4d}, K={K:4d} -> M*N*K = {ops:>15,} ops")
    
    print("=" * 60)
    
    # Run verification
    print("\nRunning verification (this will take a while)...")
    try:
        results = model_verifier.verify()
        
        print(f"\n🎉 Verification completed successfully!")
        print(f"📁 All results saved to: {verification_dir}/")
        print(f"📊 Main results file: {verification_dir}/merged_verification_results.csv")
        print(f"📈 Profiling data: {verification_dir}/filtered_events_avg_fusion.csv")
        
        # Analysis
        print("\n" + "=" * 60)
        print("RESULTS ANALYSIS")
        print("=" * 60)
        
        mape = results['Error_Percentage'].abs().mean()
        rmse = ((results['Predicted_Latency_us'] - results['Actual_Duration_us']) ** 2).mean() ** 0.5
        min_error = results['Error_Percentage'].abs().min()
        max_error = results['Error_Percentage'].abs().max()
        median_error = results['Error_Percentage'].abs().median()
        
        print(f"Total test cases: {len(results)}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Median Absolute Percentage Error: {median_error:.2f}%")
        print(f"Root Mean Square Error (RMSE): {rmse:.2f} μs")
        print(f"Error range: {min_error:.2f}% - {max_error:.2f}%")
        
        # Best and worst predictions
        best_idx = results['Error_Percentage'].abs().idxmin()
        worst_idx = results['Error_Percentage'].abs().idxmax()
        
        print(f"\n🏆 Best prediction:")
        print(f"   Shape: {results.loc[best_idx, 'Input_Shapes']}")
        print(f"   Predicted: {results.loc[best_idx, 'Predicted_Latency_us']:.2f} μs")
        print(f"   Actual: {results.loc[best_idx, 'Actual_Duration_us']:.2f} μs")
        print(f"   Error: {results.loc[best_idx, 'Error_Percentage']:.2f}%")
        
        print(f"\n⚠️  Worst prediction:")
        print(f"   Shape: {results.loc[worst_idx, 'Input_Shapes']}")
        print(f"   Predicted: {results.loc[worst_idx, 'Predicted_Latency_us']:.2f} μs")
        print(f"   Actual: {results.loc[worst_idx, 'Actual_Duration_us']:.2f} μs")
        print(f"   Error: {results.loc[worst_idx, 'Error_Percentage']:.2f}%")
        
        # Distribution of errors
        print(f"\n📊 Error Distribution:")
        error_bins = [
            (0, 5, "Excellent"),
            (5, 10, "Good"),
            (10, 20, "Fair"),
            (20, 50, "Poor"),
            (50, float('inf'), "Very Poor")
        ]
        
        for low, high, label in error_bins:
            count = len(results[(results['Error_Percentage'].abs() >= low) & 
                               (results['Error_Percentage'].abs() < high)])
            percentage = (count / len(results)) * 100
            print(f"   {label:12} ({low:3.0f}%-{high:3.0f}%): {count:3d} cases ({percentage:5.1f}%)")
        
        return results
        
    except Exception as e:
        import traceback
        print(f"❌ Error during verification: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    uniform_matmul_verification()

