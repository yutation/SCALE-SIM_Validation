#!/usr/bin/env python3
"""
Random Matrix Multiplication Data Collection Script

This script runs verification for 100 random matrix multiplication configurations
with dimensions ranging from 1024 to 4096, where M*N*K is uniformly distributed.
"""

import os
import sys
import random
import math
sys.path.append('.')
import operation_classification as oc
from model_verification import ModelVerification


def generate_uniform_configs(num_configs=256, min_dim=1024, max_dim=4096):
    """
    Generate matmul configurations where M*N*K is uniformly distributed.
    Dimensions constrained to [min_dim, max_dim].
    """
    
    # Calculate the range of possible M*N*K values
    min_ops = min_dim ** 3  # 1024^3
    max_ops = max_dim ** 3  # 4096^3
    
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


def random_matmul_data_collection():
    """Run data collection for 100 random matrix multiplication operations with uniform M*N*K."""
    
    print("=" * 60)
    print("RANDOM MATMUL DATA COLLECTION - 100 Configurations")
    print("M * N * K uniformly distributed")
    print("Dimension range: 1024 to 4096")
    print("=" * 60)
    
    verification_dir = "./matmul_data_collection_results3"
    os.makedirs(verification_dir, exist_ok=True)
    
    model_verifier = ModelVerification(profile_dir=verification_dir)
    
    print("\nGenerating 100 matrix multiplication configurations...")
    print("Dimension constraints: M, N, K ∈ [1024, 4096]\n")
    
    # Generate configurations with uniform M*N*K
    
    
    # Add configurations to verifier
    for K in range(128, 2049, 128):
        for N in range(128, 2049, 128):
            for M in range(128, 2049, 128):
                model_verifier.add_verification_config(
                    operation_type=oc.OperationType.MATMUL,
                    operation=oc.OperationMatmul.LINEAR,
                    shapes=[(M, K), (K, N)],
                    operation_params={'M': M, 'N': N, 'K': K}
                )



    
    # Show distribution statistics


    

    # Show sample configurations (evenly spaced by M*N*K)

    
    # Run verification
    print("\nRunning data collection (this will take a while)...")
    results = model_verifier.verify_matmul_sim()
    
    print(f"\n🎉 Data collection completed successfully!")
    print(f"📁 All results saved to: {verification_dir}/")
    print(f"📊 Main results file: {verification_dir}/merged_verification_results.csv")
    print(f"📈 Profiling data: {verification_dir}/filtered_events_avg_fusion.csv")
    
    return results


if __name__ == "__main__":
    random_matmul_data_collection()

