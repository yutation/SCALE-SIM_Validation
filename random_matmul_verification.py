#!/usr/bin/env python3
"""
Random Matrix Multiplication Verification Script

This script runs verification for 100 random matrix multiplication configurations
with dimensions ranging from 32 to 4096.
"""

import os
import sys
import random
sys.path.append('.')
import operation_classification as oc
from model_verification import ModelVerification


def generate_aligned_value(min_dim, max_dim, alignment, fluctuation):
    """
    Generate a value close to a multiple of alignment with specified fluctuation.
    
    Args:
        min_dim: Minimum dimension value
        max_dim: Maximum dimension value
        alignment: The base to align to (e.g., 128 or 512)
        fluctuation: Fluctuation percentage (e.g., 0.10 for 10%)
    
    Returns:
        An integer value close to a multiple of alignment within [min_dim, max_dim]
    """
    # Find valid multiples of alignment within the range
    first_multiple = ((min_dim + alignment - 1) // alignment) * alignment
    last_multiple = (max_dim // alignment) * alignment
    
    # Choose a random multiple
    num_multiples = (last_multiple - first_multiple) // alignment + 1
    chosen_multiple = first_multiple + random.randint(0, num_multiples - 1) * alignment
    
    # Apply fluctuation: ±10% of the alignment value
    max_fluctuation = int(alignment * fluctuation)
    fluctuation_value = random.randint(-max_fluctuation, max_fluctuation)
    
    # Calculate final value and clamp to range
    value = chosen_multiple + fluctuation_value
    value = max(min_dim, min(max_dim, value))
    
    return value


def random_matmul_verification():
    """Run unified verification for 100 random matrix multiplication operations."""
    
    print("=" * 60)
    print("RANDOM MATMUL VERIFICATION - 100 Configurations")
    print("=" * 60)
    
    verification_dir = "./random_matmul_verification_results2"
    os.makedirs(verification_dir, exist_ok=True)
    
    model_verifier = ModelVerification(profile_dir=verification_dir)
    
    print("Generating 100 random matrix multiplication configurations...")
    print("Distribution:")
    print("  - 1/3 (33 configs) from range 32-128 (fully random)")
    print("  - 1/3 (33 configs) from range 128-1024 (aligned to 128 ±10%)")
    print("  - 1/3 (34 configs) from range 1024-4096 (aligned to 512 ±10%)")
    
    # Set seed for reproducibility (optional - remove if you want different configs each time)
    random.seed(42)
    
    # Store configurations for display
    configs = []
    
    # Define the three ranges and their counts
    ranges = [
        (32, 128, 50),      # Small: 32-128, 33 configs
        (128, 1024, 50),    # Medium: 128-1024, 33 configs
        (1024, 4096, 50)    # Large: 1024-4096, 34 configs
    ]
    
    config_count = 0
    
    # Generate configurations for each range
    for range_idx, (min_dim, max_dim, count) in enumerate[tuple[int, int, int]](ranges):
        print(f"\nGenerating {count} configs from range [{min_dim}, {max_dim}]...")
        
        for i in range(count):
            if range_idx == 0:
                # Small range (32-128): fully random
                M = random.randint(min_dim, max_dim)
                N = random.randint(min_dim, max_dim)
                K = random.randint(min_dim, max_dim)
            elif range_idx == 1:
                # Medium range (128-1024): values close to multiples of 128 with 10% fluctuation
                M = generate_aligned_value(min_dim, max_dim, 128, 0.05)
                N = generate_aligned_value(min_dim, max_dim, 128, 0.05)
                K = generate_aligned_value(min_dim, max_dim, 128, 0.05)
            else:
                # Large range (1024-4096): values close to multiples of 512 with 10% fluctuation
                M = generate_aligned_value(min_dim, max_dim, 512, 0.05)
                N = generate_aligned_value(min_dim, max_dim, 512, 0.05)
                K = generate_aligned_value(min_dim, max_dim, 512, 0.05)
            
            configs.append((M, N, K, f"{min_dim}-{max_dim}"))
            
            model_verifier.add_verification_config(
                operation_type=oc.OperationType.MATMUL,
                operation=oc.OperationMatmul.LINEAR,
                shapes=[(M, K), (K, N)],
                operation_params={'M': M, 'N': N, 'K': K})
            
            config_count += 1
            if config_count % 20 == 0:
                print(f"  Progress: {config_count}/100 configurations...")
    
    print(f"\n✓ Total configurations: {len(model_verifier.prediction_manager.config_list)}")
    print("\nSample configurations from each range:")
    
    # Show samples from each range
    small_configs = [c for c in configs if c[3] == "32-128"]
    medium_configs = [c for c in configs if c[3] == "128-1024"]
    large_configs = [c for c in configs if c[3] == "1024-4096"]
    
    print(f"\n  Small (32-128) - First 3:")
    for i, (M, N, K, _) in enumerate(small_configs[:3]):
        print(f"    {i+1}. M={M:4d}, N={N:4d}, K={K:4d} -> ({M}x{K}) @ ({K}x{N}) = ({M}x{N})")
    
    print(f"\n  Medium (128-1024) - First 3:")
    for i, (M, N, K, _) in enumerate(medium_configs[:3]):
        print(f"    {i+1}. M={M:4d}, N={N:4d}, K={K:4d} -> ({M}x{K}) @ ({K}x{N}) = ({M}x{N})")
    
    print(f"\n  Large (1024-4096) - First 3:")
    for i, (M, N, K, _) in enumerate(large_configs[:3]):
        print(f"    {i+1}. M={M:4d}, N={N:4d}, K={K:4d} -> ({M}x{K}) @ ({K}x{N}) = ({M}x{N})")
    
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
    random_matmul_verification()

