import pandas as pd
import math
import ast


def matmul_scale_sim_model(m: int, n: int, k: int, systolic_array_size: int = 128) -> int:
    """
    Calculate the minimum cycles for matmul operation.
    Input shapes: (M, K) @ (K, N)
    """
    v1 = (2*systolic_array_size + systolic_array_size + m - 2) * math.ceil(n / systolic_array_size) * math.ceil(k / systolic_array_size)
    # m, n = n, m
    # v2 = (2*systolic_array_size + systolic_array_size + m - 2) * math.ceil(n / systolic_array_size) * math.ceil(k / systolic_array_size)
    v2 = v1
    return min(v1, v2)


def parse_input_shapes(shape_str):
    """
    Parse input shapes string like "[(32, 128), (128, 128)]"
    Returns M, K, N for matmul (M, K) @ (K, N)
    """
    shapes = ast.literal_eval(shape_str)
    # First matrix is (M, K), second matrix is (K, N)
    M, K = shapes[0]
    K2, N = shapes[1]
    assert K == K2, f"Matrix dimensions don't match: K={K}, K2={K2}"
    return M, K, N


def add_model_column(input_csv, output_csv):
    """
    Read CSV, add a column with matmul_scale_sim_model output, and save to new CSV.
    """
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Calculate the model output for each row
    model_outputs = []
    for idx, row in df.iterrows():
        try:
            M, K, N = parse_input_shapes(row['Input_Shapes'])
            model_output = matmul_scale_sim_model(M, N, K)
            model_outputs.append(model_output)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            model_outputs.append(None)
    
    # Add the new column
    df['Model_Output'] = model_outputs
    
    # Save to output CSV
    df.to_csv(output_csv, index=False)
    print(f"Successfully added 'Model_Output' column to {output_csv}")
    print(f"Total rows processed: {len(df)}")
    print(f"\nFirst few rows:")
    print(df.head())


if __name__ == "__main__":
    input_file = "merged_verification_results.csv"
    output_file = "merged_verification_results_with_model2.csv"
    
    add_model_column(input_file, output_file)
