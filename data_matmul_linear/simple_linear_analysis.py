import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

def calculate_linear_function(csv_file_path):
    """
    Calculate the linear function between total_cycles and fusion_avg
    """
    # Load the CSV data
    df = pd.read_csv(csv_file_path)
    
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()
    
    # Extract the relevant columns
    X = df['total_cycles'].values.reshape(-1, 1)  # Independent variable
    y = df['fusion_avg'].values  # Dependent variable
    
    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get coefficients
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # Calculate R-squared
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    
    return slope, intercept, r2

def generate_output_filename(input_file_path):
    """
    Generate output filename based on input filename
    """
    # Get the base name without extension
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    return f"{base_name}_linear_function_results.txt"

def main():
    """
    Main function to calculate and display the linear function
    """
    csv_file = "kernel_report_updated_2.csv"
    
    try:
        # Calculate the linear function
        slope, intercept, r2 = calculate_linear_function(csv_file)
        
        # Generate output filename
        output_file = generate_output_filename(csv_file)
        
        # Display results
        print("=" * 50)
        print("LINEAR FUNCTION CALCULATION")
        print("=" * 50)
        print(f"Input file: {csv_file}")
        print(f"Output file: {output_file}")
        print(f"Linear Function: fusion_avg = {slope:.6f} * total_cycles + {intercept:.6f}")
        print(f"Slope: {slope:.6f}")
        print(f"Intercept: {intercept:.6f}")
        print(f"R-squared: {r2:.6f}")
        print("=" * 50)
        
        # Example predictions
        print("\nExample Predictions:")
        print("-" * 30)
        test_cycles = [400, 450, 500]
        for cycles in test_cycles:
            prediction = slope * cycles + intercept
            print(f"total_cycles = {cycles} → fusion_avg = {prediction:.6f}")
        
        # Save results to file
        with open(output_file, 'w') as f:
            f.write("SIMPLE LINEAR FUNCTION ANALYSIS\n")
            f.write("=" * 40 + "\n")
            f.write(f"Input file: {csv_file}\n")
            f.write(f"Linear Function: fusion_avg = {slope:.6f} * total_cycles + {intercept:.6f}\n")
            f.write(f"Slope: {slope:.6f}\n")
            f.write(f"Intercept: {intercept:.6f}\n")
            f.write(f"R-squared: {r2:.6f}\n")
            f.write(f"Interpretation: {r2*100:.2f}% of variance explained\n")
            f.write("\nExample Predictions:\n")
            for cycles in test_cycles:
                prediction = slope * cycles + intercept
                f.write(f"total_cycles = {cycles} → fusion_avg = {prediction:.6f}\n")
        
        print(f"\nResults saved to '{output_file}'")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
