import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import os

def load_and_analyze_data(csv_file_path):
    """
    Load CSV data and perform linear regression analysis between total_cycles and fusion_avg
    """
    # Load the CSV data
    print("Loading data from CSV file...")
    df = pd.read_csv(csv_file_path)
    
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()
    
    # Display basic information about the data
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Check for missing values
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Extract the relevant columns
    X = df['total_cycles'].values.reshape(-1, 1)  # Independent variable
    y = df['fusion_avg'].values  # Dependent variable
    
    print(f"\nData summary:")
    print(f"Total cycles - Min: {X.min():.2f}, Max: {X.max():.2f}, Mean: {X.mean():.2f}")
    print(f"Fusion avg - Min: {y.min():.2f}, Max: {y.max():.2f}, Mean: {y.mean():.2f}")
    
    return df, X, y

def perform_linear_regression(X, y):
    """
    Perform linear regression and return the model and metrics
    """
    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Get coefficients
    slope = model.coef_[0]
    intercept = model.intercept_
    
    return model, y_pred, slope, intercept, r2, mse, rmse

def print_results(slope, intercept, r2, mse, rmse):
    """
    Print the linear regression results
    """
    print("\n" + "="*60)
    print("LINEAR REGRESSION RESULTS")
    print("="*60)
    print(f"Linear Function: fusion_avg = {slope:.6f} * total_cycles + {intercept:.6f}")
    print(f"Slope (coefficient): {slope:.6f}")
    print(f"Intercept: {intercept:.6f}")
    print(f"R-squared (R²): {r2:.6f}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print("="*60)
    
    # Interpretation
    print("\nINTERPRETATION:")
    print(f"- For every 1 unit increase in total_cycles, fusion_avg changes by {slope:.6f} units")
    print(f"- When total_cycles = 0, fusion_avg = {intercept:.6f}")
    print(f"- R² = {r2:.6f} means that {r2*100:.2f}% of the variance in fusion_avg is explained by total_cycles")
    
    if r2 > 0.7:
        print("- Strong linear relationship (R² > 0.7)")
    elif r2 > 0.5:
        print("- Moderate linear relationship (R² > 0.5)")
    else:
        print("- Weak linear relationship (R² < 0.5)")

def generate_output_filenames(input_file_path):
    """
    Generate output filenames based on input filename
    """
    # Get the base name without extension
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    results_file = f"{base_name}_linear_regression_results.txt"
    plot_file = f"{base_name}_linear_regression_analysis.png"
    return results_file, plot_file

def create_visualizations(df, X, y, y_pred, slope, intercept, plot_filename):
    """
    Create comprehensive visualizations of the linear regression
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Linear Regression Analysis: total_cycles vs fusion_avg', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot with regression line
    axes[0, 0].scatter(X, y, alpha=0.6, s=30, label='Data points')
    axes[0, 0].plot(X, y_pred, color='red', linewidth=2, label=f'Regression line: y = {slope:.6f}x + {intercept:.6f}')
    axes[0, 0].set_xlabel('Total Cycles')
    axes[0, 0].set_ylabel('Fusion Average')
    axes[0, 0].set_title('Linear Regression Fit')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals plot
    residuals = y - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=30)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Histogram of residuals
    axes[1, 0].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Q-Q plot for normality check
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional analysis: Correlation matrix
    correlation_matrix = df[['total_cycles', 'fusion_avg']].corr()
    print(f"\nCorrelation Matrix:")
    print(correlation_matrix)
    
    # Statistical summary
    print(f"\nStatistical Summary:")
    print(df[['total_cycles', 'fusion_avg']].describe())

def predict_values(model, total_cycles_values):
    """
    Make predictions for given total_cycles values
    """
    print(f"\nPREDICTIONS:")
    print("-" * 40)
    for cycles in total_cycles_values:
        prediction = model.predict([[cycles]])[0]
        print(f"Total cycles: {cycles:>8.1f} -> Predicted fusion_avg: {prediction:.6f}")

def main():
    """
    Main function to run the complete analysis
    """
    # File path - now in the same directory
    csv_file = "kernel_report_updated_2.csv"
    
    try:
        # Generate output filenames
        results_file, plot_file = generate_output_filenames(csv_file)
        
        # Load and analyze data
        df, X, y = load_and_analyze_data(csv_file)
        
        # Perform linear regression
        model, y_pred, slope, intercept, r2, mse, rmse = perform_linear_regression(X, y)
        
        # Print results
        print_results(slope, intercept, r2, mse, rmse)
        
        # Create visualizations
        create_visualizations(df, X, y, y_pred, slope, intercept, plot_file)
        
        # Make some predictions
        sample_cycles = [400, 450, 500, 550, 600]
        predict_values(model, sample_cycles)
        
        # Save the linear function to a file
        with open(results_file, 'w') as f:
            f.write("COMPREHENSIVE LINEAR REGRESSION RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Input file: {csv_file}\n")
            f.write(f"Linear Function: fusion_avg = {slope:.6f} * total_cycles + {intercept:.6f}\n")
            f.write(f"Slope: {slope:.6f}\n")
            f.write(f"Intercept: {intercept:.6f}\n")
            f.write(f"R-squared: {r2:.6f}\n")
            f.write(f"MSE: {mse:.6f}\n")
            f.write(f"RMSE: {rmse:.6f}\n")
            f.write(f"Interpretation: {r2*100:.2f}% of variance explained\n")
            f.write(f"\nCorrelation: {np.sqrt(r2):.6f}\n")
            f.write(f"\nExample Predictions:\n")
            for cycles in sample_cycles:
                prediction = model.predict([[cycles]])[0]
                f.write(f"Total cycles: {cycles} -> Predicted fusion_avg: {prediction:.6f}\n")
        
        print(f"\nResults saved to '{results_file}'")
        print(f"Visualization saved to '{plot_file}'")
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found!")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
