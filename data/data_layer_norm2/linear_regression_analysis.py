#!/usr/bin/env python3
"""
Linear Regression Analysis for Layer Norm Kernel Performance
Analyzes the relationship between dimension product and average duration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import sys
from pathlib import Path

def perform_linear_regression_analysis(input_file='kernel_analysis.csv'):
    """
    Perform comprehensive linear regression analysis on kernel performance data.
    """
    try:
        # Read the data
        print(f"Reading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        print(f"Loaded {len(df)} data points")
        print(f"Dimension product range: {df['dim_product'].min():,} to {df['dim_product'].max():,}")
        print(f"Average duration range: {df['avg_duration_us'].min():.3f} to {df['avg_duration_us'].max():.3f} μs")
        
        # Prepare data for regression
        X = df['dim_product'].values.reshape(-1, 1)
        y = df['avg_duration_us'].values
        
        # Create and fit the linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        
        # Calculate correlation
        correlation, p_value = stats.pearsonr(df['dim_product'], df['avg_duration_us'])
        
        # Print regression results
        print("\n" + "="*60)
        print("LINEAR REGRESSION ANALYSIS RESULTS")
        print("="*60)
        print(f"Regression Equation: avg_duration_us = {model.intercept_:.6f} + {model.coef_[0]:.10f} * dim_product")
        print(f"")
        print(f"Model Performance:")
        print(f"  R-squared (R²):           {r2:.6f}")
        print(f"  Correlation coefficient:  {correlation:.6f}")
        print(f"  P-value:                  {p_value:.2e}")
        print(f"  Root Mean Square Error:   {rmse:.6f} μs")
        print(f"  Mean Absolute Error:      {mae:.6f} μs")
        print(f"  Mean Square Error:        {mse:.6f} μs²")
        
        # Interpretation
        print(f"\nInterpretation:")
        if r2 >= 0.9:
            strength = "very strong"
        elif r2 >= 0.7:
            strength = "strong"
        elif r2 >= 0.5:
            strength = "moderate"
        else:
            strength = "weak"
        
        print(f"  The linear relationship is {strength} (R² = {r2:.3f})")
        print(f"  For every additional unit in dimension product, duration increases by {model.coef_[0]*1e6:.3f} nanoseconds")
        
        if p_value < 0.001:
            significance = "highly significant"
        elif p_value < 0.01:
            significance = "very significant"
        elif p_value < 0.05:
            significance = "significant"
        else:
            significance = "not significant"
        
        print(f"  The relationship is {significance} (p < {0.001 if p_value < 0.001 else p_value:.3f})")
        
        # Create visualization
        print(f"\nGenerating visualization...")
        
        # Set up the plot style
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Main scatter plot with regression line
        ax1.scatter(df['dim_product'], df['avg_duration_us'], alpha=0.6, s=50, c='blue', label='Data points')
        ax1.plot(df['dim_product'], y_pred, color='red', linewidth=2, label=f'Linear fit (R² = {r2:.3f})')
        
        ax1.set_xlabel('Dimension Product (batch_size × seq_len × hidden_dim)', fontsize=12)
        ax1.set_ylabel('Average Duration (μs)', fontsize=12)
        ax1.set_title('Layer Norm Kernel Performance\nDimension Product vs Average Duration', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis with scientific notation for large numbers
        ax1.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        # Residuals plot
        residuals = y - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, s=50, c='green')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Duration (μs)', fontsize=12)
        ax2.set_ylabel('Residuals (μs)', fontsize=12)
        ax2.set_title('Residuals Plot\n(Actual - Predicted)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_plot = Path(input_file).parent / 'linear_regression_analysis.png'
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_plot}")
        
        # Additional analysis: Performance by batch size
        print(f"\nPerformance Analysis by Batch Size:")
        batch_analysis = df.groupby('batch_size').agg({
            'avg_duration_us': ['mean', 'std', 'min', 'max'],
            'dim_product': ['mean', 'min', 'max']
        }).round(3)
        print(batch_analysis.to_string())
        
        # Create summary DataFrame
        summary_data = {
            'Metric': ['R-squared', 'Correlation', 'P-value', 'RMSE (μs)', 'MAE (μs)', 'Slope', 'Intercept'],
            'Value': [r2, correlation, p_value, rmse, mae, model.coef_[0], model.intercept_]
        }
        summary_df = pd.DataFrame(summary_data)
        
        # Save detailed results
        results_file = Path(input_file).parent / 'regression_results.csv'
        summary_df.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to {results_file}")
        
        # Save predictions
        df_with_predictions = df.copy()
        df_with_predictions['predicted_duration_us'] = y_pred
        df_with_predictions['residuals'] = residuals
        df_with_predictions['relative_error_pct'] = (residuals / y) * 100
        
        predictions_file = Path(input_file).parent / 'predictions.csv'
        df_with_predictions.to_csv(predictions_file, index=False)
        print(f"Predictions and residuals saved to {predictions_file}")
        
        return model, r2, correlation, p_value
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        sys.exit(1)

def main():
    """Main function to run the analysis."""
    script_dir = Path(__file__).parent
    input_file = script_dir / "kernel_analysis.csv"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        sys.exit(1)
    
    # Run the regression analysis
    model, r2, correlation, p_value = perform_linear_regression_analysis(str(input_file))
    
    print(f"\nAnalysis complete!")
    print(f"Files generated:")
    print(f"  - linear_regression_analysis.png (visualization)")
    print(f"  - regression_results.csv (summary metrics)")
    print(f"  - predictions.csv (detailed predictions)")

if __name__ == "__main__":
    main()
