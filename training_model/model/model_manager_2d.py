"""
Model training, saving, loading, and inference utilities for 2D operations.

Extends the basic train.py functionality with model persistence and prediction
for 2D tensor shapes (d0, d1).
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union

from train import load_and_prepare, split_by_log_bins
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import HistGradientBoostingRegressor


def make_features_2d(df: pd.DataFrame) -> np.ndarray:
    """
    Create feature matrix for 2D shapes.
    
    Args:
        df: DataFrame with columns input_dim_0, input_dim_1, size
        
    Returns:
        Feature matrix as numpy array
    """
    features = []
    
    # Original dimensions
    features.append(df["input_dim_0"].to_numpy(dtype=np.float64))
    features.append(df["input_dim_1"].to_numpy(dtype=np.float64))
    
    # Size (product of dimensions)
    features.append(df["size"].to_numpy(dtype=np.float64))
    
    # Log-scaled features
    eps = 1e-6
    features.append(np.log1p(df["input_dim_0"].to_numpy(dtype=np.float64) + eps))
    features.append(np.log1p(df["input_dim_1"].to_numpy(dtype=np.float64) + eps))
    features.append(np.log1p(df["size"].to_numpy(dtype=np.float64) + eps))
    
    # Aspect ratio and shape features
    d0 = df["input_dim_0"].to_numpy(dtype=np.float64)
    d1 = df["input_dim_1"].to_numpy(dtype=np.float64)
    
    # Aspect ratio (d0/d1)
    aspect_ratio = d0 / np.maximum(d1, eps)
    features.append(aspect_ratio)
    features.append(np.log1p(aspect_ratio))
    
    # Min and max dimensions
    features.append(np.minimum(d0, d1))
    features.append(np.maximum(d0, d1))
    
    # Powers of dimensions
    features.append(d0 ** 2)
    features.append(d1 ** 2)
    features.append(np.sqrt(d0 + eps))
    features.append(np.sqrt(d1 + eps))
    
    X = np.column_stack(features)
    return X


def save_model(model, save_path: str, op_name: str = "add", metadata: Dict = None):
    """
    Save trained model and metadata to disk.
    
    Args:
        model: Trained sklearn model
        save_path: Path to save the model file
        op_name: Operation name (e.g., "add")
        metadata: Optional dictionary with training metadata
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        "model": model,
        "op_name": op_name,
        "metadata": metadata or {},
        "dimensions": "2D"
    }
    
    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to: {save_path}")


def load_model(model_path: str):
    """
    Load trained model from disk.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Dictionary containing model and metadata
    """
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    return model_data


def train_and_save(
    csv_path: str,
    save_path: str,
    op_name: str = "add",
    seed: int = 0
) -> HistGradientBoostingRegressor:
    """
    Train model on 2D dataset and save to disk.
    
    Args:
        csv_path: Path to training dataset CSV
        save_path: Path to save the trained model
        op_name: Operation name to filter in dataset
        seed: Random seed for reproducibility
        
    Returns:
        Trained model
    """
    print(f"\n{'='*60}")
    print(f"Training 2D model on: {csv_path}")
    print(f"{'='*60}\n")
    
    df = load_and_prepare(csv_path, op_name=op_name)
    train_df, val_df = split_by_log_bins(df, val_ratio=0.2, seed=seed)

    X_train = make_features_2d(train_df)
    y_train = train_df["computation_mean"].to_numpy(dtype=np.float64)

    X_val = make_features_2d(val_df)
    y_val = val_df["computation_mean"].to_numpy(dtype=np.float64)

    # Train model
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.06,
        max_depth=8,
        max_iter=600,
        min_samples_leaf=20,
        random_state=seed,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # Validate
    pred_val = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred_val)
    
    eps = 1e-6
    mape = mean_absolute_percentage_error(
        np.maximum(y_val, eps), 
        np.maximum(pred_val, eps)
    )

    print(f"\nTraining Results:")
    print(f"  Train rows: {len(train_df)}")
    print(f"  Val rows:   {len(val_df)}")
    print(f"  Val MAE:    {mae:.6f}")
    print(f"  Val MAPE:   {mape*100:.2f}%")

    # Save model with metadata
    metadata = {
        "train_csv": str(csv_path),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "val_mae": float(mae),
        "val_mape": float(mape),
        "seed": seed,
        "dimensions": "2D"
    }
    
    save_model(model, save_path, op_name=op_name, metadata=metadata)
    
    return model


def predict_latency(
    model,
    shapes: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Predict latencies for a list of 2D shapes.
    
    Args:
        model: Trained model or model_data dict from load_model
        shapes: List of (d0, d1) shape tuples
        
    Returns:
        Array of predicted latencies
    """
    # Handle both raw model and loaded model dict
    if isinstance(model, dict):
        model = model["model"]
    
    # Create dataframe with shapes
    df = pd.DataFrame({
        "input_dim_0": [s[0] for s in shapes],
        "input_dim_1": [s[1] for s in shapes],
    })
    df["size"] = df["input_dim_0"] * df["input_dim_1"]
    
    # Make features
    X = make_features_2d(df)
    
    # Predict
    predictions = model.predict(X)
    
    return predictions


def predict_from_saved_model(
    model_path: str,
    shapes: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Load model and predict latencies for 2D shapes.
    
    Args:
        model_path: Path to saved model
        shapes: List of (d0, d1) shape tuples
        
    Returns:
        Array of predicted latencies
    """
    model_data = load_model(model_path)
    return predict_latency(model_data["model"], shapes)


if __name__ == "__main__":
    import os
    
    # Example usage
    script_dir = Path(__file__).parent
    csv_path = script_dir / "add_dataset_2d.csv"
    model_path = script_dir / "trained_models" / "add_model_2d.pkl"
    
    if csv_path.exists():
        model = train_and_save(
            str(csv_path),
            str(model_path),
            op_name="add",
            seed=42
        )
        
        # Test prediction
        test_shapes = [(100, 1), (256, 256), (1024, 512), (64, 128)]
        predictions = predict_latency(model, test_shapes)
        
        print(f"\nTest predictions:")
        for shape, pred in zip(test_shapes, predictions):
            print(f"  Shape {shape}: {pred:.6f}")
    else:
        print(f"Dataset not found: {csv_path}")

