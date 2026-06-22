import math
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import HistGradientBoostingRegressor


def load_and_prepare(csv_path: str, op_name: str = "add") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Basic cleanup: ensure expected columns exist
    required = ["operation_name", "input_dim_0", "input_dim_1", "input_dim_2", "size", "computation_mean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Filter operation
    df = df[df["operation_name"].astype(str).str.strip().str.lower() == op_name.lower()].copy()
    if len(df) == 0:
        raise ValueError(f"No rows found for operation_name == {op_name}")

    # Fill missing dims with 1, and cast to int
    for c in ["input_dim_0", "input_dim_1", "input_dim_2"]:
        df[c] = df[c].fillna(1).astype(int).clip(lower=1)

    # Ensure size exists & positive; if size is wrong, recompute from dims
    df["size"] = df["size"].fillna(0).astype(int)
    recomputed_size = df["input_dim_0"] * df["input_dim_1"] * df["input_dim_2"]
    bad_size = (df["size"] <= 0) | (df["size"] != recomputed_size)
    if bad_size.any():
        df.loc[bad_size, "size"] = recomputed_size[bad_size]

    # Label
    df["computation_mean"] = df["computation_mean"].astype(float)

    # Optional: drop obvious junk
    df = df[(df["size"] > 0) & np.isfinite(df["computation_mean"])].copy()

    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep it simple: dims + size (+ log2(size)).
    If you want even simpler, remove log2_size.
    """
    X = pd.DataFrame({
        "d0": df["input_dim_0"].astype(np.int64),
        "d1": df["input_dim_1"].astype(np.int64),
        "d2": df["input_dim_2"].astype(np.int64),
        "size": df["size"].astype(np.int64),
    })

    # Simple extra feature that helps a lot while still minimal
    X["log2_size"] = np.log2(X["size"].clip(lower=1)).astype(np.float32)

    return X


def split_by_log_bins(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 0):
    """
    Better than random split: hold out ~val_ratio from each log2(size) bin,
    so validation covers all sizes (small to large).
    """
    rng = np.random.default_rng(seed)

    sizes = df["size"].to_numpy()
    bins = np.floor(np.log2(np.maximum(sizes, 1))).astype(int)  # 0..24 for <= 16M

    val_mask = np.zeros(len(df), dtype=bool)

    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        if len(idx) < 5:
            # too small bin: put all in train
            continue
        k = max(1, int(round(len(idx) * val_ratio)))
        choose = rng.choice(idx, size=k, replace=False)
        val_mask[choose] = True

    train_df = df[~val_mask].copy()
    val_df = df[val_mask].copy()
    return train_df, val_df


def train_and_validate(csv_path: str, op_name: str = "add", seed: int = 0):
    df = load_and_prepare(csv_path, op_name=op_name)
    train_df, val_df = split_by_log_bins(df, val_ratio=0.2, seed=seed)

    X_train = make_features(train_df)
    y_train = train_df["computation_mean"].to_numpy(dtype=np.float64)

    X_val = make_features(val_df)
    y_val = val_df["computation_mean"].to_numpy(dtype=np.float64)

    # Simple, strong default model for tabular regression
    model = HistGradientBoostingRegressor(
        loss="absolute_error",      # robust to noise/outliers; can switch to "squared_error"
        learning_rate=0.06,
        max_depth=8,
        max_iter=600,
        min_samples_leaf=20,
        random_state=seed,
    )

    model.fit(X_train, y_train)

    pred_val = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred_val)

    # MAPE is unstable when y is near 0; add epsilon safeguard
    eps = 1e-6
    mape = mean_absolute_percentage_error(np.maximum(y_val, eps), np.maximum(pred_val, eps))

    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")
    print(f"Val MAE:  {mae:.6f}")
    print(f"Val MAPE: {mape*100:.2f}%")

    return model


if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "relu_dataset_20260202_193859.csv")
    model = train_and_validate(csv_path, op_name="relu", seed=42)
