"""
Hybrid pipeline: Huber regression on size + HGBR error correction on 2D shape.

Approach:
  1. Fit Huber regression:   duration_huber = a * size + b
  2. Compute residual:       residual = actual_duration - duration_huber
  3. Train HGBR on residual using only (d0, d1) — no size feature.
  4. Final prediction:       pred = huber(size) + HGBR(d0, d1)
  5. Randomly hold out 100 samples for testing; rest used for training.
  6. Generate comparison figures including HGBR-only scatter.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Add model/ to path so we can reuse load_and_prepare / split_by_log_bins
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
sys.path.insert(0, str(MODEL_DIR))
from train import load_and_prepare, split_by_log_bins

# ============================================================================
# CONFIGURATION
# ============================================================================

TRAINING_CSV = str(Path(__file__).resolve().parent.parent / "model" / "add_dataset_2d.csv")
OUTPUT_DIR = str(Path(__file__).resolve().parent / "results")
OP_NAME = "add"
SEED = 1999
N_TEST = 100

# HGBR hyper-parameters (for the residual model)
HGBR_PARAMS = dict(
    loss="absolute_error",
    learning_rate=0.06,
    max_depth=8,
    max_iter=600,
    min_samples_leaf=20,
)

# ============================================================================
# HELPERS
# ============================================================================


def make_2d_features(df: pd.DataFrame) -> np.ndarray:
    """Features for HGBR — only the two raw dimensions, no size."""
    d0 = df["input_dim_0"].to_numpy(dtype=np.float64)
    d1 = df["input_dim_1"].to_numpy(dtype=np.float64)

    eps = 1e-6
    features = [
        d0,
        d1,
    ]
    return np.column_stack(features)


# ============================================================================
# PIPELINE
# ============================================================================


def run_pipeline(
    training_csv: str = TRAINING_CSV,
    output_dir: str = OUTPUT_DIR,
    op_name: str = OP_NAME,
    seed: int = SEED,
    n_test: int = N_TEST,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"HYBRID PIPELINE (Huber + HGBR) — {op_name.upper()}")
    print("=" * 70)
    print(f"Training CSV : {training_csv}")
    print(f"Output dir   : {output_dir}")
    print(f"Seed         : {seed}")
    print(f"Test samples : {n_test}")
    print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # Load & split — randomly hold out n_test rows for testing
    # ------------------------------------------------------------------
    df = load_and_prepare(training_csv, op_name=op_name)
    rng = np.random.RandomState(seed)
    test_idx = rng.choice(len(df), size=min(n_test, len(df)), replace=False)
    test_mask = np.zeros(len(df), dtype=bool)
    test_mask[test_idx] = True
    test_df = df.iloc[test_mask].copy().reset_index(drop=True)
    train_df = df.iloc[~test_mask].copy().reset_index(drop=True)

    print(f"Total rows : {len(df)}")
    print(f"Train rows : {len(train_df)}")
    print(f"Test rows  : {len(test_df)}")

    y_train = train_df["computation_mean"].to_numpy(dtype=np.float64)
    y_test = test_df["computation_mean"].to_numpy(dtype=np.float64)
    size_train = train_df["size"].to_numpy(dtype=np.float64).reshape(-1, 1)
    size_test = test_df["size"].to_numpy(dtype=np.float64).reshape(-1, 1)

    # ------------------------------------------------------------------
    # Step 1 — Huber regression:  duration ≈ a·size + b
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Step 1: Huber Regression  (duration ~ size)")
    print("-" * 50)

    lr = HuberRegressor(max_iter=200)
    lr.fit(size_train, y_train)

    a, b = float(lr.coef_[0]), float(lr.intercept_)
    print(f"  slope (a)     = {a:.10f}")
    print(f"  intercept (b) = {b:.6f}")

    lr_pred_train = lr.predict(size_train)
    lr_pred_test = lr.predict(size_test)

    lr_mae_train = mean_absolute_error(y_train, lr_pred_train)
    lr_mae_test = mean_absolute_error(y_test, lr_pred_test)
    print(f"  Huber-only MAE  (train) : {lr_mae_train:.6f}")
    print(f"  Huber-only MAE  (test)  : {lr_mae_test:.6f}")

    # ------------------------------------------------------------------
    # Step 2 — Compute residuals
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Step 2: Compute residuals  (actual − huber)")
    print("-" * 50)

    residual_train = y_train - lr_pred_train
    residual_test = y_test - lr_pred_test

    print(f"  Residual stats (train): mean={residual_train.mean():.6f}  "
          f"std={residual_train.std():.6f}  "
          f"min={residual_train.min():.6f}  max={residual_train.max():.6f}")
    print(f"  Residual stats (test) : mean={residual_test.mean():.6f}  "
          f"std={residual_test.std():.6f}  "
          f"min={residual_test.min():.6f}  max={residual_test.max():.6f}")

    # ------------------------------------------------------------------
    # Step 3 — HGBR on residuals using only (d0, d1)
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Step 3: Train HGBR on residuals  (features: d0, d1 only)")
    print("-" * 50)

    X_train_2d = make_2d_features(train_df)
    X_test_2d = make_2d_features(test_df)

    hgbr = HistGradientBoostingRegressor(
        **HGBR_PARAMS,
        random_state=seed,
    )
    hgbr.fit(X_train_2d, residual_train)

    res_pred_train = hgbr.predict(X_train_2d)
    res_pred_test = hgbr.predict(X_test_2d)

    res_mae_train = mean_absolute_error(residual_train, res_pred_train)
    res_mae_test = mean_absolute_error(residual_test, res_pred_test)
    print(f"  Residual HGBR MAE (train) : {res_mae_train:.6f}")
    print(f"  Residual HGBR MAE (test)  : {res_mae_test:.6f}")

    # ------------------------------------------------------------------
    # Step 4 — Combined prediction:  linear(size) + HGBR(d0,d1)
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Step 4: Combined prediction  = huber(size) + HGBR(d0,d1)")
    print("-" * 50)

    hybrid_pred_train = lr_pred_train + res_pred_train
    hybrid_pred_test = lr_pred_test + res_pred_test

    eps = 1e-6

    def compute_metrics(y_true, y_pred, label):
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(
            np.maximum(y_true, eps), np.maximum(y_pred, eps)
        )
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        abs_err = np.abs(y_true - y_pred)
        rel_err = abs_err / np.maximum(y_true, eps) * 100
        print(f"\n  [{label}]")
        print(f"    MAE            : {mae:.6f}")
        print(f"    MAPE           : {mape * 100:.2f}%")
        print(f"    R²             : {r2:.6f}")
        print(f"    Max Abs Error  : {np.max(abs_err):.6f}")
        print(f"    P50 Rel Error  : {np.percentile(rel_err, 50):.2f}%")
        print(f"    P90 Rel Error  : {np.percentile(rel_err, 90):.2f}%")
        print(f"    P99 Rel Error  : {np.percentile(rel_err, 99):.2f}%")
        return dict(mae=mae, mape=mape, r2=r2, abs_err=abs_err, rel_err=rel_err)

    train_metrics = compute_metrics(y_train, hybrid_pred_train, "TRAIN — Hybrid")
    test_metrics = compute_metrics(y_test, hybrid_pred_test, "TEST  — Hybrid")

    # Also show linear-only for comparison
    lr_test_metrics = compute_metrics(y_test, lr_pred_test, "TEST  — Huber only")

    # ------------------------------------------------------------------
    # Step 5 — Build comparison dataframe & save
    # ------------------------------------------------------------------
    test_comp = pd.DataFrame({
        "dim_0": test_df["input_dim_0"].values,
        "dim_1": test_df["input_dim_1"].values,
        "size": test_df["size"].values,
        "actual": y_test,
        "linear_pred": lr_pred_test,
        "residual_actual": residual_test,
        "residual_pred": res_pred_test,
        "hybrid_pred": hybrid_pred_test,
        "abs_error_linear": np.abs(y_test - lr_pred_test),
        "abs_error_hybrid": np.abs(y_test - hybrid_pred_test),
        "rel_error_linear_pct": np.abs(y_test - lr_pred_test) / np.maximum(y_test, eps) * 100,
        "rel_error_hybrid_pct": np.abs(y_test - hybrid_pred_test) / np.maximum(y_test, eps) * 100,
    })

    comp_csv = output_dir / f"comparison_hybrid_{timestamp}.csv"
    test_comp.to_csv(comp_csv, index=False)
    print(f"\nComparison CSV saved to: {comp_csv}")

    # ------------------------------------------------------------------
    # Step 6 — Figures
    # ------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Step 6: Generating figures")
    print("-" * 50)

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Fig 1: Actual vs Predicted scatter (hybrid & linear) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(y_test, lr_pred_test, alpha=0.4, s=12, label="Huber only")
    lo = min(y_test.min(), lr_pred_test.min())
    hi = max(y_test.max(), lr_pred_test.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Actual duration")
    ax.set_ylabel("Predicted duration")
    ax.set_title(f"Huber only  (R²={lr_test_metrics['r2']:.4f})")
    ax.legend()

    ax = axes[1]
    ax.scatter(y_test, hybrid_pred_test, alpha=0.4, s=12, color="C1", label="Hybrid (Huber+HGBR)")
    lo = min(y_test.min(), hybrid_pred_test.min())
    hi = max(y_test.max(), hybrid_pred_test.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Actual duration")
    ax.set_ylabel("Predicted duration")
    ax.set_title(f"Hybrid  (R²={test_metrics['r2']:.4f})")
    ax.legend()

    fig.suptitle(f"Actual vs Predicted — {op_name.upper()} 2D", fontsize=14)
    fig.tight_layout()
    path1 = fig_dir / f"actual_vs_pred_{timestamp}.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path1}")

    # --- Fig 2: Residual distribution (before & after HGBR correction) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(residual_test, bins=60, alpha=0.7, edgecolor="k", linewidth=0.3)
    ax.axvline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("Residual  (actual − huber)")
    ax.set_ylabel("Count")
    ax.set_title("Residual before HGBR correction")

    corrected_residual = y_test - hybrid_pred_test
    ax = axes[1]
    ax.hist(corrected_residual, bins=60, alpha=0.7, color="C1", edgecolor="k", linewidth=0.3)
    ax.axvline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("Error  (actual − hybrid_pred)")
    ax.set_ylabel("Count")
    ax.set_title("Error after HGBR correction")

    fig.suptitle("Residual / Error Distribution (Test Set)", fontsize=14)
    fig.tight_layout()
    path2 = fig_dir / f"residual_dist_{timestamp}.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path2}")

    # --- Fig 3: Relative error vs size (log-scale x) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    sizes = test_comp["size"].values
    ax.scatter(sizes, test_comp["rel_error_linear_pct"].values,
               alpha=0.4, s=12, label="Huber only")
    ax.scatter(sizes, test_comp["rel_error_hybrid_pct"].values,
               alpha=0.4, s=12, label="Hybrid")
    ax.set_xscale("log")
    ax.set_xlabel("Tensor size  (d0 × d1)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Relative Error vs Tensor Size (Test Set)")
    ax.legend()
    fig.tight_layout()
    path3 = fig_dir / f"rel_error_vs_size_{timestamp}.png"
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path3}")

    # --- Fig 4: Error CDF (cumulative distribution) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_lr = np.sort(test_comp["rel_error_linear_pct"].values)
    sorted_hy = np.sort(test_comp["rel_error_hybrid_pct"].values)
    cdf = np.linspace(0, 1, len(sorted_lr))
    ax.plot(sorted_lr, cdf, label="Huber only")
    ax.plot(sorted_hy, cdf, label="Hybrid")
    ax.set_xlabel("Relative error (%)")
    ax.set_ylabel("CDF")
    ax.set_title("Cumulative Distribution of Relative Error (Test Set)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path4 = fig_dir / f"error_cdf_{timestamp}.png"
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path4}")

    # --- Fig 5: Linear fit visualization ---
    fig, ax = plt.subplots(figsize=(10, 5))
    sort_idx = np.argsort(size_test.ravel())
    ax.scatter(size_test.ravel(), y_test, alpha=0.3, s=10, label="Actual")
    ax.plot(size_test.ravel()[sort_idx], lr_pred_test[sort_idx],
            color="red", lw=1.5, label=f"Huber: {a:.2e}·size + {b:.2f}")
    ax.set_xlabel("Tensor size (d0 × d1)")
    ax.set_ylabel("Duration")
    ax.set_title("Huber Fit:  duration ~ size")
    ax.legend()
    fig.tight_layout()
    path5 = fig_dir / f"huber_fit_{timestamp}.png"
    fig.savefig(path5, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path5}")

    # --- Fig 6: HGBR-only  (x = actual residual, y = predicted residual) ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(residual_test, res_pred_test, alpha=0.5, s=18, color="C2")
    lo = min(residual_test.min(), res_pred_test.min())
    hi = max(residual_test.max(), res_pred_test.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    res_r2 = 1 - np.sum((residual_test - res_pred_test) ** 2) / max(
        np.sum((residual_test - residual_test.mean()) ** 2), 1e-12
    )
    res_mae = mean_absolute_error(residual_test, res_pred_test)
    ax.set_xlabel("Actual residual  (actual − huber)")
    ax.set_ylabel("HGBR predicted residual")
    ax.set_title(f"HGBR Error-Correction Only  (R²={res_r2:.4f}, MAE={res_mae:.4f})")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path6 = fig_dir / f"hgbr_only_{timestamp}.png"
    fig.savefig(path6, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path6}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Comparison CSV : {comp_csv}")
    print(f"  Figures        : {fig_dir}")
    print()
    print(f"  Huber-only   Test MAE  : {lr_test_metrics['mae']:.6f}")
    print(f"  Hybrid       Test MAE  : {test_metrics['mae']:.6f}")
    print(f"  Huber-only   Test MAPE : {lr_test_metrics['mape'] * 100:.2f}%")
    print(f"  Hybrid       Test MAPE : {test_metrics['mape'] * 100:.2f}%")
    print(f"  Huber-only   Test R²   : {lr_test_metrics['r2']:.6f}")
    print(f"  Hybrid       Test R²   : {test_metrics['r2']:.6f}")
    print()

    return {
        "lr_model": lr,
        "hgbr_model": hgbr,
        "test_comp": test_comp,
        "linear_metrics": lr_test_metrics,
        "hybrid_metrics": test_metrics,
    }


if __name__ == "__main__":
    run_pipeline()
