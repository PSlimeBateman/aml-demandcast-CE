"""
train.py — Model training and MLflow logging for DemandCast
============================================================
Loads the engineered feature set, applies a temporal train/val/test split,
and trains regression models to predict hourly taxi demand per zone.
Every run is logged to MLflow — parameters, metrics, and the model artifact.

Usage (from project root with .venv active)
-------------------------------------------
    python src/train.py

Before running
--------------
1. MLflow UI must be running:
       mlflow ui
   Then open http://localhost:5000 in your browser.
2. features.parquet must exist in data/:
       python build_features.py

Functions
---------
evaluate          Compute MAE, RMSE, and R² for a set of predictions.
train_and_log     Load data, split, train one model, log everything to MLflow.
"""

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME     = "DemandCast"

DATA_PATH = Path(__file__).parent.parent / "data" / "features.parquet"

# Temporal split cutoffs — January 2025 dataset
# The data runs Jan 7 – Feb 1, 2025.
# A random split is NOT appropriate here because our lag features create
# sequential dependencies between rows, and the model must only ever learn
# from the past to predict the future (mimicking real deployment conditions).
# Using a random split would allow future rows to leak into training,
# making validation metrics artificially optimistic.
#
# Train:      Jan 7  – Jan 21  (~52% of data, ~2 full weeks)
# Validation: Jan 22 – Jan 28  (~33% of data, 1 full week)
# Test:       Jan 29 – Feb 1   (~15% of data, sealed until final evaluation)
VAL_CUTOFF  = "2025-01-22"
TEST_CUTOFF = "2025-01-29"

DATETIME_COL = "pickup_hour"
TARGET       = "demand"

FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "is_rush_hour",
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_lag_168h",
]


# ---------------------------------------------------------------------------
# evaluate() — pre-built, use as-is
# ---------------------------------------------------------------------------

def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R² for a set of predictions.

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth demand values.
    y_pred : np.ndarray
        Model predictions, same length as y_true.

    Returns
    -------
    dict[str, float]
        Keys: 'mae', 'rmse', 'r2'. Values rounded to 4 decimal places.
    """
    return {
        "mae":  round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(root_mean_squared_error(y_true, y_pred), 4),
        "r2":   round(r2_score(y_true, y_pred), 4),
    }


# ---------------------------------------------------------------------------
# train_and_log()
# ---------------------------------------------------------------------------

def train_and_log(
    model: Any,
    run_name: str,
    params: dict,
) -> str:
    """Train one regression model and log everything to MLflow.

    Steps
    -----
    1. Load data/features.parquet
    2. Apply temporal train/val/test split (test set is sealed — not used here)
    3. Separate features (X) and target (y) for train and val
    4. Fit the model on the training set
    5. Evaluate on the validation set using evaluate()
    6. Log params, val metrics, and model artifact to MLflow
    7. Print a summary line and return the MLflow run ID

    Parameters
    ----------
    model : sklearn estimator
        An unfitted sklearn-compatible regression model.
    run_name : str
        Human-readable label shown in the MLflow UI (snake_case).
    params : dict
        Parameters to log. Must include at minimum {"model": type(model).__name__}.

    Returns
    -------
    str
        The MLflow run ID.

    Raises
    ------
    FileNotFoundError
        If data/features.parquet does not exist.
    mlflow.exceptions.MlflowException
        If the MLflow server is not reachable at MLFLOW_TRACKING_URI.
    """
    # --- 1. Load data ---
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found at {DATA_PATH}. Run build_features.py first."
        )
    df = pd.read_parquet(DATA_PATH)

    # --- 2. Temporal split ---
    # Sort ascending to be safe (data should already be sorted)
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)

    train = df[df[DATETIME_COL] < VAL_CUTOFF]
    val   = df[(df[DATETIME_COL] >= VAL_CUTOFF) & (df[DATETIME_COL] < TEST_CUTOFF)]
    # Test set is defined but intentionally not used until final evaluation
    # test = df[df[DATETIME_COL] >= TEST_CUTOFF]

    # Sanity check: confirm no temporal leakage between splits
    assert train[DATETIME_COL].max() < pd.Timestamp(VAL_CUTOFF), \
        "Split error: training data bleeds into validation window."
    assert val[DATETIME_COL].min() >= pd.Timestamp(VAL_CUTOFF), \
        "Split error: validation window starts too early."

    # --- 3. Separate features and target ---
    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_val,   y_val   = val[FEATURE_COLS],   val[TARGET]

    # --- 4–7. MLflow run ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        # Log the feature list alongside params so every run is self-documenting
        mlflow.log_params({**params, "features": str(FEATURE_COLS)})

        # Fit
        model.fit(X_train, y_train)

        # Evaluate on validation set
        val_preds   = model.predict(X_val)
        val_metrics = evaluate(y_val, val_preds)

        # Log metrics with consistent "val_" prefix for MLflow comparison view
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        # Log the fitted model artifact
        mlflow.sklearn.log_model(model, "model")

        print(
            f"[{run_name}]  "
            f"val_mae={val_metrics['mae']:.2f}  "
            f"val_rmse={val_metrics['rmse']:.2f}  "
            f"val_r2={val_metrics['r2']:.3f}"
        )
        return run.info.run_id


# ---------------------------------------------------------------------------
# Main — train all three models
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- Model 1: Linear Regression baseline ---
    # Always start with the simplest possible model. It sets the performance
    # floor and reveals whether the features carry any signal at all before
    # committing to more expensive models.
    run_id_lr = train_and_log(
        model=LinearRegression(),
        run_name="linear_regression_baseline",
        params={"model": "LinearRegression"},
    )

    # --- Model 2: Random Forest ---
    # Random Forest handles non-linear interactions between temporal features
    # (e.g., rush hour on weekdays vs. weekends) without requiring explicit
    # feature crosses. 100 trees balances variance reduction with training time.
    run_id_rf = train_and_log(
        model=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        run_name="random_forest_100est_depth10",
        params={
            "model":        "RandomForestRegressor",
            "n_estimators": 100,
            "max_depth":    10,
            "random_state": 42,
        },
    )

    # --- Model 3: Gradient Boosting Regressor ---
    # Chosen because boosting builds trees sequentially, each correcting the
    # residuals of the last. For demand data with recurring temporal patterns,
    # this tends to outperform a single-pass ensemble like Random Forest.
    # Conservative learning rate (0.05) reduces overfitting risk on a small
    # training window (~2 weeks of hourly data).
    run_id_gbr = train_and_log(
        model=GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
        ),
        run_name="gradient_boosting_200est_lr005",
        params={
            "model":         "GradientBoostingRegressor",
            "n_estimators":  200,
            "learning_rate": 0.05,
            "max_depth":     4,
            "random_state":  42,
        },
    )

    print("\nAll runs complete.")
    print(f"  LR  run_id: {run_id_lr}")
    print(f"  RF  run_id: {run_id_rf}")
    print(f"  GBR run_id: {run_id_gbr}")
    print("\nOpen http://localhost:5000 to compare runs in the MLflow UI.")
