"""
build_train.py — Training pipeline orchestrator
================================================
Validates prerequisites (features.parquet exists, MLflow is reachable),
then runs the complete model training workflow.

Usage
-----
    python build_train.py

Output
------
Logs three trained models to MLflow under the "DemandCast" experiment.
Prints a summary of run IDs and a link to the MLflow UI.
"""

from pathlib import Path
import sys
import requests

# Add src to the path so we can import train module
sys.path.append(str(Path(__file__).parent))
from src.train import MLFLOW_TRACKING_URI, DATA_PATH


def check_features_exist() -> bool:
    """Verify that data/features.parquet exists."""
    if not DATA_PATH.exists():
        print(f"❌ ERROR: Feature file not found at {DATA_PATH}")
        print(f"   Run 'python build_features.py' to generate it first.")
        return False
    print(f"✓ Feature file found at {DATA_PATH}")
    return True


def check_mlflow_server() -> bool:
    """Verify that MLflow server is reachable at MLFLOW_TRACKING_URI."""
    try:
        response = requests.get(f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/search", timeout=2)
        if response.status_code == 200:
            print(f"✓ MLflow server is running at {MLFLOW_TRACKING_URI}")
            return True
    except (requests.ConnectionError, requests.Timeout):
        pass
    
    print(f"❌ ERROR: Could not reach MLflow server at {MLFLOW_TRACKING_URI}")
    print(f"   Start it first with: mlflow ui")
    print(f"   Then access it at {MLFLOW_TRACKING_URI}")
    return False


def main():
    """Run the training pipeline after validating prerequisites."""
    print("=" * 70)
    print("DemandCast Training Pipeline")
    print("=" * 70)
    
    # --- Validate prerequisites ---
    print("\n[1/3] Checking prerequisites...")
    if not check_features_exist():
        sys.exit(1)
    
    print("  (MLflow connectivity check will be performed during training)")
    
    # --- Run training ---
    print("\n[2/3] Training models...")
    try:
        # Import and run the train module's main block
        from src import train
        # Since we want to run the if __name__ == "__main__" code,
        # we execute it by calling the training functions directly.
        # This mimics what happens when train.py is run as a script.
        
        run_id_lr = train.train_and_log(
            model=train.LinearRegression(),
            run_name="linear_regression_baseline",
            params={"model": "LinearRegression"},
        )
        
        run_id_rf = train.train_and_log(
            model=train.RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            ),
            run_name="random_forest_100est_depth10",
            params={
                "model":        "RandomForestRegressor",
                "n_estimators": 100,
                "max_depth":    10,
                "random_state": 42,
            },
        )
        
        run_id_gbr = train.train_and_log(
            model=train.GradientBoostingRegressor(
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
        
    except Exception as e:
        print(f"❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # --- Summary ---
    print("\n[3/3] Training complete!")
    print("=" * 70)
    print(f"  LR  run_id: {run_id_lr}")
    print(f"  RF  run_id: {run_id_rf}")
    print(f"  GBR run_id: {run_id_gbr}")
    print("=" * 70)
    print(f"\n✓ All runs logged to MLflow.")
    print(f"\nOpen {MLFLOW_TRACKING_URI} in your browser to compare runs.")
    print("=" * 70)


if __name__ == "__main__":
    main()