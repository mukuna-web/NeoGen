"""
Baseline Models: Random Forest & XGBoost
==========================================
These are our "go-kart" models — simple but effective.
They set the performance floor that our neural network must beat.

Random Forest: Builds many decision trees, each seeing a random
subset of features. Final prediction = average of all trees.
Works well on tabular data, hard to overfit.

XGBoost: Builds trees sequentially, each one correcting the errors
of the previous one. Usually beats RF on tabular data by 5-10%.

Both models predict log10(PK_value) because PK values are
log-normally distributed (we saw this in the distribution plots).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def load_split_features(
    splits_dir: str = "data/splits",
    features_path: str = "data/processed/features.npz",
    targets_path: str = "data/processed/targets.parquet",
) -> dict:
    """
    Load train/val/test splits with their corresponding features.

    Returns dict with keys 'train', 'val', 'test', each containing:
    - X: feature matrix (fingerprints + descriptors)
    - y: log-transformed PK values
    - meta: DataFrame with molecule IDs, PK types, raw values
    """
    # Load features and targets
    features_data = np.load(features_path)
    all_features = features_data["features"]
    all_targets = pd.read_parquet(targets_path).reset_index(drop=True)

    # Create a unique key for matching records
    all_targets["_key"] = (
        all_targets["molecule_chembl_id"] + "|"
        + all_targets["pk_param"] + "|"
        + all_targets["standard_value"].astype(str)
    )
    key_to_idx = {k: i for i, k in enumerate(all_targets["_key"])}

    result = {}
    for split_name in ["train", "val", "test"]:
        split_df = pd.read_parquet(f"{splits_dir}/{split_name}.parquet")
        split_df["_key"] = (
            split_df["molecule_chembl_id"] + "|"
            + split_df["pk_param"] + "|"
            + split_df["standard_value"].astype(str)
        )

        # Look up feature indices
        indices = [key_to_idx[k] for k in split_df["_key"] if k in key_to_idx]
        matched = split_df[split_df["_key"].isin(key_to_idx)].copy()

        X = all_features[indices]
        y = matched["log_value"].values

        result[split_name] = {
            "X": X,
            "y": y,
            "meta": matched,
        }
        logger.info(
            "%s: %d samples, features shape %s",
            split_name, len(y), X.shape,
        )

    return result


def train_rf_per_param(
    data: dict,
    n_estimators: int = 200,
    max_depth: int = 20,
    random_seed: int = 42,
) -> dict:
    """
    Train a separate Random Forest for each PK parameter.

    Why separate models? Each PK parameter has different distributions
    and relationships with molecular features. A model specialized for
    Cmax will perform better than one model trying to predict everything.

    Args:
        data: Output from load_split_features()
        n_estimators: Number of trees in the forest
        max_depth: Max depth of each tree (prevents overfitting)

    Returns:
        Dict with model objects and evaluation metrics
    """
    pk_params = data["train"]["meta"]["pk_param"].unique()
    results = {}

    for param in pk_params:
        logger.info("Training RF for %s...", param)

        # Get data for this PK parameter
        train_mask = data["train"]["meta"]["pk_param"] == param
        val_mask = data["val"]["meta"]["pk_param"] == param
        test_mask = data["test"]["meta"]["pk_param"] == param

        X_train = data["train"]["X"][train_mask]
        y_train = data["train"]["y"][train_mask]
        X_val = data["val"]["X"][val_mask]
        y_val = data["val"]["y"][val_mask]
        X_test = data["test"]["X"][test_mask]
        y_test = data["test"]["y"][test_mask]

        if len(X_train) < 10:
            logger.warning("Skipping %s — too few training samples", param)
            continue

        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_seed,
            n_jobs=-1,  # Use all CPU cores
        )
        rf.fit(X_train, y_train)

        # Evaluate on validation and test sets
        y_val_pred = rf.predict(X_val)
        y_test_pred = rf.predict(X_test)

        metrics = {
            "param": param,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "val_r2": r2_score(y_val, y_val_pred),
            "val_mae": mean_absolute_error(y_val, y_val_pred),
            "val_rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
            "test_r2": r2_score(y_test, y_test_pred),
            "test_mae": mean_absolute_error(y_test, y_test_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        }

        results[param] = {
            "model": rf,
            "metrics": metrics,
            "test_predictions": y_test_pred,
            "test_actual": y_test,
        }

        logger.info(
            "  %s: val_R²=%.3f, test_R²=%.3f, test_MAE=%.3f",
            param, metrics["val_r2"], metrics["test_r2"], metrics["test_mae"],
        )

    return results


def train_xgboost_per_param(
    data: dict,
    n_estimators: int = 300,
    max_depth: int = 8,
    learning_rate: float = 0.1,
    random_seed: int = 42,
) -> dict:
    """
    Train XGBoost for each PK parameter.

    XGBoost typically beats Random Forest by 5-10% on tabular data.
    It builds trees sequentially — each new tree focuses on the
    mistakes of the previous ones (gradient boosting).

    Returns dict with models and metrics (same structure as RF).
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("XGBoost not installed. Run: pip install xgboost")
        return {}

    pk_params = data["train"]["meta"]["pk_param"].unique()
    results = {}

    for param in pk_params:
        logger.info("Training XGBoost for %s...", param)

        train_mask = data["train"]["meta"]["pk_param"] == param
        val_mask = data["val"]["meta"]["pk_param"] == param
        test_mask = data["test"]["meta"]["pk_param"] == param

        X_train = data["train"]["X"][train_mask]
        y_train = data["train"]["y"][train_mask]
        X_val = data["val"]["X"][val_mask]
        y_val = data["val"]["y"][val_mask]
        X_test = data["test"]["X"][test_mask]
        y_test = data["test"]["y"][test_mask]

        if len(X_train) < 10:
            continue

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_seed,
            tree_method="hist",  # Fast histogram-based method
            n_jobs=-1,
            verbosity=0,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        metrics = {
            "param": param,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "val_r2": r2_score(y_val, y_val_pred),
            "val_mae": mean_absolute_error(y_val, y_val_pred),
            "val_rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
            "test_r2": r2_score(y_test, y_test_pred),
            "test_mae": mean_absolute_error(y_test, y_test_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        }

        results[param] = {
            "model": model,
            "metrics": metrics,
            "test_predictions": y_test_pred,
            "test_actual": y_test,
        }

        logger.info(
            "  %s: val_R²=%.3f, test_R²=%.3f, test_MAE=%.3f",
            param, metrics["val_r2"], metrics["test_r2"], metrics["test_mae"],
        )

    return results


def print_comparison_table(rf_results: dict, xgb_results: dict) -> pd.DataFrame:
    """Print a side-by-side comparison of RF vs XGBoost."""
    rows = []
    for param in rf_results:
        rf_m = rf_results[param]["metrics"]
        xgb_m = xgb_results.get(param, {}).get("metrics", {})

        rows.append({
            "PK Param": param,
            "RF R²": f"{rf_m['test_r2']:.3f}",
            "RF MAE": f"{rf_m['test_mae']:.3f}",
            "XGB R²": f"{xgb_m.get('test_r2', 0):.3f}" if xgb_m else "N/A",
            "XGB MAE": f"{xgb_m.get('test_mae', 0):.3f}" if xgb_m else "N/A",
            "Winner": "XGB" if xgb_m and xgb_m.get("test_r2", 0) > rf_m["test_r2"] else "RF",
            "N_train": rf_m["n_train"],
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("BASELINE MODEL COMPARISON (Test Set)")
    print("=" * 80)
    print(df.to_string(index=False))
    return df


def save_results(
    rf_results: dict,
    xgb_results: dict,
    output_dir: str = "reports/benchmarks",
) -> None:
    """Save benchmark results to JSON for later comparison with GNN."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = {}
    for param in rf_results:
        benchmarks[param] = {
            "rf": rf_results[param]["metrics"],
            "xgboost": xgb_results.get(param, {}).get("metrics", {}),
        }

    with open(output_dir / "baseline_benchmarks.json", "w") as f:
        json.dump(benchmarks, f, indent=2)

    logger.info("Saved benchmarks to %s", output_dir / "baseline_benchmarks.json")


# ── Main entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    print("=" * 60)
    print("NeoGen — Baseline Model Training")
    print("=" * 60)

    # Step 1: Load data
    print("\nStep 1: Loading split data with features...")
    data = load_split_features()

    # Step 2: Train Random Forest
    print("\n" + "=" * 60)
    print("Step 2: Training Random Forest (200 trees, max_depth=20)")
    print("=" * 60)
    rf_results = train_rf_per_param(data)

    # Step 3: Train XGBoost
    print("\n" + "=" * 60)
    print("Step 3: Training XGBoost (300 trees, lr=0.1)")
    print("=" * 60)
    xgb_results = train_xgboost_per_param(data)

    # Step 4: Compare
    comparison = print_comparison_table(rf_results, xgb_results)

    # Step 5: Save results
    save_results(rf_results, xgb_results)

    print("\nBaseline training complete!")
