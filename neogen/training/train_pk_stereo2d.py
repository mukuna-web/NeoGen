"""Train stereo_2d_v1 PK regressors for Week 4 alternates."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from neogen.chem.standardize import molecular_identity_from_input
from neogen.chem.stereo import resolve_stereoisomer
from neogen.ml.feature_schemas.stereo_2d_v1 import SCHEMA_VERSION, compute as compute_stereo_2d

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "models" / "stereo_2d_v1"
SEED = 42


@dataclass(frozen=True)
class TrainingResult:
    endpoint: str
    model_path: str
    metadata_path: str
    n_rows: int
    train_r2: float
    test_r2: float
    test_mae: float
    training_hash: str


def stereo_features_from_smiles(smiles: str) -> np.ndarray | None:
    if Chem.MolFromSmiles(smiles) is None:
        return None
    try:
        _parent, identity, _mol = molecular_identity_from_input(smiles, "isomeric_smiles")
        isomer = resolve_stereoisomer(identity)
        return compute_stereo_2d(isomer)
    except Exception:
        return None


def load_tdc_rows(paths: Iterable[Path]) -> tuple[np.ndarray, np.ndarray, str]:
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    snapshot = hashlib.sha256()
    for path in paths:
        raw = path.read_bytes()
        snapshot.update(path.name.encode())
        snapshot.update(raw)
        for row in json.loads(raw):
            smiles = str(row.get("smiles", ""))
            value = row.get("value")
            if value is None:
                continue
            features = stereo_features_from_smiles(smiles)
            if features is None:
                continue
            x_rows.append(features)
            y_rows.append(float(value))
    if not x_rows:
        raise ValueError("no featurizable training rows")
    return np.vstack(x_rows), np.asarray(y_rows, dtype=np.float64), snapshot.hexdigest()[:32]


def train_endpoint(endpoint: str, dataset_paths: Iterable[Path]) -> TrainingResult:
    x, y, snapshot_hash = load_tdc_rows(dataset_paths)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    out_dir = MODEL_ROOT / endpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pkl"
    metadata_path = out_dir / "metadata.json"
    joblib.dump(model, model_path)
    metadata = {
        "endpoint": endpoint,
        "feature_schema_version": SCHEMA_VERSION,
        "feature_dimensions": int(x.shape[1]),
        "random_state": SEED,
        "training_hash": snapshot_hash,
        "n_rows": int(x.shape[0]),
        "augmentation": "TDC isomeric SMILES consumed as given; flat SMILES represented as unknown/achiral stereo records.",
        "metrics": {
            "train_r2": float(r2_score(y_train, train_pred)),
            "test_r2": float(r2_score(y_test, test_pred)),
            "test_mae": float(mean_absolute_error(y_test, test_pred)),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return TrainingResult(
        endpoint=endpoint,
        model_path=str(model_path.relative_to(REPO_ROOT)),
        metadata_path=str(metadata_path.relative_to(REPO_ROOT)),
        n_rows=int(x.shape[0]),
        train_r2=metadata["metrics"]["train_r2"],
        test_r2=metadata["metrics"]["test_r2"],
        test_mae=metadata["metrics"]["test_mae"],
        training_hash=snapshot_hash,
    )


def register_alternates(results: Iterable[TrainingResult]) -> None:
    config_path = REPO_ROOT / "models" / "ensemble_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    alternates = dict(config.get("alternates", {}))
    for result in results:
        alternates[f"stereo_2d_v1_{result.endpoint}"] = {
            "endpoint": result.endpoint,
            "model_path": result.model_path,
            "metadata_path": result.metadata_path,
            "feature_schema_version": SCHEMA_VERSION,
            "registered_as": "alternate",
        }
    config["alternates"] = alternates
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def main() -> None:
    tdc = REPO_ROOT / "data" / "tdc"
    results = [
        train_endpoint("half_life", [tdc / "Half_Life_Obach.json"]),
        train_endpoint("clearance", [tdc / "Clearance_Microsome_AZ.json", tdc / "Clearance_Hepatocyte_AZ.json"]),
    ]
    register_alternates(results)
    print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
