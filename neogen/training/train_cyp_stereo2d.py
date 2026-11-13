"""Train stereo_2d_v1 CYP classifier alternate."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier

from neogen.training.train_pk_stereo2d import REPO_ROOT, SEED, stereo_features_from_smiles

ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4")


@dataclass(frozen=True)
class CYPTrainingResult:
    endpoint: str
    model_path: str
    metadata_path: str
    n_rows: int
    macro_f1: float
    training_hash: str


def main() -> None:
    source_path = REPO_ROOT / "data" / "pharmgkb" / "cyp_training_data.json"
    raw = source_path.read_bytes()
    rows = json.loads(raw)
    x_rows: list[np.ndarray] = []
    labels: list[list[str]] = []
    for row in rows:
        features = stereo_features_from_smiles(str(row.get("smiles", "")))
        cyps = [c for c in row.get("cyps", []) if c in ISOFORMS]
        if features is None or not cyps:
            continue
        x_rows.append(features)
        labels.append(cyps)
    if not x_rows:
        raise ValueError("no featurizable CYP rows")
    x = np.vstack(x_rows)
    mlb = MultiLabelBinarizer(classes=list(ISOFORMS))
    y = mlb.fit_transform(labels)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)
    models: dict[str, XGBClassifier] = {}
    preds: list[np.ndarray] = []
    for idx, isoform in enumerate(ISOFORMS):
        clf = XGBClassifier(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
            eval_metric="logloss",
            n_jobs=1,
        )
        clf.fit(x_train, y_train[:, idx])
        models[isoform] = clf
        preds.append(clf.predict(x_test))
    y_pred = np.vstack(preds).T
    out_dir = REPO_ROOT / "models" / "stereo_2d_v1" / "cyp"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pkl"
    metadata_path = out_dir / "metadata.json"
    joblib.dump({"models": models, "classes": list(ISOFORMS)}, model_path)
    training_hash = hashlib.sha256(source_path.name.encode() + raw).hexdigest()[:32]
    metadata = {
        "endpoint": "cyp",
        "feature_schema_version": "stereo_2d_v1",
        "feature_dimensions": int(x.shape[1]),
        "random_state": SEED,
        "training_hash": training_hash,
        "n_rows": int(x.shape[0]),
        "classes": list(ISOFORMS),
        "metrics": {"macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0))},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    config_path = REPO_ROOT / "models" / "ensemble_config.json"
    config = json.loads(config_path.read_text())
    alternates = dict(config.get("alternates", {}))
    alternates["stereo_2d_v1_cyp"] = {
        "endpoint": "cyp",
        "model_path": str(model_path.relative_to(REPO_ROOT)),
        "metadata_path": str(metadata_path.relative_to(REPO_ROOT)),
        "feature_schema_version": "stereo_2d_v1",
        "registered_as": "alternate",
    }
    config["alternates"] = alternates
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    result = CYPTrainingResult(
        endpoint="cyp",
        model_path=str(model_path.relative_to(REPO_ROOT)),
        metadata_path=str(metadata_path.relative_to(REPO_ROOT)),
        n_rows=int(x.shape[0]),
        macro_f1=metadata["metrics"]["macro_f1"],
        training_hash=training_hash,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
