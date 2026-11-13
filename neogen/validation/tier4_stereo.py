"""Tier 4 stereochemical directionality validation."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from rdkit import Chem

from neogen.chem.standardize import molecular_identity_from_input
from neogen.chem.stereo import resolve_stereoisomer
from neogen.ml.feature_schemas.stereo_2d_v1 import compute as compute_stereo_2d

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COHORT_PATH = REPO_ROOT / "data" / "training_stereo" / "chiral_cohort_v1.json"


@dataclass(frozen=True)
class ChiralPairResult:
    pair_id: str
    published_direction: str
    predicted_direction: str
    published_ratio: float
    predicted_ratio: float
    direction_correct: bool
    magnitude_within_2x: bool
    r_prediction: float
    s_prediction: float
    citation: str
    confidence: str


@dataclass(frozen=True)
class Tier4Result:
    direction_pass_rate: float
    magnitude_pass_rate: float
    per_pair_breakdown: tuple[ChiralPairResult, ...]
    overall_pass: bool
    direction_threshold: float = 80.0
    magnitude_threshold: float = 60.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction_pass_rate": self.direction_pass_rate,
            "magnitude_pass_rate": self.magnitude_pass_rate,
            "direction_threshold": self.direction_threshold,
            "magnitude_threshold": self.magnitude_threshold,
            "overall_pass": self.overall_pass,
            "per_pair_breakdown": [asdict(row) for row in self.per_pair_breakdown],
        }


def load_chiral_cohort(path: Path | str = DEFAULT_COHORT_PATH) -> list[dict[str, Any]]:
    with Path(path).open() as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        raise ValueError("chiral cohort must be a JSON list")
    return sorted(rows, key=lambda row: row["pair_id"])


def _features_for_smiles(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid isomeric SMILES: {smiles}")
    _parent, identity, _mol = molecular_identity_from_input(smiles, "isomeric_smiles")
    isomer = resolve_stereoisomer(identity)
    features = compute_stereo_2d(isomer)
    if features is None:
        raise ValueError(f"could not compute stereo_2d_v1 features for {smiles}")
    return features


def _predict(model: Any, smiles: str, endpoint: str) -> float:
    features = _features_for_smiles(smiles)
    if hasattr(model, "predict_pair_member"):
        return float(model.predict_pair_member(smiles=smiles, endpoint=endpoint, features=features))
    if callable(model):
        return float(model(smiles=smiles, endpoint=endpoint, features=features))
    if hasattr(model, "predict"):
        pred = model.predict(features.reshape(1, -1))
        return float(np.asarray(pred).reshape(-1)[0])
    raise TypeError("model must be callable or expose predict()")


def _direction(a: float, b: float) -> str:
    if math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
        return "equal"
    return "S>R" if b > a else "R>S"


def _ratio_for_direction(r_value: float, s_value: float, direction: str) -> float:
    denom = max(min(abs(r_value), abs(s_value)), 1e-12)
    if direction == "S>R":
        return abs(s_value) / max(abs(r_value), 1e-12)
    if direction == "R>S":
        return abs(r_value) / max(abs(s_value), 1e-12)
    return max(abs(r_value), abs(s_value)) / denom


def evaluate_chiral_directionality(
    model: Any,
    cohort: Iterable[dict[str, Any]] | None = None,
) -> Tier4Result:
    rows = sorted(list(cohort if cohort is not None else load_chiral_cohort()), key=lambda row: row["pair_id"])
    breakdown: list[ChiralPairResult] = []
    for row in rows:
        endpoint = str(row.get("endpoint", ""))
        r_pred = _predict(model, row["r_isomeric_smiles"], endpoint)
        s_pred = _predict(model, row["s_isomeric_smiles"], endpoint)
        predicted_direction = _direction(r_pred, s_pred)
        published_direction = row["published_direction"]
        direction_correct = predicted_direction == published_direction
        published_ratio = float(row["published_value"])
        predicted_ratio = _ratio_for_direction(r_pred, s_pred, published_direction)
        magnitude_within = 0.5 <= (predicted_ratio / max(published_ratio, 1e-12)) <= 2.0
        breakdown.append(
            ChiralPairResult(
                pair_id=row["pair_id"],
                published_direction=published_direction,
                predicted_direction=predicted_direction,
                published_ratio=published_ratio,
                predicted_ratio=predicted_ratio,
                direction_correct=direction_correct,
                magnitude_within_2x=magnitude_within,
                r_prediction=r_pred,
                s_prediction=s_pred,
                citation=row["citation"],
                confidence=row["confidence"],
            )
        )
    n = max(len(breakdown), 1)
    direction_rate = 100.0 * sum(item.direction_correct for item in breakdown) / n
    magnitude_rate = 100.0 * sum(item.magnitude_within_2x for item in breakdown) / n
    return Tier4Result(
        direction_pass_rate=direction_rate,
        magnitude_pass_rate=magnitude_rate,
        per_pair_breakdown=tuple(breakdown),
        overall_pass=direction_rate >= 80.0 and magnitude_rate >= 60.0,
    )
