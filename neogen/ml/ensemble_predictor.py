"""Ensemble predictor — bootstrap uncertainty quantification for ADME models.

Wraps N XGBoost models (trained with different random seeds and bootstrap
samples) to produce calibrated confidence intervals for every prediction.

For each prediction, all N models are run and the result is:
  - mean: average prediction across the ensemble
  - std: standard deviation (uncertainty estimate)
  - confidence: 1 / (1 + CV), where CV = std / |mean|

A confidence score near 1.0 means all models agree (low uncertainty);
a score near 0.0 means high disagreement (unreliable prediction).

Usage::

    predictor = EnsemblePredictor(
        model_dir=Path("models/adme_pipeline"),
        model_prefix="fu_predictor_v7_all",
        n_models=10,
        feature_fn=compute_features_1044,
        transform_fn=lambda raw: 10.0 ** raw,   # log10 -> linear
        clip_range=(0.001, 1.0),
    )
    result = predictor.predict(smiles)
    if result is not None:
        print(f"{result.mean:.3f} +/- {result.std:.3f}  "
              f"(confidence={result.confidence:.2f})")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from neogen.ml._features import load_joblib_lazy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnsemblePrediction:
    """Result of an ensemble prediction with uncertainty quantification.

    Attributes:
        mean: Mean prediction across the ensemble (in natural units).
        std: Standard deviation across the ensemble (in natural units).
        confidence: Reliability score in [0, 1]. Higher = more reliable.
            Computed as 1 / (1 + CV), where CV = std / |mean|.
        n_models: Number of models that successfully produced a prediction.
    """

    mean: float
    std: float
    confidence: float
    n_models: int


class EnsemblePredictor:
    """Generic bootstrap ensemble wrapper for ADME XGBoost models.

    Loads N model files named ``{prefix}_seed{i}.joblib`` from the given
    directory. On prediction, runs all loaded models and aggregates.

    Parameters:
        model_dir: Directory containing the model files.
        model_prefix: Common prefix for ensemble model files.
        n_models: Number of ensemble members (seeds 0..n_models-1).
        feature_fn: Function that takes SMILES -> Optional[np.ndarray].
        transform_fn: Function to convert raw model output to natural
            units (e.g., ``lambda x: 10**x`` for log10-trained models).
            Applied to each individual model prediction before aggregation.
        clip_range: Optional (min, max) to clip the final mean prediction.
    """

    def __init__(
        self,
        model_dir: Path,
        model_prefix: str,
        n_models: int,
        feature_fn: Callable[[str], Optional[np.ndarray]],
        transform_fn: Callable[[float], float],
        clip_range: Optional[Tuple[float, float]] = None,
    ):
        self._model_dir = Path(model_dir)
        self._model_prefix = model_prefix
        self._n_models = n_models
        self._feature_fn = feature_fn
        self._transform_fn = transform_fn
        self._clip_range = clip_range

        # Build model paths: {prefix}_seed0.joblib .. {prefix}_seed{n-1}.joblib
        self._model_paths: List[Path] = [
            self._model_dir / f"{model_prefix}_seed{i}.joblib"
            for i in range(n_models)
        ]

    @property
    def available_model_count(self) -> int:
        """Number of ensemble models that exist on disk."""
        return sum(1 for p in self._model_paths if p.exists())

    def is_available(self) -> bool:
        """Return True if at least 2 ensemble models are loadable."""
        return self.available_model_count >= 2

    def predict(self, smiles: str) -> Optional[EnsemblePrediction]:
        """Run ensemble prediction for a SMILES string.

        Returns ``None`` if features cannot be computed or fewer than
        2 models produce valid predictions.
        """
        features = self._feature_fn(smiles)
        if features is None:
            return None

        X = features.reshape(1, -1)
        predictions: List[float] = []

        for path in self._model_paths:
            model = load_joblib_lazy(path)
            if model is None:
                continue
            try:
                raw = model.predict(X)[0]
                transformed = self._transform_fn(float(raw))
                if np.isfinite(transformed):
                    predictions.append(transformed)
            except Exception as exc:
                logger.debug(
                    "Ensemble member %s failed for SMILES %r: %s",
                    path.name, smiles, exc,
                )
                continue

        if len(predictions) < 2:
            return None

        arr = np.array(predictions)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1))  # sample std

        # Coefficient of variation -> confidence
        abs_mean = abs(mean_val)
        if abs_mean > 1e-10:
            cv = std_val / abs_mean
        else:
            cv = float("inf") if std_val > 0 else 0.0
        confidence = 1.0 / (1.0 + cv)

        # Clip mean to physiological range
        if self._clip_range is not None:
            lo, hi = self._clip_range
            mean_val = max(lo, min(mean_val, hi))

        return EnsemblePrediction(
            mean=mean_val,
            std=std_val,
            confidence=confidence,
            n_models=len(predictions),
        )


# ════════════════════════════════════════════════════════════════
# PRE-CONFIGURED ENSEMBLE PREDICTORS
# ════════════════════════════════════════════════════════════════

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "adme_pipeline"


def _log10_transform(raw: float) -> float:
    """Convert log10-space model output to natural units."""
    return 10.0 ** raw


def get_fu_ensemble() -> EnsemblePredictor:
    """Pre-configured ensemble predictor for fraction unbound (fu)."""
    from neogen.ml._features import compute_features_1044

    return EnsemblePredictor(
        model_dir=_MODELS_DIR,
        model_prefix="fu_predictor_v7_all",
        n_models=10,
        feature_fn=compute_features_1044,
        transform_fn=_log10_transform,
        clip_range=(0.001, 1.0),
    )


def get_clint_ensemble() -> EnsemblePredictor:
    """Pre-configured ensemble predictor for CLint (uL/min/mg)."""
    from neogen.ml._features import compute_features_1044

    return EnsemblePredictor(
        model_dir=_MODELS_DIR,
        model_prefix="clint_predictor_v4_all",
        n_models=10,
        feature_fn=compute_features_1044,
        transform_fn=_log10_transform,
        clip_range=(0.1, 10_000.0),
    )


def get_vd_ensemble() -> EnsemblePredictor:
    """Pre-configured ensemble predictor for Vdss (L/kg)."""
    from neogen.ml._features import compute_features_1044

    return EnsemblePredictor(
        model_dir=_MODELS_DIR,
        model_prefix="vd_predictor_v2_all",
        n_models=10,
        feature_fn=compute_features_1044,
        transform_fn=_log10_transform,
        clip_range=(0.01, 5000.0),
    )


def get_half_life_ensemble() -> EnsemblePredictor:
    """Pre-configured ensemble predictor for half-life (hours)."""
    from neogen.ml._features import compute_features_1044

    return EnsemblePredictor(
        model_dir=_MODELS_DIR,
        model_prefix="half_life_predictor_v2_all",
        n_models=10,
        feature_fn=compute_features_1044,
        transform_fn=_log10_transform,
        clip_range=(0.1, 500.0),
    )
