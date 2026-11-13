"""ML Caco-2 apparent permeability predictor v1 — trained on Wang dataset.

Loads ``models/adme_pipeline/caco2_predictor_v1.joblib`` (1044-dim
features) and exposes :func:`predict_log_papp`. Trained on 906
unique drugs from the caco2_wang dataset.

Test-set metrics (20% holdout, N=182):
  R2 = 0.726, RMSE = 0.380 (log10 scale), 63% within 2-fold.

Note: This module is NOT wired into DrugFactory yet. It is
available for future use in permeability-limited absorption
modelling as a complement to the QSAR Peff estimator.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from neogen.ml._features import compute_features_1044, load_joblib_lazy

logger = logging.getLogger(__name__)


_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "adme_pipeline"
    / "caco2_predictor_v1.joblib"
)


def predict_log_papp(smiles: str) -> Optional[float]:
    """Predict Caco-2 apparent permeability log10(Papp) from SMILES.

    Returns ``None`` when the model file is missing, RDKit is
    unavailable, or the SMILES does not parse. Otherwise returns
    a float (log10 cm/s scale, typically -7 to -4).

    Training target is log10(Papp) directly (no additional
    log transform in the wrapper).
    """
    model = load_joblib_lazy(_MODEL_PATH)
    if model is None:
        return None

    features = compute_features_1044(smiles)
    if features is None:
        return None

    try:
        raw = model.predict(features.reshape(1, -1))[0]
    except Exception as exc:
        logger.warning(
            "caco2_predictor inference failed for SMILES %r: %s",
            smiles, exc,
        )
        return None

    return float(raw)


def predict_papp_cm_s(smiles: str) -> Optional[float]:
    """Predict Caco-2 Papp in cm/s (linear scale) from SMILES.

    Convenience wrapper that exponentiates the log10 prediction.
    """
    log_papp = predict_log_papp(smiles)
    if log_papp is None:
        return None
    try:
        return float(10.0 ** log_papp)
    except (OverflowError, ValueError):
        return None


def is_model_available() -> bool:
    """Return True when the ML Caco-2 model is loaded and usable."""
    return load_joblib_lazy(_MODEL_PATH) is not None
