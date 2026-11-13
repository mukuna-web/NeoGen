"""ML aqueous solubility predictor v1 — trained on AqSolDB.

Loads ``models/adme_pipeline/solubility_predictor_v1.joblib``
(1044-dim features) and exposes :func:`predict_log_solubility`.
Trained on 9,980 unique drugs from TDC Solubility_AqSolDB.

Test-set metrics (20% holdout, N=1996):
  R2 = 0.810, RMSE = 1.019 (log scale), 39% within 2-fold.

Note: This module is NOT wired into DrugFactory yet. It is
available for future use in dissolution / absorption modelling.
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
    / "solubility_predictor_v1.joblib"
)


def predict_log_solubility(smiles: str) -> Optional[float]:
    """Predict aqueous log solubility (log mol/L) from SMILES.

    Returns ``None`` when the model file is missing, RDKit is
    unavailable, or the SMILES does not parse. Otherwise returns
    a float (log mol/L scale, typically -10 to 2).

    Training target is log solubility directly (no additional
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
            "solubility_predictor inference failed for SMILES %r: %s",
            smiles, exc,
        )
        return None

    return float(raw)


def is_model_available() -> bool:
    """Return True when the ML solubility model is loaded and usable."""
    return load_joblib_lazy(_MODEL_PATH) is not None
