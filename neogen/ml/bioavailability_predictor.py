"""ML bioavailability predictor v1 — trained on ChEMBL + DrugBank + TDC.

Loads ``models/adme_pipeline/bioavailability_predictor_v1.joblib``
(1044-dim features) and exposes :func:`predict_bioavailability`.
Trained on 978 unique drugs merged from ChEMBL F, DrugBank
bioavailability, and TDC Bioavailability_Ma (binary, mapped to
approximate continuous).

Test-set metrics (20% holdout, N=196):
  R2 = 0.194, RMSE = 0.356 (log10 scale), 71% within 2-fold.

Note: This module is NOT wired into DrugFactory yet. It is
available for future use in oral absorption modelling.
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
    / "bioavailability_predictor_v1.joblib"
)

_F_MIN = 0.01
_F_MAX = 1.0


def predict_bioavailability(smiles: str) -> Optional[float]:
    """Predict oral bioavailability (F, fraction 0-1) from SMILES.

    Returns ``None`` when the model file is missing, RDKit is
    unavailable, or the SMILES does not parse. Otherwise returns
    a float in [0.01, 1.0].

    Training target: ``log10(F)``. Wrapper exponentiates and
    clips to the physiological range.
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
            "bioavailability_predictor inference failed for SMILES %r: %s",
            smiles, exc,
        )
        return None

    try:
        f_val = float(10.0 ** float(raw))
    except (OverflowError, ValueError):
        return None

    return max(_F_MIN, min(f_val, _F_MAX))


def is_model_available() -> bool:
    """Return True when the ML bioavailability model is loaded and usable."""
    return load_joblib_lazy(_MODEL_PATH) is not None
