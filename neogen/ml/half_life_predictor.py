"""ML half-life predictor — trained on Obach + DrugBank all sources.

Loads ``models/adme_pipeline/half_life_predictor_v2_all.joblib`` (1044-dim
features) and exposes :func:`predict_half_life_h`. Trained on
1,846 drugs merged from TDC Half_Life_Obach and DrugBank all_drugs.

Test-set metrics (20% holdout, N=370):
  R2 = 0.209, RMSE = 0.522 (log10 scale), 52% within 2-fold.

Note: This module is NOT wired into DrugFactory because half-life
is a derived parameter (t1/2 = 0.693 * Vd / CL). It is provided
for future use as a cross-check or calibration target.
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
    / "half_life_predictor_v2_all.joblib"
)

_HL_MIN = 0.1    # hours
_HL_MAX = 500.0  # hours


def predict_half_life_h(smiles: str) -> Optional[float]:
    """Predict elimination half-life (hours) from SMILES.

    Returns ``None`` when the model file is missing, RDKit is
    unavailable, or the SMILES does not parse. Otherwise returns
    a float in [0.1, 500] hours.

    Training target: ``log10(t½)``. Wrapper exponentiates and
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
            "half_life_predictor inference failed for SMILES %r: %s",
            smiles, exc,
        )
        return None

    try:
        hl = float(10.0 ** float(raw))
    except (OverflowError, ValueError):
        return None

    return max(_HL_MIN, min(hl, _HL_MAX))


def is_model_available() -> bool:
    """Return True when the ML half-life model is loaded and usable."""
    return load_joblib_lazy(_MODEL_PATH) is not None
