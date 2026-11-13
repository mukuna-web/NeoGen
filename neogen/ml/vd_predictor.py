"""ML Vdss predictor — trained on Lombardo + DrugBank all sources.

Loads ``models/adme_pipeline/vd_predictor_v2_all.joblib`` (1044-dim
features) and exposes :func:`predict_vdss_L_per_kg`. Trained on
2,047 drugs merged from TDC VDss_Lombardo and DrugBank all_drugs.

Test-set metrics (20% holdout, N=410):
  R2 = 0.426, RMSE = 0.525 (log10 scale), 55% within 2-fold.
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
    / "vd_predictor_v2_all.joblib"
)

_VD_MIN = 0.01    # L/kg
_VD_MAX = 5000.0  # L/kg


def predict_vdss_L_per_kg(smiles: str) -> Optional[float]:
    """Predict steady-state volume of distribution (L/kg) from SMILES.

    Returns ``None`` when the model file is missing, RDKit is
    unavailable, or the SMILES does not parse. Otherwise returns
    a float in [0.01, 5000] L/kg.

    Training target: ``log10(Vdss)``. Wrapper exponentiates and
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
            "vd_predictor inference failed for SMILES %r: %s",
            smiles, exc,
        )
        return None

    try:
        vdss = float(10.0 ** float(raw))
    except (OverflowError, ValueError):
        return None

    return max(_VD_MIN, min(vdss, _VD_MAX))


def is_model_available() -> bool:
    """Return True when the ML Vd model is loaded and usable."""
    return load_joblib_lazy(_MODEL_PATH) is not None
