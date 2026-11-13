"""ML fu (fraction unbound) predictor v7 — trained on all sources.

Loads ``models/adme_pipeline/fu_predictor_v7_all.joblib`` (1044-dim
features) and exposes :func:`predict_fu`. Trained on 6,029 unique
drugs merged from PPBR_AZ, ppbr_az.tab, ChEMBL, DrugBank fu_training,
and DrugBank all_drugs sources.

Test-set metrics (20% holdout, N=1206):
  R2 = 0.597, RMSE = 0.467 (log10 scale), 57% within 2-fold.
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
    / "fu_predictor_v7_all.joblib"
)

_FU_MIN = 0.001
_FU_MAX = 1.0


def predict_fu(smiles: str) -> Optional[float]:
    """Predict fraction unbound in plasma from SMILES.

    Returns ``None`` when the model file is missing, RDKit is
    unavailable, or the SMILES does not parse. Otherwise returns
    a float in [0.001, 1.0].

    Training target: ``log10(fu)``. Wrapper exponentiates and
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
            "fu_predictor_v6 inference failed for SMILES %r: %s",
            smiles, exc,
        )
        return None

    try:
        fu = float(10.0 ** float(raw))
    except (OverflowError, ValueError):
        return None

    return max(_FU_MIN, min(fu, _FU_MAX))


def is_model_available() -> bool:
    """Return True when the ML fu v6 model is loaded and usable."""
    return load_joblib_lazy(_MODEL_PATH) is not None
