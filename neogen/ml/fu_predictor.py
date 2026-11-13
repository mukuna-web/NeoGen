"""ML fu (fraction unbound) predictor.

Loads ``models/adme_pipeline/fu_predictor_v4.joblib`` (1044-dim
features, same layout as the CLint predictor) and exposes
:func:`predict_fu_plasma`. This is the single highest-leverage ML
fallback for novel compounds because ``fu`` feeds:

  * :class:`HepaticClearanceEngine` — well-stirred model uses
    ``fu × CLint / (Qh + fu × CLint)``.
  * :class:`TissuePartitioner` — Rodgers-Rowland Kp math uses
    ``fu_plasma / fu_tissue``.
  * :class:`RenalClearanceEngine` — filtration clearance uses
    ``GFR × fu``.

The legacy logP fallback in ``DrugFactory``
(``_estimate_fu_from_logp``) is a rough Obach 1999 approximation
that only captures the trend from lipophilicity. The ML
predictor's R² ≈ 0.63 on held-out drugs is a meaningful
improvement for compounds outside the literature catalog.
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

# fu is a probability ∈ [0, 1]. Clamp to the physiological band so
# downstream solvers never divide by 0.
_FU_MIN = 0.001
_FU_MAX = 1.0


def predict_fu_plasma(smiles: str) -> Optional[float]:
    """Predict fraction unbound in plasma from SMILES.

    Returns ``None`` when the model file is missing, RDKit is
    unavailable, or the SMILES does not parse. Otherwise returns
    a float ∈ [0.001, 1.0].

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
            "fu_predictor inference failed for SMILES %r: %s",
            smiles, exc,
        )
        return None

    try:
        fu = float(10.0 ** float(raw))
    except (OverflowError, ValueError):
        return None

    return max(_FU_MIN, min(fu, _FU_MAX))


def is_model_available() -> bool:
    """Return True when the ML fu model is loaded and usable."""
    return load_joblib_lazy(_MODEL_PATH) is not None
