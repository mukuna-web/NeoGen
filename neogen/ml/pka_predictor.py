"""ML pKa predictors (acidic + basic).

Loads ``pka_acidic_predictor_v1.joblib`` and
``pka_basic_predictor_v1.joblib`` (2083-dim features) and exposes
:func:`predict_pka_acidic` and :func:`predict_pka_basic`.

Why this matters for Tier 1
---------------------------
The V1 engine uses pKa in two places:

  * :func:`neogen.engine.drug_factory._classify_ionization`
    classifies each drug as ``ACID``, ``BASE``, ``ZWITTERION``, or
    ``NEUTRAL`` from the presence of acidic / basic pKa values.
    The classification drives Rodgers-Rowland Kp math.

  * :class:`RenalClearanceEngine._reabsorption_fraction` uses the
    pKa directly in a Henderson-Hasselbalch calculation for
    urine reabsorption.

Without the ML predictors, novel drugs have no pKa values and
default to ``NEUTRAL``. That forces every base / acid / zwitterion
into the neutral Kp branch, which misses the dominant partition
mechanism for ~70% of drugs (bases ion-trap into acidic
phospholipids; acids bind albumin with pH-dependent fraction).

Model behavior
--------------
Both predictors are regressors trained on IUPAC pKa values. The
wrappers filter out non-ionizable molecules by returning ``None``
when the predicted value is outside a physiological band
([0, 14]) — the training set had a mix of ionizable and
neutral molecules, and the neutral ones end up at sentinel values
like -5 or 20 that should not be interpreted as real pKa.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from neogen.ml._features import compute_features_2083, load_joblib_lazy

logger = logging.getLogger(__name__)


_ACIDIC_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "adme_pipeline"
    / "pka_acidic_predictor_v1.joblib"
)
_BASIC_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "adme_pipeline"
    / "pka_basic_predictor_v1.joblib"
)

# Physiological pKa band. Predictions outside this band indicate
# the drug has no meaningful ionization in that direction (i.e.,
# the model is extrapolating to a non-ionizable compound).
_PKA_MIN = 0.0
_PKA_MAX = 14.0


def _predict_one(smiles: str, model_path: Path) -> Optional[float]:
    model = load_joblib_lazy(model_path)
    if model is None:
        return None

    features = compute_features_2083(smiles)
    if features is None:
        return None

    try:
        raw = model.predict(features.reshape(1, -1))[0]
    except Exception as exc:
        logger.warning(
            "pka_predictor inference failed for %s on SMILES %r: %s",
            model_path.name, smiles, exc,
        )
        return None

    try:
        pka = float(raw)
    except (TypeError, ValueError):
        return None

    if not (_PKA_MIN <= pka <= _PKA_MAX):
        return None  # sentinel/out-of-range → treat as no ionization
    return pka


def predict_pka_acidic(smiles: str) -> Optional[float]:
    """Predict the most acidic pKa. ``None`` if the drug is not an acid."""
    return _predict_one(smiles, _ACIDIC_MODEL_PATH)


def predict_pka_basic(smiles: str) -> Optional[float]:
    """Predict the most basic pKa. ``None`` if the drug is not a base."""
    return _predict_one(smiles, _BASIC_MODEL_PATH)


def predict_pkas(smiles: str) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(pka_acidic, pka_basic)`` — either or both may be ``None``."""
    return predict_pka_acidic(smiles), predict_pka_basic(smiles)


def is_model_available() -> bool:
    """Return True when at least one pKa model is usable."""
    return (
        load_joblib_lazy(_ACIDIC_MODEL_PATH) is not None
        or load_joblib_lazy(_BASIC_MODEL_PATH) is not None
    )
