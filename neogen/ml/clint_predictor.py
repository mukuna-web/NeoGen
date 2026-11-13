"""ML CLint predictor — lean wrapper around the v2 expanded model.

Primary model: ``models/adme_pipeline/clint_predictor_v2_expanded.joblib``
(2083-dim features). Falls back to the legacy 1044-dim model
(``clint_predictor.joblib``) when the v2 file is missing.

Exposes :func:`predict_clint_microsomal`. Shares the feature
computer and XGBoost compatibility shim with the other
``neogen.ml`` predictors via :mod:`neogen.ml._features`.

This is the ML bridge that closes gap GA-9 — novel compounds now
get ML-predicted CLint instead of ``CLint = 0``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from neogen.ml._features import (
    compute_features_1044,
    compute_features_2083,
    load_joblib_lazy,
)

logger = logging.getLogger(__name__)


_MODELS_DIR = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "adme_pipeline"
)

_MODEL_V4_PATH = _MODELS_DIR / "clint_predictor_v4_all.joblib"
_MODEL_V3_PATH = _MODELS_DIR / "clint_predictor_v3_combined.joblib"
_MODEL_V2_PATH = _MODELS_DIR / "clint_predictor_v2_expanded.joblib"
_MODEL_V1_PATH = _MODELS_DIR / "clint_predictor.joblib"

# Training target: log10(CLint_uL/min/mg). Wrapper exponentiates
# and clips to a physiological band so rogue predictions cannot
# crash the PBPK solver.
_CLINT_MIN_UL_MIN_MG = 0.1
_CLINT_MAX_UL_MIN_MG = 10_000.0


def _load_model_and_features(smiles: str):
    """Try v4 (all sources, 1044-dim) first, then v3, v2, v1.

    Returns ``(model, features)`` or ``(None, None)`` on failure.
    """
    # v4: all sources — microsome + hepatocyte + DrugBank back-calc (1044-dim).
    model = load_joblib_lazy(_MODEL_V4_PATH)
    if model is not None:
        features = compute_features_1044(smiles)
        if features is not None:
            return model, features

    # v3: combined microsome + hepatocyte training (1044-dim features).
    model = load_joblib_lazy(_MODEL_V3_PATH)
    if model is not None:
        features = compute_features_1044(smiles)
        if features is not None:
            return model, features

    # v2: expanded 2083-dim features.
    model = load_joblib_lazy(_MODEL_V2_PATH)
    if model is not None:
        features = compute_features_2083(smiles)
        if features is not None:
            return model, features

    # Fallback to legacy 1044-dim model.
    model = load_joblib_lazy(_MODEL_V1_PATH)
    if model is not None:
        features = compute_features_1044(smiles)
        if features is not None:
            return model, features

    return None, None


def predict_clint_microsomal(smiles: str) -> Optional[float]:
    """Predict microsomal intrinsic clearance (uL/min/mg) from SMILES.

    Returns ``None`` when the model file is missing, RDKit is
    unavailable, or the SMILES does not parse. Otherwise returns
    a float in [0.1, 10 000] uL/min/mg — the unit consumed by
    :class:`HepaticClearanceEngine._scale_clint`.

    Uses the v2 expanded model (2083-dim features) by default,
    falling back to the legacy 1044-dim model if v2 is unavailable.
    """
    model, features = _load_model_and_features(smiles)
    if model is None or features is None:
        return None

    try:
        raw = model.predict(features.reshape(1, -1))[0]
    except Exception as exc:
        logger.warning(
            "clint_predictor inference failed for SMILES %r: %s",
            smiles, exc,
        )
        return None

    try:
        clint = float(10.0 ** float(raw))
    except (OverflowError, ValueError):
        return None

    return max(_CLINT_MIN_UL_MIN_MG, min(clint, _CLINT_MAX_UL_MIN_MG))


def is_model_available() -> bool:
    """Return True when an ML CLint model is loaded and usable."""
    return (
        load_joblib_lazy(_MODEL_V4_PATH) is not None
        or load_joblib_lazy(_MODEL_V3_PATH) is not None
        or load_joblib_lazy(_MODEL_V2_PATH) is not None
        or load_joblib_lazy(_MODEL_V1_PATH) is not None
    )
