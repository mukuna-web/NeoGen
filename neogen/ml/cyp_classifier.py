"""ML CYP substrate classifier — predicts CYP isoform fm profile.

Loads ``models/adme_pipeline/cyp_classifier.joblib`` (2083-dim
features) and ``cyp_mlb.joblib`` (MultiLabelBinarizer) lazily.
Exposes :func:`predict_cyp_substrate_profile` which returns a dict
of CYP isoform probabilities suitable for building a
:class:`CYPMetabolismProfile`.

The classifier was trained on ~2500 drugs with known CYP fm
fractions. The MultiLabelBinarizer encodes the CYP isoform labels
from the training set. The model outputs multi-label probabilities
via ``predict_proba`` which are renormalized to sum to 1.0 for use
as fm fractions.

Follows the same lazy-loading + XGBoost compat shim pattern as
:mod:`neogen.ml.clint_predictor` and :mod:`neogen.ml.fu_predictor`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from neogen.ml._features import compute_features_2083, load_joblib_lazy

logger = logging.getLogger(__name__)


_MODELS_DIR = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "adme_pipeline"
)

_CLASSIFIER_V2_PATH = _MODELS_DIR / "cyp_classifier_v2_all.joblib"
_MLB_V2_PATH = _MODELS_DIR / "cyp_mlb_v2.joblib"
_CLASSIFIER_PATH = _MODELS_DIR / "cyp_classifier.joblib"
_MLB_PATH = _MODELS_DIR / "cyp_mlb.joblib"

# Default fm profile used when the classifier is unavailable.
# CYP3A4-dominant is the safest default since ~50% of marketed
# drugs are CYP3A4 substrates (Zanger & Schwab, 2013).
_DEFAULT_PROFILE: Dict[str, float] = {
    "CYP3A4": 0.50,
    "CYP2D6": 0.25,
    "CYP2C9": 0.10,
    "CYP2C19": 0.05,
    "CYP1A2": 0.05,
    "CYP2C8": 0.03,
    "CYP2B6": 0.02,
}

# Mapping from CYP label strings to CYPMetabolismProfile field names.
_CYP_LABEL_TO_FIELD = {
    "CYP3A4": "cyp3a4",
    "CYP2D6": "cyp2d6",
    "CYP2C9": "cyp2c9",
    "CYP2C19": "cyp2c19",
    "CYP1A2": "cyp1a2",
    "CYP2C8": "cyp2c8",
    "CYP2B6": "cyp2b6",
    "CYP2E1": "cyp2e1",
}


def predict_cyp_substrate_profile(
    smiles: str,
) -> Dict[str, float]:
    """Predict CYP substrate probabilities from SMILES.

    Returns a dict mapping CYP isoform names (e.g. ``"CYP3A4"``) to
    predicted fm fractions (probabilities renormalized to sum to 1.0).

    Returns the default profile when the model files are missing,
    RDKit is unavailable, or the SMILES does not parse.
    """
    # Try v2 (all sources) first, fall back to v1
    classifier = load_joblib_lazy(_CLASSIFIER_V2_PATH)
    mlb = load_joblib_lazy(_MLB_V2_PATH)
    if classifier is None or mlb is None:
        classifier = load_joblib_lazy(_CLASSIFIER_PATH)
        mlb = load_joblib_lazy(_MLB_PATH)
    if classifier is None or mlb is None:
        logger.debug(
            "CYP classifier or MLB not available — using default profile"
        )
        return dict(_DEFAULT_PROFILE)

    features = compute_features_2083(smiles)
    if features is None:
        return dict(_DEFAULT_PROFILE)

    try:
        # predict_proba returns a list of arrays (one per label) for
        # multi-label classifiers, or a single 2D array for multi-class.
        proba = classifier.predict_proba(features.reshape(1, -1))
    except Exception as exc:
        logger.warning(
            "CYP classifier inference failed for SMILES %r: %s",
            smiles,
            exc,
        )
        return dict(_DEFAULT_PROFILE)

    try:
        classes = list(mlb.classes_)
    except Exception:
        return dict(_DEFAULT_PROFILE)

    # Build raw probability dict from classifier output.
    profile: Dict[str, float] = {}
    try:
        if isinstance(proba, list):
            # Multi-label: proba is a list of (n_samples, 2) arrays.
            # Take P(class=1) for each label.
            for i, label in enumerate(classes):
                p = float(proba[i][0, 1]) if proba[i].shape[1] > 1 else float(proba[i][0, 0])
                profile[label] = max(p, 0.0)
        else:
            # Multi-class: proba is (n_samples, n_classes).
            for i, label in enumerate(classes):
                profile[label] = max(float(proba[0, i]), 0.0)
    except Exception as exc:
        logger.warning(
            "CYP classifier post-processing failed: %s", exc,
        )
        return dict(_DEFAULT_PROFILE)

    # Renormalize to sum to 1.0 for use as fm fractions.
    total = sum(profile.values())
    if total > 0:
        profile = {k: v / total for k, v in profile.items()}
    else:
        return dict(_DEFAULT_PROFILE)

    return profile


def predict_cyp_metabolism_kwargs(smiles: str) -> Dict[str, float]:
    """Predict CYP fm profile and return as CYPMetabolismProfile kwargs.

    Returns a dict with keys like ``cyp3a4``, ``cyp2d6``, etc. that
    can be unpacked directly into ``CYPMetabolismProfile(**kwargs)``.
    """
    profile = predict_cyp_substrate_profile(smiles)
    kwargs: Dict[str, float] = {}
    for cyp_label, field_name in _CYP_LABEL_TO_FIELD.items():
        if cyp_label in profile:
            kwargs[field_name] = round(profile[cyp_label], 4)
    return kwargs


def is_model_available() -> bool:
    """Return True when both the CYP classifier and MLB are loadable."""
    return (
        (load_joblib_lazy(_CLASSIFIER_V2_PATH) is not None
         and load_joblib_lazy(_MLB_V2_PATH) is not None)
        or (load_joblib_lazy(_CLASSIFIER_PATH) is not None
            and load_joblib_lazy(_MLB_PATH) is not None)
    )
