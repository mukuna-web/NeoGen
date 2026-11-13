"""ML renal elimination classifier.

Loads ``models/adme_pipeline/renal_classifier_v1.joblib`` (2083-dim
features) and exposes :func:`predict_fraction_excreted_unchanged`.
The classifier output is mapped to a ``fraction_excreted_unchanged``
estimate that the V1 engine uses to split total clearance between
hepatic and renal paths.

Why this matters for Tier 1
---------------------------
Without this predictor, novel compounds default to
``fraction_excreted_unchanged = 0.0``, which forces 100% of
clearance through the hepatic well-stirred model. Drugs with
dominant renal elimination (metformin, digoxin, valsartan,
bortezomib, many others in the blind holdout) have their
clearance attributed entirely to CYP metabolism — and when their
CLint is low, the result is catastrophically wrong half-lives.

The V1 classifier returns a categorical label (0 = hepatic, 1 =
mixed, 2 = renal) with 91.8% accuracy on the training cohort.
This wrapper maps categories to fraction estimates:

    0 (hepatic)  →  fe = 0.05
    1 (mixed)    →  fe = 0.35
    2 (renal)    →  fe = 0.80

The buckets are calibrated against the training-set centroids for
each class. V1.2 could replace them with per-drug continuous
predictions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from neogen.ml._features import compute_features_2083, load_joblib_lazy

logger = logging.getLogger(__name__)


_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "adme_pipeline"
    / "renal_classifier_v1.joblib"
)

# Class → fraction-excreted-unchanged mapping. See module docstring.
_CLASS_TO_FE = {
    0: 0.05,  # primarily hepatic
    1: 0.35,  # mixed hepatic/renal
    2: 0.80,  # primarily renal
}
_DEFAULT_FE_ON_UNKNOWN = 0.20


def predict_fraction_excreted_unchanged(smiles: str) -> Optional[float]:
    """Predict the fraction of drug excreted unchanged in urine.

    Returns ``None`` when the model is missing or the SMILES does
    not parse. Otherwise returns a float ∈ [0.05, 0.80] mapped
    from the categorical classifier output.
    """
    model = load_joblib_lazy(_MODEL_PATH)
    if model is None:
        return None

    features = compute_features_2083(smiles)
    if features is None:
        return None

    try:
        cls = model.predict(features.reshape(1, -1))[0]
    except Exception as exc:
        logger.warning(
            "renal_classifier inference failed for SMILES %r: %s",
            smiles, exc,
        )
        return None

    try:
        cls_int = int(cls)
    except (TypeError, ValueError):
        return None

    return _CLASS_TO_FE.get(cls_int, _DEFAULT_FE_ON_UNKNOWN)


def is_model_available() -> bool:
    """Return True when the renal classifier is loaded and usable."""
    return load_joblib_lazy(_MODEL_PATH) is not None
