"""
XGBoost Compatibility Shim — V1.1 Extension
=============================================
Safe loader for legacy XGBoost / scikit-learn pickles that may have been
serialised under older library versions. The ADME pipeline stores its
predictors as ``.joblib`` files which embed XGBoost ``Booster`` objects
inside scikit-learn wrapper classes. When the runtime XGBoost version
differs from the one used to train/save the model, ``joblib.load()``
can raise a variety of cryptic errors:

  * ``xgboost.core.XGBoostError: ... json ...`` — format migration
  * ``ModuleNotFoundError: No module named 'xgboost.compat'`` — old
    shim path removed in XGBoost ≥ 2.0
  * ``ValueError: Invalid model format`` — major version bump

This module provides a single ``safe_load_model()`` function that
tries three progressively more defensive loading strategies:

  1. ``joblib.load()`` — happy path, no overhead.
  2. ``pickle.load()`` with ``encoding='latin1'`` — handles Python 2 →
     3 legacy pickles.
  3. ``xgboost.Booster().load_model()`` on the raw ``.json`` / ``.bin``
     sidecar if it exists — manual reconstruction for completely broken
     wrappers.

If all three fail, the function returns ``None`` and logs a warning so
the ADME pipeline can continue with its fallback logic rather than
crashing the server.

Usage:
    from neogen.simulation.xgboost_compat import safe_load_model
    model = safe_load_model(Path("models/fu_predictor_v4.joblib"))
    if model is not None:
        preds = model.predict(X)
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def safe_load_model(path: Path) -> Optional[Any]:
    """Load a joblib/pickle model with XGBoost version tolerance.

    Args:
        path: Path to a ``.joblib`` or ``.pkl`` file.

    Returns:
        The unpickled model object, or ``None`` if all strategies fail.
    """
    if not path.exists():
        logger.warning("safe_load_model: file not found — %s", path)
        return None

    # ── Strategy 1: standard joblib ──────────────────────────
    try:
        import joblib

        model = joblib.load(path)
        logger.debug("safe_load_model: loaded via joblib — %s", path.name)
        return model
    except Exception as exc:
        logger.info(
            "safe_load_model: joblib.load failed for %s (%s), "
            "trying pickle fallback.",
            path.name, exc,
        )

    # ── Strategy 2: raw pickle with latin-1 encoding ─────────
    try:
        with open(path, "rb") as fh:
            model = pickle.load(fh, encoding="latin1")
        logger.debug(
            "safe_load_model: loaded via pickle (latin1) — %s", path.name
        )
        return model
    except Exception as exc:
        logger.info(
            "safe_load_model: pickle fallback failed for %s (%s), "
            "trying XGBoost native loader.",
            path.name, exc,
        )

    # ── Strategy 3: XGBoost native sidecar ───────────────────
    # Some legacy workflows saved a raw ``.json`` or ``.bin`` alongside
    # the joblib wrapper. If present, reconstruct via xgb.Booster.
    for suffix in (".json", ".bin", ".ubj"):
        sidecar = path.with_suffix(suffix)
        if sidecar.exists():
            try:
                import xgboost as xgb

                booster = xgb.Booster()
                booster.load_model(str(sidecar))
                logger.info(
                    "safe_load_model: reconstructed Booster from %s",
                    sidecar.name,
                )
                return booster
            except Exception as exc:
                logger.info(
                    "safe_load_model: XGBoost native load failed for %s (%s)",
                    sidecar.name, exc,
                )

    logger.warning(
        "safe_load_model: all strategies exhausted for %s — returning None",
        path.name,
    )
    return None
