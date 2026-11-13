"""PK consistency checker — cross-validates CL, Vd, and t1/2 predictions.

When multiple ADME properties are independently predicted by ML models,
the three-way pharmacokinetic relationship

    t1/2 = 0.693 * Vd / CL

provides a strong internal consistency check.  If the independently
predicted CLint (-> CL), Vd, and t1/2 violate this relationship by
more than 5-fold, the predictions are flagged as inconsistent and
the half-life is reconciled using the geometric mean of the direct
prediction and the CL/Vd-derived estimate.

This module is used by :class:`DrugFactory` as a post-hoc sanity
check after all ADME predictions are complete.  It does NOT require
model retraining -- it is a purely algebraic cross-validation.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsistencyResult:
    """Result of a PK consistency check.

    Attributes
    ----------
    consistent
        True if the CL/Vd-derived t1/2 and the direct t1/2 prediction
        agree within the threshold (default 5-fold).
    reconciled_half_life
        When inconsistent, the geometric mean of the two estimates.
        When consistent, None (use the original prediction).
    ratio
        t1/2(CL,Vd) / t1/2(direct).  Values far from 1.0 indicate
        inconsistency.  None if the check could not be performed.
    """

    consistent: bool
    reconciled_half_life: Optional[float] = None
    ratio: Optional[float] = None


def check_pk_consistency(
    cl_L_per_h: float,
    vd_L: float,
    half_life_h: float,
    threshold_fold: float = 5.0,
) -> ConsistencyResult:
    """Cross-check CL, Vd, t1/2 for internal consistency.

    Parameters
    ----------
    cl_L_per_h
        Total clearance in L/h.
    vd_L
        Volume of distribution in L.
    half_life_h
        Directly predicted half-life in hours.
    threshold_fold
        Maximum allowed ratio between the two t1/2 estimates before
        triggering reconciliation.  Default 5.0 (very conservative).

    Returns
    -------
    ConsistencyResult
        With ``consistent=True`` if within threshold, otherwise with
        ``reconciled_half_life`` set to the geometric mean.
    """
    if cl_L_per_h <= 0 or vd_L <= 0 or half_life_h <= 0:
        return ConsistencyResult(consistent=True)

    t_half_from_cl_vd = 0.693 * vd_L / cl_L_per_h

    ratio = t_half_from_cl_vd / half_life_h

    if ratio > threshold_fold or ratio < (1.0 / threshold_fold):
        reconciled = math.sqrt(t_half_from_cl_vd * half_life_h)
        logger.info(
            "PK inconsistency: t1/2(CL,Vd)=%.2f h vs t1/2(direct)=%.2f h "
            "(ratio=%.2f) -> reconciled=%.2f h",
            t_half_from_cl_vd, half_life_h, ratio, reconciled,
        )
        return ConsistencyResult(
            consistent=False,
            reconciled_half_life=reconciled,
            ratio=ratio,
        )

    return ConsistencyResult(consistent=True, ratio=ratio)


def reconcile_pk_params(
    cl_L_per_h: Optional[float],
    vd_L: Optional[float],
    half_life_h: Optional[float],
    threshold_fold: float = 5.0,
) -> dict:
    """Reconcile PK parameters and return any adjustments.

    Convenience wrapper around :func:`check_pk_consistency` that
    returns a dict of parameter overrides to apply.  If all
    parameters are consistent (or if any are missing), returns an
    empty dict.

    Returns
    -------
    dict
        May contain ``'half_life_override_h'`` if reconciliation was
        needed.  Empty dict if consistent or if inputs are incomplete.
    """
    if cl_L_per_h is None or vd_L is None or half_life_h is None:
        return {}

    result = check_pk_consistency(
        cl_L_per_h, vd_L, half_life_h, threshold_fold
    )

    if not result.consistent and result.reconciled_half_life is not None:
        return {"half_life_override_h": result.reconciled_half_life}

    return {}
