"""Sigmoidal Emax PD model (V1 Step 12).

Implements the canonical Hill equation for concentration-effect mapping:

    E(C) = baseline + Emax · C^h / (EC50^h + C^h)

where:
  - ``E``       is the effect on whatever unit the drug reports (%, mmHg, INR…)
  - ``C``       is the **effect-site** concentration (mg/L), not plasma
  - ``Emax``    is the maximum achievable effect above baseline
  - ``EC50``    is the concentration producing 50% of Emax
  - ``h``       is the Hill coefficient (1 = hyperbolic, >1 = sigmoidal)
  - ``baseline`` is the effect at zero drug concentration

This module is deliberately a collection of **pure functions**. No classes,
no state, no I/O. The effect compartment engine in ``effect_compartment.py``
calls ``apply_emax_array`` once per simulation to turn a ``Ce(t)`` trajectory
into an effect trajectory.

Rationale for pure functions here:
  - Stateless math is easiest to unit-test in isolation.
  - Makes the receptor kinetics layer (Step 13) trivial to plug in — it will
    call the same ``apply_emax`` with its own Ce substitute (receptor
    occupancy fraction × Emax scale).
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from neogen.core.entities.pd import EmaxParams


def apply_emax(
    concentration_mg_per_L: float,
    params: EmaxParams,
) -> float:
    """Evaluate the Hill/Emax model at a single concentration.

    Args:
        concentration_mg_per_L: effect-site concentration (mg/L). Must be ≥ 0.
        params: frozen :class:`EmaxParams` from the drug's PD profile.

    Returns:
        Effect magnitude on the ``params.effect_unit`` scale.

    Behaviour:
      - At ``C = 0`` the effect equals ``params.baseline``.
      - At ``C = EC50`` the effect equals ``baseline + emax/2``.
      - As ``C → ∞`` the effect asymptotes to ``baseline + emax``.
      - Negative concentrations are clamped to 0 (defensive — plasma should
        never be negative, but the PBPK solver can produce tiny
        floating-point negatives near zero).
    """
    if concentration_mg_per_L <= 0.0:
        return params.baseline

    c_h = concentration_mg_per_L ** params.hill
    ec50_h = params.ec50_mg_per_L ** params.hill
    fraction = c_h / (ec50_h + c_h)
    return params.baseline + params.emax * fraction


def apply_emax_array(
    concentrations_mg_per_L: Iterable[float],
    params: EmaxParams,
) -> Tuple[float, ...]:
    """Vectorised Hill/Emax evaluation over a concentration trajectory.

    Uses numpy under the hood for speed, but returns a plain ``tuple`` so the
    output can live inside a ``frozen=True`` dataclass without accidental
    downstream mutation.

    Invariants enforced:
      - Input length is preserved.
      - Negative inputs are treated as zero (clamped in-place at the numpy
        level, no mutation of the caller's array).
      - Output is always in ``[baseline, baseline + emax]``.
    """
    arr = np.asarray(tuple(concentrations_mg_per_L), dtype=float)
    # Clamp negatives to 0 without touching the caller's array.
    safe = np.where(arr > 0.0, arr, 0.0)

    c_h = np.power(safe, params.hill)
    ec50_h = params.ec50_mg_per_L ** params.hill
    fraction = np.divide(
        c_h,
        ec50_h + c_h,
        out=np.zeros_like(c_h),
        where=(ec50_h + c_h) > 0.0,
    )
    effect = params.baseline + params.emax * fraction
    return tuple(float(x) for x in effect)


def ec_for_effect_fraction(
    target_fraction: float,
    params: EmaxParams,
) -> float:
    """Inverse Hill: concentration that produces a target fraction of Emax.

    Useful for figuring out what plasma level is needed to hit e.g. 80% of
    max effect for reports and therapeutic-window calculations.

    Args:
        target_fraction: desired fraction of Emax (0 < f < 1).
        params: frozen EmaxParams.

    Returns:
        Concentration in mg/L that, if sustained at the effect site, would
        produce ``target_fraction * emax`` above baseline.

    Raises:
        ValueError if ``target_fraction`` is outside (0, 1).
    """
    if not (0.0 < target_fraction < 1.0):
        raise ValueError(
            f"target_fraction must be in (0, 1); got {target_fraction}"
        )
    # C^h / (EC50^h + C^h) = f  →  C = EC50 · (f / (1-f))^(1/h)
    ratio = target_fraction / (1.0 - target_fraction)
    return params.ec50_mg_per_L * (ratio ** (1.0 / params.hill))


def peak_effect(params: EmaxParams) -> float:
    """Theoretical maximum effect = baseline + Emax."""
    return params.baseline + params.emax
