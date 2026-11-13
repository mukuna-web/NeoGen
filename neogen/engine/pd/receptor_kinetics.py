"""Receptor binding kinetics — V1 P4 Step 13.

Models dynamic receptor occupancy RO(t) as a first-order binding ODE
driven by the plasma concentration profile:

    dRO/dt = kon · Cp(t) · (1 − RO) − k_off_eff · RO

where ``k_off_eff`` is either the chemical dissociation rate (``koff``)
for reversible binders, or the *biological recovery rate*
``k_recovery = ln(2) / recovery_half_life_h`` for irreversible/covalent
binders whose effect persists long after the drug has cleared.

Why two modes on one ODE
------------------------

Reversible binders (propranolol, caffeine, atorvastatin) equilibrate with
the free drug concentration — when Cp falls, RO falls proportionally via
``koff``. Their "duration" is determined by kinetic residence time.

Irreversible binders (aspirin, omeprazole, clopidogrel) covalently modify
the target — chemical ``koff`` is effectively zero. But the effect doesn't
last forever: the *biological pool* of targets turns over (platelet
lifespan ~10 d, H⁺/K⁺-ATPase ~36 h) and fresh, unmodified target is
regenerated. That biological turnover is the real off-rate, and it's
decoupled from plasma kinetics entirely.

The legacy ``neogen/simulation/receptor_kinetics.py`` papered over this
distinction by setting ``koff = 0.001`` for "irreversible" targets — which
happens to give roughly the right recovery half-life for a single specific
drug but breaks the moment you use a different one. V1 models the biology
explicitly via ``ReceptorTarget.recovery_half_life_h``.

Integrator
----------

Same piecewise closed-form trick as the effect compartment (Step 12):
over each interval ``[tᵢ, tᵢ₊₁]`` assume ``Cp`` is constant at the
interval midpoint, then the ODE has a scalar closed form:

    k_eff = kon · Cp_mid + k_off_eff
    ro_ss = kon · Cp_mid / k_eff            (if k_eff > 0)
    RO(tᵢ₊₁) = ro_ss + (RO(tᵢ) − ro_ss) · e^(−k_eff · Δt)

Degenerate case ``k_eff = 0`` (Cp=0 AND koff=0 AND reversible): holds RO
flat, which is what you want for an unreachable system.

This is unconditionally stable for any ``kon·Cp·Δt`` and reduces exactly to
the biological recovery exponential decay for irreversible binders once
plasma has cleared.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from neogen.core.entities.pd import PDProfile, ReceptorTarget
from neogen.engine.pd.pd_result import (
    ReceptorOccupancySummary,
    ReceptorOccupancyTrace,
)


# ════════════════════════════════════════════════════════════════
# CORE INTEGRATOR
# ════════════════════════════════════════════════════════════════


def _off_rate_for(target: ReceptorTarget) -> float:
    """Effective off-rate for the closed-form integrator.

    For reversible binders, that's the chemical ``koff``. For irreversible
    binders, it's the biological turnover rate ``ln(2)/recovery_half_life``.
    This is the one place where the two modes diverge — everything
    downstream can treat the problem as a single first-order ODE.
    """
    if target.irreversible:
        # ``__post_init__`` on ``ReceptorTarget`` guarantees this is set.
        if target.recovery_half_life_h is None:
            raise ValueError(
                f"{target.name}: irreversible binder missing "
                "recovery_half_life_h (entity invariant should have "
                "caught this)"
            )
        return math.log(2.0) / target.recovery_half_life_h
    return target.koff_per_h


def simulate_ro_trajectory(
    time_h: Sequence[float],
    plasma_conc_mg_per_L: Sequence[float],
    target: ReceptorTarget,
    *,
    ro0: float = 0.0,
) -> Tuple[float, ...]:
    """Integrate the binding ODE along a time grid.

    Piecewise closed-form solution with Cp approximated by the interval
    midpoint value (second-order accurate for smooth inputs, exact for
    piecewise-constant).

    Args:
        time_h: monotonically non-decreasing time grid (h).
        plasma_conc_mg_per_L: plasma concentrations at each time (mg/L).
        target: frozen :class:`ReceptorTarget` entity from the PD profile.
        ro0: initial receptor occupancy at ``time_h[0]`` (default 0 —
            drug-naive).

    Returns:
        Tuple of ``RO`` values in [0, 1] aligned with ``time_h``.

    Raises:
        ValueError on shape mismatch, non-monotone time, or bad ro0.
    """
    if not (0.0 <= ro0 <= 1.0):
        raise ValueError(f"ro0 must be in [0, 1] (got {ro0})")

    t = np.asarray(tuple(time_h), dtype=float)
    cp = np.asarray(tuple(plasma_conc_mg_per_L), dtype=float)

    if t.shape != cp.shape:
        raise ValueError(
            f"time_h and plasma_conc_mg_per_L must share shape "
            f"(got {t.shape} vs {cp.shape})"
        )
    if t.size == 0:
        return tuple()
    if np.any(np.diff(t) < 0.0):
        raise ValueError("time_h must be non-decreasing")

    # Defensive clamp: PBPK solvers occasionally emit tiny negatives.
    cp_safe = np.where(cp > 0.0, cp, 0.0)

    kon = target.kon_per_mg_per_L_per_h
    k_off_eff = _off_rate_for(target)

    ro = np.empty_like(cp_safe)
    ro[0] = ro0

    for i in range(1, t.size):
        dt = float(t[i] - t[i - 1])
        if dt <= 0.0:
            ro[i] = ro[i - 1]
            continue

        # Midpoint rule on the driving concentration. For a
        # piecewise-linear input the midpoint value is exact through
        # second order, matching the linear-trapezoid AUC used elsewhere
        # in the V1 engine.
        cp_mid = 0.5 * (float(cp_safe[i - 1]) + float(cp_safe[i]))

        alpha = kon * cp_mid
        k_eff = alpha + k_off_eff

        if k_eff <= 0.0:
            # Unreachable equilibrium (Cp=0 and koff=0 for a reversible
            # target — i.e. a drug-free window on a reversible binder with
            # zero chemical dissociation, which would only arise from
            # pathological data). Hold RO flat.
            ro[i] = ro[i - 1]
            continue

        ro_ss = alpha / k_eff
        decay = math.exp(-k_eff * dt)
        ro_next = ro_ss + (ro[i - 1] - ro_ss) * decay

        # Floating-point safety net — the closed form is mathematically
        # bounded in [0, 1] but rounding can leak out by < 1e-15.
        if ro_next < 0.0:
            ro_next = 0.0
        elif ro_next > 1.0:
            ro_next = 1.0
        ro[i] = ro_next

    return tuple(float(x) for x in ro)


# ════════════════════════════════════════════════════════════════
# SUMMARY BUILDER
# ════════════════════════════════════════════════════════════════


def _time_above_threshold(
    time_h: np.ndarray,
    occupancy: np.ndarray,
    threshold: float,
) -> Tuple[float, float | None]:
    """Compute total time ``occupancy ≥ threshold`` via linear interpolation.

    The raw "count samples above threshold × Δt" approach is biased — it
    overshoots by half a sample on each side of a crossing. Instead we
    interpolate the crossing times and sum the exact on-target intervals.

    Returns:
        (total_time_above, first_crossing_above or None)
    """
    n = time_h.size
    if n < 2:
        return 0.0, None

    total = 0.0
    first_crossing: float | None = None

    for i in range(1, n):
        y_prev = occupancy[i - 1]
        y_curr = occupancy[i]
        t_prev = time_h[i - 1]
        t_curr = time_h[i]
        dt = t_curr - t_prev
        if dt <= 0.0:
            continue

        above_prev = y_prev >= threshold
        above_curr = y_curr >= threshold

        if above_prev and above_curr:
            total += dt
            if first_crossing is None:
                first_crossing = float(t_prev)
        elif above_prev and not above_curr:
            # Crossing from above → below. Fraction still above:
            frac = (y_prev - threshold) / (y_prev - y_curr) if y_prev != y_curr else 0.0
            total += dt * frac
        elif not above_prev and above_curr:
            # Crossing from below → above.
            frac = (y_curr - threshold) / (y_curr - y_prev) if y_curr != y_prev else 0.0
            total += dt * frac
            if first_crossing is None:
                first_crossing = float(t_prev + (1.0 - frac) * dt)
        # else: both below → contribute nothing.

    return float(total), first_crossing


def _build_summary(
    *,
    time_h: Sequence[float],
    occupancy: Sequence[float],
    target: ReceptorTarget,
) -> ReceptorOccupancySummary:
    t = np.asarray(tuple(time_h), dtype=float)
    ro = np.asarray(tuple(occupancy), dtype=float)

    if t.size == 0:
        return ReceptorOccupancySummary(
            peak_occupancy=0.0,
            peak_occupancy_time_h=0.0,
            time_above_therapeutic_h=0.0,
            first_crossing_above_h=None,
            last_occupancy=0.0,
            is_irreversible=target.irreversible,
        )

    idx_peak = int(np.argmax(ro))
    time_above, first_above = _time_above_threshold(
        t, ro, target.therapeutic_occupancy
    )

    return ReceptorOccupancySummary(
        peak_occupancy=float(ro[idx_peak]),
        peak_occupancy_time_h=float(t[idx_peak]),
        time_above_therapeutic_h=time_above,
        first_crossing_above_h=first_above,
        last_occupancy=float(ro[-1]),
        is_irreversible=target.irreversible,
    )


# ════════════════════════════════════════════════════════════════
# ENGINE CLASS
# ════════════════════════════════════════════════════════════════


class ReceptorKineticsEngine:
    """Stateless facade for the receptor binding pass.

    Typical usage::

        engine = ReceptorKineticsEngine()
        trace = engine.simulate(
            drug_name=result.drug_name,
            time_h=result.time_h,
            plasma_conc_mg_per_L=result.plasma_conc_mg_per_L,
            pd_profile=drug.pd_profile,
        )
        print(trace.summary.peak_occupancy, trace.summary.time_above_therapeutic_h)

    The engine is pure — every call is a function of its inputs. Step 15's
    ``pd_orchestrator`` will call both ``EffectCompartmentEngine`` and
    ``ReceptorKineticsEngine`` on the same plasma profile and hand both
    traces to the clinical-outcome layer.
    """

    def simulate(
        self,
        *,
        drug_name: str,
        time_h: Sequence[float],
        plasma_conc_mg_per_L: Sequence[float],
        pd_profile: PDProfile,
    ) -> ReceptorOccupancyTrace:
        """Produce a :class:`ReceptorOccupancyTrace` from plasma + PD profile.

        Raises ``ValueError`` if the PD profile has no receptor target —
        the caller must check ``pd_profile.has_receptor_kinetics`` first.
        """
        target = pd_profile.receptor_target
        if target is None:
            raise ValueError(
                f"{drug_name}: PD profile has no receptor_target; "
                "ReceptorKineticsEngine cannot run without kon/koff."
            )

        ro_tuple = simulate_ro_trajectory(
            time_h=time_h,
            plasma_conc_mg_per_L=plasma_conc_mg_per_L,
            target=target,
        )
        summary = _build_summary(
            time_h=time_h, occupancy=ro_tuple, target=target
        )

        return ReceptorOccupancyTrace(
            drug_name=drug_name,
            target_name=target.name,
            time_h=tuple(float(x) for x in time_h),
            plasma_conc_mg_per_L=tuple(
                float(x) for x in plasma_conc_mg_per_L
            ),
            occupancy=ro_tuple,
            kon_per_mg_per_L_per_h=target.kon_per_mg_per_L_per_h,
            koff_per_h=target.koff_per_h,
            irreversible=target.irreversible,
            recovery_half_life_h=target.recovery_half_life_h,
            therapeutic_occupancy=target.therapeutic_occupancy,
            summary=summary,
        )
