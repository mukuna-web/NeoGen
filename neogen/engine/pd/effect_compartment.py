"""Effect compartment link — V1 P4 Step 12.

Implements the Sheiner (1979) hypothetical effect compartment that bridges
the gap between plasma pharmacokinetics and pharmacodynamic effect:

    dCe/dt = ke0 · (Cp(t) − Ce(t))

where ``Cp`` is the plasma concentration (delivered by the P1 solver),
``Ce`` is the effect-site concentration, and ``ke0`` is the first-order
equilibration rate constant (1/h). Small ``ke0`` → slow PD (warfarin);
large ``ke0`` → effect tracks plasma closely (propofol).

Design notes
------------

*Not Euler.* The legacy module in ``neogen/simulation/effect_compartment.py``
used explicit Euler integration over the PBPK solver's time grid. That works
for fast-equilibrating drugs (ke0 > 0.1 h⁻¹) but is **numerically fragile**
for slow drugs like warfarin (ke0 ≈ 0.003 h⁻¹) or fluoxetine
(ke0 ≈ 0.001 h⁻¹): Euler needs ``ke0·Δt << 1`` for stability, and the PBPK
time grid can easily violate that.

This V1 engine uses a **piecewise exponential closed-form integrator**. For
each interval ``[tᵢ, tᵢ₊₁]`` we assume ``Cp`` varies linearly between its
endpoint values — which matches how the PBPK solver samples its own solution
— and integrate the ODE exactly:

    Cp(t) = a + b·(t − tᵢ)      with  a = Cp_i,  b = (Cp_{i+1} − Cp_i) / Δt
    Ce(tᵢ₊₁) = Ce_i · e^(−ke0·Δt) + (a − b/ke0)·(1 − e^(−ke0·Δt))
                + b·Δt

Derivation: integrating factor, then substitute. This form is
**unconditionally stable** (no matter how large or small ``ke0·Δt`` is),
exact for piecewise-linear inputs, and 2-3 orders of magnitude more accurate
than Euler for the warfarin/fluoxetine regime.

The only pure function exposed is ``simulate_ce_trajectory``, which the
``EffectCompartmentEngine.simulate`` method wraps together with the Emax
evaluation.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import numpy as np

from neogen.core.entities.pd import (
    EffectCompartmentParams,
    EmaxParams,
    PDProfile,
)
from neogen.engine.pd.emax_model import apply_emax_array
from neogen.engine.pd.pd_result import EffectTimeCourse, EffectTimingSummary


# ════════════════════════════════════════════════════════════════
# CORE INTEGRATOR
# ════════════════════════════════════════════════════════════════


def simulate_ce_trajectory(
    time_h: Sequence[float],
    plasma_conc_mg_per_L: Sequence[float],
    ke0_per_h: float,
    *,
    ce0: float = 0.0,
) -> Tuple[float, ...]:
    """Integrate ``dCe/dt = ke0·(Cp − Ce)`` along a time grid.

    Piecewise-exact solution assuming ``Cp`` is linear between samples.

    Args:
        time_h: monotonically non-decreasing time grid (h).
        plasma_conc_mg_per_L: plasma concentrations at each time (mg/L).
        ke0_per_h: effect-site equilibration rate (1/h). Must be > 0.
        ce0: initial effect-site concentration at ``time_h[0]`` (mg/L).

    Returns:
        Tuple of ``Ce`` values aligned with ``time_h``.

    Raises:
        ValueError if the inputs are inconsistent.
    """
    if ke0_per_h <= 0.0:
        raise ValueError(f"ke0_per_h must be > 0 (got {ke0_per_h})")

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

    ce = np.empty_like(cp_safe)
    ce[0] = max(ce0, 0.0)

    for i in range(1, t.size):
        dt = float(t[i] - t[i - 1])
        if dt <= 0.0:
            # Degenerate duplicate timestamp — carry state forward.
            ce[i] = ce[i - 1]
            continue

        a = float(cp_safe[i - 1])              # Cp at left endpoint
        b = (float(cp_safe[i]) - a) / dt        # linear ramp slope

        # Closed-form solution for dCe/dt = ke0·(a + b·τ − Ce)
        # over τ ∈ [0, Δt]:
        #
        #   Ce(Δt) = Ce0·e^(−ke0·Δt)
        #          + (a − b/ke0)·(1 − e^(−ke0·Δt))
        #          + b·Δt
        decay = math.exp(-ke0_per_h * dt)
        steady = a - b / ke0_per_h
        ce_next = ce[i - 1] * decay + steady * (1.0 - decay) + b * dt

        # Guard against sub-zero floating-point noise.
        ce[i] = ce_next if ce_next > 0.0 else 0.0

    return tuple(float(x) for x in ce)


# ════════════════════════════════════════════════════════════════
# ENGINE CLASS
# ════════════════════════════════════════════════════════════════


class EffectCompartmentEngine:
    """Stateless facade for the effect compartment + Emax pass.

    Typical usage::

        engine = EffectCompartmentEngine()
        course = engine.simulate(
            drug_name=result.drug_name,
            time_h=result.time_h,
            plasma_conc_mg_per_L=result.plasma_conc_mg_per_L,
            pd_profile=drug.pd_profile,
        )
        print(course.summary.hysteresis_lag_h)

    The engine has no instance state — every call is a pure function of its
    inputs. The class exists purely so the P4 orchestrator (Step 15) can
    depend on a clean interface that can later be swapped for a receptor
    kinetics variant without touching the orchestrator.
    """

    def simulate(
        self,
        *,
        drug_name: str,
        time_h: Sequence[float],
        plasma_conc_mg_per_L: Sequence[float],
        pd_profile: PDProfile,
    ) -> EffectTimeCourse:
        """Produce an :class:`EffectTimeCourse` from plasma input + PD profile.

        Args:
            drug_name: echoed into the result for reports.
            time_h: PBPK solver time grid (hours).
            plasma_conc_mg_per_L: plasma concentration at each sample (mg/L).
            pd_profile: frozen PD profile from the drug. Must have both
                ``effect_compartment`` and ``emax`` populated — raise
                ``ValueError`` otherwise so silent misuse is impossible.

        Returns:
            Frozen :class:`EffectTimeCourse` with Ce and effect trajectories.
        """
        if pd_profile.effect_compartment is None:
            raise ValueError(
                f"{drug_name}: PD profile has no effect_compartment; "
                "EffectCompartmentEngine cannot run without ke0."
            )
        if pd_profile.emax is None:
            raise ValueError(
                f"{drug_name}: PD profile has no emax model; "
                "EffectCompartmentEngine needs EC50 + Emax to report effect."
            )

        ec_params: EffectCompartmentParams = pd_profile.effect_compartment
        emax_params: EmaxParams = pd_profile.emax

        ce_tuple = simulate_ce_trajectory(
            time_h=time_h,
            plasma_conc_mg_per_L=plasma_conc_mg_per_L,
            ke0_per_h=ec_params.ke0_per_h,
        )
        effect_tuple = apply_emax_array(ce_tuple, emax_params)

        summary = _build_summary(
            time_h=time_h,
            plasma_conc_mg_per_L=plasma_conc_mg_per_L,
            ce_values=ce_tuple,
            effect_values=effect_tuple,
            ke0_per_h=ec_params.ke0_per_h,
        )

        return EffectTimeCourse(
            drug_name=drug_name,
            effect_name=emax_params.effect_name,
            effect_unit=emax_params.effect_unit,
            time_h=tuple(float(x) for x in time_h),
            plasma_conc_mg_per_L=tuple(float(x) for x in plasma_conc_mg_per_L),
            effect_site_conc_mg_per_L=ce_tuple,
            effect_values=effect_tuple,
            ke0_per_h=ec_params.ke0_per_h,
            ec50_mg_per_L=emax_params.ec50_mg_per_L,
            emax=emax_params.emax,
            hill=emax_params.hill,
            baseline=emax_params.baseline,
            summary=summary,
        )


# ════════════════════════════════════════════════════════════════
# SUMMARY BUILDER
# ════════════════════════════════════════════════════════════════


def _build_summary(
    *,
    time_h: Sequence[float],
    plasma_conc_mg_per_L: Sequence[float],
    ce_values: Sequence[float],
    effect_values: Sequence[float],
    ke0_per_h: float,
) -> EffectTimingSummary:
    """Compute scalar timing summary from a completed trajectory."""
    if len(time_h) == 0:
        return EffectTimingSummary(
            plasma_cmax_mg_per_L=0.0,
            plasma_tmax_h=0.0,
            effect_site_cmax_mg_per_L=0.0,
            effect_site_tmax_h=0.0,
            effect_peak=0.0,
            effect_peak_time_h=0.0,
            hysteresis_lag_h=0.0,
            effect_half_life_h=math.log(2.0) / ke0_per_h,
        )

    t = np.asarray(time_h, dtype=float)
    cp = np.asarray(plasma_conc_mg_per_L, dtype=float)
    ce = np.asarray(ce_values, dtype=float)
    eff = np.asarray(effect_values, dtype=float)

    # V1.2: The two-phase IV initial condition (80% plasma / 20%
    # distributed) produces a realistic bolus spike at t=0. The old
    # impulse-skip heuristic was designed for the legacy 100% in
    # plasma artefact and is no longer needed. Always use the true
    # profile maximum so PD plasma_cmax matches PKCalculator.
    idx_cp = int(np.argmax(cp))
    idx_ce = int(np.argmax(ce))
    idx_eff = int(np.argmax(eff))

    hysteresis = float(t[idx_eff] - t[idx_cp])
    if hysteresis < 0.0:
        hysteresis = 0.0

    return EffectTimingSummary(
        plasma_cmax_mg_per_L=float(cp[idx_cp]),
        plasma_tmax_h=float(t[idx_cp]),
        effect_site_cmax_mg_per_L=float(ce[idx_ce]),
        effect_site_tmax_h=float(t[idx_ce]),
        effect_peak=float(eff[idx_eff]),
        effect_peak_time_h=float(t[idx_eff]),
        hysteresis_lag_h=hysteresis,
        effect_half_life_h=math.log(2.0) / ke0_per_h,
    )
