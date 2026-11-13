"""Toxicity evaluator — V1 P4 Step 14.

Walks a plasma concentration profile against a drug's
``toxicity_thresholds`` and emits a frozen :class:`ToxicityReport`.

Design
------

Given a monotonically-sampled plasma profile ``(t_i, Cp_i)`` and a list of
``ToxicityThreshold`` rules, the evaluator asks one question per rule:

    "What is the first simulation time at which ``Cp(t) ≥ threshold.conc``?"

Rather than returning the index of the first sample that crosses the line
— which is biased by one sample width — we interpolate linearly between
``(t_{i−1}, Cp_{i−1})`` and ``(t_i, Cp_i)`` to find the exact crossing time.
This is the same sub-sample convention used in the receptor kinetics
therapeutic window calculation (``_time_above_threshold``) and keeps the
PD layer internally consistent.

Therapeutic index
-----------------

``TI = C_toxic / C_effect`` where ``C_toxic`` is the *lowest* severe or
critical threshold concentration and ``C_effect`` is the Emax EC50. This
is an honest approximation: the canonical TD50/ED50 definition requires a
dose-response curve we don't have. What we *do* have is the first
plasma concentration at which something serious happens and the
concentration at which half the pharmacological effect is reached — their
ratio is the cleanest analogue and matches the intuition a clinician
brings to the number ("how much headroom is there between efficacy and
harm"). If either ingredient is missing the report returns ``None`` for
TI rather than fabricating one.

Purity
------

Every function in this module is pure: given the same inputs it returns
the same frozen output and never mutates the profile or the PDProfile it
reads. The engine class is a thin stateless facade used by
``pd_orchestrator`` in Step 15.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from neogen.core.entities.pd import (
    ClinicalEffect,
    EffectOrgan,
    PDProfile,
    ToxicitySeverity,
    ToxicityThreshold,
)
from neogen.engine.pd.pd_result import ToxicityReport


# ════════════════════════════════════════════════════════════════
# CORE CROSSING DETECTION
# ════════════════════════════════════════════════════════════════


def _first_crossing_time(
    time_h: np.ndarray,
    plasma_conc: np.ndarray,
    threshold_mg_per_L: float,
) -> Optional[Tuple[float, float]]:
    """Find the first time ``Cp`` crosses above ``threshold``.

    Returns ``(t_cross, cp_at_cross)`` or ``None`` if never crossed. The
    crossing is found via linear interpolation between the two samples
    that bracket the first upward crossing. Samples that start already
    above the threshold are treated as crossing at ``time_h[0]`` with the
    actual first-sample concentration (no extrapolation).
    """
    n = time_h.size
    if n == 0:
        return None

    if plasma_conc[0] >= threshold_mg_per_L:
        return float(time_h[0]), float(plasma_conc[0])

    for i in range(1, n):
        prev = plasma_conc[i - 1]
        curr = plasma_conc[i]
        if curr >= threshold_mg_per_L and prev < threshold_mg_per_L:
            # Linear interpolation for the exact crossing time.
            dt = float(time_h[i] - time_h[i - 1])
            if dt <= 0.0 or curr == prev:
                return float(time_h[i]), float(curr)
            frac = (threshold_mg_per_L - prev) / (curr - prev)
            t_cross = float(time_h[i - 1]) + frac * dt
            return t_cross, float(threshold_mg_per_L)

    return None


# ════════════════════════════════════════════════════════════════
# REPORT ASSEMBLY
# ════════════════════════════════════════════════════════════════


def _sort_effects(
    effects: Sequence[ClinicalEffect],
) -> Tuple[ClinicalEffect, ...]:
    """Sort by severity (highest first), then by trigger time (earliest first)."""
    return tuple(
        sorted(
            effects,
            key=lambda e: (-e.severity.rank, e.triggered_at_h),
        )
    )


def _organ_worst_map(
    effects: Sequence[ClinicalEffect],
) -> dict:
    worst: dict = {}
    for e in effects:
        prior = worst.get(e.organ)
        if prior is None or e.severity.rank > prior.rank:
            worst[e.organ] = e.severity
    return worst


def _overall_severity(
    effects: Sequence[ClinicalEffect],
) -> Optional[ToxicitySeverity]:
    if not effects:
        return None
    return max(effects, key=lambda e: e.severity.rank).severity


def _first_severe_time(
    effects: Sequence[ClinicalEffect],
) -> Optional[float]:
    """Earliest trigger time among SEVERE and CRITICAL effects."""
    severe = [
        e.triggered_at_h
        for e in effects
        if e.severity.rank >= ToxicitySeverity.SEVERE.rank
    ]
    if not severe:
        return None
    return float(min(severe))


def _therapeutic_index(
    pd_profile: PDProfile,
) -> Optional[float]:
    """TI = (lowest SEVERE/CRITICAL threshold concentration) / EC50.

    Rationale: see module docstring. Returns ``None`` if either ingredient
    is unavailable. Does **not** depend on the simulation — it's a
    property of the drug's PD profile alone.
    """
    if pd_profile.emax is None:
        return None
    severe_concs = [
        th.concentration_mg_per_L
        for th in pd_profile.toxicity_thresholds
        if th.severity.rank >= ToxicitySeverity.SEVERE.rank
    ]
    if not severe_concs:
        return None
    ec50 = pd_profile.emax.ec50_mg_per_L
    if ec50 <= 0.0:
        return None
    return float(min(severe_concs) / ec50)


# ════════════════════════════════════════════════════════════════
# ENGINE CLASS
# ════════════════════════════════════════════════════════════════


class ToxicityEvaluatorEngine:
    """Stateless facade for the toxicity pass.

    Typical usage::

        engine = ToxicityEvaluatorEngine()
        report = engine.evaluate(
            drug_name=result.drug_name,
            time_h=result.time_h,
            plasma_conc_mg_per_L=result.plasma_conc_mg_per_L,
            pd_profile=drug.pd_profile,
        )
        print(report.verdict)
        for eff in report.triggered_effects:
            print(f"  {eff.severity.value}: {eff.symptom} @ {eff.triggered_at_h:.1f}h")

    The engine never raises on empty threshold lists — a drug with no
    toxicity model just produces an empty report with ``has_any_toxicity
    = False``. That keeps downstream orchestration code branch-free.
    """

    def evaluate(
        self,
        *,
        drug_name: str,
        time_h: Sequence[float],
        plasma_conc_mg_per_L: Sequence[float],
        pd_profile: PDProfile,
    ) -> ToxicityReport:
        """Walk the plasma profile against the drug's toxicity thresholds.

        Raises ``ValueError`` only on structural input errors (shape
        mismatch, negative time). Missing PD data is *not* an error —
        it yields an empty, well-formed report.
        """
        t = np.asarray(tuple(time_h), dtype=float)
        cp = np.asarray(tuple(plasma_conc_mg_per_L), dtype=float)

        if t.shape != cp.shape:
            raise ValueError(
                f"time_h and plasma_conc_mg_per_L must share shape "
                f"(got {t.shape} vs {cp.shape})"
            )
        if t.size > 0 and t[0] < 0.0:
            raise ValueError("time_h must start at ≥ 0")
        if t.size > 0 and np.any(np.diff(t) < 0.0):
            raise ValueError("time_h must be non-decreasing")

        # Defensive clamp: PBPK solvers can emit tiny negatives.
        cp_safe = np.where(cp > 0.0, cp, 0.0)

        # Peak plasma (independent of threshold crossings — always reported).
        if t.size == 0:
            peak_cp = 0.0
            peak_t = 0.0
        else:
            idx_peak = int(np.argmax(cp_safe))
            peak_cp = float(cp_safe[idx_peak])
            peak_t = float(t[idx_peak])

        triggered: List[ClinicalEffect] = []
        for threshold in pd_profile.toxicity_thresholds:
            crossing = _first_crossing_time(
                t, cp_safe, threshold.concentration_mg_per_L
            )
            if crossing is None:
                continue
            t_cross, cp_cross = crossing
            triggered.append(
                ClinicalEffect(
                    organ=threshold.organ,
                    severity=threshold.severity,
                    symptom=threshold.symptom,
                    triggered_at_h=t_cross,
                    triggered_concentration_mg_per_L=cp_cross,
                    description=threshold.description,
                )
            )

        sorted_effects = _sort_effects(triggered)

        return ToxicityReport(
            drug_name=drug_name,
            peak_plasma_mg_per_L=peak_cp,
            peak_plasma_time_h=peak_t,
            triggered_effects=sorted_effects,
            organ_worst_severity=_organ_worst_map(sorted_effects),
            overall_severity=_overall_severity(sorted_effects),
            therapeutic_index=_therapeutic_index(pd_profile),
            first_severe_trigger_time_h=_first_severe_time(sorted_effects),
        )
