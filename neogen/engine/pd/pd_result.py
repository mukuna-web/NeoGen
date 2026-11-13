"""Frozen result containers for the V1 P4 pharmacodynamics engine.

These dataclasses are the *output contracts* of the PD layer. Each phase of
the pipeline emits immutable objects — never mutates inputs — and the
orchestrator composes them into the final ``SimulationResultV2``.

Step 12 introduces:
  - ``EffectTimeCourse`` / ``EffectTimingSummary``: the t → (Ce, effect)
    trajectory from an effect compartment + Emax pass.

Step 13 adds:
  - ``ReceptorOccupancyTrace`` / ``ReceptorOccupancySummary``: the t → RO
    trajectory from the kon/koff (or irreversible + biological recovery)
    binding ODE.

Step 14 adds:
  - ``ToxicityReport``: the organ-level severity ladder produced by walking
    the plasma profile against a drug's ``ToxicityThreshold`` list.

Step 15 adds:
  - ``ClinicalOutcome``: the top-level composite of effect course + receptor
    trace + toxicity report, with derived scalars (peak effect fraction,
    therapeutic index, overall severity) for one-line report verdicts.

Units: concentrations are **mg/L**, time is **hours**, occupancy is a
unit-free fraction in [0, 1], effect is on the Emax scale (% or biomarker
unit, depending on ``effect_unit``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

from neogen.core.entities.pd import (
    ClinicalEffect,
    EffectOrgan,
    ToxicitySeverity,
)


@dataclass(frozen=True)
class EffectTimingSummary:
    """Scalar summary of a single effect trajectory.

    All times are in hours. Concentrations are in mg/L. The hysteresis lag
    is defined as ``t_effect_max - t_plasma_max`` and is non-negative by
    construction (an effect that peaks *before* plasma is clamped to 0 —
    physically impossible for a passive effect compartment).
    """

    plasma_cmax_mg_per_L: float
    plasma_tmax_h: float

    effect_site_cmax_mg_per_L: float
    effect_site_tmax_h: float

    effect_peak: float
    """Peak effect magnitude on the Emax scale (e.g. 85 for 85%)."""

    effect_peak_time_h: float
    """Time at which the peak effect occurs (h)."""

    hysteresis_lag_h: float
    """``t_effect_max - t_plasma_max``, clamped to ≥ 0 (hours)."""

    effect_half_life_h: float
    """``ln(2) / ke0`` — how long effect site takes to halve its gap with plasma."""


@dataclass(frozen=True)
class EffectTimeCourse:
    """Immutable effect-compartment + Emax trajectory.

    Produced by ``EffectCompartmentEngine.simulate``. All arrays share the
    same time grid as the input plasma profile — the PD layer does *not*
    resample; it respects whatever grid the P1 solver produced.

    Arrays are ``tuple[float, ...]`` to stay frozen and hashable-friendly.
    Callers that need numpy can ``np.asarray(course.effect_site_conc_mg_per_L)``.
    """

    drug_name: str
    effect_name: str
    effect_unit: str

    # Time grid (shared with the plasma input)
    time_h: Tuple[float, ...]

    # Raw plasma input (echoed for convenience so downstream reports don't
    # have to carry two objects around)
    plasma_conc_mg_per_L: Tuple[float, ...]

    # Effect compartment concentration (mg/L at effect site)
    effect_site_conc_mg_per_L: Tuple[float, ...]

    # Effect value on the Emax scale (0 to Emax + baseline)
    effect_values: Tuple[float, ...]

    # Parameters echoed (for reproducibility / reports)
    ke0_per_h: float
    ec50_mg_per_L: float
    emax: float
    hill: float
    baseline: float

    # Scalar summary
    summary: EffectTimingSummary

    def __post_init__(self) -> None:
        n = len(self.time_h)
        if (
            len(self.plasma_conc_mg_per_L) != n
            or len(self.effect_site_conc_mg_per_L) != n
            or len(self.effect_values) != n
        ):
            raise ValueError(
                "EffectTimeCourse arrays must share a common length "
                f"(got time={n}, Cp={len(self.plasma_conc_mg_per_L)}, "
                f"Ce={len(self.effect_site_conc_mg_per_L)}, "
                f"E={len(self.effect_values)})"
            )
        if n > 0 and self.time_h[0] < 0.0:
            raise ValueError("time_h must start at ≥ 0")

    @property
    def n_points(self) -> int:
        return len(self.time_h)


# ════════════════════════════════════════════════════════════════
# RECEPTOR OCCUPANCY (Step 13)
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ReceptorOccupancySummary:
    """Scalar summary of a single receptor occupancy trajectory.

    Occupancy is a dimensionless fraction in [0, 1]. The therapeutic window
    calculation walks the trace once and reports the *first* sustained
    crossing above and below ``therapeutic_occupancy`` — so the reported
    duration always represents a contiguous on-target interval.
    """

    peak_occupancy: float
    """Maximum receptor occupancy achieved (0–1)."""

    peak_occupancy_time_h: float
    """Time at which peak occupancy occurs (h)."""

    time_above_therapeutic_h: float
    """Total time during which RO ≥ therapeutic_occupancy (h)."""

    first_crossing_above_h: Optional[float]
    """First time the trace crosses ``therapeutic_occupancy`` from below.
    ``None`` if the threshold is never reached."""

    last_occupancy: float
    """Final occupancy at the end of the simulation — diagnostic for how
    much target engagement persists past the last sample."""

    is_irreversible: bool
    """Echoed from the receptor target — helps downstream reports
    distinguish aspirin-style persistence from propranolol-style
    equilibrium binding."""


@dataclass(frozen=True)
class ReceptorOccupancyTrace:
    """Immutable RO trajectory from the receptor kinetics engine.

    Produced by ``ReceptorKineticsEngine.simulate``. Arrays share the same
    time grid as the PBPK plasma input — no resampling. The plasma input is
    echoed for reports so downstream code doesn't need to carry two
    aligned structures.
    """

    drug_name: str
    target_name: str

    # Time grid (shared with the plasma input)
    time_h: Tuple[float, ...]
    plasma_conc_mg_per_L: Tuple[float, ...]

    # The thing we're actually solving for
    occupancy: Tuple[float, ...]
    """Receptor occupancy fraction at each sample (0–1)."""

    # Echoed kinetic parameters (for reproducibility / reports)
    kon_per_mg_per_L_per_h: float
    koff_per_h: float
    irreversible: bool
    recovery_half_life_h: Optional[float]
    therapeutic_occupancy: float

    # Scalar summary
    summary: ReceptorOccupancySummary

    def __post_init__(self) -> None:
        n = len(self.time_h)
        if (
            len(self.plasma_conc_mg_per_L) != n
            or len(self.occupancy) != n
        ):
            raise ValueError(
                "ReceptorOccupancyTrace arrays must share a common length "
                f"(got time={n}, Cp={len(self.plasma_conc_mg_per_L)}, "
                f"RO={len(self.occupancy)})"
            )
        if n > 0 and self.time_h[0] < 0.0:
            raise ValueError("time_h must start at ≥ 0")
        for ro in self.occupancy:
            if not (0.0 <= ro <= 1.0 + 1e-9):
                raise ValueError(
                    f"occupancy values must be in [0, 1]; got {ro}"
                )

    @property
    def n_points(self) -> int:
        return len(self.time_h)


# ════════════════════════════════════════════════════════════════
# TOXICITY REPORT (Step 14)
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ToxicityReport:
    """Organ-level severity ladder for a single simulation.

    Produced by :class:`ToxicityEvaluatorEngine.evaluate`. The report walks
    the plasma profile against the drug's ``toxicity_thresholds`` and emits
    a frozen list of ``ClinicalEffect``s, one per threshold that was crossed.

    The engine uses *linear interpolation* between adjacent time points to
    determine the first-crossing time — this is consistent with the
    sub-sample precision used elsewhere in the V1 engine (receptor
    occupancy therapeutic window, AUC trapezoid).

    Scalar summaries are derived deterministically from the triggered
    effects so callers can render a one-line verdict without re-walking
    the list.
    """

    drug_name: str
    peak_plasma_mg_per_L: float
    peak_plasma_time_h: float

    triggered_effects: Tuple[ClinicalEffect, ...]
    """Every clinical effect triggered at any time during the simulation,
    sorted by severity rank (CRITICAL → MILD), then by trigger time."""

    organ_worst_severity: Mapping[EffectOrgan, ToxicitySeverity] = field(
        default_factory=dict
    )
    """Per-organ worst severity achieved during the simulation."""

    overall_severity: Optional[ToxicitySeverity] = None
    """Highest severity across all organs (``None`` if nothing triggered)."""

    therapeutic_index: Optional[float] = None
    """Ratio of the lowest SEVERE/CRITICAL threshold to the EC50 (effect-
    producing concentration). ``None`` if either the Emax model or a
    severe threshold is missing. Higher is safer — a TI of 2 means the
    severe-toxicity concentration is only 2× the therapeutic EC50."""

    first_severe_trigger_time_h: Optional[float] = None
    """Simulation time at which the first SEVERE or CRITICAL effect was
    triggered. ``None`` if nothing at that severity was crossed."""

    @property
    def has_any_toxicity(self) -> bool:
        return len(self.triggered_effects) > 0

    @property
    def n_effects(self) -> int:
        return len(self.triggered_effects)

    @property
    def verdict(self) -> str:
        """One-line human-readable summary for reports."""
        if self.overall_severity is None:
            return f"{self.drug_name}: no toxicity thresholds crossed"
        return (
            f"{self.drug_name}: {self.overall_severity.value.upper()} "
            f"({self.n_effects} effect"
            f"{'s' if self.n_effects != 1 else ''} triggered)"
        )

    def __post_init__(self) -> None:
        if self.peak_plasma_mg_per_L < 0.0:
            raise ValueError("peak_plasma_mg_per_L must be ≥ 0")
        if self.peak_plasma_time_h < 0.0:
            raise ValueError("peak_plasma_time_h must be ≥ 0")
        if (
            self.therapeutic_index is not None
            and self.therapeutic_index <= 0.0
        ):
            raise ValueError("therapeutic_index must be > 0 when set")


# ════════════════════════════════════════════════════════════════
# CLINICAL OUTCOME (Step 15)
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ClinicalOutcome:
    """Top-level composite result of the Phase 4 PD engine.

    Produced by :class:`PDOrchestrator.run`. This is the single object
    that ``orchestrator_v2`` hands back to the caller (wrapped inside a
    :class:`SimulationResultV2`). Every sub-field mirrors what the
    sub-engines produced — none of it is recomputed here — plus a few
    derived scalars that callers ask for repeatedly.

    All three trajectory fields are optional because a ``PDProfile`` may
    have any subset of (effect compartment + Emax), (receptor kinetics),
    and (toxicity thresholds) populated. ``toxicity_report`` is always
    present — a drug with no thresholds yields an empty report with
    ``has_any_toxicity = False`` — so callers can render the verdict line
    without a None check.
    """

    drug_name: str

    effect_course: Optional[EffectTimeCourse] = None
    """Effect compartment + Emax trajectory, if the drug has both."""

    receptor_trace: Optional[ReceptorOccupancyTrace] = None
    """Receptor occupancy trajectory, if the drug has a receptor target."""

    toxicity_report: Optional[ToxicityReport] = None
    """Organ severity ladder. Always populated by the orchestrator (may
    be empty) — ``Optional`` only to allow construction of outcome
    objects in tests without running the full toxicity pass."""

    # ── Derived scalars (echoed / computed from the trajectories) ──

    peak_effect_fraction: Optional[float] = None
    """``effect_peak / emax`` — the fraction of maximum pharmacological
    effect reached during the simulation (0 to 1+). ``None`` if no Emax
    model was run."""

    time_to_peak_effect_h: Optional[float] = None
    """Simulation time at which ``effect_peak`` occurs. ``None`` if no
    effect compartment was run."""

    hysteresis_lag_h: Optional[float] = None
    """``t_effect_max - t_plasma_max``. ``None`` if no effect compartment."""

    receptor_peak_occupancy: Optional[float] = None
    """Max receptor occupancy (0–1). ``None`` if no receptor kinetics."""

    therapeutic_index: Optional[float] = None
    """Echo of ``toxicity_report.therapeutic_index`` — surfaced at the
    outcome level so the one-line verdict can render without peeking
    into the sub-report."""

    overall_severity: Optional[ToxicitySeverity] = None
    """Echo of ``toxicity_report.overall_severity``. ``None`` if no
    thresholds were crossed."""

    @property
    def has_effect_data(self) -> bool:
        return self.effect_course is not None

    @property
    def has_receptor_data(self) -> bool:
        return self.receptor_trace is not None

    @property
    def has_toxicity(self) -> bool:
        return (
            self.toxicity_report is not None
            and self.toxicity_report.has_any_toxicity
        )

    @property
    def verdict(self) -> str:
        """One-line human-readable summary for reports.

        Prefers overall severity if toxicity is present, else falls back
        to peak effect %, else "no PD data". Always starts with the
        drug name for easy log grepping.
        """
        if self.has_toxicity and self.overall_severity is not None:
            return (
                f"{self.drug_name}: "
                f"{self.overall_severity.value.upper()}"
                + (
                    f" (peak effect "
                    f"{100.0 * self.peak_effect_fraction:.0f}%)"
                    if self.peak_effect_fraction is not None
                    else ""
                )
            )
        if self.peak_effect_fraction is not None:
            return (
                f"{self.drug_name}: peak effect "
                f"{100.0 * self.peak_effect_fraction:.0f}%, no toxicity"
            )
        if self.has_receptor_data and self.receptor_peak_occupancy is not None:
            return (
                f"{self.drug_name}: peak RO "
                f"{100.0 * self.receptor_peak_occupancy:.0f}%"
            )
        return f"{self.drug_name}: no PD data"

    def to_summary_lines(self) -> Tuple[str, ...]:
        """Layered multi-line summary for clinical-style reports."""
        lines = [self.verdict]
        if self.effect_course is not None:
            s = self.effect_course.summary
            lines.append(
                f"  Effect:   peak {s.effect_peak:.1f} "
                f"{self.effect_course.effect_unit} at "
                f"t = {s.effect_peak_time_h:.1f} h "
                f"(lag {s.hysteresis_lag_h:.1f} h, "
                f"t½_eff {s.effect_half_life_h:.1f} h)"
            )
        if self.receptor_trace is not None:
            rs = self.receptor_trace.summary
            lines.append(
                f"  Receptor: peak RO {rs.peak_occupancy:.2f} at "
                f"t = {rs.peak_occupancy_time_h:.1f} h, "
                f"time above therapeutic = "
                f"{rs.time_above_therapeutic_h:.1f} h"
            )
        if self.toxicity_report is not None and self.has_toxicity:
            for eff in self.toxicity_report.triggered_effects:
                lines.append(
                    f"  [{eff.severity.value.upper():8s}] "
                    f"{eff.organ.value}: {eff.symptom} "
                    f"@ t = {eff.triggered_at_h:.1f} h "
                    f"(Cp {eff.triggered_concentration_mg_per_L:.2f} mg/L)"
                )
        if self.therapeutic_index is not None:
            lines.append(
                f"  Therapeutic index: {self.therapeutic_index:.1f}"
            )
        return tuple(lines)

    def __post_init__(self) -> None:
        if self.peak_effect_fraction is not None and self.peak_effect_fraction < 0.0:
            raise ValueError("peak_effect_fraction must be ≥ 0 when set")
        if (
            self.receptor_peak_occupancy is not None
            and not (0.0 <= self.receptor_peak_occupancy <= 1.0 + 1e-9)
        ):
            raise ValueError("receptor_peak_occupancy must be in [0, 1]")
        if (
            self.therapeutic_index is not None
            and self.therapeutic_index <= 0.0
        ):
            raise ValueError("therapeutic_index must be > 0 when set")
