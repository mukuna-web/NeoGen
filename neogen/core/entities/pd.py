"""Pharmacodynamic entities — V1 P4 foundation.

These frozen dataclasses define the data contracts that the Phase 4 PD engine
consumes. They are intentionally small and parameter-only — no physics lives
here. Physics modules under ``neogen/engine/pd/`` will read these structures
to drive effect-site kinetics, Emax/Hill saturation, receptor occupancy, and
organ-level toxicity severity ladders.

Architecture reference: V1 Blueprint — Layer 5 (Pharmacodynamics).

Design rules:
  - Every class is ``frozen=True``. Variants are produced via ``replace()``.
  - Units are explicit in every field docstring (SI, hours, mg/L, µM).
  - ``Optional`` fields default to ``None`` so drugs without PD data remain
    valid (the PD engine will skip the corresponding sub-model).
  - No imports from ``neogen.engine.*`` — entities must stay engine-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


# ════════════════════════════════════════════════════════════════
# ENUMS
# ════════════════════════════════════════════════════════════════


class ToxicitySeverity(Enum):
    """Canonical severity ladder for clinical toxicity effects.

    The ordering matters: ``MILD < MODERATE < SEVERE < CRITICAL``. The
    ``rank`` property returns an integer for comparisons.
    """

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {
            ToxicitySeverity.MILD: 1,
            ToxicitySeverity.MODERATE: 2,
            ToxicitySeverity.SEVERE: 3,
            ToxicitySeverity.CRITICAL: 4,
        }[self]


class EffectOrgan(Enum):
    """Organ systems recognised by the toxicity evaluator.

    Kept as a closed enum rather than a free string so downstream reports
    stay consistent and typo-free. "GENERAL" is the catch-all bucket used
    when a drug has non-organ-specific effects.
    """

    BRAIN = "brain"
    HEART = "heart"
    LIVER = "liver"
    KIDNEY = "kidney"
    LUNG = "lung"
    GI = "gi"
    SKIN = "skin"
    MUSCLE = "muscle"
    BLOOD = "blood"
    GENERAL = "general"


# ════════════════════════════════════════════════════════════════
# EFFECT COMPARTMENT + EMAX
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class EffectCompartmentParams:
    """Parameters for the Sheiner (1979) effect compartment link.

    The effect compartment is a hypothetical pharmacokinetic compartment
    that equilibrates with plasma at a first-order rate ``ke0``. The effect
    site concentration ``Ce(t)`` is what drives the PD response — it lags
    plasma ``Cp(t)`` whenever ``ke0`` is small.

    ODE: ``dCe/dt = ke0 * (Cp(t) - Ce(t))``.

    Typical ranges:
      - Fast-equilibrating (IV anaesthetics, propofol): ``ke0 ≈ 1-3 h⁻¹``
      - Analgesics (morphine): ``ke0 ≈ 0.1-0.3 h⁻¹``
      - Slow PD (warfarin INR): ``ke0 ≈ 0.001-0.005 h⁻¹``
    """

    ke0_per_h: float
    """Effect-site equilibration rate constant (1/h). Must be > 0."""

    t_peak_lag_h: float = 0.0
    """Approximate lag between Cp_max and Ce_max (h). Informational only."""

    description: str = ""
    """Human-readable description of the lag mechanism."""

    def __post_init__(self) -> None:
        if self.ke0_per_h <= 0.0:
            raise ValueError(
                f"ke0_per_h must be > 0 (got {self.ke0_per_h})"
            )


@dataclass(frozen=True)
class EmaxParams:
    """Sigmoidal Emax model parameters.

    Effect model: ``E(Ce) = baseline + Emax * Ce^h / (EC50^h + Ce^h)``.

    Concentrations are in **mg/L at the effect site** to stay consistent
    with the P1 plasma profile units. Convert from µM if needed using the
    drug molecular weight.
    """

    emax: float
    """Maximum achievable effect (arbitrary % scale or biomarker unit)."""

    ec50_mg_per_L: float
    """Effect-site concentration producing 50% of Emax (mg/L). Must be > 0."""

    hill: float = 1.0
    """Hill coefficient (steepness). 1 = hyperbolic, >1 = sigmoidal, <1 = shallow."""

    baseline: float = 0.0
    """Effect at zero drug concentration (same unit as ``emax``)."""

    effect_name: str = ""
    """Human-readable description of the effect (e.g. "Pain relief")."""

    effect_unit: str = "%"
    """Unit of the effect axis (e.g. "%", "mmHg", "INR")."""

    def __post_init__(self) -> None:
        if self.ec50_mg_per_L <= 0.0:
            raise ValueError(
                f"ec50_mg_per_L must be > 0 (got {self.ec50_mg_per_L})"
            )
        if self.hill <= 0.0:
            raise ValueError(f"hill must be > 0 (got {self.hill})")


# ════════════════════════════════════════════════════════════════
# RECEPTOR BINDING KINETICS
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ReceptorTarget:
    """Dynamic receptor binding kinetics (kon/koff model).

    Used for drugs where the time course of target engagement is decoupled
    from plasma PK — most importantly *irreversible* or *slow-off* binders
    whose effect persists long after the drug has been cleared.

    ODE: ``dRO/dt = kon * Cp * (1 - RO) - koff * RO``

    Classic examples:
      - Aspirin + COX-1: effectively irreversible, platelet turnover (~10 d)
        governs recovery, not plasma PK.
      - Omeprazole + H⁺/K⁺-ATPase: covalent inhibition, same story.
      - Clopidogrel + P2Y12: irreversible, ~7 d recovery.
    """

    name: str
    """Target name (e.g. "COX-1", "H+/K+-ATPase")."""

    kon_per_mg_per_L_per_h: float
    """Association rate constant. Units: (mg/L)⁻¹·h⁻¹."""

    koff_per_h: float
    """Dissociation rate constant (1/h). 0 for irreversible binders."""

    irreversible: bool = False
    """If True, the PD engine uses receptor-turnover recovery instead of koff."""

    recovery_half_life_h: Optional[float] = None
    """Biological recovery half-life when ``irreversible`` (e.g. 240 h for
    platelet COX-1). Required if ``irreversible`` is True."""

    therapeutic_occupancy: float = 0.5
    """Occupancy fraction required for clinical effect (0-1)."""

    @property
    def kd_mg_per_L(self) -> Optional[float]:
        """Equilibrium dissociation constant. Undefined for irreversible binders."""
        if self.kon_per_mg_per_L_per_h <= 0.0 or self.koff_per_h <= 0.0:
            return None
        return self.koff_per_h / self.kon_per_mg_per_L_per_h

    def __post_init__(self) -> None:
        if self.kon_per_mg_per_L_per_h <= 0.0:
            raise ValueError("kon_per_mg_per_L_per_h must be > 0")
        if self.koff_per_h < 0.0:
            raise ValueError("koff_per_h must be ≥ 0")
        if self.irreversible and self.recovery_half_life_h is None:
            raise ValueError(
                "recovery_half_life_h is required for irreversible binders"
            )
        if not (0.0 <= self.therapeutic_occupancy <= 1.0):
            raise ValueError("therapeutic_occupancy must be in [0, 1]")


# ════════════════════════════════════════════════════════════════
# TOXICITY THRESHOLDS & CLINICAL EFFECTS
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ToxicityThreshold:
    """A single concentration threshold that triggers a clinical effect.

    The PD engine walks the drug's threshold list and flips on any effect
    whose ``concentration_mg_per_L`` is exceeded anywhere in the simulated
    plasma profile. Ordering within a drug's threshold list should go from
    least to most severe.
    """

    organ: EffectOrgan
    severity: ToxicitySeverity
    concentration_mg_per_L: float
    """Plasma concentration above which this effect is triggered (mg/L)."""

    symptom: str
    """Short human-readable symptom name (e.g. "Nausea")."""

    description: str = ""
    """Longer explanation of the mechanism and clinical meaning."""

    vital_sign: str = ""
    """Which vital sign is affected (e.g. "Heart rate", "BP")."""

    vital_change: str = ""
    """Direction/magnitude of change (e.g. "HR +20-40 bpm")."""

    def __post_init__(self) -> None:
        if self.concentration_mg_per_L < 0.0:
            raise ValueError(
                f"concentration_mg_per_L must be ≥ 0 "
                f"(got {self.concentration_mg_per_L})"
            )


@dataclass(frozen=True)
class ClinicalEffect:
    """A triggered clinical effect — output of the toxicity evaluator.

    Distinct from ``ToxicityThreshold`` (which is an *input* rule): this
    object represents an *observed* effect in a specific simulation, with
    the actual triggering concentration and time.
    """

    organ: EffectOrgan
    severity: ToxicitySeverity
    symptom: str
    triggered_at_h: float
    """Simulation time (h) at which the threshold was first crossed."""

    triggered_concentration_mg_per_L: float
    """Plasma concentration at the trigger time."""

    description: str = ""


# ════════════════════════════════════════════════════════════════
# PD PROFILE (TOP-LEVEL CONTAINER)
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PDProfile:
    """Aggregate PD parameters attached to a ``Drug``.

    Every sub-field is optional: a drug with no PD data at all is still a
    valid drug, and the PD engine will simply return empty results for the
    missing sub-models. Drugs with partial PD data (e.g. toxicity thresholds
    only, no Emax) are also fully supported.

    Typical wiring inside ``DrugFactory``:

        PDProfile(
            effect_compartment=EffectCompartmentParams(ke0_per_h=0.12, ...),
            emax=EmaxParams(emax=100, ec50_mg_per_L=0.025, hill=2.5, ...),
            toxicity_thresholds=(
                ToxicityThreshold(EffectOrgan.BRAIN, ToxicitySeverity.MILD, ...),
                ...
            ),
        )
    """

    effect_compartment: Optional[EffectCompartmentParams] = None
    emax: Optional[EmaxParams] = None
    receptor_target: Optional[ReceptorTarget] = None
    toxicity_thresholds: Tuple[ToxicityThreshold, ...] = field(
        default_factory=tuple
    )

    # Therapeutic window — used for the TI metric in the clinical outcome.
    therapeutic_range_low_mg_per_L: Optional[float] = None
    therapeutic_range_high_mg_per_L: Optional[float] = None

    # Free-text notes (not used by engine, purely for reports).
    notes: str = ""

    @property
    def has_effect_compartment(self) -> bool:
        return self.effect_compartment is not None

    @property
    def has_emax_model(self) -> bool:
        return self.emax is not None

    @property
    def has_receptor_kinetics(self) -> bool:
        return self.receptor_target is not None

    @property
    def has_toxicity_model(self) -> bool:
        return len(self.toxicity_thresholds) > 0

    @property
    def has_any_pd_data(self) -> bool:
        return (
            self.has_effect_compartment
            or self.has_emax_model
            or self.has_receptor_kinetics
            or self.has_toxicity_model
        )

    def max_severity_at(
        self, plasma_conc_mg_per_L: float
    ) -> Optional[ToxicitySeverity]:
        """Return the highest severity triggered by a given plasma level.

        Pure function over the threshold list — does not mutate state and
        does not depend on time. Returns ``None`` if no threshold is crossed.
        """
        worst: Optional[ToxicitySeverity] = None
        for th in self.toxicity_thresholds:
            if plasma_conc_mg_per_L >= th.concentration_mg_per_L:
                if worst is None or th.severity.rank > worst.rank:
                    worst = th.severity
        return worst

    def __post_init__(self) -> None:
        lo = self.therapeutic_range_low_mg_per_L
        hi = self.therapeutic_range_high_mg_per_L
        if lo is not None and lo < 0.0:
            raise ValueError("therapeutic_range_low_mg_per_L must be ≥ 0")
        if hi is not None and hi < 0.0:
            raise ValueError("therapeutic_range_high_mg_per_L must be ≥ 0")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(
                "therapeutic_range_low must be ≤ therapeutic_range_high"
            )
