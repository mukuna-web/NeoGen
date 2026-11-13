"""Formulation, DoseEvent, and DoseSchedule entities.

How a drug enters the body determines the initial conditions of the
simulation. IV bypasses absorption entirely (F=1.0). Oral drugs must
dissolve, cross the GI wall, and survive first-pass metabolism.

Architecture reference: Layer 2 — Formulation & Administration (Blueprint p.3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Route(Enum):
    """Drug administration route."""

    IV_BOLUS = "iv_bolus"
    IV_INFUSION = "iv_infusion"
    ORAL = "oral"
    IM = "intramuscular"  # V2+
    SC = "subcutaneous"  # V2+
    INHALATION = "inhalation"  # V2+

    @property
    def bypasses_first_pass(self) -> bool:
        """Whether this route skips gut wall + hepatic first-pass."""
        return self in (Route.IV_BOLUS, Route.IV_INFUSION)

    @property
    def bioavailability_is_one(self) -> bool:
        """Whether F = 1.0 by definition (IV routes)."""
        return self in (Route.IV_BOLUS, Route.IV_INFUSION)


class FormulationType(Enum):
    """Drug formulation type — affects dissolution rate."""

    SOLUTION = "solution"  # Instant dissolution
    IR_TABLET = "ir_tablet"  # Immediate release: disintegration → dissolution
    IR_CAPSULE = "ir_capsule"
    ENTERIC_COATED = "enteric_coated"  # Bypasses stomach (acid-labile drugs)
    SUSTAINED_RELEASE = "sustained_release"  # V2+
    IV_SOLUTION = "iv_solution"  # Pre-dissolved for injection


@dataclass(frozen=True)
class Formulation:
    """Physical form of the drug product.

    For V1 MVP: only SOLUTION, IR_TABLET, and IV_SOLUTION.
    Sustained-release is deferred to V2+.
    """

    formulation_type: FormulationType = FormulationType.IR_TABLET

    # Particle size (affects dissolution rate for BCS II drugs)
    particle_radius_um: float = 25.0  # µm — typical IR tablet

    # Disintegration time (tablet → granules)
    disintegration_time_min: float = 5.0

    # For enteric-coated: pH threshold for coating dissolution
    enteric_ph_threshold: float = 5.5

    # For sustained-release: release rate constant (V2+)
    release_rate_h: Optional[float] = None


@dataclass(frozen=True)
class DoseEvent:
    """A single drug administration event.

    Represents one dose given at a specific time via a specific route.
    The SimulationOrchestrator converts DoseEvents into ODE initial
    conditions or forcing functions.
    """

    drug_name: str  # Links to Drug entity
    dose_mg: float
    route: Route = Route.ORAL
    time_h: float = 0.0  # Time of administration (hours from simulation start)
    formulation: Formulation = field(default_factory=Formulation)

    # IV infusion duration (only for IV_INFUSION route)
    infusion_duration_h: Optional[float] = None

    @property
    def dose_umol(self) -> Optional[float]:
        """Dose in micromoles — requires MW (set by orchestrator)."""
        return None  # Computed when linked to Drug entity

    def dose_in_umol(self, molecular_weight: float) -> float:
        """Convert dose from mg to µmol given MW."""
        return (self.dose_mg / molecular_weight) * 1000.0

    @property
    def is_iv(self) -> bool:
        return self.route in (Route.IV_BOLUS, Route.IV_INFUSION)

    @property
    def is_oral(self) -> bool:
        return self.route == Route.ORAL

    def __repr__(self) -> str:
        return (
            f"Dose({self.drug_name}, {self.dose_mg}mg, "
            f"{self.route.value}, t={self.time_h}h)"
        )


@dataclass(frozen=True)
class DoseSchedule:
    """A series of dose events for one or more drugs.

    Supports multi-dose regimens (e.g., 500mg BID for 7 days)
    and multi-drug scenarios (Drug A + Drug B co-administered).

    The simulation window extends from the first dose to
    end_time_h (default: 96 hours post-last-dose for DDI).
    """

    events: tuple[DoseEvent, ...] = ()
    end_time_h: float = 96.0  # Total simulation duration

    @property
    def drugs_involved(self) -> tuple[str, ...]:
        """Unique drug names in this schedule."""
        return tuple(sorted(set(e.drug_name for e in self.events)))

    @property
    def n_drugs(self) -> int:
        return len(self.drugs_involved)

    @property
    def is_multi_drug(self) -> bool:
        return self.n_drugs > 1

    @property
    def first_dose_time(self) -> float:
        if not self.events:
            return 0.0
        return min(e.time_h for e in self.events)

    @property
    def last_dose_time(self) -> float:
        if not self.events:
            return 0.0
        return max(e.time_h for e in self.events)

    def events_for_drug(self, drug_name: str) -> tuple[DoseEvent, ...]:
        """Get all dose events for a specific drug."""
        return tuple(e for e in self.events if e.drug_name == drug_name)

    def events_at_time(self, time_h: float, tolerance_h: float = 0.01) -> tuple[DoseEvent, ...]:
        """Get dose events occurring at a specific time."""
        return tuple(
            e for e in self.events
            if abs(e.time_h - time_h) < tolerance_h
        )

    @staticmethod
    def single_dose(drug_name: str, dose_mg: float,
                    route: Route = Route.ORAL,
                    end_time_h: float = 96.0) -> DoseSchedule:
        """Convenience: create a single-dose schedule."""
        event = DoseEvent(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=route,
            time_h=0.0,
        )
        return DoseSchedule(events=(event,), end_time_h=end_time_h)

    @staticmethod
    def multiple_dose(drug_name: str, dose_mg: float,
                      interval_h: float, n_doses: int,
                      route: Route = Route.ORAL,
                      end_time_h: Optional[float] = None) -> DoseSchedule:
        """Convenience: create a repeated-dose schedule (e.g., BID, TID).

        Example:
            DoseSchedule.multiple_dose("ibuprofen", 400, interval_h=8, n_doses=9)
            → 400mg every 8h for 3 days (9 doses)
        """
        events = tuple(
            DoseEvent(
                drug_name=drug_name,
                dose_mg=dose_mg,
                route=route,
                time_h=i * interval_h,
            )
            for i in range(n_doses)
        )
        if end_time_h is None:
            end_time_h = (n_doses - 1) * interval_h + 96.0
        return DoseSchedule(events=events, end_time_h=end_time_h)

    @staticmethod
    def ddi_scenario(
        perpetrator_name: str,
        perpetrator_dose_mg: float,
        perpetrator_interval_h: float,
        perpetrator_n_doses: int,
        victim_name: str,
        victim_dose_mg: float,
        victim_time_h: float = 0.0,
        victim_route: Route = Route.ORAL,
        end_time_h: float = 96.0,
    ) -> DoseSchedule:
        """Convenience: standard DDI study design.

        Perpetrator is given at steady-state. Victim is given once
        at victim_time_h (typically after perpetrator reaches steady-state).

        Example (ketoconazole + midazolam):
            DoseSchedule.ddi_scenario(
                perpetrator_name="ketoconazole",
                perpetrator_dose_mg=400,
                perpetrator_interval_h=24,
                perpetrator_n_doses=5,
                victim_name="midazolam",
                victim_dose_mg=5,
                victim_time_h=96.0,  # Day 5
            )
        """
        perp_events = tuple(
            DoseEvent(
                drug_name=perpetrator_name,
                dose_mg=perpetrator_dose_mg,
                route=Route.ORAL,
                time_h=i * perpetrator_interval_h,
            )
            for i in range(perpetrator_n_doses)
        )
        victim_event = DoseEvent(
            drug_name=victim_name,
            dose_mg=victim_dose_mg,
            route=victim_route,
            time_h=victim_time_h,
        )
        return DoseSchedule(
            events=perp_events + (victim_event,),
            end_time_h=end_time_h,
        )

    def __repr__(self) -> str:
        return (
            f"DoseSchedule({self.n_drugs} drugs, "
            f"{len(self.events)} events, "
            f"{self.end_time_h}h window)"
        )
