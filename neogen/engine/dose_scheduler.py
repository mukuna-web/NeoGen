"""DoseScheduler — route, timing, and regimen management.

Converts clinical dosing instructions into DoseSchedule objects
that the SimulationOrchestrator can execute.

Architecture reference: Module #2 — DoseScheduler (Blueprint p.5)
"""

from __future__ import annotations

import logging
from typing import Optional

from neogen.core.entities.formulation import (
    DoseEvent,
    DoseSchedule,
    Formulation,
    FormulationType,
    Route,
)

logger = logging.getLogger(__name__)


# Standard dosing regimen abbreviations
REGIMEN_INTERVALS: dict[str, float] = {
    "QD": 24.0,  # Once daily
    "BID": 12.0,  # Twice daily
    "TID": 8.0,  # Three times daily
    "QID": 6.0,  # Four times daily
    "Q8H": 8.0,  # Every 8 hours
    "Q12H": 12.0,  # Every 12 hours
    "Q24H": 24.0,  # Every 24 hours
    "STAT": 0.0,  # Single dose, immediately
}


class DoseScheduler:
    """Creates DoseSchedule objects from clinical dosing parameters.

    Usage:
        scheduler = DoseScheduler()

        # Single oral dose
        schedule = scheduler.single_oral("midazolam", 5.0)

        # IV bolus
        schedule = scheduler.iv_bolus("midazolam", 2.0)

        # Repeated dosing
        schedule = scheduler.repeated("ibuprofen", 400.0, "TID", days=3)

        # DDI study design
        schedule = scheduler.ddi_study(
            perpetrator="ketoconazole", perp_dose=400, perp_regimen="QD",
            victim="midazolam", victim_dose=5,
            pretreatment_days=3,
        )
    """

    def single_oral(
        self,
        drug_name: str,
        dose_mg: float,
        formulation: Optional[Formulation] = None,
        end_time_h: float = 96.0,
    ) -> DoseSchedule:
        """Single oral dose at t=0."""
        form = formulation or Formulation(
            formulation_type=FormulationType.IR_TABLET
        )
        event = DoseEvent(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=Route.ORAL,
            time_h=0.0,
            formulation=form,
        )
        return DoseSchedule(events=(event,), end_time_h=end_time_h)

    def iv_bolus(
        self,
        drug_name: str,
        dose_mg: float,
        end_time_h: float = 96.0,
    ) -> DoseSchedule:
        """Single IV bolus at t=0."""
        event = DoseEvent(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=Route.IV_BOLUS,
            time_h=0.0,
            formulation=Formulation(
                formulation_type=FormulationType.IV_SOLUTION
            ),
        )
        return DoseSchedule(events=(event,), end_time_h=end_time_h)

    def iv_infusion(
        self,
        drug_name: str,
        dose_mg: float,
        duration_h: float = 1.0,
        end_time_h: float = 96.0,
    ) -> DoseSchedule:
        """Single IV infusion over specified duration."""
        event = DoseEvent(
            drug_name=drug_name,
            dose_mg=dose_mg,
            route=Route.IV_INFUSION,
            time_h=0.0,
            infusion_duration_h=duration_h,
            formulation=Formulation(
                formulation_type=FormulationType.IV_SOLUTION
            ),
        )
        return DoseSchedule(events=(event,), end_time_h=end_time_h)

    def repeated(
        self,
        drug_name: str,
        dose_mg: float,
        regimen: str = "QD",
        days: int = 1,
        route: Route = Route.ORAL,
        end_time_h: Optional[float] = None,
    ) -> DoseSchedule:
        """Repeated dosing regimen.

        Args:
            drug_name: Drug name
            dose_mg: Dose per administration
            regimen: Dosing frequency ("QD", "BID", "TID", etc.)
            days: Number of days
            route: Administration route
            end_time_h: Simulation end time (default: last dose + 96h)
        """
        interval_h = REGIMEN_INTERVALS.get(regimen.upper())
        if interval_h is None:
            raise ValueError(
                f"Unknown regimen: {regimen}. "
                f"Available: {list(REGIMEN_INTERVALS.keys())}"
            )

        if interval_h == 0.0:
            # STAT = single dose
            return self.single_oral(drug_name, dose_mg)

        n_doses = int(days * 24.0 / interval_h)
        formulation = Formulation(
            formulation_type=(
                FormulationType.IV_SOLUTION
                if route in (Route.IV_BOLUS, Route.IV_INFUSION)
                else FormulationType.IR_TABLET
            )
        )

        events = tuple(
            DoseEvent(
                drug_name=drug_name,
                dose_mg=dose_mg,
                route=route,
                time_h=i * interval_h,
                formulation=formulation,
            )
            for i in range(n_doses)
        )

        if end_time_h is None:
            end_time_h = (n_doses - 1) * interval_h + 96.0

        schedule = DoseSchedule(events=events, end_time_h=end_time_h)
        logger.info(
            f"[DoseScheduler] {drug_name} {dose_mg}mg {regimen} × {days}d "
            f"= {n_doses} doses over {(n_doses-1)*interval_h:.0f}h"
        )
        return schedule

    def ddi_study(
        self,
        perpetrator: str,
        perp_dose_mg: float,
        perp_regimen: str = "QD",
        victim: str = "",
        victim_dose_mg: float = 0.0,
        victim_route: Route = Route.ORAL,
        pretreatment_days: int = 3,
        victim_time_offset_h: float = 0.0,
        end_time_h: float = 96.0,
    ) -> DoseSchedule:
        """Standard clinical DDI study design.

        Pattern:
          1. Give perpetrator for pretreatment_days to reach steady-state
          2. On the last day, co-administer victim
          3. Monitor victim PK for comparison vs. victim alone

        This is the standard FDA DDI study design.
        """
        perp_interval = REGIMEN_INTERVALS.get(perp_regimen.upper(), 24.0)
        perp_n_doses = int(pretreatment_days * 24.0 / perp_interval)
        victim_time = (pretreatment_days - 1) * 24.0 + victim_time_offset_h

        perp_events = tuple(
            DoseEvent(
                drug_name=perpetrator,
                dose_mg=perp_dose_mg,
                route=Route.ORAL,
                time_h=i * perp_interval,
            )
            for i in range(perp_n_doses)
        )

        victim_event = DoseEvent(
            drug_name=victim,
            dose_mg=victim_dose_mg,
            route=victim_route,
            time_h=victim_time,
        )

        total_end = victim_time + end_time_h
        schedule = DoseSchedule(
            events=perp_events + (victim_event,),
            end_time_h=total_end,
        )
        logger.info(
            f"[DoseScheduler] DDI study: {perpetrator} {perp_dose_mg}mg "
            f"{perp_regimen} × {pretreatment_days}d, then {victim} "
            f"{victim_dose_mg}mg at t={victim_time:.0f}h"
        )
        return schedule

    @staticmethod
    def combine(*schedules: DoseSchedule) -> DoseSchedule:
        """Combine multiple DoseSchedules into one.

        Useful for multi-drug simulations not covered by ddi_study().
        """
        all_events: list[DoseEvent] = []
        max_end = 0.0
        for s in schedules:
            all_events.extend(s.events)
            max_end = max(max_end, s.end_time_h)
        # Sort events by time
        sorted_events = tuple(sorted(all_events, key=lambda e: e.time_h))
        return DoseSchedule(events=sorted_events, end_time_h=max_end)
