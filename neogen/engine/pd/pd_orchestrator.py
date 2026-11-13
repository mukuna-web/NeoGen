"""PD orchestrator — V1 P4 Step 15.

Stateless facade that composes the three Phase 4 sub-engines into a
single :class:`ClinicalOutcome`:

  * :class:`EffectCompartmentEngine` — plasma → effect-site → Emax effect
  * :class:`ReceptorKineticsEngine`  — plasma → receptor occupancy
  * :class:`ToxicityEvaluatorEngine` — plasma → severity ladder

Dispatch is driven entirely by what the caller's :class:`PDProfile`
contains. Every sub-engine is optional; a drug with no PD data at all
still produces a well-formed (mostly-None) ``ClinicalOutcome`` instead
of raising — this keeps ``orchestrator_v2`` branch-free when it iterates
over drugs with uneven PD coverage.

Design rules
------------

1. The orchestrator owns *no* physics. It only calls sub-engines and
   assembles their outputs.
2. The orchestrator is pure: given the same inputs it produces the same
   frozen output, and it never mutates the incoming profile.
3. Sub-engine exceptions propagate unchanged. The orchestrator does not
   "rescue" them — if the effect compartment ODE blows up, the caller
   should see that, not a silently-empty outcome.
4. All derived scalars (peak effect %, hysteresis lag, TI) are computed
   from the sub-engine summaries, never from the raw trajectories — the
   summaries are the contract, and duplicating the summary math here
   would be a source of drift.
"""
from __future__ import annotations

from typing import Optional, Sequence

from neogen.core.entities.pd import PDProfile
from neogen.engine.pd.effect_compartment import EffectCompartmentEngine
from neogen.engine.pd.pd_result import (
    ClinicalOutcome,
    EffectTimeCourse,
    ReceptorOccupancyTrace,
    ToxicityReport,
)
from neogen.engine.pd.receptor_kinetics import ReceptorKineticsEngine
from neogen.engine.pd.toxicity_evaluator import ToxicityEvaluatorEngine


class PDOrchestrator:
    """Compose the three P4 sub-engines into a ``ClinicalOutcome``.

    Typical usage::

        orchestrator = PDOrchestrator()
        outcome = orchestrator.run(
            drug_name=result.drug_name,
            time_h=result.time_h,
            plasma_conc_mg_per_L=result.plasma_conc_mg_per_L,
            pd_profile=drug.pd_profile,
        )
        for line in outcome.to_summary_lines():
            print(line)

    The class is reusable and stateless — hold onto a single instance
    and call ``run`` once per drug, or create a fresh one per call. Both
    patterns are equivalent.
    """

    def __init__(self) -> None:
        self._effect_engine = EffectCompartmentEngine()
        self._receptor_engine = ReceptorKineticsEngine()
        self._toxicity_engine = ToxicityEvaluatorEngine()

    def run(
        self,
        *,
        drug_name: str,
        time_h: Sequence[float],
        plasma_conc_mg_per_L: Sequence[float],
        pd_profile: PDProfile,
    ) -> ClinicalOutcome:
        """Dispatch to each sub-engine and assemble a ``ClinicalOutcome``.

        Raises ``ValueError`` only if the input trajectories are
        structurally invalid (shape mismatch, negative time, etc.) —
        missing PD sub-models are silently skipped.
        """
        # ── Effect compartment + Emax ─────────────────────────────
        effect_course: Optional[EffectTimeCourse] = None
        if pd_profile.has_effect_compartment and pd_profile.has_emax_model:
            effect_course = self._effect_engine.simulate(
                drug_name=drug_name,
                time_h=time_h,
                plasma_conc_mg_per_L=plasma_conc_mg_per_L,
                pd_profile=pd_profile,
            )

        # ── Receptor kinetics ─────────────────────────────────────
        receptor_trace: Optional[ReceptorOccupancyTrace] = None
        if pd_profile.has_receptor_kinetics:
            receptor_trace = self._receptor_engine.simulate(
                drug_name=drug_name,
                time_h=time_h,
                plasma_conc_mg_per_L=plasma_conc_mg_per_L,
                pd_profile=pd_profile,
            )

        # ── Toxicity (always runs; empty report if no thresholds) ─
        toxicity_report: ToxicityReport = self._toxicity_engine.evaluate(
            drug_name=drug_name,
            time_h=time_h,
            plasma_conc_mg_per_L=plasma_conc_mg_per_L,
            pd_profile=pd_profile,
        )

        # ── Derived scalars ───────────────────────────────────────
        peak_effect_fraction: Optional[float] = None
        time_to_peak_effect_h: Optional[float] = None
        hysteresis_lag_h: Optional[float] = None
        if effect_course is not None and pd_profile.emax is not None:
            emax = pd_profile.emax.emax
            if emax > 0.0:
                # Subtract baseline so the "fraction of Emax" reading is
                # honest — a drug with baseline 50% and peak 60% has
                # used only 10/Emax of its capacity, not 60/Emax.
                peak_effect_fraction = max(
                    0.0,
                    (
                        effect_course.summary.effect_peak
                        - pd_profile.emax.baseline
                    )
                    / emax,
                )
            time_to_peak_effect_h = effect_course.summary.effect_peak_time_h
            hysteresis_lag_h = effect_course.summary.hysteresis_lag_h

        receptor_peak_occupancy: Optional[float] = None
        if receptor_trace is not None:
            receptor_peak_occupancy = receptor_trace.summary.peak_occupancy

        return ClinicalOutcome(
            drug_name=drug_name,
            effect_course=effect_course,
            receptor_trace=receptor_trace,
            toxicity_report=toxicity_report,
            peak_effect_fraction=peak_effect_fraction,
            time_to_peak_effect_h=time_to_peak_effect_h,
            hysteresis_lag_h=hysteresis_lag_h,
            receptor_peak_occupancy=receptor_peak_occupancy,
            therapeutic_index=toxicity_report.therapeutic_index,
            overall_severity=toxicity_report.overall_severity,
        )
