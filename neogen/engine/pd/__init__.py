"""NeoGen V1 — Phase 4 Pharmacodynamics Engine.

Translates plasma concentration into pharmacological effect and clinical
outcome. Built step-by-step per the V1 execution plan:

Step 12 — effect_compartment.py + emax_model.py   (complete)
Step 13 — receptor_kinetics.py                     (complete)
Step 14 — toxicity_evaluator.py                    (complete)
Step 15 — pd_orchestrator.py                       (THIS STEP — V1 P4 close)

Every module is stateless at the class level; state lives only inside the
frozen result dataclasses that flow out of each pass.
"""
from __future__ import annotations

from neogen.engine.pd.effect_compartment import (
    EffectCompartmentEngine,
    simulate_ce_trajectory,
)
from neogen.engine.pd.emax_model import (
    apply_emax,
    apply_emax_array,
    ec_for_effect_fraction,
    peak_effect,
)
from neogen.engine.pd.pd_orchestrator import PDOrchestrator
from neogen.engine.pd.pd_result import (
    ClinicalOutcome,
    EffectTimeCourse,
    EffectTimingSummary,
    ReceptorOccupancySummary,
    ReceptorOccupancyTrace,
    ToxicityReport,
)
from neogen.engine.pd.receptor_kinetics import (
    ReceptorKineticsEngine,
    simulate_ro_trajectory,
)
from neogen.engine.pd.toxicity_evaluator import ToxicityEvaluatorEngine

__all__ = [
    # Engines
    "EffectCompartmentEngine",
    "simulate_ce_trajectory",
    "ReceptorKineticsEngine",
    "simulate_ro_trajectory",
    "ToxicityEvaluatorEngine",
    "PDOrchestrator",
    # Emax pure functions
    "apply_emax",
    "apply_emax_array",
    "ec_for_effect_fraction",
    "peak_effect",
    # Frozen results
    "EffectTimeCourse",
    "EffectTimingSummary",
    "ReceptorOccupancyTrace",
    "ReceptorOccupancySummary",
    "ToxicityReport",
    "ClinicalOutcome",
]
