"""Population & Patient Variability — V1 Engine Phase 5.

This package turns the V1 engine from a *reference individual* simulator
into a *patient-aware* simulator. It composes small, pure functions that
transform a ``Drug`` / ``PhysiologyModel`` in response to a
``PatientProfile``:

    * CYP phenotype activity scaling (PM/IM/EM/UM)
    * Hepatic impairment (Child-Pugh-style CLint scaling)
    * Renal impairment (GFR-based CLr scaling)
    * Smoker induction (CYP1A2)
    * Pediatric / geriatric age scaling
    * Monte Carlo population sampling

All transforms are **pure**, **frozen**, and **backward compatible** —
when ``patient`` is ``None`` the engine behaves exactly as before.
"""
from __future__ import annotations

from neogen.engine.population.patient_adapter import (
    age_cyp_maturation_factor,
    apply_patient_to_drug,
    apply_patient_to_physiology,
    effective_cyp_activity_factor,
    gfr_for_patient,
    hepatic_impairment_factor,
    smoker_cyp1a2_factor,
    total_clint_scale,
)

__all__ = [
    "age_cyp_maturation_factor",
    "apply_patient_to_drug",
    "apply_patient_to_physiology",
    "effective_cyp_activity_factor",
    "gfr_for_patient",
    "hepatic_impairment_factor",
    "smoker_cyp1a2_factor",
    "total_clint_scale",
]
