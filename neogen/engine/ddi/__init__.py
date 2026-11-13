"""NeoGen V1 — Phase 3 DDI Engine.

Mechanistic drug-drug interaction layer that makes clearance state-dependent.
Built step-by-step per the V1 execution plan:

Step 7 — cyp_inhibition.py          (static reversible competitive)
Step 8 — cyp_induction.py           (steady-state Emax induction)
Step 9 — enzyme_pool_evolver.py     (time-dependent MBI + turnover ODE)
Step 10 — ddi_orchestrator.py       (runtime PBPK coupling)
V1.2-A — gut_wall_mbi.py            (steady-state enterocyte CYP3A4 MBI)
V1.2-B — transporter_ddi.py         (steady-state OATP1B1 IC50 modulation)

Everything is stateless at the module level. Results are frozen dataclasses
that flow between phases without mutation.
"""
from __future__ import annotations

from neogen.engine.ddi.autoinduction import (
    AutoinductionSimulator,
    AutoinductionSummary,
    is_autoinducer,
    overlapping_cyps,
)
from neogen.engine.ddi.cyp_inhibition import (
    CYP_NAMES,
    CYPInhibitionContribution,
    CYPInhibitionEngine,
    InhibitionState,
)
from neogen.engine.ddi.cyp_induction import (
    INDUCIBLE_CYP_NAMES,
    CYPInductionContribution,
    CYPInductionEngine,
    InductionState,
)
from neogen.engine.ddi.enzyme_pool_evolver import (
    DEFAULT_KDEG_PER_H,
    EnzymePoolEvolver,
    EnzymePoolTrace,
    EnzymeSnapshot,
)
from neogen.engine.ddi.ddi_result import (
    CLintBreakdown,
    DDISummary,
    PerpetratorExposure,
)
from neogen.engine.ddi.ddi_orchestrator import (
    CLintProvider,
    DDIOrchestrator,
    well_stirred_clh,
)
from neogen.engine.ddi.gut_wall_mbi import (
    DEFAULT_GUT_EXTRACTION_ALPHA,
    ENTEROCYTE_CYP3A4_HALF_LIFE_H,
    ENTEROCYTE_CYP3A4_KDEG_PER_H,
    GUT_LIVER_CYP3A4_RATIO,
    GutWallMBIDiagnostic,
    GutWallInductionExposure,
    GutWallMBIExposure,
    baseline_segment_fg,
    compute_cumulative_fg,
    compute_steady_state_fg_per_segment,
    diagnose_gut_wall_mbi,
    mbi_segment_fg,
    qgut_segment_fg,
)
from neogen.engine.ddi.transporter_ddi import (
    DEFAULT_OATP1B1_UPTAKE_FACTOR,
    OATP1B1DDIDiagnostic,
    OATP1B1InhibitorExposure,
    compute_oatp1b1_uptake_factor,
    diagnose_oatp1b1_ddi,
)

__all__ = [
    "AutoinductionSimulator",
    "AutoinductionSummary",
    "is_autoinducer",
    "overlapping_cyps",
    "CYP_NAMES",
    "CYPInhibitionContribution",
    "CYPInhibitionEngine",
    "InhibitionState",
    "INDUCIBLE_CYP_NAMES",
    "CYPInductionContribution",
    "CYPInductionEngine",
    "InductionState",
    "DEFAULT_KDEG_PER_H",
    "EnzymePoolEvolver",
    "EnzymePoolTrace",
    "EnzymeSnapshot",
    "CLintBreakdown",
    "CLintProvider",
    "DDIOrchestrator",
    "DDISummary",
    "PerpetratorExposure",
    "well_stirred_clh",
    # V1.2 gut-wall MBI (Track A)
    "DEFAULT_GUT_EXTRACTION_ALPHA",
    "ENTEROCYTE_CYP3A4_HALF_LIFE_H",
    "ENTEROCYTE_CYP3A4_KDEG_PER_H",
    "GutWallInductionExposure",
    "GutWallMBIExposure",
    "GutWallMBIDiagnostic",
    "baseline_segment_fg",
    "mbi_segment_fg",
    "qgut_segment_fg",
    "GUT_LIVER_CYP3A4_RATIO",
    "compute_steady_state_fg_per_segment",
    "compute_cumulative_fg",
    "diagnose_gut_wall_mbi",
    # V1.2 OATP1B1 transporter DDI (Track B)
    "DEFAULT_OATP1B1_UPTAKE_FACTOR",
    "OATP1B1InhibitorExposure",
    "OATP1B1DDIDiagnostic",
    "compute_oatp1b1_uptake_factor",
    "diagnose_oatp1b1_ddi",
]
