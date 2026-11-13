"""NeoGen V1 Mechanistic Simulation Engine.

This package contains the clean mechanistic PBPK engine built according
to the V1 Architecture Blueprint. It operates alongside the existing
neogen.simulation package and will replace it once validated.

Phase 0 (Foundation):
  - DrugFactory: SMILES → Drug entity
  - DoseScheduler: dosing regimen builder

Phase 1 (Core PK Engine):
  - TissuePartitioner: Rodgers-Rowland Kp prediction
  - HepaticClearanceEngine: well-stirred liver model
  - RenalClearanceEngine: mechanistic renal clearance
  - PBPKSolver: 14-compartment BDF ODE system
  - SimulationOrchestrator: end-to-end simulation pipeline
  - PKCalculator: Cmax, AUC, t½ computation

Phase 4 (Pharmacodynamics):
  - EffectCompartmentEngine: Sheiner ke0 + Emax model
  - ReceptorKineticsEngine: kon/koff + irreversible recovery
  - ToxicityEvaluatorEngine: organ severity ladder
  - PDOrchestrator: composition facade
  - OrchestratorV2: full P1 → P4 chain + SimulationResultV2
"""

# P0
from neogen.engine.drug_factory import DrugFactory
from neogen.engine.dose_scheduler import DoseScheduler

# P1
from neogen.engine.tissue_partitioner import TissuePartitioner, PartitionCoefficients
from neogen.engine.hepatic_clearance import HepaticClearanceEngine, HepaticClearanceResult
from neogen.engine.renal_clearance import RenalClearanceEngine, RenalClearanceResult
from neogen.engine.pbpk_solver import PBPKSolver, SolverConfig
from neogen.engine.simulation_result import SimulationResult, PKMetrics, PKCalculator
from neogen.engine.orchestrator import SimulationOrchestrator

# P4
from neogen.engine.pd import (
    ClinicalOutcome,
    EffectCompartmentEngine,
    EffectTimeCourse,
    EffectTimingSummary,
    PDOrchestrator,
    ReceptorKineticsEngine,
    ReceptorOccupancySummary,
    ReceptorOccupancyTrace,
    ToxicityEvaluatorEngine,
    ToxicityReport,
)
from neogen.engine.orchestrator_v2 import (
    ENGINE_VERSION,
    OrchestratorV2,
    SimulationResultV2,
)

__all__ = [
    # P0
    "DrugFactory",
    "DoseScheduler",
    # P1
    "TissuePartitioner",
    "PartitionCoefficients",
    "HepaticClearanceEngine",
    "HepaticClearanceResult",
    "RenalClearanceEngine",
    "RenalClearanceResult",
    "PBPKSolver",
    "SolverConfig",
    "SimulationResult",
    "PKMetrics",
    "PKCalculator",
    "SimulationOrchestrator",
    # P4
    "EffectCompartmentEngine",
    "ReceptorKineticsEngine",
    "ToxicityEvaluatorEngine",
    "PDOrchestrator",
    "EffectTimeCourse",
    "EffectTimingSummary",
    "ReceptorOccupancyTrace",
    "ReceptorOccupancySummary",
    "ToxicityReport",
    "ClinicalOutcome",
    "OrchestratorV2",
    "SimulationResultV2",
    "ENGINE_VERSION",
]
