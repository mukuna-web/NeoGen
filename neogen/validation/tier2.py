"""Tier 2 — DDI pair AUCR validation.

For each pair in the cohort, simulate the victim alone and then
the victim under the perpetrator using the P3 DDIOrchestrator's
CLint provider path. Compute AUCR_predicted = AUC_ddi / AUC_baseline
and compare to the clinical AUCR.

This harness deliberately uses the same low-level PBPKSolver +
DDIOrchestrator path that tests/test_v1_p3_acceptance_gate.py
uses. OrchestratorV2 does not yet wire DDI end-to-end (the ``ddi``
slot on SimulationResultV2 is always None); that is a known gap
and will be closed in P7 API work. Tier 2 does not depend on that
integration.

Cohort pass criterion: ≥60 percent of pairs within 2-fold of
observed AUCR (blueprint Tier 2 threshold).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

from neogen.core.constants.reference_physiology import build_reference_male
from neogen.core.entities.physiology import PhysiologyModel
from neogen.engine.ddi.ddi_orchestrator import DDIOrchestrator
from neogen.engine.drug_factory import DrugFactory
from neogen.engine.dose_scheduler import DoseScheduler
from neogen.engine.hepatic_clearance import HepaticClearanceEngine
from neogen.engine.pbpk_solver import PBPKSolver, SolverConfig
from neogen.engine.simulation_result import PKCalculator
from neogen.validation.cohort import Tier2Case, load_tier2_pairs
from neogen.validation.metrics import (
    fold_error,
    percent_within_fold,
    within_n_fold,
)

logger = logging.getLogger(__name__)


# Long-tail solver config matching the P3 acceptance gate. DDI runs
# need a long simulation window to capture the slow elimination tail
# under induction/MBI perturbation.
_DDI_SOLVER_CONFIG = SolverConfig(
    method="BDF",
    rtol=1e-6,
    atol=1e-9,
    n_timepoints=2000,
    t_max_multiplier=80.0,
)


# ════════════════════════════════════════════════════════════════
# RESULT TYPES
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Tier2CaseResult:
    """Per-pair Tier 2 result."""

    case: Tier2Case
    auc_baseline: Optional[float] = None
    auc_ddi: Optional[float] = None
    aucr_predicted: Optional[float] = None
    fold_error_vs_clinical: Optional[float] = None
    within_2_fold: bool = False
    error: str = ""


@dataclass(frozen=True)
class Tier2Report:
    """Aggregate Tier 2 report."""

    results: Tuple[Tier2CaseResult, ...] = field(default_factory=tuple)
    acceptance_threshold_pct: float = 60.0

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_errored(self) -> int:
        return sum(1 for r in self.results if r.error)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.within_2_fold)

    @property
    def pct_passed(self) -> float:
        n_valid = self.n_total - self.n_errored
        if n_valid == 0:
            return 0.0
        return 100.0 * self.n_passed / n_valid

    @property
    def gate_pass(self) -> bool:
        return self.pct_passed >= self.acceptance_threshold_pct

    def predicted_aucrs(self) -> Tuple[Optional[float], ...]:
        return tuple(r.aucr_predicted for r in self.results)

    def clinical_aucrs(self) -> Tuple[float, ...]:
        return tuple(r.case.clinical_aucr for r in self.results)


# ════════════════════════════════════════════════════════════════
# HARNESS
# ════════════════════════════════════════════════════════════════


class Tier2Harness:
    """Run DDI pair validation against the clinical AUCR cohort."""

    ACCEPTANCE_THRESHOLD_PCT = 60.0

    def __init__(
        self,
        physiology: PhysiologyModel | None = None,
        factory: DrugFactory | None = None,
        solver: PBPKSolver | None = None,
        ddi_orch: DDIOrchestrator | None = None,
    ) -> None:
        self._physiology = physiology or build_reference_male()
        self._factory = factory or DrugFactory()
        self._solver = solver or PBPKSolver(_DDI_SOLVER_CONFIG)
        self._ddi_orch = ddi_orch or DDIOrchestrator()
        self._hepatic = HepaticClearanceEngine()
        self._scheduler = DoseScheduler()

    # ── helpers ─────────────────────────────────────────────────

    def _baseline_auc(self, victim, schedule, dose_mg: float) -> float:
        """Run victim alone and return AUC_0_inf."""
        out = self._solver.solve(victim, schedule, self._physiology)
        pk = PKCalculator.compute(
            out.time_h, out.concentrations[1], dose_mg=dose_mg
        )
        return pk.auc_0_inf_mg_h_per_L

    def _build_clint_provider(
        self,
        case: Tier2Case,
        victim,
        perpetrator,
        base_clint_L_per_h: float,
    ) -> Callable[[float], float]:
        """Build the CLint(t) provider for a pair. Mirrors the P3 gate logic.

        For reversible inhibition we use an instantaneous Ki/[I]u provider
        with no enzyme pool. For MBI and induction we build an enzyme pool
        trace out to 14 days and clamp to the plateau value — this matches
        the clinical reference protocols which dose the victim at the
        perpetrator's steady state.
        """
        if case.mode == "reversible":
            exposure = self._ddi_orch.build_exposure(
                perpetrator,
                steady_state_conc_um=case.perpetrator_iu_um,
                duration_on_h=500.0,
                include_enzyme_pool=False,
            )
            return self._ddi_orch.build_clint_provider(
                victim=victim,
                base_clint_scaled_L_per_h=base_clint_L_per_h,
                exposures=[exposure],
            )

        # Induction or MBI — build pool trace then clamp to plateau.
        exposure = self._ddi_orch.build_exposure(
            perpetrator,
            steady_state_conc_um=case.perpetrator_iu_um,
            duration_on_h=14 * 24,
            include_enzyme_pool=True,
        )
        raw_provider = self._ddi_orch.build_clint_provider(
            victim=victim,
            base_clint_scaled_L_per_h=base_clint_L_per_h,
            exposures=[exposure],
        )
        plateau_clint = raw_provider(13 * 24)

        def steady(_t: float) -> float:
            return plateau_clint

        return steady

    # ── public API ──────────────────────────────────────────────

    def run_case(self, case: Tier2Case) -> Tier2CaseResult:
        """Run one pair. Errors become a failed Tier2CaseResult."""
        try:
            victim = self._factory.create(case.victim_name)
            perpetrator = self._factory.create(case.perpetrator_name)
        except Exception as exc:
            logger.warning("Failed to build pair %s: %s", case.label, exc)
            return Tier2CaseResult(case=case, error=f"drug build: {exc}")

        try:
            schedule = self._scheduler.iv_bolus(
                case.victim_name, dose_mg=case.victim_dose_mg
            )
        except Exception as exc:
            return Tier2CaseResult(case=case, error=f"schedule build: {exc}")

        try:
            auc_baseline = self._baseline_auc(
                victim, schedule, case.victim_dose_mg
            )
        except Exception as exc:
            logger.warning("Baseline failed for %s: %s", case.label, exc)
            return Tier2CaseResult(case=case, error=f"baseline: {exc}")

        if auc_baseline <= 0:
            return Tier2CaseResult(case=case, error="baseline AUC non-positive")

        try:
            base_clint = self._hepatic.compute(
                victim, self._physiology
            ).clint_scaled_L_per_h
            provider = self._build_clint_provider(
                case, victim, perpetrator, base_clint
            )
            out_ddi = self._solver.solve(
                victim, schedule, self._physiology, clint_provider=provider
            )
            pk_ddi = PKCalculator.compute(
                out_ddi.time_h,
                out_ddi.concentrations[1],
                dose_mg=case.victim_dose_mg,
            )
        except Exception as exc:
            logger.warning("DDI solve failed for %s: %s", case.label, exc)
            return Tier2CaseResult(case=case, error=f"ddi: {exc}")

        auc_ddi = pk_ddi.auc_0_inf_mg_h_per_L
        aucr_predicted = auc_ddi / auc_baseline if auc_baseline > 0 else 0.0
        fe = fold_error(aucr_predicted, case.clinical_aucr)
        passed = within_n_fold(aucr_predicted, case.clinical_aucr, n=2.0)

        return Tier2CaseResult(
            case=case,
            auc_baseline=auc_baseline,
            auc_ddi=auc_ddi,
            aucr_predicted=aucr_predicted,
            fold_error_vs_clinical=fe,
            within_2_fold=passed,
        )

    def run(
        self, cases: Optional[Tuple[Tier2Case, ...]] = None
    ) -> Tier2Report:
        """Run all pairs. Loads default cohort if none provided."""
        if cases is None:
            cases = load_tier2_pairs()
        results = tuple(self.run_case(c) for c in cases)
        return Tier2Report(
            results=results,
            acceptance_threshold_pct=self.ACCEPTANCE_THRESHOLD_PCT,
        )
