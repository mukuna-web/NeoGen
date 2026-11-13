"""Benchmark runner — executes :class:`BenchmarkCase` lists through the engine.

The runner is stateless: each :meth:`BenchmarkRunner.run_case` call is
independent. Construction is a one-shot wiring of the default
:class:`OrchestratorV2`, :class:`DrugFactory`, and
:class:`DoseScheduler`; callers who want to inject custom components
(e.g. a solver with tighter tolerances for a specific benchmark row)
can pass them in.

Error handling policy
---------------------

* A case whose simulation **raises** is recorded as a :class:`BenchmarkResult`
  with ``success=False`` and the exception stringified on ``error_message``.
  It does not abort the batch — the scorecard treats failed runs as
  automatic "outside tolerance" rows.
* A case with no observed references at all is **skipped** by
  :meth:`run_case` (``success=False``, ``error_message="no reference data"``).
  This keeps partially-specified datasets auditable.
* Fold-error calculation uses the symmetric definition
  ``fold_error = max(pred / obs, obs / pred)`` — always ≥ 1.0, always
  symmetric across order-of-magnitude shifts.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from neogen.engine.ddi.ddi_orchestrator import DDIOrchestrator, PerpetratorExposure
from neogen.engine.dose_scheduler import DoseScheduler
from neogen.engine.drug_factory import DrugFactory
from neogen.engine.orchestrator_v2 import (
    ENGINE_VERSION,
    OrchestratorV2,
    SimulationResultV2,
)

from neogen.validation.benchmark_case import (
    BenchmarkCase,
    DDIBenchmarkCase,
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# BENCHMARK RESULT — one row per case
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class MetricScore:
    """Score for a single metric (Cmax, AUC, t½, or AUCR).

    Attributes
    ----------
    name
        Metric label for rendering.
    observed
        Published reference value. ``None`` → "not scored".
    predicted
        Engine-predicted value. ``None`` → simulation failed.
    fold_error
        Symmetric fold error = ``max(pred/obs, obs/pred)``. Always ≥ 1.0.
        ``None`` if either side is missing.
    within_tolerance
        True if ``fold_error <= tolerance_fold``. False if scored but
        out of tolerance. ``None`` if not scored (neither side avail).
    """

    name: str
    observed: Optional[float]
    predicted: Optional[float]
    fold_error: Optional[float]
    within_tolerance: Optional[bool]

    @property
    def is_scored(self) -> bool:
        return self.fold_error is not None


@dataclass(frozen=True)
class BenchmarkResult:
    """Per-case outcome bundled with metric scores.

    ``success=False`` means the simulation raised or the case had no
    reference data. All metric scores are then ``None``-valued but the
    result is still included so the scorecard can render the failure.
    """

    case_name: str
    drug_name: str
    success: bool
    engine_version: str
    cmax: MetricScore
    auc: MetricScore
    half_life: MetricScore
    error_message: Optional[str] = None
    role: str = "victim"

    @property
    def scored_metrics(self) -> Tuple[MetricScore, ...]:
        """All metrics with non-None fold_error."""
        return tuple(m for m in (self.cmax, self.auc, self.half_life) if m.is_scored)

    @property
    def all_within_tolerance(self) -> bool:
        """True iff every scored metric is within tolerance."""
        scored = self.scored_metrics
        if not scored:
            return False
        return all(m.within_tolerance for m in scored)


@dataclass(frozen=True)
class DDIBenchmarkResult:
    """Per-case outcome for a DDI pair."""

    case_name: str
    victim: str
    perpetrator: str
    success: bool
    engine_version: str
    aucr: MetricScore
    error_message: Optional[str] = None

    @property
    def within_tolerance(self) -> bool:
        return bool(self.aucr.within_tolerance)


# ════════════════════════════════════════════════════════════════
# RUNNER
# ════════════════════════════════════════════════════════════════


def _fold_error(predicted: float, observed: float) -> float:
    """Symmetric fold-error, guarded against zero/near-zero values."""
    if predicted <= 0 or observed <= 0:
        return math.inf
    ratio = predicted / observed
    return max(ratio, 1.0 / ratio)


def _score_metric(
    name: str,
    predicted: Optional[float],
    observed: Optional[float],
    tolerance_fold: float,
) -> MetricScore:
    if observed is None or predicted is None:
        return MetricScore(
            name=name,
            observed=observed,
            predicted=predicted,
            fold_error=None,
            within_tolerance=None,
        )
    fe = _fold_error(predicted, observed)
    return MetricScore(
        name=name,
        observed=observed,
        predicted=predicted,
        fold_error=fe,
        within_tolerance=fe <= tolerance_fold,
    )


class BenchmarkRunner:
    """Runs :class:`BenchmarkCase` / :class:`DDIBenchmarkCase` instances.

    Usage::

        runner = BenchmarkRunner()
        result = runner.run_case(case)
        results = runner.run_all(cases)

    The runner is deliberately thin — heavy lifting goes to the
    engine, aggregation goes to :class:`Scorecard`.
    """

    def __init__(
        self,
        engine: Optional[OrchestratorV2] = None,
        factory: Optional[DrugFactory] = None,
        scheduler: Optional[DoseScheduler] = None,
        ddi_orchestrator: Optional[DDIOrchestrator] = None,
    ) -> None:
        self._engine = engine or OrchestratorV2()
        self._factory = factory or DrugFactory()
        self._scheduler = scheduler or DoseScheduler()
        self._ddi = ddi_orchestrator or DDIOrchestrator()

    # ── Single-drug case ────────────────────────────────────────

    def run_case(self, case: BenchmarkCase) -> BenchmarkResult:  # noqa: C901
        # (complexity from the failure fan-out, not from the happy path)
        if not case.has_any_reference:
            return _empty_result(
                case,
                success=False,
                error_message="no reference data",
            )

        try:
            drug = self._factory.create(case.drug_name)
            schedule = self._build_schedule(case)
            from neogen.core.constants.reference_physiology import (
                build_reference_male,
            )
            physiology = build_reference_male()
            result = self._engine.run(
                drug, schedule, physiology, patient=case.patient
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Benchmark case %r failed: %s", case.name, exc
            )
            return _empty_result(
                case,
                success=False,
                error_message=f"{type(exc).__name__}: {exc}",
            )

        pk = result.pk_result.pk_metrics
        cmax = _score_metric(
            "Cmax",
            pk.cmax_mg_per_L,
            case.observed_cmax_mg_per_L,
            case.tolerance_fold,
        )
        auc = _score_metric(
            "AUC",
            pk.auc_0_inf_mg_h_per_L,
            case.observed_auc_mg_h_per_L,
            case.tolerance_fold,
        )
        half_life = _score_metric(
            "t½",
            pk.half_life_h,
            case.observed_half_life_h,
            case.tolerance_fold,
        )
        return BenchmarkResult(
            case_name=case.name,
            drug_name=case.drug_name,
            success=True,
            engine_version=result.engine_version,
            cmax=cmax,
            auc=auc,
            half_life=half_life,
            error_message=None,
            role=getattr(case, "role", "victim"),
        )

    def run_all(
        self, cases: Sequence[BenchmarkCase]
    ) -> Tuple[BenchmarkResult, ...]:
        return tuple(self.run_case(c) for c in cases)

    # ── DDI case ────────────────────────────────────────────────

    def run_ddi_case(self, case: DDIBenchmarkCase) -> DDIBenchmarkResult:
        try:
            victim = self._factory.create(case.victim_drug_name)
            perpetrator = self._factory.create(case.perpetrator_drug_name)
            if case.victim_route == "IV":
                schedule = self._scheduler.iv_bolus(
                    case.victim_drug_name, dose_mg=case.victim_dose_mg
                )
            else:
                schedule = self._scheduler.single_oral(
                    case.victim_drug_name, dose_mg=case.victim_dose_mg
                )
            from neogen.core.constants.reference_physiology import (
                build_reference_male,
            )
            physiology = build_reference_male()

            # Baseline (monotherapy)
            mono = self._engine.run(
                victim, schedule, physiology, patient=case.patient
            )
            auc_mono = mono.pk_result.pk_metrics.auc_0_inf_mg_h_per_L

            # DDI exposure — pass the clinical pretreatment window so the
            # enzyme-pool trace reaches steady state before the victim is
            # dosed. This is a no-op for pure reversible inhibitors and is
            # essential for inducers (rifampicin) and MBIs (clarithromycin,
            # itraconazole) to match clinical AUCR.
            exposure = self._ddi.build_exposure(
                perpetrator,
                steady_state_conc_um=case.perpetrator_steady_state_conc_um,
                duration_on_h=case.perpetrator_duration_on_h,
                pretreatment_h=case.pretreatment_h,
                include_enzyme_pool=case.include_enzyme_pool,
            )
            # V1.2 Track C: if the perpetrator has CYP3A4 induction
            # params AND the victim is oral, build a gut-wall induction
            # exposure so enterocyte CYP3A4 upregulation is modelled.
            # This is what makes rifampicin + oral midazolam AUCR drop
            # from 0.44 to ~0.07 (clinical 0.04).
            gut_ind_list = None
            if (
                case.victim_route == "ORAL"
                and perpetrator.induction.indmax_cyp3a4 is not None
                and perpetrator.induction.indmax_cyp3a4 > 0
                and victim.metabolism.cyp3a4 >= 0.70
            ):
                from neogen.engine.ddi.gut_wall_mbi import (
                    GutWallInductionExposure,
                )
                # Luminal [I]: oral dose / ~250 mL / MW × 1e6 → µM.
                # At saturation (>> EC50) the exact value doesn't
                # matter; use 3000 µM as a reasonable estimate for
                # 600 mg rifampicin dissolved in small-intestinal fluid.
                gut_ind_list = [
                    GutWallInductionExposure(
                        perpetrator=perpetrator,
                        luminal_conc_um=3000.0,
                    )
                ]

            # Gut-wall MBI channel — when the victim is oral AND the
            # perpetrator has a luminal concentration, construct a
            # GutWallMBIExposure so the absorption engine's fg_j
            # accounts for enterocyte CYP3A4 MBI.
            gut_mbi_list = None
            if (
                case.victim_route == "ORAL"
                and case.perpetrator_luminal_conc_um is not None
                and case.perpetrator_luminal_conc_um > 0
            ):
                from neogen.engine.ddi.gut_wall_mbi import (
                    GutWallMBIExposure,
                )
                gut_mbi_list = [
                    GutWallMBIExposure(
                        perpetrator=perpetrator,
                        luminal_conc_um=case.perpetrator_luminal_conc_um,
                        induction_conc_um=case.perpetrator_steady_state_conc_um,
                    ),
                ]

            # V1.2: if the perpetrator is an OATP1B1 inhibitor, build
            # an OATP1B1InhibitorExposure so the hepatic uptake
            # modulation DDI pathway fires for statin victims.
            oatp_list = None
            if getattr(perpetrator.transporters, "is_oatp1b1_inhibitor", False):
                from neogen.engine.ddi.transporter_ddi import OATP1B1InhibitorExposure
                oatp_list = [OATP1B1InhibitorExposure(
                    perpetrator=perpetrator,
                    unbound_portal_conc_um=case.perpetrator_steady_state_conc_um,
                )]

            ddi_result = self._engine.run(
                victim,
                schedule,
                physiology,
                perpetrators=[exposure],
                gut_wall_mbi_exposures=gut_mbi_list,
                gut_wall_induction_exposures=gut_ind_list,
                oatp1b1_inhibitors=oatp_list,
                patient=case.patient,
            )
            auc_ddi = ddi_result.pk_result.pk_metrics.auc_0_inf_mg_h_per_L

            if auc_mono <= 0:
                raise ValueError(
                    f"monotherapy AUC is zero for {case.victim_drug_name}"
                )
            predicted_aucr = auc_ddi / auc_mono

        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "DDI benchmark case %r failed: %s", case.name, exc
            )
            return DDIBenchmarkResult(
                case_name=case.name,
                victim=case.victim_drug_name,
                perpetrator=case.perpetrator_drug_name,
                success=False,
                engine_version=ENGINE_VERSION,
                aucr=MetricScore(
                    name="AUCR",
                    observed=case.observed_aucr,
                    predicted=None,
                    fold_error=None,
                    within_tolerance=None,
                ),
                error_message=f"{type(exc).__name__}: {exc}",
            )

        aucr_score = _score_metric(
            "AUCR",
            predicted_aucr,
            case.observed_aucr,
            case.tolerance_fold,
        )
        return DDIBenchmarkResult(
            case_name=case.name,
            victim=case.victim_drug_name,
            perpetrator=case.perpetrator_drug_name,
            success=True,
            engine_version=ddi_result.engine_version,
            aucr=aucr_score,
            error_message=None,
        )

    def run_all_ddi(
        self, cases: Sequence[DDIBenchmarkCase]
    ) -> Tuple[DDIBenchmarkResult, ...]:
        return tuple(self.run_ddi_case(c) for c in cases)

    # ── helpers ─────────────────────────────────────────────────

    def _build_schedule(self, case: BenchmarkCase):
        from neogen.core.entities.formulation import Route

        if case.schedule_kind == "single":
            if case.route == "IV":
                return self._scheduler.iv_bolus(
                    case.drug_name, dose_mg=case.dose_mg
                )
            return self._scheduler.single_oral(
                case.drug_name, dose_mg=case.dose_mg
            )

        # Multi-dose — delegate to DoseScheduler.repeated, which takes
        # a regimen string rather than a bare interval. Convert the
        # case's dosing_interval_h to the closest regimen tag.
        interval_h = case.dosing_interval_h or 24.0
        if interval_h <= 6.1:
            regimen = "QID"
        elif interval_h <= 8.1:
            regimen = "Q8H"
        elif interval_h <= 12.1:
            regimen = "BID"
        elif interval_h <= 24.1:
            regimen = "QD"
        else:
            regimen = "QD"  # fall back to daily
        days = max(1, int(round((case.total_duration_h or 24.0) / 24.0)))
        route = (
            Route.IV_BOLUS if case.route == "IV" else Route.ORAL
        )
        return self._scheduler.repeated(
            case.drug_name,
            dose_mg=case.dose_mg,
            regimen=regimen,
            days=days,
            route=route,
        )


def _empty_result(
    case: BenchmarkCase,
    success: bool,
    error_message: Optional[str],
) -> BenchmarkResult:
    def _blank(name: str, obs: Optional[float]) -> MetricScore:
        return MetricScore(
            name=name,
            observed=obs,
            predicted=None,
            fold_error=None,
            within_tolerance=None,
        )

    return BenchmarkResult(
        case_name=case.name,
        drug_name=case.drug_name,
        success=success,
        engine_version=ENGINE_VERSION,
        cmax=_blank("Cmax", case.observed_cmax_mg_per_L),
        auc=_blank("AUC", case.observed_auc_mg_h_per_L),
        half_life=_blank("t½", case.observed_half_life_h),
        error_message=error_message,
    )
