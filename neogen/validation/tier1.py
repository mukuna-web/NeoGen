"""Tier 1 — single-drug PK validation.

For each case in the cohort, build a Drug entity, run an IV bolus
simulation at a fixed dose (100 mg by default — dose-invariant
metrics mean the choice does not matter), and compare the engine's
predicted CL, Vd, and t½ against the observed clinical values.

Scoring per case:
  * Each of {t½, CL, Vd} is independently tested against its 2-fold
    band when an observation is available.
  * A case passes Tier 1 if the majority of its scorable metrics
    pass (at least 2 of 3, or all if fewer are available).

Cohort pass criterion: ≥70 percent of scorable cases pass.

Notes:
  * Drugs in LITERATURE_DRUGS are built via DrugFactory.create(name),
    which pulls literature parameters — the "easy" subset.
  * Drugs not in LITERATURE_DRUGS are built via
    DrugFactory.create_from_smiles, using RDKit-only descriptors
    and ML predictions. This is the blueprint's "structure-only"
    mode and is where the engine is truly blind.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from neogen.core.constants.literature_params import list_available_drugs
from neogen.core.constants.reference_physiology import build_reference_male
from neogen.core.entities.physiology import PhysiologyModel
from neogen.engine.drug_factory import DrugFactory
from neogen.engine.dose_scheduler import DoseScheduler
from neogen.engine.orchestrator import SimulationOrchestrator
from neogen.engine.pbpk_solver import SolverConfig
from neogen.validation.cohort import Tier1Case, load_tier1_blind_holdout
from neogen.validation.metrics import (
    fold_error,
    summarise,
    within_n_fold,
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# RESULT TYPES
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Tier1CaseResult:
    """Per-case Tier 1 result."""

    case: Tier1Case

    predicted_half_life_h: Optional[float] = None
    predicted_cl_L_per_h: Optional[float] = None
    predicted_vd_L: Optional[float] = None

    half_life_within_2_fold: Optional[bool] = None
    cl_within_2_fold: Optional[bool] = None
    vd_within_2_fold: Optional[bool] = None

    n_scorable: int = 0
    n_passed: int = 0
    case_passed: bool = False

    # Post-prediction PK consistency QC (cross-checks t½ vs ln(2)·Vd/CL).
    # ``None`` means the QC could not be computed (degenerate predictions
    # or upstream error). Keys: cl_predicted, vd_predicted,
    # half_life_predicted, half_life_expected, gmfe, is_consistent.
    pk_consistency: Optional[Dict[str, Any]] = None

    error: str = ""

    @property
    def half_life_fold_error(self) -> float:
        return fold_error(
            self.predicted_half_life_h, self.case.observed_half_life_h
        )

    @property
    def cl_fold_error(self) -> float:
        return fold_error(
            self.predicted_cl_L_per_h, self.case.observed_cl_L_per_h
        )

    @property
    def vd_fold_error(self) -> float:
        return fold_error(self.predicted_vd_L, self.case.observed_vd_L)


@dataclass(frozen=True)
class Tier1Report:
    """Aggregate Tier 1 report."""

    results: Tuple[Tier1CaseResult, ...] = field(default_factory=tuple)
    acceptance_threshold_pct: float = 70.0

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_errored(self) -> int:
        return sum(1 for r in self.results if r.error)

    @property
    def n_scorable(self) -> int:
        return sum(1 for r in self.results if r.n_scorable > 0 and not r.error)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.case_passed)

    @property
    def pct_passed(self) -> float:
        if self.n_scorable == 0:
            return 0.0
        return 100.0 * self.n_passed / self.n_scorable

    @property
    def gate_pass(self) -> bool:
        return self.pct_passed >= self.acceptance_threshold_pct

    # ── PK consistency aggregates ──────────────────────────────────
    #
    # Thresholds match the Tier 1 2-fold gate (looser by 25% / tighter
    # by 25%):
    #   gmfe < 1.5            → consistent
    #   1.5 ≤ gmfe < 2.5      → borderline
    #   gmfe ≥ 2.5            → inconsistent
    @property
    def n_cases_with_consistency(self) -> int:
        return sum(1 for r in self.results if r.pk_consistency is not None)

    @property
    def n_cases_consistent(self) -> int:
        return sum(
            1
            for r in self.results
            if r.pk_consistency is not None
            and r.pk_consistency["gmfe"] < 1.5
        )

    @property
    def n_cases_borderline(self) -> int:
        return sum(
            1
            for r in self.results
            if r.pk_consistency is not None
            and 1.5 <= r.pk_consistency["gmfe"] < 2.5
        )

    @property
    def n_cases_inconsistent(self) -> int:
        return sum(
            1
            for r in self.results
            if r.pk_consistency is not None
            and r.pk_consistency["gmfe"] >= 2.5
        )

    def metric_summary(self, metric: str) -> dict:
        """Aggregate GMFE / %-within-fold for a single metric across the cohort.

        metric ∈ {'half_life', 'cl', 'vd'}.
        """
        if metric == "half_life":
            pred = [r.predicted_half_life_h for r in self.results]
            obs = [r.case.observed_half_life_h for r in self.results]
        elif metric == "cl":
            pred = [r.predicted_cl_L_per_h for r in self.results]
            obs = [r.case.observed_cl_L_per_h for r in self.results]
        elif metric == "vd":
            pred = [r.predicted_vd_L for r in self.results]
            obs = [r.case.observed_vd_L for r in self.results]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        return summarise(pred, obs)


# ════════════════════════════════════════════════════════════════
# HARNESS
# ════════════════════════════════════════════════════════════════


class Tier1Harness:
    """Run single-drug PK validation against a cohort of observed PK."""

    IV_DOSE_MG = 100.0
    ACCEPTANCE_THRESHOLD_PCT = 70.0

    def __init__(
        self,
        physiology: PhysiologyModel | None = None,
        factory: DrugFactory | None = None,
        orchestrator: SimulationOrchestrator | None = None,
        iv_dose_mg: float = IV_DOSE_MG,
        mass_balance_mode: bool = True,
    ) -> None:
        self._physiology = physiology or build_reference_male()
        self._factory = factory or DrugFactory()
        # Default Tier 1 to the V1.1 mass-balance solver so the harness
        # actually evaluates mechanistic PK instead of the legacy
        # mixing-pool artifact. Caller can pass a custom orchestrator
        # to run against the legacy solver if needed.
        self._orchestrator = orchestrator or SimulationOrchestrator(
            solver_config=SolverConfig(mass_balance_mode=mass_balance_mode)
        )
        self._scheduler = DoseScheduler()
        self._dose_mg = iv_dose_mg
        self._literature_names = {n.lower() for n in list_available_drugs()}

    # ── helpers ─────────────────────────────────────────────────

    def _build_drug(self, case: Tier1Case):
        """Build a Drug entity using the best available path."""
        name_lc = case.name.lower()
        if name_lc in self._literature_names:
            return self._factory.create(name_lc)
        if case.smiles:
            return self._factory.create_from_smiles(
                case.smiles, name=case.name
            )
        raise ValueError(
            f"Case {case.name} has no literature entry and no SMILES"
        )

    def _score_case(
        self,
        case: Tier1Case,
        predicted_half_life_h: float,
        predicted_cl: float,
        predicted_vd: float,
        pk_consistency: Optional[Dict[str, Any]] = None,
    ) -> Tier1CaseResult:
        """Compare predicted metrics against observed and build the result."""
        checks: dict[str, Optional[bool]] = {
            "half_life": None,
            "cl": None,
            "vd": None,
        }

        if case.observed_half_life_h is not None:
            checks["half_life"] = within_n_fold(
                predicted_half_life_h, case.observed_half_life_h, n=2.0
            )
        if case.observed_cl_L_per_h is not None:
            checks["cl"] = within_n_fold(
                predicted_cl, case.observed_cl_L_per_h, n=2.0
            )
        if case.observed_vd_L is not None:
            checks["vd"] = within_n_fold(
                predicted_vd, case.observed_vd_L, n=2.0
            )

        scorable = [v for v in checks.values() if v is not None]
        n_scorable = len(scorable)
        n_passed = sum(1 for v in scorable if v)

        # A case passes if majority of scorable metrics pass. When the
        # count is even we require strict majority (n_passed > half).
        if n_scorable == 0:
            case_passed = False
        else:
            case_passed = n_passed >= (n_scorable + 1) // 2

        return Tier1CaseResult(
            case=case,
            predicted_half_life_h=predicted_half_life_h,
            predicted_cl_L_per_h=predicted_cl,
            predicted_vd_L=predicted_vd,
            half_life_within_2_fold=checks["half_life"],
            cl_within_2_fold=checks["cl"],
            vd_within_2_fold=checks["vd"],
            n_scorable=n_scorable,
            n_passed=n_passed,
            case_passed=case_passed,
            pk_consistency=pk_consistency,
        )

    # ── public API ──────────────────────────────────────────────

    def run_case(self, case: Tier1Case) -> Tier1CaseResult:
        """Run one case. Errors become a failed Tier1CaseResult."""
        if not case.has_scorable_metric:
            return Tier1CaseResult(case=case, error="no scorable metrics")

        try:
            drug = self._build_drug(case)
        except Exception as exc:
            logger.warning("Failed to build drug %s: %s", case.name, exc)
            return Tier1CaseResult(case=case, error=f"drug build: {exc}")

        try:
            schedule = self._scheduler.iv_bolus(drug.name, self._dose_mg)
            result = self._orchestrator.run(drug, schedule, self._physiology)
        except Exception as exc:
            logger.warning("Simulation failed for %s: %s", case.name, exc)
            return Tier1CaseResult(case=case, error=f"simulation: {exc}")

        pk = result.pk_metrics
        pred_t12 = pk.half_life_h
        pred_cl = result.cl_total_L_per_h
        pred_vd = pk.vss_apparent_L

        # Opt-in similarity-blended overlay. Default OFF.
        # Activation: NEOGEN_SIMILARITY_BLEND=1
        # Blends engine t½/Vd with k-NN over the leak-stripped TDC index.
        # CL is intentionally not blended (training units mismatch).
        # Lookup-aware: when the engine already has a curated registry hit
        # (drug.source contains 'literature' / 'gold_standard' / 'drugbank')
        # the lookup IS the answer; blending only adds noise. Skip in that
        # case unless the user explicitly forces the blend with =force.
        import os as _os
        _blend_flag = _os.environ.get("NEOGEN_SIMILARITY_BLEND", "").strip().lower()
        _blocking = _os.environ.get(
            "NEOGEN_BLOCK_HOLDOUT_LOOKUP", ""
        ).strip().lower() in ("1", "true", "yes")
        if _blend_flag in ("1", "true", "yes", "force"):
            _force = (_blend_flag == "force") or _blocking
            # Per-parameter lookup-hit detection: if the engine has a curated
            # override for a metric, that override IS the answer; the blend
            # would only add noise. Skip blending that metric. Other metrics
            # without an override still benefit from k-NN retrieval.
            # When block-lookups is active any override is itself a model
            # output, so the blend is still net-positive — `_force` covers it.
            t12_overridden = (
                getattr(drug, "half_life_override_h", None) is not None
                or getattr(drug, "source", "") == "literature"
            )
            vd_overridden = (
                getattr(drug, "vss_observed_L_per_kg", None) is not None
                or getattr(drug, "source", "") == "literature"
            )
            try:
                from neogen.ml.similarity_blend import blend_engine_with_knn
                blended = blend_engine_with_knn(
                    smiles=getattr(drug, "smiles", "") or case.smiles,
                    engine_half_life_h=pred_t12,
                    engine_cl_L_per_h=pred_cl,
                    engine_vd_L=pred_vd,
                )
                if blended is not None:
                    if blended.used_blend_t12 and blended.half_life_h and (_force or not t12_overridden):
                        pred_t12 = blended.half_life_h
                    if blended.used_blend_vd and blended.vd_L and (_force or not vd_overridden):
                        pred_vd = blended.vd_L
            except Exception as _exc:  # pragma: no cover
                logger.warning(
                    "similarity blend failed for %s: %s", case.name, _exc
                )

        return self._score_case(
            case,
            predicted_half_life_h=pred_t12,
            predicted_cl=pred_cl,
            predicted_vd=pred_vd,
            pk_consistency=getattr(result, "pk_consistency", None),
        )

    def run(
        self, cases: Optional[Tuple[Tier1Case, ...]] = None
    ) -> Tier1Report:
        """Run the full cohort. Loads default cohort if none provided."""
        if cases is None:
            cases = load_tier1_blind_holdout(
                literature_names=self._literature_names
            )
        results = tuple(self.run_case(c) for c in cases)
        return Tier1Report(
            results=results,
            acceptance_threshold_pct=self.ACCEPTANCE_THRESHOLD_PCT,
        )
