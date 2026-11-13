"""Tier 3 — bioavailability validation.

For each oral drug in the cohort, compute the empirical
bioavailability F = (AUC_oral / Dose_oral) / (AUC_iv / Dose_iv)
by running the V1 engine twice (once oral, once IV) and compare
against the clinically observed F.

Acceptance band: ±20 percent of observed F per case (blueprint
Tier 3 criterion). We do not gate the cohort on an aggregate
percent — Tier 3 is per-case reportable because clinical F values
vary widely with formulation, fed state, and genotype.

fa/fg/fh decomposition is reported when available. If the
engine's absorption engine exposes fa and fg via its
AbsorptionProfile, we pair them with the hepatic clearance
engine's fh to report the decomposed F = fa × fg × fh alongside
the empirical value.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

from neogen.core.constants.reference_physiology import build_reference_male
from neogen.core.entities.formulation import Route
from neogen.core.entities.physiology import PhysiologyModel
from neogen.engine.drug_factory import DrugFactory
from neogen.engine.dose_scheduler import DoseScheduler
from neogen.engine.hepatic_clearance import HepaticClearanceEngine
from neogen.engine.orchestrator import SimulationOrchestrator
from neogen.engine.pbpk_solver import SolverConfig
from neogen.validation.cohort import Tier3Case, load_tier3_cohort
from neogen.validation.metrics import percent_error, within_pct

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# RESULT TYPES
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Tier3CaseResult:
    """Per-case Tier 3 result."""

    case: Tier3Case

    auc_oral: Optional[float] = None
    auc_iv: Optional[float] = None
    f_empirical: Optional[float] = None

    # Optional decomposition (only set when absorption engine yielded it)
    predicted_fa: Optional[float] = None
    predicted_fg: Optional[float] = None
    predicted_fh: Optional[float] = None
    f_decomposed: Optional[float] = None

    pct_err_empirical: Optional[float] = None
    within_acceptance: bool = False

    error: str = ""


@dataclass(frozen=True)
class Tier3Report:
    """Aggregate Tier 3 report."""

    results: Tuple[Tier3CaseResult, ...] = field(default_factory=tuple)
    acceptance_band_pct: float = 20.0

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_errored(self) -> int:
        return sum(1 for r in self.results if r.error)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.within_acceptance)

    @property
    def pct_passed(self) -> float:
        n_valid = self.n_total - self.n_errored
        if n_valid == 0:
            return 0.0
        return 100.0 * self.n_passed / n_valid

    @property
    def gate_pass(self) -> bool:
        # Tier 3 is advisory in V1 — treat majority-pass as green.
        return self.pct_passed >= 50.0


# ════════════════════════════════════════════════════════════════
# HARNESS
# ════════════════════════════════════════════════════════════════


class Tier3Harness:
    """Run bioavailability validation against the Tier 3 cohort."""

    ACCEPTANCE_BAND_PCT = 20.0

    def __init__(
        self,
        physiology: PhysiologyModel | None = None,
        factory: DrugFactory | None = None,
        orchestrator: SimulationOrchestrator | None = None,
        mass_balance_mode: bool = True,
    ) -> None:
        self._physiology = physiology or build_reference_male()
        self._factory = factory or DrugFactory()
        # Default Tier 3 to the V1.1 mass-balance solver — the entire
        # point of this tier is to check F = AUC_oral/AUC_iv, which
        # only has physical meaning when mass is conserved.
        self._orchestrator = orchestrator or SimulationOrchestrator(
            solver_config=SolverConfig(mass_balance_mode=mass_balance_mode)
        )
        self._hepatic_engine = HepaticClearanceEngine()
        self._scheduler = DoseScheduler()

    # ── helpers ─────────────────────────────────────────────────

    def _extract_absorption_decomposition(
        self, orch_result
    ) -> Tuple[Optional[float], Optional[float]]:
        """Pull ``(fa, fg)`` from the first absorption profile on the
        result, or ``(None, None)`` if the oral run did not produce
        one (e.g. the permeability engine fell back to legacy ka).

        V1.1: :class:`SimulationResult.absorption_profiles` is the
        tuple of :class:`AbsorptionProfile` objects produced by the
        P2 permeability engine and attached by
        :meth:`SimulationOrchestrator.run`. Each profile has ``fa``
        and ``fg`` fields directly.
        """
        profiles = getattr(orch_result, "absorption_profiles", None)
        if not profiles:
            return None, None
        first = profiles[0]
        fa = getattr(first, "fa", None)
        fg = getattr(first, "fg", None)
        return fa, fg

    # ── public API ──────────────────────────────────────────────

    def run_case(self, case: Tier3Case) -> Tier3CaseResult:
        """Run one case: oral + IV simulations, compute F_empirical."""
        try:
            drug = self._factory.create(case.drug_name)
        except Exception as exc:
            logger.warning(
                "Failed to build drug %s: %s", case.drug_name, exc
            )
            return Tier3CaseResult(case=case, error=f"drug build: {exc}")

        # ── IV arm ─────────────────────────────────────────────
        try:
            iv_schedule = self._scheduler.iv_bolus(
                case.drug_name, case.iv_dose_mg
            )
            iv_result = self._orchestrator.run(
                drug, iv_schedule, self._physiology
            )
        except Exception as exc:
            logger.warning("IV sim failed for %s: %s", case.drug_name, exc)
            return Tier3CaseResult(case=case, error=f"iv: {exc}")

        auc_iv = iv_result.pk_metrics.auc_0_inf_mg_h_per_L
        if auc_iv <= 0:
            return Tier3CaseResult(case=case, error="IV AUC non-positive")

        # ── Oral arm ───────────────────────────────────────────
        try:
            oral_schedule = self._scheduler.single_oral(
                case.drug_name, case.oral_dose_mg
            )
            oral_result = self._orchestrator.run(
                drug, oral_schedule, self._physiology
            )
        except Exception as exc:
            logger.warning("Oral sim failed for %s: %s", case.drug_name, exc)
            return Tier3CaseResult(case=case, error=f"oral: {exc}")

        auc_oral = oral_result.pk_metrics.auc_0_inf_mg_h_per_L
        if auc_oral < 0:
            return Tier3CaseResult(case=case, error="Oral AUC negative")

        # ── Empirical F ────────────────────────────────────────
        # F = (AUC_oral / Dose_oral) / (AUC_iv / Dose_iv)
        f_empirical = (auc_oral / case.oral_dose_mg) / (
            auc_iv / case.iv_dose_mg
        )
        # Report raw F_empirical without clamping to [0, 1]. The V1
        # engine produces F > 1.0 for most drugs due to the IV
        # initial-condition spike artifact — the IV bolus deposits
        # ``dose/V_plasma`` into a 3 L plasma pool which decays 20×
        # between the first two time points and barely contributes
        # to AUC. Meanwhile the oral path builds a physiological
        # Cmax via the P2 portal-forcing function. Net effect: oral
        # AUC > IV AUC scaled by dose, so F_empirical > 1. This is
        # a direct consequence of the Tier 4 advisory (plasma is a
        # venous mixing pool, not a mass-balance node) and is
        # scheduled to be lifted in V1.1 along with the IV initial
        # condition redistribution across Vd-weighted organ volumes.
        f_empirical_clamped = max(0.0, f_empirical)

        # Statin prodrug correction: published F for atorvastatin and
        # simvastatin is for the acid form only. The active_fraction
        # corrects Cmax/AUC in the orchestrator, but the T3 AUC ratio
        # (oral/IV) cancels the correction. Apply it here.
        if drug.active_fraction is not None:
            f_empirical_clamped *= drug.active_fraction

        pct_err = percent_error(f_empirical_clamped, case.observed_f_oral)
        within = within_pct(
            f_empirical_clamped,
            case.observed_f_oral,
            pct=self.ACCEPTANCE_BAND_PCT,
        )

        # Optional decomposition (V1.1):
        #   fa, fg come from the P2 absorption profile attached to
        #   the oral ``SimulationResult`` by the orchestrator.
        #   fh is the steady-state well-stirred hepatic availability
        #   from the :class:`HepaticClearanceEngine` well-stirred
        #   model, ``fh = 1 − Eh``. The engine's Eh is computed from
        #   ``fu × CLint / (Qh + fu × CLint)`` — the same formula the
        #   ODE uses internally for the well-stirred liver term.
        fa, fg = self._extract_absorption_decomposition(oral_result)
        fh: Optional[float] = None
        try:
            fh = self._hepatic_engine.compute(drug, self._physiology).fh
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("Hepatic fh lookup failed for %s: %s", case.drug_name, exc)
        f_decomposed: Optional[float] = None
        if fa is not None and fg is not None and fh is not None:
            f_decomposed = fa * fg * fh

        return Tier3CaseResult(
            case=case,
            auc_oral=auc_oral,
            auc_iv=auc_iv,
            f_empirical=f_empirical_clamped,
            predicted_fa=fa,
            predicted_fg=fg,
            predicted_fh=fh,
            f_decomposed=f_decomposed,
            pct_err_empirical=pct_err,
            within_acceptance=within,
        )

    def run(
        self, cases: Optional[Tuple[Tier3Case, ...]] = None
    ) -> Tier3Report:
        """Run all cases. Loads default cohort if none provided."""
        if cases is None:
            cases = load_tier3_cohort()
        results = tuple(self.run_case(c) for c in cases)
        return Tier3Report(
            results=results,
            acceptance_band_pct=self.ACCEPTANCE_BAND_PCT,
        )
