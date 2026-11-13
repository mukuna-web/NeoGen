"""Tier 4 — mechanistic sanity invariants.

Tier 4 is the blueprint's "things that must always be true" tier.
Unlike Tier 1-3 (statistical accuracy against clinical data),
Tier 4 is deterministic: every invariant must pass, or the engine
has a bug.

These invariants guard against the "looks smart but is wrong"
failure modes enumerated in the blueprint:
  * Wrong fu propagation
  * Non-unity IV bioavailability
  * Mass imbalance
  * Negative concentrations from solver drift
  * Numerical divergence in CL × AUC consistency
  * Well-stirred liver algebra mismatch

Gate policy: 100 percent must pass. A single failing invariant is
a blocker for engine releases, not a statistical signal.

Each invariant is a small, named predicate function. The harness
runs each one and collects structured pass/fail results.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np

from neogen.core.constants.reference_physiology import build_reference_male
from neogen.core.entities.physiology import PhysiologyModel
from neogen.engine.drug_factory import DrugFactory
from neogen.engine.dose_scheduler import DoseScheduler
from neogen.engine.hepatic_clearance import HepaticClearanceEngine
from neogen.engine.orchestrator import SimulationOrchestrator
from neogen.engine.pbpk_solver import SolverConfig
from neogen.engine.renal_clearance import RenalClearanceEngine
from neogen.engine.simulation_result import SimulationResult
from neogen.engine.tissue_partitioner import TissuePartitioner

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# RESULT TYPES
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class InvariantResult:
    """One invariant's pass/fail outcome.

    ``gating`` controls whether a failure blocks Tier 4's overall
    gate. Non-gating invariants are still reported — they surface
    known V1 architectural limitations (e.g., simplified plasma
    mixing pool) that are scheduled to be lifted in V1.1 — but
    they do not cause an otherwise-green run to fail.
    """

    name: str
    description: str
    passed: bool
    detail: str = ""
    error: str = ""
    gating: bool = True


@dataclass(frozen=True)
class Tier4Report:
    """Aggregate Tier 4 report."""

    results: Tuple[InvariantResult, ...] = field(default_factory=tuple)

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        return self.n_total - self.n_passed

    @property
    def n_gating(self) -> int:
        return sum(1 for r in self.results if r.gating)

    @property
    def n_gating_failed(self) -> int:
        return sum(1 for r in self.results if r.gating and not r.passed)

    @property
    def n_advisory_failed(self) -> int:
        return sum(1 for r in self.results if not r.gating and not r.passed)

    @property
    def gate_pass(self) -> bool:
        """True if every *gating* invariant passed.

        Non-gating invariants (marked ``gating=False``) can fail
        without blocking the gate. These track known V1
        architectural limitations and are reported prominently
        in the Markdown output.
        """
        return self.n_total > 0 and self.n_gating_failed == 0

    def failed_invariants(self) -> Tuple[InvariantResult, ...]:
        return tuple(r for r in self.results if not r.passed)

    def failed_gating_invariants(self) -> Tuple[InvariantResult, ...]:
        return tuple(
            r for r in self.results if r.gating and not r.passed
        )


# ════════════════════════════════════════════════════════════════
# HARNESS
# ════════════════════════════════════════════════════════════════


class Tier4Harness:
    """Run mechanistic sanity invariants against the V1 engine.

    Reference drug: midazolam (CYP3A4 substrate with well-known PK,
    literature-override available via DrugFactory). A single
    simulation is shared across most invariants to keep the gate
    fast (≤1 s per full Tier 4 run).

    ``mass_balance_mode`` (default True) runs the reference
    simulation through the V1.1 closed-loop plasma ODE so the
    ``cl_auc_equals_dose`` invariant can enforce true mass balance
    as a gating check. Pass ``False`` to audit the legacy solver
    behaviour (the invariant will then flip to advisory and fail
    by ~70% as documented in the session-3 findings).
    """

    REFERENCE_DRUG = "midazolam"
    REFERENCE_IV_DOSE_MG = 100.0
    REL_TOLERANCE_CL_AUC = 0.02  # organ-integrated mass balance within 2%
    REL_TOLERANCE_WELL_STIRRED = 0.02
    NEGATIVE_CONC_FLOOR = -1.0e-9  # allow solver noise

    # Tight solver config used ONLY for the strict mass-balance
    # invariant. The default config uses t_max_multiplier=5 and
    # n_timepoints=500 which is fine for PK metrics but truncates
    # the elimination tail at ~96 % of total mass for fast-
    # distributing high-extraction drugs. For strict balance we
    # integrate to 20 half-lives and sample densely.
    STRICT_MB_SOLVER_CONFIG = SolverConfig(
        method="BDF",
        rtol=1e-8,
        atol=1e-11,
        n_timepoints=2000,
        t_max_multiplier=20.0,
        mass_balance_mode=True,
    )

    def __init__(
        self,
        physiology: PhysiologyModel | None = None,
        factory: DrugFactory | None = None,
        mass_balance_mode: bool = True,
    ) -> None:
        self._physiology = physiology or build_reference_male()
        self._factory = factory or DrugFactory()
        self._mass_balance_mode = mass_balance_mode
        self._orchestrator = SimulationOrchestrator(
            solver_config=SolverConfig(mass_balance_mode=mass_balance_mode)
        )
        # Second orchestrator with extra-tight integration — used by
        # the strict mass-balance invariant only.
        self._strict_mb_orchestrator = SimulationOrchestrator(
            solver_config=self.STRICT_MB_SOLVER_CONFIG
        )
        self._hepatic_engine = HepaticClearanceEngine()
        self._renal_engine = RenalClearanceEngine()
        self._partitioner = TissuePartitioner()
        self._scheduler = DoseScheduler()

    # ── helpers ─────────────────────────────────────────────────

    def _run_reference_iv(self) -> SimulationResult:
        """Run the reference IV simulation used by most invariants."""
        drug = self._factory.create(self.REFERENCE_DRUG)
        schedule = self._scheduler.iv_bolus(
            self.REFERENCE_DRUG, self.REFERENCE_IV_DOSE_MG
        )
        return self._orchestrator.run(drug, schedule, self._physiology)

    def _wrap(
        self,
        name: str,
        description: str,
        fn: Callable[[], Tuple[bool, str]],
        gating: bool = True,
    ) -> InvariantResult:
        """Call an invariant and turn exceptions into InvariantResult.

        ``passed`` is coerced to a plain Python ``bool`` because
        some invariants compare numpy scalar concentrations; the
        resulting ``numpy.bool`` is not JSON-serializable via
        Pydantic v2, so we normalize at this single boundary.
        """
        try:
            passed, detail = fn()
            return InvariantResult(
                name=name,
                description=description,
                passed=bool(passed),
                detail=str(detail),
                gating=gating,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Invariant %s raised", name)
            return InvariantResult(
                name=name,
                description=description,
                passed=False,
                error=f"{type(exc).__name__}: {exc}",
                gating=gating,
            )

    # ── invariants ──────────────────────────────────────────────

    def invariant_iv_peak_at_zero(self) -> Tuple[bool, str]:
        """IV bolus: peak plasma concentration occurs at t=0 (or very
        close to it). For a true IV bolus the dose is deposited
        instantaneously, so Cmax and tmax≈0 are a necessary condition.
        """
        result = self._run_reference_iv()
        tmax = result.pk_metrics.tmax_h
        passed = tmax < 0.5  # generous — solver samples may not hit 0
        return passed, f"tmax = {tmax:.4f} h (expect < 0.5)"

    def invariant_non_negative_concentrations(self) -> Tuple[bool, str]:
        """All plasma + organ concentrations must be non-negative."""
        result = self._run_reference_iv()
        min_plasma = min(result.plasma_conc_mg_per_L)
        worst_organ = ("plasma", min_plasma)
        for name, oc in result.organ_profiles.items():
            if not oc.concentration_mg_per_L:
                continue
            m = min(oc.concentration_mg_per_L)
            if m < worst_organ[1]:
                worst_organ = (name, m)
        passed = worst_organ[1] >= self.NEGATIVE_CONC_FLOOR
        return (
            passed,
            f"worst = {worst_organ[0]} @ {worst_organ[1]:.3e} mg/L",
        )

    def invariant_monotone_post_cmax(self) -> Tuple[bool, str]:
        """After Cmax the plasma profile must not rise (within 1 percent
        solver tolerance). Catches spurious re-peaks from numerical noise
        or mis-wired redistribution.
        """
        result = self._run_reference_iv()
        plasma = list(result.plasma_conc_mg_per_L)
        if len(plasma) < 3:
            return False, "profile too short"
        idx_max = plasma.index(max(plasma))
        worst_rise = 0.0
        for i in range(idx_max, len(plasma) - 1):
            if plasma[i] <= 0:
                continue
            rise = (plasma[i + 1] - plasma[i]) / plasma[i]
            if rise > worst_rise:
                worst_rise = rise
        passed = worst_rise <= 0.01
        return passed, f"worst relative rise post-Cmax = {worst_rise:.4%}"

    def invariant_cl_auc_equals_dose(self) -> Tuple[bool, str]:
        """Strict mass balance for IV bolus.

        The correct elimination integral for a multi-compartment
        PBPK is:

            Eliminated(t) = ∫₀ᵗ ( CLₕ·C_liver/Kpₗᵢᵥₑᵣ
                              + CLᵣ·C_kidney/Kp_kidney ) ds

        Not ``CL_total × AUC(C_plasma)`` — that proxy systematically
        overshoots for drugs with high kp_liver and dominant
        hepatic extraction (midazolam, theophylline) because
        plasma AUC rises faster than the liver/kidney concentrations
        during the initial redistribution.

        This invariant runs the reference drug with tight solver
        tolerances (rtol=1e-8) and an extended time horizon
        (20 half-lives) so the elimination integral resolves to
        the full dose, then checks:

            (eliminated_integral + remaining_in_body_at_t_end) ≈ Dose
            within REL_TOLERANCE_CL_AUC (2 %).
        """
        drug = self._factory.create(self.REFERENCE_DRUG)
        schedule = self._scheduler.iv_bolus(
            self.REFERENCE_DRUG, self.REFERENCE_IV_DOSE_MG
        )
        # Use the strict-tolerance orchestrator for this invariant.
        result = self._strict_mb_orchestrator.run(
            drug, schedule, self._physiology
        )

        hepatic = self._hepatic_engine.compute(drug, self._physiology)
        renal = self._renal_engine.compute(drug, self._physiology)
        kp = self._partitioner.predict(drug, self._physiology)

        t = np.asarray(result.time_h, dtype=float)
        c_liver = np.asarray(
            result.organ_profiles["liver"].concentration_mg_per_L,
            dtype=float,
        )
        c_kidney = np.asarray(
            result.organ_profiles["kidney"].concentration_mg_per_L,
            dtype=float,
        )

        # Elimination integral on the time grid.
        try:
            trap = np.trapezoid
        except AttributeError:  # pragma: no cover — numpy < 2.0
            trap = np.trapz
        eliminated = (
            hepatic.cl_hepatic_L_per_h * trap(c_liver / kp.liver, t)
            + renal.cl_renal_net_L_per_h * trap(c_kidney / kp.kidney, t)
        )

        # Mass remaining in body at t_end (should be ~0 if tail is
        # long enough). Accounts for the truncation residual.
        remaining = self._remaining_body_mass(result)

        total = eliminated + remaining
        dose = result.dose_mg
        if dose <= 0:
            return False, "dose is non-positive"
        rel_err = abs(total - dose) / dose
        passed = rel_err <= self.REL_TOLERANCE_CL_AUC
        return (
            passed,
            f"eliminated={eliminated:.3f} + remaining={remaining:.3f} = "
            f"{total:.3f} mg vs dose={dose:.1f} mg "
            f"(rel err {rel_err:.3%}, tol {self.REL_TOLERANCE_CL_AUC:.0%})",
        )

    def _remaining_body_mass(self, result: SimulationResult) -> float:
        """Sum C×V across all non-lumen compartments at the last time point."""
        phys = self._physiology
        organ_to_compartment = {
            "plasma":   phys.plasma,
            "lung":     phys.lung,
            "liver":    phys.liver,
            "gut_wall": phys.gut,
            "kidney":   phys.kidney,
            "brain":    phys.brain,
            "heart":    phys.heart,
            "muscle":   phys.muscle,
            "fat":      phys.fat,
            "skin":     phys.skin,
            "bone":     phys.bone,
            "spleen":   phys.spleen,
            "pancreas": phys.pancreas,
        }
        total = 0.0
        for name, organ in organ_to_compartment.items():
            oc = result.organ_profiles.get(name)
            if oc is None or not oc.concentration_mg_per_L:
                continue
            c_end = oc.concentration_mg_per_L[-1]
            total += c_end * organ.volume_L
        return total

    def invariant_half_life_positive_finite(self) -> Tuple[bool, str]:
        """Terminal half-life must be strictly positive and physiological
        (0.1 < t½ < 500 h). PKCalculator clamps to this band — if the
        clamp is hit it's still a valid number but we report it.
        """
        result = self._run_reference_iv()
        t_half = result.pk_metrics.half_life_h
        in_band = 0.1 < t_half < 500.0
        return in_band, f"t½ = {t_half:.3f} h"

    def invariant_well_stirred_algebra(self) -> Tuple[bool, str]:
        """Validate the well-stirred liver identity used by HepaticClearance:

            CLh = Qh × fu × CLint / (Qh + fu × CLint)

        This is a pure-algebra check on the engine's own numbers. If the
        result's `cl_hepatic_L_per_h` drifts from the formula, the engine
        has a bug in the hepatic clearance path (not a validation error).
        """
        drug = self._factory.create(self.REFERENCE_DRUG)
        res = self._hepatic_engine.compute(drug, self._physiology)
        qh = res.qh_L_per_h
        fu = res.fu_plasma
        clint = res.clint_scaled_L_per_h
        if qh <= 0 or clint <= 0:
            return False, f"qh={qh}, clint={clint}"
        expected = qh * fu * clint / (qh + fu * clint)
        actual = res.cl_hepatic_L_per_h
        rel_err = abs(expected - actual) / expected if expected > 0 else 1.0
        passed = rel_err <= self.REL_TOLERANCE_WELL_STIRRED
        return (
            passed,
            f"expected {expected:.4f} vs actual {actual:.4f} "
            f"(rel err {rel_err:.2%})",
        )

    def invariant_zero_dose_near_zero(self) -> Tuple[bool, str]:
        """A near-zero dose should produce near-zero concentrations
        everywhere. Catches initialization bugs that inject spurious mass.
        Uses 1e-6 mg rather than literal zero to avoid degenerate solver
        behavior at exactly Dose=0.
        """
        drug = self._factory.create(self.REFERENCE_DRUG)
        schedule = self._scheduler.iv_bolus(self.REFERENCE_DRUG, 1.0e-6)
        try:
            result = self._orchestrator.run(drug, schedule, self._physiology)
        except Exception as exc:
            return False, f"zero-dose run failed: {exc}"
        max_plasma = max(result.plasma_conc_mg_per_L, default=0.0)
        # Cmax should scale roughly linearly with dose; 1e-6 mg should
        # produce at most 1e-6 / V_ss mg/L which is << 1e-3.
        passed = max_plasma < 1.0e-3
        return passed, f"max plasma at 1e-6 mg IV = {max_plasma:.3e} mg/L"

    def invariant_nonzero_dose_positive_response(self) -> Tuple[bool, str]:
        """A normal dose must produce a strictly positive Cmax and a
        finite AUC. Catches solver silent-fail modes where the engine
        returns zeros instead of raising.
        """
        result = self._run_reference_iv()
        cmax = result.pk_metrics.cmax_mg_per_L
        auc = result.pk_metrics.auc_0_inf_mg_h_per_L
        finite = math.isfinite(cmax) and math.isfinite(auc)
        positive = cmax > 0 and auc > 0
        passed = finite and positive
        return (
            passed,
            f"Cmax={cmax:.4f} mg/L, AUC_inf={auc:.4f} mg·h/L",
        )

    # ── runner ──────────────────────────────────────────────────

    def run(self) -> Tier4Report:
        """Execute all invariants and return a Tier4Report.

        ``cl_auc_equals_dose`` is GATING when the harness runs in
        ``mass_balance_mode=True`` (V1.1 solver) and ADVISORY when
        run against the legacy mixing-pool solver. The gating
        promotion is the primary motivation for the V1.1 fix.
        """
        cl_auc_gating = self._mass_balance_mode
        cl_auc_description = (
            "IV strict: CL_h·∫(C_liver/Kp_liver) + "
            "CL_r·∫(C_kidney/Kp_kidney) + body_remaining ≈ Dose "
            "within 2% (V1.1 gating)"
            if cl_auc_gating
            else (
                "IV: CL_total × AUC_0_inf ≈ Dose (advisory — V1 "
                "plasma is a mixing pool; switch the harness to "
                "mass_balance_mode=True to promote)"
            )
        )
        invariants: list[
            tuple[str, str, Callable[[], Tuple[bool, str]], bool]
        ] = [
            (
                "iv_peak_at_zero",
                "IV bolus Cmax must occur at tmax≈0 h",
                self.invariant_iv_peak_at_zero,
                True,
            ),
            (
                "non_negative_concentrations",
                "All plasma + organ concentrations must be ≥ 0",
                self.invariant_non_negative_concentrations,
                True,
            ),
            (
                "monotone_post_cmax",
                "Plasma must not re-rise by >1% after Cmax",
                self.invariant_monotone_post_cmax,
                True,
            ),
            (
                "cl_auc_equals_dose",
                cl_auc_description,
                self.invariant_cl_auc_equals_dose,
                cl_auc_gating,
            ),
            (
                "half_life_physiological",
                "Terminal t½ must be in (0.1, 500) h",
                self.invariant_half_life_positive_finite,
                True,
            ),
            (
                "well_stirred_algebra",
                "CLh = Qh·fu·CLint / (Qh + fu·CLint) identity",
                self.invariant_well_stirred_algebra,
                True,
            ),
            (
                "zero_dose_near_zero",
                "Near-zero dose produces near-zero concentrations",
                self.invariant_zero_dose_near_zero,
                True,
            ),
            (
                "positive_response",
                "Normal dose produces finite positive Cmax and AUC",
                self.invariant_nonzero_dose_positive_response,
                True,
            ),
        ]

        results = tuple(
            self._wrap(name, desc, fn, gating=gating)
            for name, desc, fn, gating in invariants
        )
        return Tier4Report(results=results)
