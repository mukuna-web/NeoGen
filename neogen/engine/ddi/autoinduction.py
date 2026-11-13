"""
Autoinduction — V2 DDI Engine Extension
========================================
Time-dependent self-induction model for drugs that induce the CYPs
responsible for clearing them. This is the mechanism behind rifampicin's
characteristic 30-40 % AUC reduction over 5-7 days, and the analogous
signatures of efavirenz, carbamazepine, and nevirapine.

Why a new module (not a flag on EnzymePoolEvolver)
---------------------------------------------------
The Step 9 evolver treats the perpetrator concentration ``[I]u`` as an
*exogenous* input — either a scalar or a caller-supplied time function.
Autoinduction violates that contract because ``[I]u(t)`` depends on the
current enzyme pool state ``e(t)``:

    [I]u(t) = [I]u_baseline · AUCR_self(t)
    AUCR_self(t) = 1 / ( (1 − Σ fm_CYP) + Σ fm_CYP · e_CYP(t) )

With the feedback loop closed, the system becomes a coupled nonlinear
ODE in ``e`` — scalar ``e`` on the LHS, fold-dependent ``[I]u`` inside
the induction saturation on the RHS. ``EnzymePoolEvolver`` explicitly
documents this case as V2+ out-of-scope; rather than retrofit a
feedback hook into its stateless ``derivative()`` API, this module
provides a dedicated simulator that integrates the closed system.

Coupled ODE system (one state variable per inducible CYP)
----------------------------------------------------------
::

    de_i/dt = kdeg_i · (1 + f_ind_i([I]u(t))) − (kdeg_i + kobs_i([I]u(t))) · e_i(t)
    [I]u(t) = [I]u_baseline / ( (1 − Σ fm_i) + Σ fm_i · e_i(t) )
    f_ind_i(C) = Indmax_i · C / (IndC50_i + C)
    kobs_i(C)  = kinact_i · C / (KI_i + C)   (V1: CYP3A4 only)

The trace produced by ``simulate()`` is a regular :class:`EnzymePoolTrace`
so the downstream :class:`DDIOrchestrator` can consume it without any
special-casing. The only observable difference vs Step 9 is that the
``inducer_conc_um`` field on each :class:`EnzymeSnapshot` carries the
*post-attenuation* concentration — useful for diagnostics.

Scope of this V2 extension
--------------------------
  * Time-dependent self-induction for any drug whose ``fm_CYP`` overlaps
    with its own induction targets.
  * Fully mechanism-based — uses the existing ``InductionProfile``
    parameters, no new drug fields required.
  * Integrates cleanly with :class:`DDIOrchestrator.build_exposure` via
    the new ``autoinduction=True`` flag (see that module).

Out of scope (later V2+):
  * Gut vs liver compartment separation.
  * Concurrent third-party perpetrator on top of autoinduction.
  * Non-linear protein binding during self-attenuation.
  * Transporter-mediated feedback.

References
----------
    Yamashita F et al. Drug Metab Pharmacokinet 2013; 28: 37-45.
    Baneyx G et al. Eur J Pharm Sci 2014; 56: 1-15.
    Hariparsad N et al. Drug Metab Dispos 2022; 50: 795-807 — Simcyp
        ADAM-driven autoinduction calibration for rifampicin, efavirenz,
        carbamazepine.
    Chen Y et al. Drug Metab Dispos 2012; 40: 1170-1177 — rifampicin
        self-clearance build-up measured in healthy volunteers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from neogen.core.entities.drug import Drug
from neogen.engine.ddi.cyp_inhibition import CYP_NAMES
from neogen.engine.ddi.enzyme_pool_evolver import (
    DEFAULT_KDEG_PER_H,
    EnzymePoolTrace,
    EnzymeSnapshot,
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# DETECTION HELPERS
# ════════════════════════════════════════════════════════════════


def is_autoinducer(drug: Drug) -> bool:
    """Return ``True`` iff the drug induces any CYP that also clears it.

    An autoinducer must simultaneously satisfy:

      * ``fm_CYP > 0``          — at least one CYP contributes to its
        clearance (``MetabolismProfile.fm_for``).
      * ``Indmax_CYP > 0`` on the **same** CYP — it is a positive
        inducer of that isoform (``InductionProfile.params_for``).

    Rifampicin is the canonical example: CYP3A4 fm ≈ 0.6 and Indmax
    CYP3A4 ≈ 8-16, so self-induction drives its own clearance upward
    over 5-7 days.
    """
    if not drug.induction.is_inducer:
        return False
    for cyp in CYP_NAMES:
        fm = drug.metabolism.fm_for(cyp)
        indmax, indc50 = drug.induction.params_for(cyp)
        if fm > 0.0 and indmax is not None and indmax > 0.0 and indc50 is not None and indc50 > 0.0:
            return True
    return False


def overlapping_cyps(drug: Drug) -> Tuple[str, ...]:
    """Ordered tuple of CYPs that both clear the drug and are induced by it."""
    out: List[str] = []
    for cyp in CYP_NAMES:
        fm = drug.metabolism.fm_for(cyp)
        indmax, indc50 = drug.induction.params_for(cyp)
        if fm > 0.0 and indmax is not None and indmax > 0.0 and indc50 is not None and indc50 > 0.0:
            out.append(cyp)
    return tuple(out)


# ════════════════════════════════════════════════════════════════
# FROZEN RESULT DATACLASS
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AutoinductionSummary:
    """Steady-state summary of a self-induction run.

    Attributes:
        drug_name: Perpetrator drug name.
        baseline_conc_um: Raw unbound liver concentration supplied by
            the caller (µM, "day-1" exposure before any induction).
        effective_conc_um_ss: Steady-state ``[I]u`` after self-attenuation.
        aucr_self_ss: Self-on-self AUC ratio at steady state,
            ``effective / baseline``. Values < 1 indicate that chronic
            dosing produces lower exposure than the first dose.
        fold_by_cyp_ss: Normalised enzyme fold-change at the end of the
            on-window, keyed by CYP name.
        time_to_90pct_h: Time to reach 90 % of the dominant-overlap CYP's
            steady-state fold. ``None`` if induction never materialises
            or the on-window was too short to capture it.
    """

    drug_name: str
    baseline_conc_um: float
    effective_conc_um_ss: float
    aucr_self_ss: float
    fold_by_cyp_ss: Mapping[str, float]
    time_to_90pct_h: Optional[float]


# ════════════════════════════════════════════════════════════════
# SIMULATOR
# ════════════════════════════════════════════════════════════════


class AutoinductionSimulator:
    """Integrator for the coupled self-induction ODE.

    Produces an :class:`EnzymePoolTrace` byte-compatible with the
    outputs of :class:`EnzymePoolEvolver`, so the downstream
    :class:`DDIOrchestrator` can consume it unchanged.

    Usage:
        sim = AutoinductionSimulator()
        trace = sim.simulate(
            rifampicin,
            baseline_conc_um=1.0,
            duration_on_h=10 * 24,
        )
        summary = sim.summarize(trace, rifampicin, baseline_conc_um=1.0)
        print(summary.aucr_self_ss)   # ~0.3-0.7 depending on parameters
    """

    #: Defensive ceiling on the fold-change state (matches Step 9).
    _FOLD_CEILING: float = 50.0

    #: Lower bound on the fold state used when computing the AUCR_self
    #: feedback term. Prevents division instabilities if the solver
    #: transiently produces slightly-negative ``e`` during stiff steps.
    _E_FLOOR: float = 1.0e-6

    def __init__(
        self,
        kdeg_per_h: Optional[Mapping[str, float]] = None,
        tracked_cyps: Tuple[str, ...] = CYP_NAMES,
    ) -> None:
        self._kdeg: Dict[str, float] = (
            dict(kdeg_per_h) if kdeg_per_h else dict(DEFAULT_KDEG_PER_H)
        )
        self._tracked: Tuple[str, ...] = tuple(tracked_cyps)
        missing = [c for c in self._tracked if c not in self._kdeg]
        if missing:
            raise ValueError(
                f"kdeg_per_h is missing entries for CYPs: {missing}"
            )

    # ── Public API ───────────────────────────────────────────

    @property
    def tracked_cyps(self) -> Tuple[str, ...]:
        return self._tracked

    def simulate(
        self,
        perpetrator: Drug,
        *,
        baseline_conc_um: float,
        duration_on_h: float,
        duration_off_h: float = 0.0,
        n_points: int = 240,
        rtol: float = 1.0e-6,
        atol: float = 1.0e-9,
    ) -> EnzymePoolTrace:
        """Integrate the coupled autoinduction ODE.

        Args:
            perpetrator: The self-inducer drug.
            baseline_conc_um: Unbound liver concentration (µM) the drug
                would reach in the **absence** of any induction — i.e.
                day-1 exposure before the enzyme pool has ramped up.
            duration_on_h: Length of the dosing window.
            duration_off_h: Length of the washout phase (default 0).
            n_points: Number of snapshots in the output trace.
            rtol, atol: Tolerances forwarded to ``solve_ivp``.

        Returns:
            An :class:`EnzymePoolTrace` covering both phases. The
            ``inducer_conc_um`` field on each snapshot carries the
            **post-attenuation** ``[I]u`` at that time.
        """
        if baseline_conc_um < 0.0:
            raise ValueError(
                f"baseline_conc_um must be ≥ 0, got {baseline_conc_um}"
            )
        if duration_on_h < 0.0 or duration_off_h < 0.0:
            raise ValueError("durations must be non-negative")
        total_h = duration_on_h + duration_off_h
        if total_h <= 0.0:
            raise ValueError(
                "At least one of duration_on_h / duration_off_h must be > 0"
            )
        if n_points < 2:
            raise ValueError("n_points must be ≥ 2")

        fm_by_cyp: Tuple[Tuple[str, float], ...] = tuple(
            (cyp, perpetrator.metabolism.fm_for(cyp)) for cyp in self._tracked
        )
        total_fm = sum(fm for _, fm in fm_by_cyp)
        non_cyp = max(1.0 - total_fm, 0.0)

        if not is_autoinducer(perpetrator):
            logger.info(
                "AutoinductionSimulator: %s is not a self-inducer — "
                "the coupled ODE collapses to the exogenous-[I]u baseline.",
                perpetrator.name,
            )

        def effective_conc(t: float, y: np.ndarray) -> float:
            """``[I]u(t)`` after self-attenuation by the current pool."""
            if t > duration_on_h:
                return 0.0
            denom = non_cyp
            for i, (_, fm) in enumerate(fm_by_cyp):
                if fm > 0.0:
                    denom += fm * max(float(y[i]), self._E_FLOOR)
            if denom <= 0.0:
                return baseline_conc_um
            return baseline_conc_um / denom

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            conc = effective_conc(t, y)
            dy = np.empty_like(y)
            for i, cyp in enumerate(self._tracked):
                kdeg = self._kdeg[cyp]
                f_ind = self._induction_factor(perpetrator, cyp, conc)
                kobs = self._mbi_rate_per_h(perpetrator, cyp, conc)
                dy[i] = kdeg * (1.0 + f_ind) - (kdeg + kobs) * y[i]
            return dy

        n = len(self._tracked)
        e0 = np.ones(n, dtype=float)
        t_eval = np.linspace(0.0, total_h, n_points)

        sol = solve_ivp(
            rhs,
            (0.0, total_h),
            e0,
            method="LSODA",
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            max_step=min(6.0, total_h),
        )
        if not sol.success:
            raise RuntimeError(
                f"AutoinductionSimulator: ODE solver failed — {sol.message}"
            )

        snapshots: List[EnzymeSnapshot] = []
        for j, t in enumerate(sol.t):
            y_col = sol.y[:, j]
            fold_map: Dict[str, float] = {}
            for i, cyp in enumerate(self._tracked):
                fold_map[cyp] = float(
                    min(max(y_col[i], 0.0), self._FOLD_CEILING)
                )
            snapshots.append(
                EnzymeSnapshot(
                    time_h=float(t),
                    fold_by_cyp=fold_map,
                    inducer_conc_um=float(effective_conc(float(t), y_col)),
                )
            )

        trace = EnzymePoolTrace(
            perpetrator_name=perpetrator.name,
            duration_h=total_h,
            snapshots=tuple(snapshots),
            tracked_cyps=self._tracked,
        )

        dominant = self._dominant_overlap(perpetrator)
        if dominant is not None:
            plateau = trace.max_fold(dominant)
            if plateau > 1.1:
                logger.info(
                    "Autoinduction: %s self-induces %s to %.2f× over %.1f h",
                    perpetrator.name, dominant, plateau, total_h,
                )

        return trace

    def summarize(
        self,
        trace: EnzymePoolTrace,
        perpetrator: Drug,
        *,
        baseline_conc_um: float,
    ) -> AutoinductionSummary:
        """Extract the steady-state summary from an autoinduction trace.

        "Steady state" is taken as the last snapshot inside the on-window
        (last snapshot whose ``inducer_conc_um > 0``). For a well-sized
        window (e.g. ≥ 5 · t½_CYP) this is within 1-2 % of the analytical
        fixed point.
        """
        on_snapshots = [s for s in trace.snapshots if s.inducer_conc_um > 0.0]
        if not on_snapshots:
            return AutoinductionSummary(
                drug_name=perpetrator.name,
                baseline_conc_um=baseline_conc_um,
                effective_conc_um_ss=baseline_conc_um,
                aucr_self_ss=1.0,
                fold_by_cyp_ss={c: 1.0 for c in self._tracked},
                time_to_90pct_h=None,
            )

        last = on_snapshots[-1]
        ss_fold: Dict[str, float] = dict(last.fold_by_cyp)

        fm_by_cyp = tuple(
            (cyp, perpetrator.metabolism.fm_for(cyp)) for cyp in self._tracked
        )
        total_fm = sum(fm for _, fm in fm_by_cyp)
        non_cyp = max(1.0 - total_fm, 0.0)
        denom = non_cyp
        for cyp, fm in fm_by_cyp:
            if fm > 0.0:
                denom += fm * ss_fold.get(cyp, 1.0)
        aucr_self = 1.0 / denom if denom > 0.0 else 1.0
        effective_conc_ss = baseline_conc_um * aucr_self

        dominant = self._dominant_overlap(perpetrator)
        t90: Optional[float] = None
        if dominant is not None:
            t90 = trace.time_to_fraction(dominant, 0.9)

        return AutoinductionSummary(
            drug_name=perpetrator.name,
            baseline_conc_um=baseline_conc_um,
            effective_conc_um_ss=effective_conc_ss,
            aucr_self_ss=aucr_self,
            fold_by_cyp_ss=ss_fold,
            time_to_90pct_h=t90,
        )

    # ── Internal helpers ─────────────────────────────────────

    def _dominant_overlap(self, drug: Drug) -> Optional[str]:
        overlap = overlapping_cyps(drug)
        if not overlap:
            return None
        return max(overlap, key=lambda c: drug.metabolism.fm_for(c))

    @staticmethod
    def _induction_factor(
        perpetrator: Drug, cyp: str, inducer_conc_um: float
    ) -> float:
        """Emax induction saturation ``Indmax · C / (IndC50 + C)``."""
        indmax, indc50 = perpetrator.induction.params_for(cyp)
        if indmax is None or indmax <= 0.0:
            return 0.0
        if indc50 is None or indc50 <= 0.0:
            return 0.0
        if inducer_conc_um <= 0.0:
            return 0.0
        return indmax * inducer_conc_um / (indc50 + inducer_conc_um)

    @staticmethod
    def _mbi_rate_per_h(
        perpetrator: Drug, cyp: str, inducer_conc_um: float
    ) -> float:
        """Mechanism-based inactivation ``kinact · C / (KI + C)`` (CYP3A4)."""
        if cyp.upper() != "CYP3A4":
            return 0.0
        inhibition = perpetrator.inhibition
        kinact_per_min = inhibition.kinact_cyp3a4
        ki_mbi_um = inhibition.k_i_mbi_cyp3a4
        if kinact_per_min is None or kinact_per_min <= 0.0:
            return 0.0
        if ki_mbi_um is None or ki_mbi_um <= 0.0:
            return 0.0
        if inducer_conc_um <= 0.0:
            return 0.0
        kinact_per_h = kinact_per_min * 60.0
        return kinact_per_h * inducer_conc_um / (ki_mbi_um + inducer_conc_um)
