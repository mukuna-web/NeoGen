"""
Enzyme Pool Evolver — V1 DDI Engine (Step 9)
==============================================
Time-dependent CYP enzyme abundance ODE. This is the module that makes the
NeoGen DDI layer *dynamic* rather than a static post-hoc scaling.

Physics
-------
For each inducible / inhibitable CYP, the liver enzyme pool follows a
first-order turnover balance. Let ``E(t)`` be the enzyme abundance (or,
equivalently, the fold-change relative to baseline). Three processes act
on it simultaneously:

    dE/dt = ksyn(t) − kdeg · E − kobs(t) · E

where

    ksyn(t) = kdeg · E_baseline · (1 + f_ind(t))   [synthesis]
    f_ind(t) = Indmax · [I]u(t) / (IndC50 + [I]u(t))   [Emax induction, Step 8]
    kobs(t) = kinact · [I]u(t) / (KI + [I]u(t))   [MBI destruction, V1 CYP3A4 only]

At steady state ``dE/dt = 0`` and the normalised fold-change collapses to::

    E_ss / E_baseline  =  kdeg · (1 + f_ind) / (kdeg + kobs)

Induction alone (kobs = 0): fold = 1 + f_ind (matches Step 8).
MBI alone (f_ind = 0):      fold = kdeg / (kdeg + kobs)  (≤ 1, suppression).
Combined (rifampicin + ritonavir style): both effects multiply cleanly.

Working on the normalised variable ``e(t) = E(t) / E_baseline`` eliminates
the baseline abundance term and gives the canonical linear ODE::

    de/dt = kdeg · (1 + f_ind) − (kdeg + kobs) · e          [linear]

which has an analytical solution for a constant perpetrator concentration
but is integrated numerically so that time-varying ``[I]u(t)`` and eventual
Step 10 PBPK coupling work without a second code path.

Per-CYP degradation rate constants
----------------------------------
``kdeg = ln(2) / t_half``. Protein half-lives taken from Obach RS 2008,
Yang et al. 2008, Almond LM 2016. These set the time scale of induction
reaching steady state (~3-5 half-lives) — CYP3A4 hits 90 % of the
rifampicin plateau in ~5 days, CYP2C9 takes ~10-12 days.

Scope of Step 9
---------------
  * Time-dependent enzyme pool ODE for all 8 tracked CYPs.
  * Consumes the Emax induction math from Step 8 (no duplication).
  * Consumes the MBI parameters on ``InhibitionProfile`` (currently only
    ``kinact_cyp3a4`` / ``k_i_mbi_cyp3a4`` are populated in V1).
  * Provides a standalone ``simulate()`` method for isolation tests.
  * Provides a stateless ``derivative()`` method that Step 10's PBPK RHS
    will call at every solver step.

Out of scope:
  * Autoinduction — the perpetrator's own clearance does not yet feed back
    into [I]u. V2+ feature.
  * Gut vs liver separation — a single pool per CYP for V1.
  * Per-patient kdeg variability — uses the average adult values.

References
----------
    Obach RS et al. J Pharmacol Exp Ther 2008; 326: 336-348.
    Yang J et al. Curr Drug Metab 2008; 9: 384-394.
    Almond LM et al. Drug Metab Dispos 2016; 44: 821-832.
    Fahmi OA et al. Drug Metab Dispos 2008; 36: 1698-1708.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
from scipy.integrate import solve_ivp

from neogen.core.entities.drug import Drug
from neogen.engine.ddi.cyp_inhibition import CYP_NAMES

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# PER-CYP PROTEIN TURNOVER CONSTANTS
# ════════════════════════════════════════════════════════════════

#: Protein half-life (hours) for each tracked CYP. Values are the
#: mid-range of Obach 2008 and Yang 2008 reference tables.
_CYP_HALF_LIFE_H: Mapping[str, float] = {
    "CYP3A4": 36.0,
    "CYP2D6": 51.0,
    "CYP2C9": 55.0,
    "CYP2C19": 26.0,
    "CYP1A2": 39.0,
    "CYP2C8": 23.0,
    "CYP2B6": 23.0,
    "CYP2E1": 50.0,
}

#: Degradation rate constant ``kdeg = ln(2) / t_half`` (h⁻¹).
DEFAULT_KDEG_PER_H: Mapping[str, float] = {
    cyp: math.log(2.0) / t_half for cyp, t_half in _CYP_HALF_LIFE_H.items()
}

#: Type alias: either a constant concentration or a time-varying callable.
InducerConcInput = Union[float, Callable[[float], float]]


# ════════════════════════════════════════════════════════════════
# FROZEN RESULT DATACLASSES
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class EnzymeSnapshot:
    """Single time-point view of the enzyme pool state.

    Attributes:
        time_h: Elapsed time in hours since perpetrator exposure began.
        fold_by_cyp: Fold-change relative to baseline for each tracked
            CYP. ``1.0`` = baseline, ``>1`` = induced, ``<1`` = suppressed
            by MBI.
        inducer_conc_um: Unbound perpetrator concentration used at this
            time point (µM).
    """

    time_h: float
    fold_by_cyp: Mapping[str, float]
    inducer_conc_um: float

    def fold_for(self, cyp: str) -> float:
        """Return fold-change for ``cyp`` (case-insensitive). 1.0 if absent."""
        return self.fold_by_cyp.get(cyp.upper(), 1.0)


@dataclass(frozen=True)
class EnzymePoolTrace:
    """Full time course of the enzyme pool over a single simulation run.

    Attributes:
        perpetrator_name: Name of the drug being studied.
        duration_h: Total simulated duration (hours).
        snapshots: Ordered tuple of per-time-step snapshots. The first
            snapshot is always at ``t = 0`` with every fold = 1.0.
        tracked_cyps: The CYP isoforms present in every snapshot.
    """

    perpetrator_name: str
    duration_h: float
    snapshots: Tuple[EnzymeSnapshot, ...]
    tracked_cyps: Tuple[str, ...]

    def times_h(self) -> Tuple[float, ...]:
        """Tuple of snapshot times in hours."""
        return tuple(s.time_h for s in self.snapshots)

    def fold_trace(self, cyp: str) -> Tuple[float, ...]:
        """Full fold-vs-time trace for a single CYP."""
        key = cyp.upper()
        return tuple(s.fold_by_cyp.get(key, 1.0) for s in self.snapshots)

    def fold_at(self, time_h: float, cyp: str) -> float:
        """Linear interpolation of the fold-change at an arbitrary time.

        Returns 1.0 outside the simulated window. Used by Step 10's
        runtime coupling to pull CLint(t) mid-solve without recomputing.
        """
        if not self.snapshots:
            return 1.0
        if time_h <= self.snapshots[0].time_h:
            return self.snapshots[0].fold_by_cyp.get(cyp.upper(), 1.0)
        if time_h >= self.snapshots[-1].time_h:
            return self.snapshots[-1].fold_by_cyp.get(cyp.upper(), 1.0)

        key = cyp.upper()
        # Binary search would be cleaner but snapshots are small (<500).
        for i in range(len(self.snapshots) - 1):
            t0 = self.snapshots[i].time_h
            t1 = self.snapshots[i + 1].time_h
            if t0 <= time_h <= t1:
                f0 = self.snapshots[i].fold_by_cyp.get(key, 1.0)
                f1 = self.snapshots[i + 1].fold_by_cyp.get(key, 1.0)
                if t1 == t0:
                    return f0
                w = (time_h - t0) / (t1 - t0)
                return f0 * (1.0 - w) + f1 * w
        return 1.0

    def max_fold(self, cyp: str) -> float:
        """Maximum fold-change observed for ``cyp`` over the trace."""
        trace = self.fold_trace(cyp)
        return max(trace) if trace else 1.0

    def time_to_fraction(
        self,
        cyp: str,
        fraction: float,
        *,
        target_fold: Optional[float] = None,
    ) -> Optional[float]:
        """Time (h) to reach ``fraction`` of the steady-state fold on ``cyp``.

        If ``target_fold`` is None, uses the trace's own ``max_fold(cyp)``
        as the steady-state reference — the classic "time to 90 % of
        plateau" measurement for rifampicin.

        Returns ``None`` if the target is never reached within the window.
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")

        trace = self.fold_trace(cyp)
        times = self.times_h()
        if not trace:
            return None

        plateau = target_fold if target_fold is not None else max(trace)
        if plateau <= 1.0:
            return None  # No meaningful induction

        threshold = 1.0 + fraction * (plateau - 1.0)
        for t, f in zip(times, trace):
            if f >= threshold:
                return t
        return None


# ════════════════════════════════════════════════════════════════
# ENGINE
# ════════════════════════════════════════════════════════════════


class EnzymePoolEvolver:
    """Time-dependent CYP enzyme pool integrator.

    Usage:
        evolver = EnzymePoolEvolver()

        # Standalone isolated simulation (Step 9)
        trace = evolver.simulate(
            perpetrator=rifampicin,
            inducer_conc_um=0.2,
            duration_on_h=7 * 24,
            duration_off_h=14 * 24,
        )
        print(trace.fold_at(5 * 24, "CYP3A4"))   # ~0.9 of plateau

        # Stateless ODE hook (Step 10 PBPK RHS coupling)
        de_dt = evolver.derivative(
            e_current=1.0,
            cyp="CYP3A4",
            perpetrator=rifampicin,
            inducer_conc_um=0.5,
        )
    """

    #: Defensive ceiling on the fold-change state to guard against
    #: pathological Indmax values. Matches the Step 8 ceiling.
    _FOLD_CEILING: float = 50.0

    def __init__(
        self,
        kdeg_per_h: Optional[Mapping[str, float]] = None,
        tracked_cyps: Iterable[str] = CYP_NAMES,
    ) -> None:
        self._kdeg = dict(kdeg_per_h) if kdeg_per_h else dict(DEFAULT_KDEG_PER_H)
        self._tracked: Tuple[str, ...] = tuple(tracked_cyps)
        # Sanity: every tracked CYP must have a kdeg.
        missing = [c for c in self._tracked if c not in self._kdeg]
        if missing:
            raise ValueError(
                f"kdeg_per_h is missing entries for CYPs: {missing}"
            )

    # ── Public API ───────────────────────────────────────────

    @property
    def tracked_cyps(self) -> Tuple[str, ...]:
        """CYPs integrated by this evolver (ordered)."""
        return self._tracked

    def kdeg(self, cyp: str) -> float:
        """Return the degradation rate constant (h⁻¹) for ``cyp``."""
        return self._kdeg[cyp.upper()]

    def derivative(
        self,
        *,
        e_current: float,
        cyp: str,
        perpetrator: Drug,
        inducer_conc_um: float,
    ) -> float:
        """Instantaneous ``de/dt`` for a single CYP fold-change state.

        Used by Step 10's PBPK RHS at every solver step. Pure function —
        no internal state, no logging, hot-path safe.

        Args:
            e_current: Current normalised enzyme abundance (1.0 = baseline).
            cyp: CYP isoform whose derivative is being evaluated.
            perpetrator: Drug providing Indmax/IndC50/kinact/KI parameters.
            inducer_conc_um: Current unbound perpetrator concentration (µM).

        Returns:
            ``de/dt`` (h⁻¹) for the normalised abundance.
        """
        key = cyp.upper()
        kdeg = self._kdeg[key]
        f_ind = self._induction_factor(perpetrator, key, inducer_conc_um)
        kobs = self._mbi_rate_per_h(perpetrator, key, inducer_conc_um)
        # de/dt = kdeg*(1 + f_ind) - (kdeg + kobs) * e
        return kdeg * (1.0 + f_ind) - (kdeg + kobs) * e_current

    def steady_state_fold(
        self,
        perpetrator: Drug,
        inducer_conc_um: float,
        cyp: str,
    ) -> float:
        """Analytical steady-state fold-change for one CYP.

        ``e_ss = kdeg · (1 + f_ind) / (kdeg + kobs)``.
        Faster than ``simulate()`` when only the asymptotic value matters.
        """
        key = cyp.upper()
        kdeg = self._kdeg[key]
        f_ind = self._induction_factor(perpetrator, key, inducer_conc_um)
        kobs = self._mbi_rate_per_h(perpetrator, key, inducer_conc_um)
        e_ss = kdeg * (1.0 + f_ind) / (kdeg + kobs)
        return min(e_ss, self._FOLD_CEILING)

    def simulate(
        self,
        perpetrator: Drug,
        inducer_conc_um: InducerConcInput,
        *,
        duration_on_h: float,
        duration_off_h: float = 0.0,
        n_points: int = 200,
        rtol: float = 1e-6,
        atol: float = 1e-9,
    ) -> EnzymePoolTrace:
        """Integrate the enzyme pool ODE for a single perpetrator exposure.

        The simulation has two phases:

          * ``[0, duration_on_h]`` — perpetrator present; ``[I]u`` equals
            the supplied value (or the callable's output at each t).
          * ``(duration_on_h, duration_on_h + duration_off_h]`` — washout.
            ``[I]u = 0``, so induction synthesis drops back to baseline
            and MBI-suppressed enzyme recovers via de novo synthesis.

        Args:
            perpetrator: Perpetrator ``Drug``.
            inducer_conc_um: Constant µM value, or a callable ``t -> µM``.
            duration_on_h: Length of the "on" phase (hours).
            duration_off_h: Length of the washout phase (hours, default 0).
            n_points: Target number of time-grid samples returned
                in the trace (at least ~50 recommended for smooth curves).
            rtol, atol: Relative / absolute tolerances for ``solve_ivp``.

        Returns:
            Frozen :class:`EnzymePoolTrace` covering both phases.
        """
        if duration_on_h < 0 or duration_off_h < 0:
            raise ValueError("durations must be non-negative")
        if n_points < 2:
            raise ValueError("n_points must be ≥ 2")

        total_h = duration_on_h + duration_off_h
        if total_h <= 0:
            raise ValueError(
                "At least one of duration_on_h / duration_off_h must be > 0"
            )

        conc_fn = self._coerce_conc_fn(
            inducer_conc_um, duration_on_h=duration_on_h
        )

        # State vector: one fold per tracked CYP, starts at 1.0.
        n = len(self._tracked)
        e0 = np.ones(n, dtype=float)

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            conc = conc_fn(t)
            if conc < 0.0:
                conc = 0.0
            dy = np.empty_like(y)
            for i, cyp in enumerate(self._tracked):
                kdeg = self._kdeg[cyp]
                f_ind = self._induction_factor(perpetrator, cyp, conc)
                kobs = self._mbi_rate_per_h(perpetrator, cyp, conc)
                dy[i] = kdeg * (1.0 + f_ind) - (kdeg + kobs) * y[i]
            return dy

        t_eval = np.linspace(0.0, total_h, n_points)
        sol = solve_ivp(
            rhs,
            (0.0, total_h),
            e0,
            method="LSODA",
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            max_step=min(6.0, total_h),  # keep resolution on the transients
        )
        if not sol.success:
            raise RuntimeError(
                f"EnzymePoolEvolver: ODE solver failed — {sol.message}"
            )

        snapshots: list[EnzymeSnapshot] = []
        for j, t in enumerate(sol.t):
            fold_map: Dict[str, float] = {}
            for i, cyp in enumerate(self._tracked):
                fold_map[cyp] = float(
                    min(max(sol.y[i, j], 0.0), self._FOLD_CEILING)
                )
            snapshots.append(
                EnzymeSnapshot(
                    time_h=float(t),
                    fold_by_cyp=fold_map,
                    inducer_conc_um=float(conc_fn(float(t))),
                )
            )

        trace = EnzymePoolTrace(
            perpetrator_name=perpetrator.name,
            duration_h=total_h,
            snapshots=tuple(snapshots),
            tracked_cyps=self._tracked,
        )

        strongest = max(
            self._tracked,
            key=lambda c: trace.max_fold(c),
            default=None,
        )
        if strongest and trace.max_fold(strongest) > 1.1:
            logger.info(
                "Enzyme pool: %s on %s reaches %.2f× over %.1f h",
                perpetrator.name, strongest,
                trace.max_fold(strongest), total_h,
            )

        return trace

    # ── Internal helpers ─────────────────────────────────────

    @staticmethod
    def _coerce_conc_fn(
        conc: InducerConcInput,
        *,
        duration_on_h: float,
    ) -> Callable[[float], float]:
        """Turn either a float or a callable into a time function that
        drops to zero after the ``duration_on_h`` mark."""
        if callable(conc):
            fn = conc  # type: ignore[assignment]

            def _callable_with_washout(t: float) -> float:
                if t > duration_on_h:
                    return 0.0
                return float(fn(t))

            return _callable_with_washout

        value = float(conc)  # type: ignore[arg-type]
        if value < 0.0:
            raise ValueError(f"inducer_conc_um must be ≥ 0, got {value}")

        def _constant_with_washout(t: float) -> float:
            return value if t <= duration_on_h else 0.0

        return _constant_with_washout

    @staticmethod
    def _induction_factor(
        perpetrator: Drug,
        cyp: str,
        inducer_conc_um: float,
    ) -> float:
        """Emax induction saturation term ``f_ind = Indmax · C / (IndC50 + C)``.

        Returns 0.0 (no induction) when the CYP is not an inducible
        target on this perpetrator or the concentration is zero.
        """
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
        perpetrator: Drug,
        cyp: str,
        inducer_conc_um: float,
    ) -> float:
        """Mechanism-based inactivation rate ``kobs = kinact · C / (KI + C)``.

        ``kinact`` is stored in min⁻¹ on the V1 ``InhibitionProfile`` per
        the PBPK convention; we convert to h⁻¹ here so the ODE state
        equations all use a single time base.

        V1 populates MBI parameters only for CYP3A4; any other CYP
        returns 0.0.
        """
        key = cyp.upper()
        inhibition = perpetrator.inhibition
        if key == "CYP3A4":
            kinact_per_min = inhibition.kinact_cyp3a4
            ki_mbi_um = inhibition.k_i_mbi_cyp3a4
        else:
            return 0.0

        if kinact_per_min is None or kinact_per_min <= 0.0:
            return 0.0
        if ki_mbi_um is None or ki_mbi_um <= 0.0:
            return 0.0
        if inducer_conc_um <= 0.0:
            return 0.0

        kinact_per_h = kinact_per_min * 60.0
        return kinact_per_h * inducer_conc_um / (ki_mbi_um + inducer_conc_um)
