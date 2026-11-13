"""
Simulation Result — V1 Engine
===============================
Immutable containers for PBPK simulation output.

Computes standard PK metrics from time-concentration profiles:
  - Cmax: peak plasma concentration
  - tmax: time to peak
  - AUC: area under the curve (trapezoidal rule)
  - t½: terminal elimination half-life
  - CL/F: apparent clearance
  - Vss: steady-state volume of distribution
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PKMetrics:
    """Standard pharmacokinetic metrics derived from a plasma profile."""

    cmax_mg_per_L: float         # Peak plasma concentration
    tmax_h: float                # Time to peak
    auc_0_inf_mg_h_per_L: float  # AUC extrapolated to infinity
    auc_0_last_mg_h_per_L: float # AUC to last measurable time
    half_life_h: float           # Terminal elimination half-life
    cl_apparent_L_per_h: float   # CL/F = Dose / AUC
    vss_apparent_L: float        # Vss/F = CL/F × MRT
    mrt_h: float                 # Mean residence time

    @property
    def auc_ratio(self) -> float:
        """AUC ratio (for DDI: AUC_inhibited / AUC_control). Default 1.0."""
        return 1.0


@dataclass(frozen=True)
class OrganTimeCourse:
    """Time-concentration data for a single organ."""
    name: str
    time_h: Tuple[float, ...]
    concentration_mg_per_L: Tuple[float, ...]

    @property
    def cmax(self) -> float:
        return max(self.concentration_mg_per_L) if self.concentration_mg_per_L else 0.0

    @property
    def tmax(self) -> float:
        if not self.concentration_mg_per_L:
            return 0.0
        idx = self.concentration_mg_per_L.index(max(self.concentration_mg_per_L))
        return self.time_h[idx]


@dataclass(frozen=True)
class SimulationResult:
    """
    Complete immutable output from a PBPK simulation.

    Contains:
      - PK metrics (Cmax, AUC, t½, etc.)
      - Time-concentration profiles per organ
      - Drug and dose metadata
      - Clearance breakdown
      - Optional absorption-engine outputs (for oral simulations)
    """

    # Identity
    drug_name: str
    dose_mg: float
    route: str

    # Primary PK metrics
    pk_metrics: PKMetrics

    # Plasma time course (primary output)
    time_h: Tuple[float, ...]
    plasma_conc_mg_per_L: Tuple[float, ...]

    # Organ profiles (name → OrganTimeCourse)
    organ_profiles: Dict[str, OrganTimeCourse]

    # Clearance breakdown
    cl_hepatic_L_per_h: float
    cl_renal_L_per_h: float
    cl_total_L_per_h: float

    # Optional absorption-engine outputs.
    #
    # Populated only when the schedule contains oral doses and the
    # P2 permeability engine produced a profile. Each element is an
    # ``AbsorptionProfile`` with fa, fg, ka_eff, tlag, and the
    # cumulative dissolution / permeation / portal traces. Typed as
    # ``Any`` to avoid a hard import from the absorption subpackage
    # at result-class definition time.
    absorption_profiles: Tuple[Any, ...] = field(default_factory=tuple)

    # Solver metadata
    solver_method: str = "BDF"
    solver_success: bool = True
    n_timepoints: int = 0

    # Post-prediction PK consistency QC. Populated by the orchestrator
    # after the result is assembled. ``None`` when the QC could not be
    # computed (e.g. degenerate values). Keys:
    #   cl_predicted, vd_predicted, half_life_predicted,
    #   half_life_expected, gmfe, is_consistent
    pk_consistency: Optional[Dict[str, Any]] = None


# ════════════════════════════════════════════════════════════════
# PK CONSISTENCY QC
# ════════════════════════════════════════════════════════════════

# Threshold mirrors the Tier 1 2-fold gate (looser by 25% / tighter
# by 25%) used by the validation harness. Do not redefine elsewhere.
PK_CONSISTENCY_GMFE_OK: float = 1.5
PK_CONSISTENCY_GMFE_BORDERLINE: float = 2.5


def compute_pk_consistency_qc(
    cl_L_per_h: Optional[float],
    vd_L: Optional[float],
    half_life_h: Optional[float],
) -> Optional[Dict[str, Any]]:
    """Cross-check the algebraic PK relation t½ = ln(2) · Vd / CL.

    Computes the expected half-life from CL+Vd and reports the
    geometric-mean fold error between the predicted and expected
    half-life. Returns ``None`` when any input is missing or
    non-positive (the relation is undefined there).

    The returned dict is suitable for storage on
    :class:`SimulationResult.pk_consistency` and downstream surfacing
    in validation reports.
    """
    if (
        cl_L_per_h is None
        or vd_L is None
        or half_life_h is None
        or cl_L_per_h <= 0.0
        or vd_L <= 0.0
        or half_life_h <= 0.0
    ):
        return None

    half_life_expected = math.log(2.0) * vd_L / cl_L_per_h
    if half_life_expected <= 0.0:
        return None

    ratio = half_life_h / half_life_expected
    # Geometric-mean fold error: max(ratio, 1/ratio).
    gmfe = ratio if ratio >= 1.0 else 1.0 / ratio

    return {
        "cl_predicted": round(cl_L_per_h, 6),
        "vd_predicted": round(vd_L, 6),
        "half_life_predicted": round(half_life_h, 6),
        "half_life_expected": round(half_life_expected, 6),
        "gmfe": round(gmfe, 4),
        "is_consistent": gmfe < PK_CONSISTENCY_GMFE_OK,
    }


# ════════════════════════════════════════════════════════════════
# PK METRIC COMPUTATION
# ════════════════════════════════════════════════════════════════

class PKCalculator:
    """
    Computes PK metrics from raw time-concentration arrays.
    Pure functions — no state.
    """

    @staticmethod
    def compute(
        time_h: np.ndarray,
        plasma_conc: np.ndarray,
        dose_mg: float,
        route: str = "iv",
    ) -> PKMetrics:
        """
        Compute standard PK metrics from a plasma concentration-time profile.

        Args:
            time_h: Time points (hours)
            plasma_conc: Plasma concentration at each time point (mg/L)
            dose_mg: Administered dose (mg)
            route: "iv" or "oral"

        Returns:
            Frozen PKMetrics
        """
        if len(time_h) < 3 or len(plasma_conc) < 3:
            return PKMetrics(0, 0, 0, 0, 0, 0, 0, 0)

        # Cmax and tmax
        idx_max = int(np.argmax(plasma_conc))
        cmax = float(plasma_conc[idx_max])
        tmax = float(time_h[idx_max])

        # AUC by linear-log trapezoidal rule
        auc_last = PKCalculator._auc_trapz(time_h, plasma_conc)

        # Terminal half-life from log-linear regression on terminal phase
        half_life, lambda_z = PKCalculator._terminal_half_life(time_h, plasma_conc)

        # AUC extrapolated to infinity
        c_last = float(plasma_conc[-1])
        if lambda_z > 0 and c_last > 0:
            auc_extrap = c_last / lambda_z
            auc_inf = auc_last + auc_extrap
        else:
            auc_inf = auc_last

        # Apparent clearance: CL/F = Dose / AUC
        cl_apparent = dose_mg / auc_inf if auc_inf > 0 else 0.0

        # MRT (mean residence time)
        aumc = PKCalculator._aumc_trapz(time_h, plasma_conc)
        mrt = aumc / auc_last if auc_last > 0 else 0.0

        # For IV: MRT = AUMC/AUC. For oral: MRT = AUMC/AUC - 1/ka (absorption correction)
        # We report the raw MRT here; oral correction is done by orchestrator.

        # Apparent Vss: Vss/F = CL/F × MRT
        vss_apparent = cl_apparent * mrt if mrt > 0 else 0.0

        return PKMetrics(
            cmax_mg_per_L=round(cmax, 6),
            tmax_h=round(tmax, 4),
            auc_0_inf_mg_h_per_L=round(auc_inf, 4),
            auc_0_last_mg_h_per_L=round(auc_last, 4),
            half_life_h=round(half_life, 4),
            cl_apparent_L_per_h=round(cl_apparent, 4),
            vss_apparent_L=round(vss_apparent, 2),
            mrt_h=round(mrt, 4),
        )

    @staticmethod
    def _auc_trapz(time: np.ndarray, conc: np.ndarray) -> float:
        """
        Linear-log trapezoidal AUC.

        Linear rule for ascending portions (absorption), log rule for
        descending (elimination). More accurate than pure linear trapezoid.
        """
        auc = 0.0
        for i in range(1, len(time)):
            dt = time[i] - time[i - 1]
            c1, c2 = conc[i - 1], conc[i]
            if dt <= 0:
                continue
            if c1 <= 0 or c2 <= 0 or abs(c1 - c2) < 1e-15:
                # Linear trapezoid
                auc += 0.5 * (c1 + c2) * dt
            elif c2 < c1:
                # Log trapezoid (descending)
                auc += (c1 - c2) * dt / math.log(c1 / c2)
            else:
                # Linear trapezoid (ascending)
                auc += 0.5 * (c1 + c2) * dt
        return auc

    @staticmethod
    def _aumc_trapz(time: np.ndarray, conc: np.ndarray) -> float:
        """Area under the first moment curve (AUMC) via trapezoidal rule."""
        moment = time * conc
        aumc = 0.0
        for i in range(1, len(time)):
            dt = time[i] - time[i - 1]
            if dt <= 0:
                continue
            aumc += 0.5 * (moment[i - 1] + moment[i]) * dt
        return aumc

    @staticmethod
    def _terminal_half_life(
        time: np.ndarray,
        conc: np.ndarray,
        min_points: int = 3,
    ) -> Tuple[float, float]:
        """
        Estimate terminal elimination half-life via log-linear regression
        on the terminal phase (post-Cmax, declining portion).

        Returns:
            (half_life_h, lambda_z)
        """
        # Find terminal phase: start from Cmax, use descending points
        idx_max = int(np.argmax(conc))

        # Collect log-linear terminal points (post-Cmax, positive concentrations)
        term_times = []
        term_log_conc = []
        for i in range(idx_max + 1, len(conc)):
            if conc[i] > 0:
                term_times.append(time[i])
                term_log_conc.append(math.log(conc[i]))

        if len(term_times) < min_points:
            # Not enough points — estimate from last points
            return 4.0, 0.693 / 4.0  # Default 4h

        # Use last 2/3 of terminal phase for best fit
        n_use = max(min_points, len(term_times) * 2 // 3)
        t_arr = np.array(term_times[-n_use:])
        lnc_arr = np.array(term_log_conc[-n_use:])

        # Linear regression: ln(C) = -λz × t + intercept
        if len(t_arr) < 2:
            return 4.0, 0.693 / 4.0

        n = len(t_arr)
        sum_t = np.sum(t_arr)
        sum_lnc = np.sum(lnc_arr)
        sum_t2 = np.sum(t_arr ** 2)
        sum_t_lnc = np.sum(t_arr * lnc_arr)

        denom = n * sum_t2 - sum_t ** 2
        if abs(denom) < 1e-15:
            return 4.0, 0.693 / 4.0

        slope = (n * sum_t_lnc - sum_t * sum_lnc) / denom
        lambda_z = -slope  # Should be positive for elimination

        if lambda_z <= 0:
            return 4.0, 0.693 / 4.0

        half_life = 0.693 / lambda_z

        # Sanity: half-life should be between 0.1h and 500h for most drugs
        half_life = max(0.1, min(half_life, 500.0))

        return half_life, lambda_z
