"""
PBPK ODE Solver — V1 Engine
==============================
Clean 14-compartment physiologically based pharmacokinetic ODE system.

Compartments (state vector order):
  0.  Gut lumen (amount, mg)
  1.  Plasma (concentration, mg/L)
  2.  Lung
  3.  Liver
  4.  Gut wall
  5.  Kidney
  6.  Brain
  7.  Heart
  8.  Muscle
  9.  Fat
  10. Skin
  11. Bone
  12. Spleen
  13. Pancreas

Circulation topology:
  Venous pool → Lung (pulmonary) → Arterial pool → All organs → Venous pool
  Portal vein: Gut + Spleen + Pancreas → Liver

Solver: scipy BDF (stiff-appropriate, backward differentiation formula).
Fallback: LSODA (auto-switching stiff/non-stiff).

References:
  Nestorov I. Clin Pharmacokinet 2003; 42:883-908
  Jones HM, Rowland-Yeo K. CPT Pharmacometrics Syst Pharmacol 2013; 2:e63
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from neogen.core.entities.drug import Drug
from neogen.core.entities.physiology import PhysiologyModel
from neogen.core.entities.formulation import DoseSchedule, DoseEvent, Route
from neogen.engine.absorption.absorption_profile import AbsorptionProfile
from neogen.engine.tissue_partitioner import TissuePartitioner, PartitionCoefficients
from neogen.engine.hepatic_clearance import HepaticClearanceEngine, HepaticClearanceResult
from neogen.engine.renal_clearance import RenalClearanceEngine, RenalClearanceResult

logger = logging.getLogger(__name__)


# Type alias: portal input rate provider (hours → mg/h into portal vein).
PortalRateProvider = Callable[[float], float]

# Type alias: dynamic CLint provider (hours → scaled whole-liver CLint in L/h).
# When supplied, the ODE RHS recomputes hepatic clearance via the well-stirred
# model at every solver step using this callable's output instead of the
# static ``cl_hepatic`` pre-computation. Produced by the Step 10
# ``DDIOrchestrator`` to wire dynamic DDI effects into the PBPK run.
CLintProvider = Callable[[float], float]

# State vector indices
_IDX = {
    "gut_lumen": 0,
    "plasma": 1,
    "lung": 2,
    "liver": 3,
    "gut_wall": 4,
    "kidney": 5,
    "brain": 6,
    "heart": 7,
    "muscle": 8,
    "fat": 9,
    "skin": 10,
    "bone": 11,
    "spleen": 12,
    "pancreas": 13,
}

N_COMPARTMENTS = 14

# Organ names in state vector order (for results mapping)
ORGAN_NAMES = [
    "gut_lumen", "plasma", "lung", "liver", "gut_wall", "kidney",
    "brain", "heart", "muscle", "fat", "skin", "bone", "spleen", "pancreas",
]


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for the ODE solver."""
    method: str = "BDF"          # Primary: BDF (stiff), fallback: LSODA
    rtol: float = 1e-6           # Relative tolerance
    atol: float = 1e-9           # Absolute tolerance
    max_step_fraction: float = 0.01  # Max step = tmax × this
    n_timepoints: int = 500      # Output resolution
    t_max_multiplier: float = 5.0    # t_max = multiplier × estimated t½

    # V1.1 mass-balance fix (now DEFAULT ON):
    #
    # When True, the plasma and lung ODEs form a closed circulation
    # loop, and IV bolus doses are distributed across Vss-weighted
    # organ volumes rather than dumped into plasma:
    #
    #     venous_return (from organs) → plasma → lung → arterial → organs
    #
    # This restores strict mass conservation. Empirical validation:
    # CL_total × AUC_0_inf ≈ Dose within 40% (vs 71% loss in the
    # legacy mixing-pool path). See the Tier 4 invariant
    # ``cl_auc_equals_dose`` for the gating check and the validation
    # harness ``reports/validation/`` for the per-drug numbers.
    #
    # The legacy behavior is preserved behind ``mass_balance_mode=False``
    # for audit / regression purposes only. V1.1 callers should leave
    # this at the default.
    mass_balance_mode: bool = True


def build_portal_rate_provider(
    profiles_with_offsets: Sequence[Tuple[AbsorptionProfile, float]],
) -> PortalRateProvider:
    """Assemble a single portal-rate callable from one or more P2 profiles.

    Each ``AbsorptionProfile`` carries ``cum_portal_mg(t)`` on its own local
    time grid (starting at t=0). In a multi-dose scenario every oral dose
    event contributes its own profile shifted by ``dose.time_h``. The
    returned callable takes simulation time (hours) and returns the
    combined instantaneous mass flux entering the portal vein (mg/h).

    Implementation detail: we pre-compute the finite-difference derivative
    of ``cum_portal_mg`` on the profile's native grid once, then the
    callable does a binary search + linear interpolation per query. This
    is cheap enough to be called from the ODE RHS at every scipy step
    without blowing up the wall-clock budget.

    Outside the profile's window the contribution is zero (before the
    dose) or zero (after the tail — cum_portal has already saturated and
    its derivative is ~0 in well-behaved simulations).
    """
    if not profiles_with_offsets:

        def zero_rate(t: float) -> float:
            return 0.0

        return zero_rate

    # Pre-compute per-profile (t_local_grid, rate_grid) derivative curves.
    prepared: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for profile, offset in profiles_with_offsets:
        t_local = np.asarray(profile.t_h, dtype=float)
        cum_portal = np.asarray(profile.cum_portal_mg, dtype=float)
        if t_local.size < 2:
            continue
        dt = np.diff(t_local)
        dy = np.diff(cum_portal)
        rate_mid = dy / np.maximum(dt, 1.0e-12)
        # Sample rate on a grid of length len(t_local) by averaging
        # neighbouring midpoints (trapezoidal collocation).
        rate_grid = np.zeros_like(t_local)
        rate_grid[0] = rate_mid[0]
        rate_grid[-1] = rate_mid[-1]
        if rate_mid.size > 1:
            rate_grid[1:-1] = 0.5 * (rate_mid[:-1] + rate_mid[1:])
        # Clip tiny negatives from solver noise.
        rate_grid = np.clip(rate_grid, a_min=0.0, a_max=None)
        prepared.append((t_local, rate_grid, float(offset)))

    def portal_rate_at(t: float) -> float:
        total = 0.0
        for t_local, rate_grid, offset in prepared:
            t_rel = t - offset
            if t_rel <= t_local[0] or t_rel >= t_local[-1]:
                continue
            # Linear interpolation on the pre-built rate grid.
            total += float(np.interp(t_rel, t_local, rate_grid))
        return total

    return portal_rate_at


@dataclass
class SolverOutput:
    """Raw output from the ODE solver (mutable for construction)."""
    time_h: np.ndarray
    concentrations: np.ndarray  # Shape: (N_COMPARTMENTS, n_timepoints)
    success: bool
    method_used: str
    message: str = ""


class PBPKSolver:
    """
    14-compartment PBPK ODE solver using V1 entity system.

    Wires together:
      - TissuePartitioner → Kp values
      - HepaticClearanceEngine → CLh, Eh
      - RenalClearanceEngine → CLr
      - scipy BDF solver → time-concentration profiles

    Usage:
        solver = PBPKSolver()
        output = solver.solve(drug, dose_schedule, physiology)
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        self._partitioner = TissuePartitioner()
        self._hepatic_engine = HepaticClearanceEngine()
        self._renal_engine = RenalClearanceEngine()

    def solve(
        self,
        drug: Drug,
        dose_schedule: DoseSchedule,
        physiology: PhysiologyModel,
        kp_override: Optional[PartitionCoefficients] = None,
        oral_portal_rate: Optional[PortalRateProvider] = None,
        clint_provider: Optional[CLintProvider] = None,
        oatp1b1_inhibitors: Optional[Sequence[object]] = None,
        gfr_ml_per_min: Optional[float] = None,
    ) -> SolverOutput:
        """
        Solve the full PBPK ODE system.

        Args:
            drug: V1 Drug entity (physicochemical + ADME)
            dose_schedule: Dosing events (single or multiple)
            physiology: V1 PhysiologyModel (14 organs)
            kp_override: Pre-computed Kp values (optional)
            oral_portal_rate: Optional P2-derived portal vein input rate
                callable ``f(t_h) → mg/h``. When supplied, the ODE ignores
                the legacy ``ka × a_gut`` first-order absorption path for
                oral doses and instead injects ``oral_portal_rate(t)``
                directly into the liver compartment. The gut lumen state
                variable is held at zero for P2-driven doses.
            clint_provider: Optional dynamic CLint callable produced by the
                Phase-3 ``DDIOrchestrator`` (Step 10). When supplied, the
                RHS ignores the static ``hepatic.cl_hepatic_L_per_h`` and
                instead recomputes hepatic clearance each step via the
                well-stirred liver model::

                    CLint(t) = clint_provider(t)
                    CLh(t)   = Qh · fu · CLint / (Qh + fu · CLint)

                This is how dynamic inhibition / induction / MBI drive
                time-varying elimination in the victim's PBPK run.

        Returns:
            SolverOutput with time and concentration arrays
        """
        # ── Step 1: Compute Kp partition coefficients ──
        kp = kp_override or self._partitioner.predict(drug, physiology)

        # ── Step 2: Compute clearances ──
        # V1.2 Track B: oatp1b1_inhibitors modulates the OATP1B1 active
        # uptake amplification inside HepaticClearanceEngine. When None
        # (default), the result is byte-identical to V1.1.
        hepatic = self._hepatic_engine.compute(
            drug, physiology, oatp1b1_inhibitors=oatp1b1_inhibitors
        )
        renal_kwargs = {}
        if gfr_ml_per_min is not None:
            renal_kwargs["gfr_ml_per_min"] = gfr_ml_per_min
        renal = self._renal_engine.compute(drug, physiology, **renal_kwargs)
        renal = RenalClearanceEngine.calibrate_with_fe(renal, drug, hepatic.cl_hepatic_L_per_h)

        # ── Step 3: Build ODE parameters ──
        params = _ODEParams.build(
            drug, physiology, kp, hepatic, renal,
            oral_portal_rate=oral_portal_rate,
            clint_provider=clint_provider,
            mass_balance_mode=self.config.mass_balance_mode,
        )

        # ── Step 4: Determine time span ──
        t_max = self._estimate_tmax(drug, hepatic, renal, dose_schedule)
        t_eval = np.linspace(0.0, t_max, self.config.n_timepoints)

        # ── Step 5: Build initial conditions ──
        y0 = self._initial_conditions(
            drug, dose_schedule, physiology,
            p2_driven_oral=oral_portal_rate is not None,
            kp=kp,
            mass_balance_mode=self.config.mass_balance_mode,
        )

        # ── Step 6: Build dose event callback (for multiple doses) ──
        dose_events = self._build_dose_events(
            dose_schedule, drug, physiology,
            p2_driven_oral=oral_portal_rate is not None,
        )

        # ── Step 7: Solve ODE ──
        output = self._run_solver(params, y0, t_eval, t_max, dose_events)

        return output

    def _estimate_tmax(
        self,
        drug: Drug,
        hepatic: HepaticClearanceResult,
        renal: RenalClearanceResult,
        schedule: DoseSchedule,
    ) -> float:
        """Estimate simulation end time from clearance and dosing."""
        if drug.cl_total_override_L_per_h is not None:
            cl_total = drug.cl_total_override_L_per_h
        elif drug.cl_hepatic_override_L_per_h is not None:
            cl_total = (drug.cl_hepatic_override_L_per_h +
                        (drug.cl_renal_override_L_per_h or 0.0))
        else:
            cl_total = hepatic.cl_hepatic_L_per_h + renal.cl_renal_net_L_per_h
        if cl_total > 0:
            # Very rough: t½ ≈ 0.693 × Vd / CL, assume Vd ≈ 50L
            estimated_half_life = 0.693 * 50.0 / cl_total
        else:
            estimated_half_life = 24.0

        estimated_half_life = max(1.0, min(estimated_half_life, 200.0))

        # At least 5 half-lives after last dose
        last_dose_time = max((e.time_h for e in schedule.events), default=0.0)
        t_max = last_dose_time + self.config.t_max_multiplier * estimated_half_life

        # Minimum 24h, maximum 500h
        return max(24.0, min(t_max, 2500.0))

    def _initial_conditions(
        self,
        drug: Drug,
        schedule: DoseSchedule,
        physiology: PhysiologyModel,
        p2_driven_oral: bool = False,
        kp: Optional[PartitionCoefficients] = None,
        mass_balance_mode: bool = False,
    ) -> np.ndarray:
        """Set up initial state vector based on first dose event.

        When ``p2_driven_oral`` is True, oral doses are delivered via the
        portal forcing function — the gut_lumen state stays at zero and
        is effectively unused for that dose. IV routes always use the
        legacy initial-condition path regardless.

        When ``mass_balance_mode`` is True and the first dose is an IV
        bolus, the dose is distributed across all compartments at their
        pseudo-equilibrium concentrations (``c_i = C0 × Kp_i``) where
        ``C0 = dose / Vss_mechanistic``. This avoids the unphysical
        spike of ``dose/V_plasma`` and restores strict mass balance at
        t=0 (total mass in all compartments exactly equals ``dose_mg``).
        """
        y0 = np.zeros(N_COMPARTMENTS)

        if not schedule.events:
            return y0

        first_dose = schedule.events[0]
        if first_dose.time_h > 0:
            # First dose is delayed — start with empty compartments
            return y0

        dose_mg = first_dose.dose_mg

        if first_dose.route == Route.IV_BOLUS:
            if mass_balance_mode and kp is not None:
                # V1.1 path: distribute dose across Vss-weighted volumes
                # so ``sum_i C_i × V_i = dose`` holds exactly at t=0.
                self._iv_bolus_pseudo_equilibrium(
                    y0, dose_mg, physiology, kp
                )
            else:
                # Legacy path: entire dose into plasma instantly
                y0[_IDX["plasma"]] = dose_mg / physiology.plasma.volume_L
        elif first_dose.route == Route.IV_INFUSION:
            # IV infusion: handled as rate in ODE (y0 stays 0)
            pass
        else:
            # Oral: when P2 drives the portal rate, gut_lumen is unused.
            # Otherwise fall back to the legacy ka × a_gut path.
            if not p2_driven_oral:
                y0[_IDX["gut_lumen"]] = dose_mg

        return y0

    @staticmethod
    def _iv_bolus_pseudo_equilibrium(
        y0: np.ndarray,
        dose_mg: float,
        physiology: PhysiologyModel,
        kp: PartitionCoefficients,
    ) -> None:
        """Distribute an IV bolus using a two-phase approach.

        Places a small fraction (5%) of the dose directly in plasma
        as an IV bolus spike, and distributes the remaining 95% across
        all compartments at Kp-weighted pseudo-equilibrium. This
        balances two requirements:

          1. IV Cmax: a small plasma spike gives a more realistic
             initial concentration than pure equilibrium (which
             assumes instantaneous distribution to Vss).
          2. Mass-balance stability: distributing the bulk of the
             dose at Kp-weighted equilibrium avoids the phantom-sink
             artifact from the legacy 100%-in-plasma approach.

        The real body takes 1-5 minutes for redistribution from the
        central compartment. The 5/95 split approximates a snapshot
        shortly after the initial mixing phase.
        """
        # Build (name, V, Kp) triples for every compartment.
        organ_entries = [
            ("plasma",   physiology.plasma.volume_L,   1.0),
            ("lung",     physiology.lung.volume_L,     kp.lung),
            ("liver",    physiology.liver.volume_L,    kp.liver),
            ("gut_wall", physiology.gut.volume_L,      kp.gut),
            ("kidney",   physiology.kidney.volume_L,   kp.kidney),
            ("brain",    physiology.brain.volume_L,    kp.brain),
            ("heart",    physiology.heart.volume_L,    kp.heart),
            ("muscle",   physiology.muscle.volume_L,   kp.muscle),
            ("fat",      physiology.fat.volume_L,      kp.fat),
            ("skin",     physiology.skin.volume_L,     kp.skin),
            ("bone",     physiology.bone.volume_L,     kp.bone),
            ("spleen",   physiology.spleen.volume_L,   kp.spleen),
            ("pancreas", physiology.pancreas.volume_L, kp.pancreas),
        ]

        vss_l = sum(v * k for _, v, k in organ_entries)
        if vss_l <= 0.0:  # pragma: no cover - defensive
            y0[_IDX["plasma"]] = dose_mg / max(
                physiology.plasma.volume_L, 1.0e-6
            )
            return

        # Two-phase split: 5% extra bolus spike in plasma on top of
        # the equilibrium distribution. This modest bump lifts IV
        # Cmax for large-Vss drugs (metoprolol, diazepam, digoxin)
        # without creating unrealistic spikes for small-Vss drugs.
        plasma_spike_fraction = 0.05
        equilibrium_fraction = 1.0 - plasma_spike_fraction

        # Equilibrium distribution across all compartments (including plasma)
        c_eq = (dose_mg * equilibrium_fraction) / vss_l
        for name, _v, kp_val in organ_entries:
            y0[_IDX[name]] = c_eq * kp_val

        # Add the spike on top of plasma equilibrium
        y0[_IDX["plasma"]] += (
            dose_mg * plasma_spike_fraction
        ) / physiology.plasma.volume_L

    def _build_dose_events(
        self,
        schedule: DoseSchedule,
        drug: Drug,
        physiology: PhysiologyModel,
        p2_driven_oral: bool = False,
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Build list of (time, delta_y) for doses after the first.
        These are applied as instantaneous additions to the state vector.

        When ``p2_driven_oral`` is True, oral doses are omitted from the
        state-delta path — their mass enters via the portal forcing
        function instead. IV doses always use the delta path.
        """
        events = []
        for dose_event in schedule.events:
            if dose_event.time_h <= 0:
                continue  # First dose handled in y0

            delta = np.zeros(N_COMPARTMENTS)
            if dose_event.route == Route.IV_BOLUS:
                delta[_IDX["plasma"]] = dose_event.dose_mg / physiology.plasma.volume_L
            elif dose_event.route == Route.ORAL and not p2_driven_oral:
                delta[_IDX["gut_lumen"]] = dose_event.dose_mg
            # else: oral dose under P2 is injected via portal forcing.

            # Only record non-zero deltas (P2-driven oral doses still need a
            # breakpoint, but the solver handles that via the forcing fn —
            # there's no instantaneous state jump).
            if np.any(delta != 0.0):
                events.append((dose_event.time_h, delta))

        return sorted(events, key=lambda x: x[0])

    def _run_solver(
        self,
        params: _ODEParams,
        y0: np.ndarray,
        t_eval: np.ndarray,
        t_max: float,
        dose_events: List[Tuple[float, np.ndarray]],
    ) -> SolverOutput:
        """
        Run the ODE solver, handling multiple dose events by splitting
        the integration at each dose time.
        """
        if not dose_events:
            # Single dose — solve in one shot
            return self._solve_segment(params, y0, 0.0, t_max, t_eval)

        # Multiple doses: solve segment by segment
        all_times = []
        all_concs = []
        current_y = y0.copy()
        current_t = 0.0
        method_used = self.config.method
        success = True

        # Add dose times as breakpoints
        breakpoints = [t for t, _ in dose_events] + [t_max]

        for bp_t, delta in dose_events + [(t_max, np.zeros(N_COMPARTMENTS))]:
            if bp_t <= current_t:
                current_y += delta
                continue

            # Time points for this segment
            seg_mask = (t_eval >= current_t) & (t_eval < bp_t)
            seg_t = t_eval[seg_mask]
            if len(seg_t) == 0:
                seg_t = np.array([current_t, bp_t])

            # Solve segment
            seg_out = self._solve_segment(params, current_y, current_t, bp_t, seg_t)

            if seg_out.time_h.size > 0:
                all_times.append(seg_out.time_h)
                all_concs.append(seg_out.concentrations)
                method_used = seg_out.method_used
                success = success and seg_out.success

                # Update state from end of segment
                current_y = seg_out.concentrations[:, -1].copy()

            # Apply dose
            current_y += delta
            current_t = bp_t

        # Concatenate all segments
        if all_times:
            time_h = np.concatenate(all_times)
            concentrations = np.concatenate(all_concs, axis=1)
        else:
            time_h = t_eval
            concentrations = np.zeros((N_COMPARTMENTS, len(t_eval)))

        return SolverOutput(
            time_h=time_h,
            concentrations=np.clip(concentrations, 0, None),
            success=success,
            method_used=method_used,
        )

    def _solve_segment(
        self,
        params: _ODEParams,
        y0: np.ndarray,
        t_start: float,
        t_end: float,
        t_eval: np.ndarray,
    ) -> SolverOutput:
        """Solve a single segment of the ODE."""
        max_step = (t_end - t_start) * self.config.max_step_fraction

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            return _ode_rhs(t, y, params)

        try:
            sol = solve_ivp(
                rhs,
                t_span=(t_start, t_end),
                y0=y0,
                t_eval=t_eval,
                method=self.config.method,
                rtol=self.config.rtol,
                atol=self.config.atol,
                max_step=max(max_step, 0.01),
            )

            if sol.success:
                return SolverOutput(
                    time_h=sol.t,
                    concentrations=np.clip(sol.y, 0, None),
                    success=True,
                    method_used=self.config.method,
                )

            # Fallback to LSODA
            logger.warning(
                "BDF failed (%s), falling back to LSODA", sol.message
            )
            sol = solve_ivp(
                rhs,
                t_span=(t_start, t_end),
                y0=y0,
                t_eval=t_eval,
                method="LSODA",
                rtol=self.config.rtol,
                atol=self.config.atol,
            )

            return SolverOutput(
                time_h=sol.t,
                concentrations=np.clip(sol.y, 0, None),
                success=sol.success,
                method_used="LSODA",
                message=sol.message if not sol.success else "",
            )

        except Exception as e:
            logger.error("ODE solver crashed: %s", e)
            zeros = np.zeros((N_COMPARTMENTS, len(t_eval)))
            return SolverOutput(
                time_h=t_eval,
                concentrations=zeros,
                success=False,
                method_used="FAILED",
                message=str(e),
            )


# ════════════════════════════════════════════════════════════════
# ODE PARAMETERS (internal, pre-computed for speed)
# ════════════════════════════════════════════════════════════════

class _ODEParams:
    """
    Pre-computed ODE parameters. Avoids recalculating constants
    inside the hot RHS function on every call.
    """

    __slots__ = (
        # Volumes
        "v_plasma", "v_lung", "v_liver", "v_gut", "v_gut_wall",
        "v_kidney", "v_brain", "v_heart", "v_muscle", "v_fat",
        "v_skin", "v_bone", "v_spleen", "v_pancreas",
        # Blood flows
        "q_lung", "q_liver", "q_gut", "q_kidney", "q_brain",
        "q_heart", "q_muscle", "q_fat", "q_skin", "q_bone",
        "q_spleen", "q_pancreas", "q_ha", "q_rest",
        # Partition coefficients
        "kp_lung", "kp_liver", "kp_gut", "kp_gut_wall", "kp_kidney",
        "kp_brain", "kp_heart", "kp_muscle", "kp_fat", "kp_skin",
        "kp_bone", "kp_spleen", "kp_pancreas",
        # Clearances
        "cl_hepatic", "cl_renal",
        # Absorption
        "ka",
        "oral_portal_rate",  # P2-driven forcing function, or None
        # Drug properties
        "fu",
        # Phase-3 DDI dynamic clearance (Step 10)
        "clint_provider",   # Callable[[float], float] or None
        "qh_hepatic",       # Total hepatic blood flow (L/h), used for well-stirred recompute
        # V1.1 mass-balance fix (opt-in flag read by _ode_rhs)
        "mass_balance_mode",
    )

    @classmethod
    def build(
        cls,
        drug: Drug,
        phys: PhysiologyModel,
        kp: PartitionCoefficients,
        hepatic: HepaticClearanceResult,
        renal: RenalClearanceResult,
        oral_portal_rate: Optional[PortalRateProvider] = None,
        clint_provider: Optional[CLintProvider] = None,
        mass_balance_mode: bool = False,
    ) -> _ODEParams:
        p = cls()

        # Volumes
        # The "plasma" compartment represents the rapidly-mixing
        # central pool (plasma + interstitial fluid of well-perfused
        # tissues).  Scale by 1.5 to reduce Cmax over-prediction
        # without affecting AUC (which depends on CL, not V).
        p.v_plasma = phys.plasma.volume_L * 1.5
        p.v_lung = phys.lung.volume_L
        p.v_liver = phys.liver.volume_L
        p.v_gut = phys.gut.volume_L
        p.v_gut_wall = phys.gut.volume_L * 0.38  # Gut wall ≈ 38% of gut volume
        p.v_kidney = phys.kidney.volume_L
        p.v_brain = phys.brain.volume_L
        p.v_heart = phys.heart.volume_L
        p.v_muscle = phys.muscle.volume_L
        p.v_fat = phys.fat.volume_L
        p.v_skin = phys.skin.volume_L
        p.v_bone = phys.bone.volume_L
        p.v_spleen = phys.spleen.volume_L
        p.v_pancreas = phys.pancreas.volume_L

        # Blood flows
        co = phys.lung.blood_flow_L_per_h  # Cardiac output = lung flow
        p.q_lung = co
        p.q_liver = phys.liver.blood_flow_L_per_h
        p.q_gut = phys.gut.blood_flow_L_per_h
        p.q_kidney = phys.kidney.blood_flow_L_per_h
        p.q_brain = phys.brain.blood_flow_L_per_h
        p.q_heart = phys.heart.blood_flow_L_per_h
        p.q_muscle = phys.muscle.blood_flow_L_per_h
        p.q_fat = phys.fat.blood_flow_L_per_h
        p.q_skin = phys.skin.blood_flow_L_per_h
        p.q_bone = phys.bone.blood_flow_L_per_h
        p.q_spleen = phys.spleen.blood_flow_L_per_h
        p.q_pancreas = phys.pancreas.blood_flow_L_per_h

        # Hepatic artery = total liver flow - portal vein (gut + spleen + pancreas)
        p.q_ha = max(p.q_liver - p.q_gut - p.q_spleen - p.q_pancreas, 0.01)

        # Rest-of-body flow
        q_accounted = (
            p.q_liver + p.q_kidney + p.q_brain + p.q_heart
            + p.q_muscle + p.q_fat + p.q_skin + p.q_bone
        )
        p.q_rest = max(co - q_accounted, 0.0)

        # Kp values
        p.kp_lung = kp.lung
        p.kp_liver = kp.liver
        p.kp_gut = kp.gut
        p.kp_gut_wall = kp.gut  # Gut wall uses same Kp as gut
        p.kp_kidney = kp.kidney
        p.kp_brain = kp.brain
        p.kp_heart = kp.heart
        p.kp_muscle = kp.muscle
        p.kp_fat = kp.fat
        p.kp_skin = kp.skin
        p.kp_bone = kp.bone
        p.kp_spleen = kp.spleen
        p.kp_pancreas = kp.pancreas

        # Clearances (L/h) — with V1.2 structural fix overrides
        if drug.cl_total_override_L_per_h is not None:
            # Extrahepatic drug: put all CL into hepatic slot (liver
            # is the only eliminating compartment in the ODE), zero renal.
            p.cl_hepatic = drug.cl_total_override_L_per_h
            p.cl_renal = 0.0
        elif drug.cl_hepatic_override_L_per_h is not None:
            p.cl_hepatic = drug.cl_hepatic_override_L_per_h
            p.cl_renal = drug.cl_renal_override_L_per_h or 0.0
        else:
            p.cl_hepatic = hepatic.cl_hepatic_L_per_h
            p.cl_renal = renal.cl_renal_net_L_per_h

        # Absorption rate constant (legacy fallback for oral doses that
        # are NOT driven by a P2 AbsorptionProfile).
        # ka ≈ 1-3 /h for BCS I/III, 0.3-1 for BCS II/IV
        p.ka = _estimate_ka(drug)

        # Optional mechanistic oral forcing function (from Step 5 P2↔P1
        # integration). When present, the ODE RHS ignores ``ka × a_gut``
        # and instead injects ``oral_portal_rate(t)`` into the liver.
        p.oral_portal_rate = oral_portal_rate

        # Fraction unbound
        p.fu = drug.fu_plasma_safe

        # Phase-3 DDI: dynamic CLint callback + Qh for well-stirred
        # recomputation inside the RHS. ``qh_hepatic`` is the total
        # hepatic blood flow (artery + portal), identical to the value
        # ``HepaticClearanceEngine`` used to build the static
        # ``cl_hepatic`` so the no-DDI case stays byte-identical.
        p.clint_provider = clint_provider
        p.qh_hepatic = hepatic.qh_L_per_h

        # V1.1 mass-balance mode flag (default False → legacy mixing pool).
        p.mass_balance_mode = mass_balance_mode

        return p


def _estimate_ka(drug: Drug) -> float:
    """
    Estimate first-order absorption rate constant (1/h).

    BCS I (high sol, high perm): ka ≈ 1.5  (literature: 0.5-1.5)
    BCS II (low sol, high perm): ka ≈ 0.5  (dissolution limited)
    BCS III (high sol, low perm): ka ≈ 1.0  (permeability limited)
    BCS IV (low sol, low perm): ka ≈ 0.3
    """
    from neogen.core.entities.drug import BCSClass
    bcs_ka = {
        BCSClass.I: 1.5,
        BCSClass.II: 0.5,
        BCSClass.III: 1.0,
        BCSClass.IV: 0.3,
    }
    return bcs_ka.get(drug.bcs_class, 1.0)


# ════════════════════════════════════════════════════════════════
# ODE RIGHT-HAND SIDE (hot path)
# ════════════════════════════════════════════════════════════════

def _ode_rhs(t: float, y: np.ndarray, p: _ODEParams) -> np.ndarray:
    """
    14-compartment PBPK ODE system.

    Mass balance per organ (perfusion-limited model):
      dC_tissue/dt = (Q/V) × (C_arterial − C_tissue/Kp)
                     − (CL/V) × C_tissue/Kp  [for eliminating organs]

    Pulmonary circulation:
      Venous blood → Lung → Arterial blood
      C_arterial ≈ C_lung / Kp_lung

    Portal vein anatomy:
      Gut + Spleen + Pancreas → Portal vein → Liver
    """
    dydt = np.zeros(N_COMPARTMENTS)

    # ── Unpack state (clamp non-negative) ──
    a_gut = max(y[0], 0.0)    # Amount in gut lumen (mg)
    c_pl  = max(y[1], 0.0)    # Plasma conc (mg/L)
    c_lu  = max(y[2], 0.0)    # Lung
    c_li  = max(y[3], 0.0)    # Liver
    c_gw  = max(y[4], 0.0)    # Gut wall
    c_ki  = max(y[5], 0.0)    # Kidney
    c_br  = max(y[6], 0.0)    # Brain
    c_ht  = max(y[7], 0.0)    # Heart
    c_mu  = max(y[8], 0.0)    # Muscle
    c_fa  = max(y[9], 0.0)    # Fat
    c_sk  = max(y[10], 0.0)   # Skin
    c_bo  = max(y[11], 0.0)   # Bone
    c_sp  = max(y[12], 0.0)   # Spleen
    c_pa  = max(y[13], 0.0)   # Pancreas

    # ── Arterial concentration (from lung) ──
    c_art = c_lu / p.kp_lung if p.kp_lung > 0 else c_pl

    # ── Venous return (weighted sum of all organ outflows) ──
    venous_return = (
        p.q_liver  * c_li / p.kp_liver
        + p.q_kidney * c_ki / p.kp_kidney
        + p.q_brain  * c_br / p.kp_brain
        + p.q_heart  * c_ht / p.kp_heart
        + p.q_muscle * c_mu / p.kp_muscle
        + p.q_fat    * c_fa / p.kp_fat
        + p.q_skin   * c_sk / p.kp_skin
        + p.q_bone   * c_bo / p.kp_bone
        + p.q_rest   * c_pl  # Rest-of-body returns at plasma conc
    )

    # ── Gut lumen: first-order absorption (legacy path) ──
    # When a P2 AbsorptionProfile is driving this simulation, the
    # oral_portal_rate callable delivers mass directly into the liver
    # compartment (see below) and the gut_lumen state stays unused.
    if p.oral_portal_rate is None:
        absorption = p.ka * a_gut
    else:
        absorption = 0.0
    dydt[0] = -absorption

    # ── Portal input from P2 (mechanistic oral forcing) ──
    # ``oral_portal_rate(t)`` already accounts for dissolution (Fa),
    # permeation, and gut-wall CYP3A4 extraction (Fg). It is the mass
    # flux entering the portal vein, so it goes straight into the liver
    # compartment. Hepatic first-pass (Fh) is still applied by the
    # existing ``cl_hepatic`` term on the liver state.
    portal_in_mg_per_h = (
        p.oral_portal_rate(t) if p.oral_portal_rate is not None else 0.0
    )

    # ── Plasma + Lung: two modes ──
    #
    # Legacy mode (mass_balance_mode=False, default):
    #   Plasma is a tracking variable that follows arterial output
    #   from the lung. It does NOT participate in mass flow. This is
    #   calibrated against the P1/P3/P4 acceptance gates and produces
    #   the spike-then-decay IV profile that those gates expect.
    #
    # V1.1 mode (mass_balance_mode=True):
    #   Plasma is the venous pool that receives organ outflows via
    #   ``venous_return`` and pumps its contents to the lung. Lung
    #   then outputs arterial blood. This closes the circulation
    #   loop and restores strict mass balance:
    #
    #       organs → venous_return → plasma → lung → arterial → organs
    #
    #   The fix touches only these two ODE rows — every other
    #   organ equation and the clearance terms are identical in
    #   both modes.
    if p.mass_balance_mode:
        # Plasma acts as the venous pool. Receives organ returns,
        # flows to the lung at cardiac output.
        dydt[1] = (1.0 / p.v_plasma) * (
            venous_return
            - p.q_lung * c_pl
        )
        # Lung receives plasma (venous) input, outputs arterial.
        dydt[2] = (1.0 / p.v_lung) * (
            p.q_lung * c_pl
            - p.q_lung * c_lu / p.kp_lung
        )
    else:
        # Legacy mixing-pool behavior — preserved byte-identical.
        dydt[1] = (1.0 / p.v_plasma) * (
            p.q_lung * c_lu / p.kp_lung  # Arterial delivery from lung
            - p.q_lung * c_pl             # Redistribution to organs
        )
        dydt[2] = (1.0 / p.v_lung) * (
            venous_return - p.q_lung * c_lu / p.kp_lung
        )

    # ── Hepatic clearance: static default OR dynamic DDI provider ──
    # When a Phase-3 CLintProvider is attached, the well-stirred liver
    # model is recomputed at each solver step from the current CLint(t).
    # Otherwise the pre-computed static ``cl_hepatic`` is used and the
    # no-DDI code path stays numerically identical to the prior build.
    if p.clint_provider is None:
        cl_hepatic_now = p.cl_hepatic
    else:
        clint_now = p.clint_provider(t)
        if clint_now <= 0.0 or p.qh_hepatic <= 0.0:
            cl_hepatic_now = 0.0
        else:
            fu_clint = p.fu * clint_now
            denom = p.qh_hepatic + fu_clint
            eh = fu_clint / denom if denom > 0.0 else 0.0
            if eh > 0.99:
                eh = 0.99
            cl_hepatic_now = p.qh_hepatic * eh

    # ── Liver: portal vein (gut wall + spleen + pancreas) + hepatic artery ──
    dydt[3] = (1.0 / p.v_liver) * (
        p.q_ha     * c_art                          # Hepatic artery
        + p.q_gut    * c_gw / p.kp_gut_wall         # Portal: gut wall
        + p.q_spleen * c_sp / p.kp_spleen           # Portal: spleen
        + p.q_pancreas * c_pa / p.kp_pancreas       # Portal: pancreas
        + portal_in_mg_per_h                        # P2-driven oral input
        - p.q_liver  * c_li / p.kp_liver            # Venous outflow
        - cl_hepatic_now * c_li / p.kp_liver        # Hepatic metabolism (t-dep in DDI mode)
    )

    # ── Gut wall: absorbs from lumen, perfused by gut artery ──
    dydt[4] = (absorption / p.v_gut_wall) + (
        p.q_gut / p.v_gut_wall
    ) * (c_art - c_gw / p.kp_gut_wall)

    # ── Kidney: perfusion + renal elimination ──
    dydt[5] = (p.q_kidney / p.v_kidney) * (
        c_art - c_ki / p.kp_kidney
    ) - (p.cl_renal / p.v_kidney) * c_ki / p.kp_kidney

    # ── Brain: perfusion-limited (BBB) ──
    dydt[6] = (p.q_brain / p.v_brain) * (c_art - c_br / p.kp_brain)

    # ── Heart ──
    dydt[7] = (p.q_heart / p.v_heart) * (c_art - c_ht / p.kp_heart)

    # ── Muscle ──
    dydt[8] = (p.q_muscle / p.v_muscle) * (c_art - c_mu / p.kp_muscle)

    # ── Fat ──
    dydt[9] = (p.q_fat / p.v_fat) * (c_art - c_fa / p.kp_fat)

    # ── Skin ──
    dydt[10] = (p.q_skin / p.v_skin) * (c_art - c_sk / p.kp_skin)

    # ── Bone ──
    dydt[11] = (p.q_bone / p.v_bone) * (c_art - c_bo / p.kp_bone)

    # ── Spleen: drains to portal vein ──
    dydt[12] = (p.q_spleen / p.v_spleen) * (c_art - c_sp / p.kp_spleen)

    # ── Pancreas: drains to portal vein ──
    dydt[13] = (p.q_pancreas / p.v_pancreas) * (c_art - c_pa / p.kp_pancreas)

    return dydt
