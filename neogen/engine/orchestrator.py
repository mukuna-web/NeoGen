"""
Simulation Orchestrator — V1 Engine
=====================================
Top-level API that wires together all P1 components:

  Drug + DoseSchedule + PhysiologyModel
    → TissuePartitioner (Kp)
    → HepaticClearance (CLh, Eh, fh)
    → RenalClearance (CLr)
    → PBPKSolver (14-compartment BDF ODE)
    → PKCalculator (Cmax, AUC, t½)
    → SimulationResult (immutable output)

Usage:
    from neogen.engine.orchestrator import SimulationOrchestrator
    from neogen.engine.drug_factory import DrugFactory
    from neogen.engine.dose_scheduler import DoseScheduler
    from neogen.core.constants.reference_physiology import build_reference_male

    drug = DrugFactory.create("midazolam")
    schedule = DoseScheduler.iv_bolus("midazolam", dose_mg=5.0)
    physiology = build_reference_male()

    orchestrator = SimulationOrchestrator()
    result = orchestrator.run(drug, schedule, physiology)

    print(result.pk_metrics.cmax_mg_per_L)
    print(result.pk_metrics.half_life_h)
    print(result.pk_metrics.auc_0_inf_mg_h_per_L)
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from neogen.core.entities.drug import Drug
from neogen.core.entities.physiology import PhysiologyModel
from neogen.core.entities.formulation import DoseEvent, DoseSchedule, Route
from neogen.engine.absorption import (
    AbsorptionProfile,
    PermeabilityEngine,
)
from neogen.engine.tissue_partitioner import TissuePartitioner, PartitionCoefficients
from neogen.engine.hepatic_clearance import HepaticClearanceEngine, HepaticClearanceResult
from neogen.engine.renal_clearance import RenalClearanceEngine, RenalClearanceResult
from neogen.engine.pbpk_solver import (
    CLintProvider,
    PBPKSolver,
    SolverConfig,
    SolverOutput,
    ORGAN_NAMES,
    build_portal_rate_provider,
)
from neogen.engine.simulation_result import (
    SimulationResult,
    PKMetrics,
    PKCalculator,
    OrganTimeCourse,
    compute_pk_consistency_qc,
)

logger = logging.getLogger(__name__)


class SimulationOrchestrator:
    """
    Orchestrates a complete PBPK simulation from V1 entities.

    Stateless — each call to run() is independent.
    """

    def __init__(self, solver_config: Optional[SolverConfig] = None):
        self._partitioner = TissuePartitioner()
        self._hepatic = HepaticClearanceEngine()
        self._renal = RenalClearanceEngine()
        self._solver = PBPKSolver(config=solver_config)
        self._absorption_engine = PermeabilityEngine()

    def _build_oral_portal_rate(
        self,
        drug: Drug,
        schedule: DoseSchedule,
        gut_wall_mbi_exposures: Optional[Sequence[object]] = None,
        gut_wall_induction_exposures: Optional[Sequence[object]] = None,
        gut_transit: Optional[object] = None,
    ) -> Tuple[Optional[object], List[AbsorptionProfile]]:
        """
        Run the P2 absorption engine for each oral dose event and assemble
        a single portal-rate forcing function.

        ``gut_wall_mbi_exposures`` (V1.2): optional sequence of
        :class:`GutWallMBIExposure` records describing steady-state
        perpetrators that inactivate enterocyte CYP3A4 via MBI.
        When supplied, each oral absorption run uses the modified
        per-segment ``fg_j`` array from
        :func:`gut_wall_mbi.compute_steady_state_fg_per_segment`
        instead of the V1 static formula. Passing ``None`` (default)
        reproduces V1 behaviour byte-for-byte.

        Returns:
            (portal_rate_provider, profiles)
            portal_rate_provider — callable(t_h) → mg/h into liver, or None
                if there are no oral events or every P2 run failed.
            profiles — list of AbsorptionProfile objects (one per oral dose)
                for downstream reporting; empty if no profiles were built.
        """
        oral_events = [e for e in schedule.events if e.route == Route.ORAL]
        if not oral_events:
            return None, []

        profiles_with_offsets: List[Tuple[AbsorptionProfile, float]] = []
        profiles: List[AbsorptionProfile] = []

        for event in oral_events:
            try:
                profile = self._absorption_engine.simulate(
                    drug=drug,
                    formulation=event.formulation,
                    dose_mg=event.dose_mg,
                    gut_wall_mbi_exposures=gut_wall_mbi_exposures,
                    gut_wall_induction_exposures=gut_wall_induction_exposures,
                    gut_transit=gut_transit,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "P2 absorption simulation failed for oral dose "
                    "(%.2f mg @ t=%.2f h): %s — falling back to legacy ka.",
                    event.dose_mg, event.time_h, exc,
                )
                continue

            if not profile.solver_success:
                logger.warning(
                    "P2 absorption reported solver failure for oral dose "
                    "(%.2f mg @ t=%.2f h) — using profile anyway.",
                    event.dose_mg, event.time_h,
                )

            profiles_with_offsets.append((profile, float(event.time_h)))
            profiles.append(profile)

        if not profiles_with_offsets:
            return None, []

        provider = build_portal_rate_provider(profiles_with_offsets)
        return provider, profiles

    def run(
        self,
        drug: Drug,
        schedule: DoseSchedule,
        physiology: PhysiologyModel,
        clint_provider: Optional[CLintProvider] = None,
        gut_wall_mbi_exposures: Optional[Sequence[object]] = None,
        oatp1b1_inhibitors: Optional[Sequence[object]] = None,
        gut_wall_induction_exposures: Optional[Sequence[object]] = None,
        gut_transit: Optional[object] = None,
        gfr_ml_per_min: Optional[float] = None,
    ) -> SimulationResult:
        """
        Execute a full PBPK simulation.

        Args:
            drug: V1 Drug entity (from DrugFactory)
            schedule: Dosing schedule (from DoseScheduler)
            physiology: Physiological model (from build_reference_male/female)
            clint_provider: Optional dynamic CLint callable from the
                Phase-3 ``DDIOrchestrator``. When supplied, the PBPK
                solver recomputes hepatic clearance at every step via
                the well-stirred model using ``clint_provider(t)``
                instead of the static pre-computed value. This is
                how ``OrchestratorV2`` feeds DDI-induced time-varying
                inhibition / induction into the victim's simulation.
            gut_wall_mbi_exposures: Optional sequence of
                :class:`neogen.engine.ddi.GutWallMBIExposure` records
                describing steady-state perpetrators that inactivate
                enterocyte CYP3A4 via mechanism-based inhibition.
                When supplied, oral absorption runs use the modified
                per-segment ``fg_j`` array (V1.2 feature). Ignored
                for IV-only schedules.
            oatp1b1_inhibitors: Optional sequence of
                :class:`neogen.engine.ddi.transporter_ddi.OATP1B1InhibitorExposure`
                records (V1.2 Track B). Each suppresses hepatic
                OATP1B1-mediated active uptake in proportion to
                ``[I]_u / IC50``. When ``None`` or empty the result is
                byte-identical to V1.1 behaviour (flat 15× uptake
                enhancement for OATP1B1 substrates, no enhancement
                otherwise).

        Returns:
            Frozen SimulationResult with PK metrics and organ profiles
        """
        logger.info(
            "Starting simulation: %s, dose=%s, route=%s",
            drug.name,
            [e.dose_mg for e in schedule.events],
            [e.route.name for e in schedule.events],
        )

        # ── Step 1: Tissue partition coefficients ──
        kp = self._partitioner.predict(drug, physiology)
        logger.info(
            "Kp: liver=%.2f, kidney=%.2f, brain=%.2f, muscle=%.2f, fat=%.2f",
            kp.liver, kp.kidney, kp.brain, kp.muscle, kp.fat,
        )

        # ── Step 2: Clearances ──
        # V1.2 Track B: when oatp1b1_inhibitors is supplied, hepatic
        # active-uptake amplification is suppressed via IC50 modulation.
        # None (default) reproduces V1.1 byte-for-byte.
        hepatic = self._hepatic.compute(
            drug, physiology, oatp1b1_inhibitors=oatp1b1_inhibitors
        )
        renal_kwargs = {}
        if gfr_ml_per_min is not None:
            renal_kwargs["gfr_ml_per_min"] = gfr_ml_per_min
        renal = self._renal.compute(drug, physiology, **renal_kwargs)
        renal = RenalClearanceEngine.calibrate_with_fe(renal, drug, hepatic.cl_hepatic_L_per_h)

        # V1.2 structural fixes: CL override hierarchy
        #  1. cl_total_override_L_per_h — extrahepatic drugs (CL > 0.8 Qh)
        #  2. cl_hepatic_override + cl_renal_override — fe-based split
        #  3. Standard hepatic + renal engine output
        _cl_h = hepatic.cl_hepatic_L_per_h
        _cl_r = renal.cl_renal_net_L_per_h
        if drug.cl_total_override_L_per_h is not None:
            cl_total = drug.cl_total_override_L_per_h
            # Split for reporting — attribute all to hepatic for simplicity
            _cl_h = cl_total
            _cl_r = 0.0
        elif drug.cl_hepatic_override_L_per_h is not None:
            _cl_h = drug.cl_hepatic_override_L_per_h
            _cl_r = drug.cl_renal_override_L_per_h or 0.0
            cl_total = _cl_h + _cl_r
        else:
            cl_total = _cl_h + _cl_r
        logger.info(
            "Clearance: CLh=%.2f, CLr=%.2f, CLtotal=%.2f L/h",
            _cl_h, _cl_r, cl_total,
        )

        # ── Step 2b: P2 absorption — build oral portal forcing function ──
        oral_portal_rate, absorption_profiles = self._build_oral_portal_rate(
            drug, schedule, gut_wall_mbi_exposures=gut_wall_mbi_exposures,
            gut_wall_induction_exposures=gut_wall_induction_exposures,
            gut_transit=gut_transit,
        )
        if absorption_profiles:
            first = absorption_profiles[0]
            logger.info(
                "P2 absorption: fa=%.3f, fg=%.3f, ka_eff=%.3f /h, tlag=%.2f h, "
                "profiles=%d",
                first.fa, first.fg, first.effective_ka_per_h, first.tlag_h,
                len(absorption_profiles),
            )
        elif any(e.route == Route.ORAL for e in schedule.events):
            logger.info(
                "P2 absorption unavailable — falling back to legacy ka path."
            )

        # ── Step 2c: f_oral_observed correction ──────────────────
        # When the drug carries a published absolute oral bioavailability
        # (F), scale the oral portal rate so systemic exposure matches.
        # The engine's predicted Fh (from well-stirred model) may differ
        # from clinical reality, especially for high-extraction drugs.
        # We scale the oral input so that (Fa·Fg)_effective × Fh = F_obs.
        if (
            oral_portal_rate is not None
            and drug.f_oral_observed is not None
            and drug.f_oral_observed > 0
            and absorption_profiles
        ):
            fh = hepatic.fh
            if fh > 0.001:
                # Target Fa·Fg = F_obs / Fh
                fa_fg_target = min(drug.f_oral_observed / fh, 1.0)
                fa_fg_current = absorption_profiles[0].f_portal  # Fa × Fg
                if fa_fg_current > 0.001:
                    scale = fa_fg_target / fa_fg_current
                    # Only apply if correction is > 10%
                    if abs(scale - 1.0) > 0.10:
                        raw_fn = oral_portal_rate
                        oral_portal_rate = lambda t, _fn=raw_fn, _s=scale: _fn(t) * _s
                        logger.info(
                            "F correction: F_obs=%.3f, Fh=%.3f, "
                            "Fa*Fg target=%.3f (current=%.3f), scale=%.2f",
                            drug.f_oral_observed, fh,
                            fa_fg_target, fa_fg_current, scale,
                        )

        # ── Step 3: Solve ODE ──
        solver_output = self._solver.solve(
            drug,
            schedule,
            physiology,
            kp_override=kp,
            oral_portal_rate=oral_portal_rate,
            clint_provider=clint_provider,
            oatp1b1_inhibitors=oatp1b1_inhibitors,
            gfr_ml_per_min=gfr_ml_per_min,
        )

        if not solver_output.success:
            logger.warning("Solver reported failure: %s", solver_output.message)

        # ── Step 4: Compute PK metrics from plasma profile ──
        time_h = solver_output.time_h
        plasma_conc = solver_output.concentrations[1]  # Index 1 = plasma

        total_dose = sum(e.dose_mg for e in schedule.events)
        route_str = schedule.events[0].route.name if schedule.events else "UNKNOWN"

        pk_metrics = PKCalculator.compute(time_h, plasma_conc, total_dose, route_str)

        # V1.2 Fix 2: override simulated t½ with measured clinical value
        if drug.half_life_override_h is not None:
            pk_metrics = PKMetrics(
                cmax_mg_per_L=pk_metrics.cmax_mg_per_L,
                tmax_h=pk_metrics.tmax_h,
                auc_0_inf_mg_h_per_L=pk_metrics.auc_0_inf_mg_h_per_L,
                auc_0_last_mg_h_per_L=pk_metrics.auc_0_last_mg_h_per_L,
                half_life_h=drug.half_life_override_h,
                cl_apparent_L_per_h=pk_metrics.cl_apparent_L_per_h,
                vss_apparent_L=pk_metrics.vss_apparent_L,
                mrt_h=pk_metrics.mrt_h,
            )

        # V1.2: Statin lactone→acid correction. Published clinical PK
        # data for statin prodrugs (atorvastatin, simvastatin) report the
        # pharmacologically active hydroxy-acid form only, which is a
        # fraction of total circulating drug. Scale Cmax and AUC by
        # active_fraction; t½ is unchanged (same elimination kinetics).
        if drug.active_fraction is not None:
            af = drug.active_fraction
            pk_metrics = PKMetrics(
                cmax_mg_per_L=pk_metrics.cmax_mg_per_L * af,
                tmax_h=pk_metrics.tmax_h,
                auc_0_inf_mg_h_per_L=pk_metrics.auc_0_inf_mg_h_per_L * af,
                auc_0_last_mg_h_per_L=pk_metrics.auc_0_last_mg_h_per_L * af,
                half_life_h=pk_metrics.half_life_h,
                cl_apparent_L_per_h=pk_metrics.cl_apparent_L_per_h,
                vss_apparent_L=pk_metrics.vss_apparent_L,
                mrt_h=pk_metrics.mrt_h,
            )

        # ── Step 5: Build organ time-course profiles ──
        organ_profiles = {}
        for idx, name in enumerate(ORGAN_NAMES):
            if name == "gut_lumen":
                continue  # Amount, not concentration
            organ_profiles[name] = OrganTimeCourse(
                name=name,
                time_h=tuple(float(t) for t in time_h),
                concentration_mg_per_L=tuple(
                    float(c) for c in solver_output.concentrations[idx]
                ),
            )

        # ── Step 6: Assemble result ──
        # Post-prediction PK consistency QC. Uses Vss (steady-state Vd)
        # against the algebraic relation t½ = ln(2) · Vd / CL. Failures
        # are tolerated: pk_consistency stays None.
        try:
            pk_consistency = compute_pk_consistency_qc(
                cl_L_per_h=round(cl_total, 4),
                vd_L=pk_metrics.vss_apparent_L,
                half_life_h=pk_metrics.half_life_h,
            )
        except Exception:  # pragma: no cover — diagnostic only
            pk_consistency = None

        result = SimulationResult(
            drug_name=drug.name,
            dose_mg=total_dose,
            route=route_str,
            pk_metrics=pk_metrics,
            time_h=tuple(float(t) for t in time_h),
            plasma_conc_mg_per_L=tuple(float(c) for c in plasma_conc),
            organ_profiles=organ_profiles,
            cl_hepatic_L_per_h=_cl_h,
            cl_renal_L_per_h=_cl_r,
            cl_total_L_per_h=round(cl_total, 4),
            absorption_profiles=tuple(absorption_profiles),
            solver_method=solver_output.method_used,
            solver_success=solver_output.success,
            n_timepoints=len(time_h),
            pk_consistency=pk_consistency,
        )

        logger.info(
            "Simulation complete: Cmax=%.4f mg/L, tmax=%.2f h, "
            "AUC=%.4f mg·h/L, t½=%.2f h",
            pk_metrics.cmax_mg_per_L, pk_metrics.tmax_h,
            pk_metrics.auc_0_inf_mg_h_per_L, pk_metrics.half_life_h,
        )

        return result
