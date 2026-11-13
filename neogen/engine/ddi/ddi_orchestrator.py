"""
DDI Orchestrator — V1 DDI Engine (Step 10)
============================================
Runtime coupling between the Phase-3 DDI layer and the Phase-1 PBPK solver.

This module bridges the three mechanisms built in Steps 7-9 into a single
``CLintProvider`` callable (hours → scaled CLint in L/h) that the PBPK ODE
RHS consults at every solver step. It is the module that turns NeoGen's
DDI engine from a post-hoc AUCR calculator into a dynamic, state-aware
simulator.

Composition rule — single victim, multiple perpetrators
---------------------------------------------------------
For every tracked CYP isoform the victim's intrinsic clearance scales by
three independent mechanisms:

1. **Reversible competitive inhibition** (Step 7) via ``R_cyp(t)``:
   ``R_cyp = 1 + [I]u(t) / Ki``   →  instantaneous, concentration-driven.
2. **Enzyme pool fold-change** (Steps 8 + 9) via ``fold_cyp(t)``:
   Captures induction build-up, MBI destruction, and their combined
   turnover ODE. Read from a :class:`EnzymePoolTrace` produced by Step 9.
3. **Baseline** pathway (non-CYP, renal, biliary) — untouched.

Per-CYP effective CLint scaling::

    s_cyp(t) = Π_k  fold_cyp_k(t) / R_cyp_k(t)

Product across perpetrators ``k`` gives a multiplicative combination which
is exact at steady state for independent mechanisms and a pragmatic first
approximation for simultaneous effects during transients. V2+ can replace
this with a composite enzyme-pool integration driven by the sum of all
perpetrator inputs.

Victim CLint under the load::

    CLint_eff(t) = CLint_base · [ (1 − Σ fm_cyp) + Σ fm_cyp · s_cyp(t) ]

The bracket is 1.0 in the absence of any perpetrator. Induction pushes it
above 1.0 (faster clearance, AUC down). Inhibition and MBI pull it below
1.0 (slower clearance, AUC up). All three cleanly collapse to a no-op
when their respective parameters are unset.

The well-stirred liver model then recovers CLh from CLint *inside* the
PBPK RHS::

    fu_clint(t) = fu · CLint_eff(t)
    CLh(t)      = Qh · fu_clint / (Qh + fu_clint)

so the nonlinear saturation between CLint and CLh is preserved — critical
for high-extraction victims where the usual "AUCR = 1 / (Σ fm/R + …)"
FDA static equation systematically overpredicts DDI magnitude.

OATP1B1 perpetrator-side coupling (Pass 2)
------------------------------------------
The static :class:`HepaticClearanceEngine` already boosts CLint for
OATP1B1-substrate victims (simvastatin, atorvastatin, pravastatin,
rosuvastatin, valsartan, repaglinide) by a flat 15× factor,
:data:`neogen.engine.hepatic_clearance.OATP1B1_UPTAKE_FACTOR`. That
captures the fact that hepatic uptake, not CYP metabolism, is
rate-limiting for these drugs. When a perpetrator inhibits OATP1B1 at
the sinusoid, the effective uptake factor shrinks and the drug
accumulates in plasma — that is the dominant mechanism behind the
clinical azole + simvastatin AUCR ≈ 10 result.

For an OATP1B1-substrate victim, the orchestrator computes::

    r_oatp(t)          = Π_k (1 + [I]u_k(t) / Ki_oatp_k)      over perpetrators
    u_eff(t)           = 1 + (U_base − 1) / r_oatp(t)          effective uptake factor
    uptake_corr(t)     = u_eff(t) / U_base                     multiplier on base CLint

where ``U_base = OATP1B1_UPTAKE_FACTOR``. At r_oatp=1 the correction is
identity (no DDI); at r_oatp→∞ it floors at ``1/U_base`` — i.e. the
baseline CLint drops back to the pure metabolic value, as if no OATP1B1
boost existed. The final scaling returned by the provider is the product
of the CYP factor and the uptake correction::

    CLint(t) = base × [ (1 − Σfm) + Σ fm · s_cyp(t) ] · uptake_corr(t)

For non-substrate victims, ``uptake_corr(t) ≡ 1`` and the orchestrator
reduces to the pre-Pass-2 behaviour exactly. Perpetrators without an
``oatp1b1_ic50`` contribute nothing to ``r_oatp``.

Scope
-----
  * Victim = single drug; perpetrator list = arbitrary length.
  * Perpetrator exposure is constant during its ``on`` window (the V1
    assumption — real perpetrator PK profiles are a V2 feature).
  * Step 9 enzyme trace is pre-simulated once per perpetrator and then
    linearly interpolated during the victim run.
  * Transporter DDIs limited to OATP1B1 sinusoidal uptake (Pass 2).
    P-gp, BCRP, OCT2 coupling are deferred to a later pass.

Out of scope (V2+):
  * Autoinduction: perpetrator metabolised by an enzyme it induces.
  * P-gp / BCRP / OCT2 transporter DDIs.
  * Time-varying perpetrator concentration (full co-administration PBPK).
  * OATP1B1 induction (rifampicin chronic-dosing net effect near zero).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

from neogen.core.entities.drug import Drug
from neogen.engine.ddi.autoinduction import (
    AutoinductionSimulator,
    AutoinductionSummary,
    is_autoinducer,
)
from neogen.engine.ddi.cyp_inhibition import CYP_NAMES
from neogen.engine.ddi.enzyme_pool_evolver import (
    EnzymePoolEvolver,
    EnzymePoolTrace,
)
from neogen.engine.ddi.transporter_ddi import DEFAULT_OATP1B1_UPTAKE_FACTOR as OATP1B1_UPTAKE_FACTOR

logger = logging.getLogger(__name__)


#: Canonical type alias for the dynamic-CLint callback consumed by the
#: PBPK solver RHS. Takes simulation time in hours, returns the scaled
#: whole-liver intrinsic clearance (L/h) at that instant.
CLintProvider = Callable[[float], float]

#: Numerical floor for CLint so the well-stirred recomputation never
#: divides by zero even under extreme MBI suppression.
_CLINT_FLOOR_L_PER_H: float = 1.0e-6


# ════════════════════════════════════════════════════════════════
# FROZEN INPUT / OUTPUT DATACLASSES
# ════════════════════════════════════════════════════════════════
# Canonical definitions live in ``ddi_result.py``. Re-exported here
# so existing ``from ddi_orchestrator import PerpetratorExposure``
# imports continue to work without changes.

from neogen.engine.ddi.ddi_result import (  # noqa: E402
    CLintBreakdown,
    DDISummary,
    PerpetratorExposure,
)


# ════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ════════════════════════════════════════════════════════════════


class DDIOrchestrator:
    """Assemble PerpetratorExposure records into a CLintProvider.

    Usage:
        orch = DDIOrchestrator()

        # Build exposures (one per perpetrator). The evolver is called
        # internally so the enzyme trace is pre-computed.
        ketoconazole_exp = orch.build_exposure(
            ketoconazole,
            steady_state_conc_um=0.5,
            duration_on_h=72.0,
            include_enzyme_pool=False,  # pure reversible; save the solve
        )

        # Build the CLint(t) callable for the victim.
        provider = orch.build_clint_provider(
            victim=midazolam,
            base_clint_scaled_L_per_h=hepatic_base.clint_scaled_L_per_h,
            exposures=[ketoconazole_exp],
        )

        # Hand it to the solver.
        out = pbpk_solver.solve(midazolam, schedule, phys, clint_provider=provider)
    """

    def __init__(self, evolver: Optional[EnzymePoolEvolver] = None) -> None:
        self._evolver = evolver or EnzymePoolEvolver()

    @property
    def evolver(self) -> EnzymePoolEvolver:
        return self._evolver

    # ── Exposure construction ───────────────────────────────

    def build_exposure(
        self,
        perpetrator: Drug,
        steady_state_conc_um: float,
        *,
        duration_on_h: float,
        duration_off_h: float = 0.0,
        pretreatment_h: float = 0.0,
        include_enzyme_pool: bool = True,
        n_points: int = 200,
    ) -> PerpetratorExposure:
        """Build a :class:`PerpetratorExposure` for the given drug.

        Args:
            perpetrator: The DDI perpetrator drug.
            steady_state_conc_um: Unbound liver-site concentration (µM).
            duration_on_h: Dosing window length (hours).
            duration_off_h: Washout length (hours, default 0).
            pretreatment_h: Hours of perpetrator exposure before the
                victim simulation starts. The enzyme trace is evolved
                over ``[0, pretreatment_h + duration_on_h + duration_off_h]``
                and the orchestrator shifts its lookup by ``pretreatment_h``
                so the victim clock ``t = 0`` lines up with the plateau
                induction/MBI state. Default 0 (no pretreatment).
            include_enzyme_pool: If True, pre-run Step 9 to get the
                enzyme trace. Set False for pure reversible inhibitors
                (no Indmax, no kinact) to skip the ODE solve.
            n_points: Snapshot resolution for the enzyme trace.

        Returns:
            Frozen :class:`PerpetratorExposure` record.
        """
        if steady_state_conc_um < 0.0:
            raise ValueError(
                f"steady_state_conc_um must be ≥ 0, got {steady_state_conc_um}"
            )
        if duration_on_h <= 0.0:
            raise ValueError(
                f"duration_on_h must be > 0, got {duration_on_h}"
            )
        if duration_off_h < 0.0:
            raise ValueError(
                f"duration_off_h must be ≥ 0, got {duration_off_h}"
            )
        if pretreatment_h < 0.0:
            raise ValueError(
                f"pretreatment_h must be ≥ 0, got {pretreatment_h}"
            )

        trace: Optional[EnzymePoolTrace] = None
        if include_enzyme_pool:
            # Extend the evolver's on-window to cover the pretreatment
            # plateau plus the scored victim window. This is the single
            # point at which pretreatment affects the enzyme pool; after
            # this, the lookup shift inside ``_combined_cyp_scaling``
            # does all the rest of the work.
            trace = self._evolver.simulate(
                perpetrator,
                inducer_conc_um=steady_state_conc_um,
                duration_on_h=pretreatment_h + duration_on_h,
                duration_off_h=duration_off_h,
                n_points=n_points,
            )

        return PerpetratorExposure(
            perpetrator=perpetrator,
            steady_state_conc_um=steady_state_conc_um,
            duration_on_h=duration_on_h,
            duration_off_h=duration_off_h,
            pretreatment_h=pretreatment_h,
            enzyme_trace=trace,
        )

    # ── PK-derived perpetrator exposure (V1.3 enhancement) ──

    def build_exposure_from_pk(
        self,
        perpetrator: Drug,
        perpetrator_dose_mg: float,
        *,
        physiology: "PhysiologyModel",
        perpetrator_route: str = "po",
        perpetrator_dose_interval_h: float = 24.0,
        duration_on_h: float = 500.0,
        duration_off_h: float = 0.0,
        pretreatment_h: float = 0.0,
        inlet_factor: float = 1.5,
        include_enzyme_pool: bool = False,
        n_points: int = 200,
        solver_config: Optional["SolverConfig"] = None,
    ) -> Tuple[PerpetratorExposure, float]:
        """Build a :class:`PerpetratorExposure` with ``[I]u`` derived from PBPK.

        Runs the perpetrator alone through the V1 PBPK solver to obtain
        its steady-state plasma Cmax, then converts that to an unbound
        hepatic-inlet concentration in micromolar units::

            [I]u_uM = inlet_factor × fu × Cmax_mg_per_L × 1000 / MW_g_per_mol

        ``inlet_factor`` defaults to ~1.5 for oral dosing (FDA guidance
        approximation: portal-vein inlet > systemic plasma) and should
        be set to 1.0 for IV. The literal-value path
        (:meth:`build_exposure`) is retained unchanged — this method is
        purely additive for callers who want a PK-derived ``[I]u``
        instead of a hand-tuned literal.

        Args:
            perpetrator: The DDI perpetrator drug.
            perpetrator_dose_mg: Per-administration dose (mg) used to
                generate the perpetrator's own PK profile.
            physiology: Physiology model used by the perpetrator's PK
                simulation. Must be supplied — this method does not
                fall back to a default to keep the caller in charge of
                the human model.
            perpetrator_route: ``"po"`` for oral (default) or ``"iv"``
                for an IV bolus.
            perpetrator_dose_interval_h: Dosing interval (h) used to
                build a multiple-dose schedule that reaches steady
                state. ``24`` ≈ QD, ``12`` ≈ BID. Ignored when only a
                single dose is needed.
            duration_on_h: Length of the resulting exposure on-window.
            duration_off_h: Washout window (default 0).
            pretreatment_h: Hours of perpetrator dosing before the
                victim simulation starts. Forwarded to
                :class:`PerpetratorExposure`.
            inlet_factor: Multiplier applied to systemic Cmax to
                approximate the unbound hepatic-inlet concentration.
                Defaults to 1.5 (oral) per FDA inlet-Cmax guidance;
                use 1.0 for IV routes.
            include_enzyme_pool: If True, run Step 9 to pre-compute the
                enzyme trace at the derived ``[I]u``. Default False
                for the pure-reversible fast path.
            n_points: Snapshot resolution for the enzyme trace.
            solver_config: Optional solver configuration. Defaults to
                a BDF config sized for steady-state perpetrator runs.

        Returns:
            ``(exposure, derived_iu_um)`` — the exposure is ready for
            :meth:`build_clint_provider`, and ``derived_iu_um`` is the
            scalar that was plugged into ``steady_state_conc_um`` so
            callers can log or sanity-check it.
        """
        # Local imports keep this method optional — the literal path
        # in build_exposure must not pay for solver startup costs.
        from neogen.engine.dose_scheduler import DoseScheduler
        from neogen.engine.pbpk_solver import PBPKSolver, SolverConfig
        from neogen.engine.simulation_result import PKCalculator

        if perpetrator_dose_mg <= 0.0:
            raise ValueError(
                f"perpetrator_dose_mg must be > 0, got {perpetrator_dose_mg}"
            )
        if duration_on_h <= 0.0:
            raise ValueError(
                f"duration_on_h must be > 0, got {duration_on_h}"
            )
        if inlet_factor <= 0.0:
            raise ValueError(
                f"inlet_factor must be > 0, got {inlet_factor}"
            )

        route = perpetrator_route.lower()
        if route not in ("po", "oral", "iv"):
            raise ValueError(
                f"perpetrator_route must be 'po'/'oral' or 'iv', got "
                f"{perpetrator_route!r}"
            )

        # Build a multiple-dose schedule that pushes through to steady
        # state. We dose for ~5 half-lives or 7 days (whichever is
        # smaller) to keep solver runtime bounded; for the canonical
        # perpetrators (ketoconazole, rifampicin, clarithromycin) this
        # is more than enough to reach a stable Cmax_ss.
        scheduler = DoseScheduler()
        interval_h = max(perpetrator_dose_interval_h, 1.0)
        n_doses = max(int(168.0 / interval_h), 5)  # ~7 days, ≥5 doses
        sim_end_h = n_doses * interval_h + 24.0
        if route == "iv":
            from neogen.core.entities.formulation import (
                DoseEvent,
                DoseSchedule,
                Formulation,
                FormulationType,
                Route,
            )
            events = tuple(
                DoseEvent(
                    drug_name=perpetrator.name,
                    dose_mg=perpetrator_dose_mg,
                    route=Route.IV_BOLUS,
                    time_h=i * interval_h,
                    formulation=Formulation(
                        formulation_type=FormulationType.IV_SOLUTION
                    ),
                )
                for i in range(n_doses)
            )
            schedule = DoseSchedule(events=events, end_time_h=sim_end_h)
        else:
            schedule = scheduler.repeated(
                perpetrator.name,
                dose_mg=perpetrator_dose_mg,
                regimen="QD" if interval_h >= 23.0 else (
                    "BID" if interval_h >= 11.0 else "TID"
                ),
                days=max(int(round(n_doses * interval_h / 24.0)), 5),
                end_time_h=sim_end_h,
            )

        cfg = solver_config or SolverConfig(
            method="BDF",
            rtol=1e-6,
            atol=1e-9,
            n_timepoints=1500,
            t_max_multiplier=10.0,
        )
        solver = PBPKSolver(cfg)
        out = solver.solve(perpetrator, schedule, physiology)

        # Use the last dosing interval to estimate Cmax_ss — this avoids
        # the loading-phase transient.
        time_h = out.time_h
        plasma = out.concentrations[1]
        last_dose_t = (n_doses - 1) * interval_h
        ss_mask = time_h >= last_dose_t
        if ss_mask.any():
            cmax_ss_mg_per_L = float(plasma[ss_mask].max())
        else:
            # Fallback: take the global Cmax via the standard PK calc.
            pk = PKCalculator.compute(
                time_h, plasma, dose_mg=perpetrator_dose_mg
            )
            cmax_ss_mg_per_L = pk.cmax_mg_per_L

        if cmax_ss_mg_per_L <= 0.0:
            raise RuntimeError(
                f"Perpetrator {perpetrator.name} simulation produced "
                f"non-positive Cmax_ss; cannot derive [I]u from PK."
            )

        # Convert mg/L → µM:  C[µM] = C[mg/L] × 1000 / MW[g/mol]
        mw = perpetrator.molecular_weight
        if mw <= 0.0:
            raise ValueError(
                f"Perpetrator {perpetrator.name} has invalid MW={mw}"
            )
        fu = perpetrator.fu_plasma_safe
        derived_iu_um = inlet_factor * fu * cmax_ss_mg_per_L * 1000.0 / mw

        exposure = self.build_exposure(
            perpetrator,
            steady_state_conc_um=derived_iu_um,
            duration_on_h=duration_on_h,
            duration_off_h=duration_off_h,
            pretreatment_h=pretreatment_h,
            include_enzyme_pool=include_enzyme_pool,
            n_points=n_points,
        )
        logger.info(
            "DDIOrchestrator.build_exposure_from_pk: %s @ %.0f mg %s — "
            "Cmax_ss=%.4f mg/L, fu=%.3f, MW=%.1f, inlet_factor=%.2f → "
            "[I]u=%.4f µM",
            perpetrator.name, perpetrator_dose_mg, route.upper(),
            cmax_ss_mg_per_L, fu, mw, inlet_factor, derived_iu_um,
        )
        return exposure, derived_iu_um

    # ── Autoinduction (V2 extension) ────────────────────────

    def build_autoinduction_exposure(
        self,
        perpetrator: Drug,
        baseline_conc_um: float,
        *,
        duration_on_h: float,
        duration_off_h: float = 0.0,
        pretreatment_h: float = 0.0,
        simulator: Optional[AutoinductionSimulator] = None,
        n_points: int = 240,
    ) -> Tuple[PerpetratorExposure, AutoinductionSummary]:
        """Build an exposure that tracks the perpetrator's self-induction.

        This is the V2 extension that closes the feedback loop for drugs
        like rifampicin, efavirenz, and carbamazepine: their clearance
        grows over 5-7 days as they induce the CYPs that metabolise
        them, and the resulting lower liver exposure back-attenuates the
        induction saturation. The Step 9 ``EnzymePoolEvolver`` cannot
        model this because it treats ``[I]u`` as an exogenous input;
        the :class:`AutoinductionSimulator` integrates the coupled ODE
        directly and hands back an :class:`EnzymePoolTrace` that the
        orchestrator's runtime coupling can consume unchanged.

        Args:
            perpetrator: The self-inducing ``Drug``. If the drug is not
                flagged as an autoinducer (no overlap between its ``fm``
                profile and its induction targets), this method falls
                back to the standard :meth:`build_exposure` path.
            baseline_conc_um: Unbound liver ``[I]u`` (µM) the drug would
                reach on day 1, i.e. before any self-induction. The
                effective concentration at steady state is this value
                scaled by ``AUCR_self_ss``.
            duration_on_h: Dosing-window length. For rifampicin a full
                plateau takes ~5-7 days (CYP3A4 t½ ≈ 36 h).
            duration_off_h: Washout length.
            pretreatment_h: Hours of perpetrator exposure **before** the
                victim simulation starts. The autoinduction trace is
                extended over ``[0, pretreatment_h + duration_on_h + duration_off_h]``
                and the orchestrator's runtime lookup shifts by
                ``pretreatment_h`` exactly as in :meth:`build_exposure`.
            simulator: Optional :class:`AutoinductionSimulator` instance.
                A default one is created when ``None``.
            n_points: Grid resolution for the coupled ODE.

        Returns:
            ``(exposure, summary)`` — the exposure is ready to drop into
            :meth:`build_clint_provider` alongside any other perpetrators;
            the summary carries the steady-state ``AUCR_self``, effective
            concentration, and per-CYP fold values for diagnostics.
        """
        if baseline_conc_um < 0.0:
            raise ValueError(
                f"baseline_conc_um must be ≥ 0, got {baseline_conc_um}"
            )
        if duration_on_h <= 0.0:
            raise ValueError(
                f"duration_on_h must be > 0, got {duration_on_h}"
            )
        if duration_off_h < 0.0:
            raise ValueError(
                f"duration_off_h must be ≥ 0, got {duration_off_h}"
            )
        if pretreatment_h < 0.0:
            raise ValueError(
                f"pretreatment_h must be ≥ 0, got {pretreatment_h}"
            )

        if not is_autoinducer(perpetrator):
            logger.info(
                "DDIOrchestrator.build_autoinduction_exposure: %s is not "
                "a self-inducer — delegating to build_exposure.",
                perpetrator.name,
            )
            exposure = self.build_exposure(
                perpetrator,
                steady_state_conc_um=baseline_conc_um,
                duration_on_h=duration_on_h,
                duration_off_h=duration_off_h,
                pretreatment_h=pretreatment_h,
                include_enzyme_pool=True,
                n_points=n_points,
            )
            trivial_summary = AutoinductionSummary(
                drug_name=perpetrator.name,
                baseline_conc_um=baseline_conc_um,
                effective_conc_um_ss=baseline_conc_um,
                aucr_self_ss=1.0,
                fold_by_cyp_ss={c: 1.0 for c in self._evolver.tracked_cyps},
                time_to_90pct_h=None,
            )
            return exposure, trivial_summary

        sim = simulator or AutoinductionSimulator(
            tracked_cyps=self._evolver.tracked_cyps,
        )
        trace = sim.simulate(
            perpetrator,
            baseline_conc_um=baseline_conc_um,
            duration_on_h=pretreatment_h + duration_on_h,
            duration_off_h=duration_off_h,
            n_points=n_points,
        )
        summary = sim.summarize(
            trace, perpetrator, baseline_conc_um=baseline_conc_um
        )

        exposure = PerpetratorExposure(
            perpetrator=perpetrator,
            # The reversible R factor uses this scalar over the on-window.
            # At plateau the victim sees the attenuated concentration, so
            # feeding the steady-state effective [I]u here is the right
            # first-order approximation — the transient reversible
            # contribution during the ramp is already a minor effect for
            # the drugs this extension targets (rifampicin and friends
            # are weak reversible inhibitors of their own metabolising
            # CYPs).
            steady_state_conc_um=summary.effective_conc_um_ss,
            duration_on_h=duration_on_h,
            duration_off_h=duration_off_h,
            pretreatment_h=pretreatment_h,
            enzyme_trace=trace,
        )
        return exposure, summary

    # ── Main API: build the CLint provider ──────────────────

    def build_clint_provider(
        self,
        *,
        victim: Drug,
        base_clint_scaled_L_per_h: float,
        exposures: Sequence[PerpetratorExposure],
    ) -> CLintProvider:
        """Return a ``CLint(t)`` callable for the victim under load.

        The returned function is pure (closes over the perpetrators and
        the victim fm profile) and safe to call from the PBPK RHS at any
        simulation time. Fold/R lookups are O(N_perpetrators × N_cyp)
        with a binary interpolation per trace — fast enough for scipy
        BDF / LSODA hot loops in V1 (millisecond-class scenes).

        Args:
            victim: The V1 ``Drug`` whose CLint is being modulated. Its
                ``metabolism.fm_for(cyp)`` values drive the combination.
                Its ``transporters.is_oatp1b1_substrate`` flag enables
                the uptake-inhibition correction when any perpetrator
                has an ``oatp1b1_ic50``.
            base_clint_scaled_L_per_h: Pre-computed whole-liver CLint for
                the victim in the ABSENCE of any DDI (L/h). For OATP1B1
                substrates this already includes the flat ``U_base``
                multiplier from :class:`HepaticClearanceEngine`; the
                uptake correction below multiplies that by
                ``u_eff(t) / U_base`` so the result collapses cleanly
                when no inhibitor is present.
            exposures: One record per perpetrator.

        Returns:
            A :data:`CLintProvider` callable.
        """
        if base_clint_scaled_L_per_h < 0.0:
            raise ValueError(
                f"base_clint must be ≥ 0, got {base_clint_scaled_L_per_h}"
            )

        fm = victim.metabolism
        # Precompute per-CYP fm once — avoids repeated attribute lookups.
        fm_pairs: Tuple[Tuple[str, float], ...] = tuple(
            (cyp, fm.fm_for(cyp)) for cyp in CYP_NAMES
        )
        total_fm = sum(f for _, f in fm_pairs)
        remaining_fraction = max(1.0 - total_fm, 0.0)

        exposures_tuple = tuple(exposures)

        victim_is_oatp1b1_substrate = bool(
            getattr(victim.transporters, "is_oatp1b1_substrate", False)
        )

        def clint_at(t_h: float) -> float:
            cyp_scale = self._compute_scaling(
                t_h,
                fm_pairs=fm_pairs,
                remaining_fraction=remaining_fraction,
                exposures=exposures_tuple,
            )
            uptake_corr = self._oatp1b1_uptake_correction(
                t_h,
                is_substrate=victim_is_oatp1b1_substrate,
                exposures=exposures_tuple,
            )
            clint = base_clint_scaled_L_per_h * cyp_scale * uptake_corr
            return max(clint, _CLINT_FLOOR_L_PER_H)

        return clint_at

    # ── Diagnostic: full breakdown at a given time ──────────

    def breakdown_at(
        self,
        t_h: float,
        *,
        victim: Drug,
        base_clint_scaled_L_per_h: float,
        exposures: Sequence[PerpetratorExposure],
    ) -> CLintBreakdown:
        """Return a full diagnostic decomposition of CLint at one time.

        Useful for unit tests and regulatory audit trails — callers can
        ask "why was CYP3A4 scaled by 0.097 at t = 36 h?" and get the
        per-CYP contribution map back.
        """
        fm = victim.metabolism
        fm_pairs = tuple((cyp, fm.fm_for(cyp)) for cyp in CYP_NAMES)
        total_fm = sum(f for _, f in fm_pairs)
        remaining = max(1.0 - total_fm, 0.0)
        exposures_tuple = tuple(exposures)

        per_cyp: list[Tuple[str, float]] = []
        contrib_sum = 0.0
        for cyp, fm_i in fm_pairs:
            s_cyp = self._combined_cyp_scaling(t_h, cyp, exposures_tuple)
            per_cyp.append((cyp, s_cyp))
            if fm_i > 0.0:
                contrib_sum += fm_i * s_cyp

        cyp_scale = remaining + contrib_sum
        uptake_corr = self._oatp1b1_uptake_correction(
            t_h,
            is_substrate=bool(
                getattr(victim.transporters, "is_oatp1b1_substrate", False)
            ),
            exposures=exposures_tuple,
        )
        scale = cyp_scale * uptake_corr
        clint_eff = max(
            base_clint_scaled_L_per_h * scale, _CLINT_FLOOR_L_PER_H
        )
        return CLintBreakdown(
            time_h=t_h,
            clint_base_L_per_h=base_clint_scaled_L_per_h,
            clint_eff_L_per_h=clint_eff,
            scaling=scale,
            cyp_scaling=cyp_scale,
            uptake_correction=uptake_corr,
            per_cyp_scaling=tuple(per_cyp),
        )

    # ── Internal math ───────────────────────────────────────

    @classmethod
    def _compute_scaling(
        cls,
        t_h: float,
        *,
        fm_pairs: Tuple[Tuple[str, float], ...],
        remaining_fraction: float,
        exposures: Tuple[PerpetratorExposure, ...],
    ) -> float:
        """``(1 − Σfm) + Σ fm_cyp · s_cyp(t)``. Returns dimensionless."""
        if not exposures:
            return 1.0  # No DDI → identity scaling.

        total = remaining_fraction
        for cyp, fm_i in fm_pairs:
            if fm_i <= 0.0:
                continue
            s_cyp = cls._combined_cyp_scaling(t_h, cyp, exposures)
            total += fm_i * s_cyp
        return total

    @staticmethod
    def _oatp1b1_uptake_correction(
        t_h: float,
        *,
        is_substrate: bool,
        exposures: Tuple[PerpetratorExposure, ...],
    ) -> float:
        """Multiplicative correction on base CLint for OATP1B1 DDI.

        Returns 1.0 when the victim is not an OATP1B1 substrate, when
        no perpetrator inhibits OATP1B1, or when perpetrator
        concentration is zero (outside the on-window). Otherwise shrinks
        the pre-baked ``U_base`` uptake boost in proportion to the
        perpetrator-driven R-factor on OATP1B1.

        Math::

            r_oatp = Π_k (1 + [I]u_k(t) / Ki_oatp_k)   over inhibitors
            u_eff  = 1 + (U_base − 1) / r_oatp
            corr   = u_eff / U_base                     ∈ [1/U_base, 1.0]
        """
        if not is_substrate or not exposures:
            return 1.0

        r_oatp = 1.0
        for exp in exposures:
            transporters = getattr(exp.perpetrator, "transporters", None)
            if transporters is None:
                continue
            if not getattr(transporters, "is_oatp1b1_inhibitor", False):
                continue
            ic50 = getattr(transporters, "oatp1b1_ic50", None)
            if ic50 is None or ic50 <= 0.0:
                continue
            conc = exp.conc_at(t_h)
            if conc <= 0.0:
                continue
            r_oatp *= 1.0 + conc / ic50

        if r_oatp <= 1.0:
            return 1.0

        u_base = OATP1B1_UPTAKE_FACTOR
        if u_base <= 0.0:
            return 1.0
        u_eff = 1.0 + (u_base - 1.0) / r_oatp
        return u_eff / u_base

    @staticmethod
    def _is_mbi_perpetrator_for_cyp(perpetrator: Drug, cyp: str) -> bool:
        """Return True when the perpetrator carries MBI parameters that
        deplete the given CYP's enzyme pool.

        V1 only models MBI on CYP3A4 (mirroring Step 9 / autoinduction).
        Both ``kinact_cyp3a4`` and ``k_i_mbi_cyp3a4`` must be populated
        for the inactivation ODE to fire — drugs with only one set are
        silently no-ops and therefore are not "MBI" for this purpose.
        """
        if cyp.upper() != "CYP3A4":
            return False
        inh = perpetrator.inhibition
        kinact = getattr(inh, "kinact_cyp3a4", None)
        ki_mbi = getattr(inh, "k_i_mbi_cyp3a4", None)
        return (
            kinact is not None
            and kinact > 0.0
            and ki_mbi is not None
            and ki_mbi > 0.0
        )

    @staticmethod
    def _scale_with_mbi_aware_inhibition(
        fold: float, r: float, mbi_active: bool
    ) -> float:
        """Combine the enzyme-pool fold-change with reversible R, while
        avoiding the MBI × reversible double-count.

        Background: the reversible factor ``R = 1 + [I]u / Ki`` describes
        competitive inhibition of the **active** enzyme pool. When MBI has
        already destroyed most of that pool (``fold << 1``), ``R`` is
        effectively re-inhibiting enzyme that no longer exists, which
        compounds suppression and over-predicts AUCR.

        Resolution (Option A from the V1.2 fix plan): when MBI is the
        active mechanism for this CYP **and** the pool is meaningfully
        depleted (``fold < 0.5``, i.e. >50 % destroyed), drop the
        reversible R correction. Pure-reversible and pure-induction
        regimes are unchanged.
        """
        if mbi_active and fold < 0.5:
            return fold
        return fold / r

    @staticmethod
    def _combined_cyp_scaling(
        t_h: float,
        cyp: str,
        exposures: Tuple[PerpetratorExposure, ...],
    ) -> float:
        """Product across perpetrators of the per-perpetrator scaling
        factor for one CYP.

        For each perpetrator we combine reversible competitive R with the
        enzyme-pool fold-change via :meth:`_scale_with_mbi_aware_inhibition`
        so MBI does not double-count against R. Returns 1.0 when no
        perpetrator touches the CYP.
        """
        scale = 1.0
        for exp in exposures:
            conc = exp.conc_at(t_h)

            # Reversible competitive (Step 7 instantaneous model)
            ki = exp.perpetrator.inhibition.ki_for(cyp)
            if ki is not None and ki > 0.0 and conc > 0.0:
                r = 1.0 + conc / ki
            else:
                r = 1.0

            # Induction + MBI fold (from pre-simulated Step 9 trace).
            # Shift the victim clock by the pretreatment window so the
            # lookup lands on the plateau portion of the trace, not on
            # the ramp — this is what matches "rifampicin was dosed for
            # a week before the probe drug" in the clinical protocol.
            if exp.enzyme_trace is not None:
                fold = exp.enzyme_trace.fold_at(exp.trace_time(t_h), cyp)
            else:
                fold = 1.0

            # Guard against pathological zeros
            if r <= 0.0:
                r = 1.0
            if fold < 0.0:
                fold = 0.0

            mbi_active = DDIOrchestrator._is_mbi_perpetrator_for_cyp(
                exp.perpetrator, cyp
            )
            scale *= DDIOrchestrator._scale_with_mbi_aware_inhibition(
                fold, r, mbi_active
            )
        return scale


# ════════════════════════════════════════════════════════════════
# CLint → CLh (well-stirred) helper for the PBPK RHS
# ════════════════════════════════════════════════════════════════


def well_stirred_clh(
    clint_scaled_L_per_h: float,
    qh_L_per_h: float,
    fu_plasma: float,
    *,
    extraction_ceiling: float = 0.99,
) -> float:
    """Well-stirred hepatic clearance from a scaled CLint value.

    Used inside the PBPK ODE RHS to recompute CLh at every solver step
    once the :data:`CLintProvider` has produced a new CLint. Mirrors the
    static calculation in :class:`HepaticClearanceEngine` so single-drug
    PK stays byte-identical when no DDI is present.

    Formula::

        Eh = fu · CLint / (Qh + fu · CLint)   (clipped to 0.99)
        CLh = Qh · Eh
    """
    if clint_scaled_L_per_h <= 0.0 or qh_L_per_h <= 0.0:
        return 0.0
    fu_clint = fu_plasma * clint_scaled_L_per_h
    denom = qh_L_per_h + fu_clint
    if denom <= 0.0:
        return 0.0
    eh = fu_clint / denom
    if eh > extraction_ceiling:
        eh = extraction_ceiling
    return qh_L_per_h * eh
