"""Orchestrator V2 — V1 P4 Step 15 (final).

Chains the NeoGen V1 engine end-to-end:

    P0 (entities) → P2 (absorption) → P3 (DDI) → P1 (PBPK) → P4 (PD)

Current scope
-------------

The V1 blueprint describes a five-phase pipeline. As of Step 15 the
layers live in different states:

  * **P0** — live. ``DrugFactory``, ``DoseScheduler``, reference
    physiology.
  * **P1** — live. ``SimulationOrchestrator`` (14-compartment PBPK,
    82 tests passing).
  * **P2** — not yet built. Plan §4 Steps 2-5 will add it.
  * **P3** — not yet built. Plan §4 Steps 7-10 will add it.
  * **P4** — live. Effect compartment, receptor kinetics, toxicity,
    orchestrator (Steps 12-15, ~87 tests passing after this step).

``OrchestratorV2`` therefore wires **P1 → P4** today and keeps explicit
keyword slots for ``absorption`` and ``ddi`` so the eventual P2/P3
builds can plug in without a breaking API change. Those slots accept
any object today (they're stored and echoed as-is) — they become typed
and physics-connected in their respective phase builds.

Why build the v2 orchestrator now, before P2/P3?

1. It is the **output contract** that the API layer (P6) will serialize.
   Locking that contract before P2/P3 means the API work in a later
   phase doesn't reshape everything.
2. It forces the P1/P4 integration to actually run on real drugs, which
   is a much sharper test than unit-level stubs.
3. It provides the single end-to-end call site (``run(drug, schedule,
   physiology)``) that reports, notebooks, and eventual web UIs depend
   on.

Design rules
------------

* Stateless at the facade level. All state lives inside the frozen
  result object.
* Sub-orchestrators are instantiated once on ``__init__`` and reused —
  they are themselves stateless but reusing them is a tiny perf win.
* Never mutates inputs. Every return is a fresh frozen dataclass.
* Never swallows sub-engine exceptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from neogen.core.entities.drug import Drug
from neogen.engine.ddi.ddi_orchestrator import (
    DDIOrchestrator,
    PerpetratorExposure,
)
from neogen.engine.ddi.gut_wall_mbi import (
    GutWallInductionExposure,
    GutWallMBIExposure,
)
from neogen.engine.ddi.transporter_ddi import OATP1B1InhibitorExposure
from neogen.engine.dose_scheduler import DoseSchedule
from neogen.engine.hepatic_clearance import HepaticClearanceEngine
from neogen.engine.orchestrator import SimulationOrchestrator
from neogen.engine.pd.pd_orchestrator import PDOrchestrator
from neogen.engine.pd.pd_result import ClinicalOutcome
from neogen.engine.population.patient_adapter import apply_patient_to_drug
from neogen.engine.simulation_result import SimulationResult


ENGINE_VERSION = "v1.2.0"
"""Current engine version tag.

Bumped when a phase gate closes.

  * ``v1.0.0-p1`` first PBPK-green tag.
  * ``v1.0.0-p4`` complete PD layer.
  * ``v1.1.0``    closed-loop plasma mass balance (SolverConfig
                  .mass_balance_mode default True) + OrchestratorV2
                  DDI wiring (SimulationResultV2.ddi now populated)
                  + validation harness + population Monte Carlo +
                  v2 API.
  * ``v1.2.0``    V1.2 Track A (gut-wall CYP3A4 MBI, enterocyte
                  pool steady state, simvastatin + clarithromycin
                  oral AUCR 10x inside 2-fold band) + V1.2 Track B
                  (OATP1B1 transporter DDI, IC50-based uptake
                  modulation, atorvastatin + rifampicin single-dose
                  OATP inhibition, itraconazole + atorvastatin
                  oral migrated from xfail — all previously
                  deferred pairs now active).
"""


@dataclass(frozen=True)
class DDISummary:
    """Compact DDI result populated on :class:`SimulationResultV2`.

    Holds the victim/perpetrator identities, the predicted AUC
    ratio (computed in the caller when a baseline run is available),
    and the list of :class:`PerpetratorExposure` records fed into
    the engine. Absent ``auc_ratio`` means :class:`OrchestratorV2`
    was run in DDI mode but the caller did not request a baseline
    comparison.
    """

    victim_name: str
    perpetrators: Tuple[str, ...]
    n_perpetrators: int
    modes: Tuple[str, ...]
    auc_victim_ddi_mg_h_per_L: float
    auc_victim_baseline_mg_h_per_L: Optional[float] = None
    auc_ratio: Optional[float] = None
    clint_scaling_at_dose_time: Optional[float] = None

    @property
    def has_baseline(self) -> bool:
        return self.auc_victim_baseline_mg_h_per_L is not None


# ════════════════════════════════════════════════════════════════
# SIMULATION RESULT V2 — top-level composite
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SimulationResultV2:
    """Top-level frozen composite returned by :class:`OrchestratorV2`.

    Holds the P1 ``SimulationResult`` plus the P4 ``ClinicalOutcome`` in
    a single immutable object, with forward-compatibility slots for the
    P2 absorption and P3 DDI results that will be wired in their
    respective phase builds.

    All access to PK data goes through ``pk_result`` (drug name, time
    grid, plasma profile, PK metrics, organ courses); all access to PD
    data goes through ``clinical_outcome``. That two-object split is
    deliberate — the alternative (flattening everything into one
    dataclass with 30+ fields) makes the object harder to inspect and
    harder to evolve.
    """

    pk_result: SimulationResult
    """Complete P1 PBPK output (plasma + organ courses + PK metrics)."""

    clinical_outcome: Optional[ClinicalOutcome] = None
    """P4 PD output, if the drug had a ``pd_profile``. ``None`` means
    the drug was run for PK only — which is a legitimate mode, not an
    error."""

    # ── Forward-compat slots (Steps 2-11 scope) ──────────────────

    absorption: Optional[Any] = None
    """Will be an ``AbsorptionProfile`` from P2. Currently always ``None``."""

    ddi: Optional[Any] = None
    """Will be a ``DDISummary`` from P3. Currently always ``None``."""

    engine_version: str = ENGINE_VERSION
    """Engine tag at the time the simulation was run. Useful for
    reproducibility when results are persisted and re-examined later."""

    # ── Convenience properties ────────────────────────────────────

    @property
    def drug_name(self) -> str:
        return self.pk_result.drug_name

    @property
    def has_pd_layer(self) -> bool:
        return self.clinical_outcome is not None

    @property
    def has_toxicity(self) -> bool:
        return (
            self.clinical_outcome is not None
            and self.clinical_outcome.has_toxicity
        )

    @property
    def has_ddi(self) -> bool:
        """True when DDI perpetrators were applied to this simulation."""
        return self.ddi is not None

    @property
    def absorption_profiles(self) -> Tuple[Any, ...]:
        """Convenience alias mirroring ``SimulationResult.absorption_profiles``.

        Returns the absorption tuple from the underlying PK result when
        the V2 ``absorption`` slot is populated, or falls back to the raw
        PK result's ``absorption_profiles``.  Returns an empty tuple when
        neither source has data.
        """
        if self.absorption is not None:
            if isinstance(self.absorption, tuple):
                return self.absorption
            return (self.absorption,)
        return self.pk_result.absorption_profiles

    def to_clinical_report(self) -> Tuple[str, ...]:
        """Layered text summary — top line is the verdict, then PK, then PD.

        Returns a tuple of lines so the caller decides how to render
        them (stdout, HTML, PDF, log entry, …). Keeping it as lines
        rather than a joined string also makes it trivial to slice off
        the header for a one-line log.
        """
        lines = [
            f"== {self.drug_name} (engine {self.engine_version}) =="
        ]
        pk = self.pk_result.pk_metrics
        lines.append(
            f"  PK:       Cmax {pk.cmax_mg_per_L:.3f} mg/L at "
            f"t = {pk.tmax_h:.2f} h, "
            f"AUC_inf {pk.auc_0_inf_mg_h_per_L:.2f} mg·h/L, "
            f"t½ {pk.half_life_h:.2f} h"
        )
        if self.clinical_outcome is not None:
            lines.extend(self.clinical_outcome.to_summary_lines())
        else:
            lines.append("  (PK only — no PD profile attached to drug)")
        return tuple(lines)


# ════════════════════════════════════════════════════════════════
# ORCHESTRATOR V2
# ════════════════════════════════════════════════════════════════


class OrchestratorV2:
    """End-to-end NeoGen V1 engine facade.

    Usage::

        engine = OrchestratorV2()
        result = engine.run(drug, schedule, physiology)
        for line in result.to_clinical_report():
            print(line)

    The class is stateless — reusing a single instance across many
    simulations is fine and slightly cheaper than constructing a new
    one each call.

    Today this chains P1 → P4 only. When P2 (absorption) and P3 (DDI)
    are built, they will be wired in *before* the P1 call (P2 feeds a
    forcing function into the gut compartment; P3 feeds a time-varying
    CLint into the hepatic clearance engine). The public signature of
    ``run`` will not change — the new phases will appear as additional
    populated fields on the returned ``SimulationResultV2``.
    """

    def __init__(self) -> None:
        self._pk = SimulationOrchestrator()
        self._pd = PDOrchestrator()
        self._ddi = DDIOrchestrator()
        self._hepatic = HepaticClearanceEngine()

    def run(
        self,
        drug: Drug,
        schedule: DoseSchedule,
        physiology,  # PhysiologyModel — avoid import cycle at typing time
        perpetrators: Optional[Sequence[PerpetratorExposure]] = None,
        gut_wall_mbi_exposures: Optional[Sequence[GutWallMBIExposure]] = None,
        gut_wall_induction_exposures: Optional[Sequence[GutWallInductionExposure]] = None,
        oatp1b1_inhibitors: Optional[Sequence[OATP1B1InhibitorExposure]] = None,
        compute_ddi_baseline: bool = True,
        *,
        patient=None,
        perpetrator_exposures: Optional[Sequence[PerpetratorExposure]] = None,
    ) -> SimulationResultV2:
        """Run the full V1 pipeline for a single drug + schedule + physiology.

        Steps:
          1. If ``perpetrators`` is provided and non-empty, build a
             dynamic CLint provider from the DDI orchestrator and
             (optionally) run a baseline PK simulation without the
             perpetrators for an AUC ratio.
          2. P1 PBPK solve (14-compartment) with the CLint provider
             AND the optional ``gut_wall_mbi_exposures`` (V1.2),
             surfacing ``AbsorptionProfile`` objects on the result.
          3. If ``drug.pd_profile`` is populated, run the P4
             orchestrator on the plasma trace.
          4. Assemble into a frozen :class:`SimulationResultV2`,
             populating the ``absorption`` and ``ddi`` slots.

        Args:
            drug: The victim drug.
            schedule: Victim dose schedule.
            physiology: Patient physiology (or reference).
            perpetrators: Optional list of :class:`PerpetratorExposure`
                records for HEPATIC DDI. Each contributes its mechanism
                (reversible, induction, or MBI) via the DDI orchestrator's
                fold / R combination rule. These are the hepatic-inlet
                unbound concentrations used by the CLint provider.
            gut_wall_mbi_exposures: Optional list of
                :class:`GutWallMBIExposure` records for GUT-WALL MBI
                (V1.2 feature). These carry the LUMINAL concentration
                (typically 100-10 000 uM), which is orders of magnitude
                higher than the hepatic-inlet unbound concentration
                used by ``perpetrators``. A single co-administered
                perpetrator can legitimately appear in both lists
                simultaneously — hepatic via PerpetratorExposure,
                gut-wall via GutWallMBIExposure. Callers typically
                build both from the same Drug with different
                concentrations, as in the simvastatin + clarithromycin
                validation case.
            oatp1b1_inhibitors: Optional list of
                :class:`OATP1B1InhibitorExposure` records for V1.2
                Track B hepatic OATP1B1 transporter DDI. Each record
                carries the unbound hepatic-inlet concentration of the
                perpetrator at steady state and, via the perpetrator's
                :class:`TransporterAffinities`, the OATP1B1 IC50.
                Modulation is multiplicative with the baseline active
                uptake factor. As with gut-wall MBI a single drug can
                appear in multiple lists (e.g. rifampicin is both a
                hepatic CYP3A4 inducer and an OATP1B1 inhibitor; it
                would appear in both ``perpetrators`` and
                ``oatp1b1_inhibitors`` for a full-mechanism study).
            compute_ddi_baseline: When ``True`` (default) and any
                DDI exposures are non-empty, an extra baseline PK run
                is performed without any DDI perturbation so an AUC
                ratio can be reported.

        Returns:
            :class:`SimulationResultV2` with populated slots.
        """
        # ── Patient adapter: scale drug CLint before simulation ──
        if patient is not None:
            drug = apply_patient_to_drug(drug, patient)

        # ── Alias: perpetrator_exposures → perpetrators ──
        if perpetrator_exposures is not None and perpetrators is None:
            perpetrators = perpetrator_exposures

        # ── Build patient-aware gut transit for oral absorption ──
        gut_transit = None
        if patient is not None and hasattr(patient, "is_fasted"):
            from neogen.engine.absorption.gut_transit import (
                build_gi_tract_for_patient,
            )
            gut_transit = build_gi_tract_for_patient(
                is_fasted=patient.is_fasted,
            )

        # ── Patient GFR for renal clearance scaling ──
        gfr_ml_per_min = None
        if patient is not None:
            from neogen.engine.population import gfr_for_patient
            patient_gfr = gfr_for_patient(patient)
            if patient_gfr != 120.0:
                gfr_ml_per_min = patient_gfr

        # ── Build CLint provider when hepatic perpetrators are present ──
        clint_provider = None
        ddi_summary: Optional[DDISummary] = None
        baseline_auc: Optional[float] = None

        perp_list: List[PerpetratorExposure] = (
            list(perpetrators) if perpetrators else []
        )
        gut_mbi_list: List[GutWallMBIExposure] = (
            list(gut_wall_mbi_exposures) if gut_wall_mbi_exposures else []
        )
        gut_ind_list: List[GutWallInductionExposure] = (
            list(gut_wall_induction_exposures) if gut_wall_induction_exposures else []
        )
        oatp_list: List[OATP1B1InhibitorExposure] = (
            list(oatp1b1_inhibitors) if oatp1b1_inhibitors else []
        )
        any_ddi = bool(perp_list) or bool(gut_mbi_list) or bool(gut_ind_list) or bool(oatp_list)

        if any_ddi and compute_ddi_baseline:
            # Baseline: no DDI perturbations at all, including no OATP
            # suppression — this gives us the "V1.1 equivalent" AUC
            # denominator for the AUCR ratio.
            baseline_result = self._pk.run(
                drug, schedule, physiology,
                gfr_ml_per_min=gfr_ml_per_min,
            )
            baseline_auc = (
                baseline_result.pk_metrics.auc_0_inf_mg_h_per_L
            )

        if perp_list:
            # When building the dynamic CYP CLint provider, the base
            # hepatic CLint must already incorporate OATP1B1 uptake
            # suppression if co-dosed OATP inhibitors are supplied.
            # This is the mixed-mechanism pathway (rifampicin on
            # atorvastatin, cyclosporine on statins, etc).
            hepatic = self._hepatic.compute(
                drug, physiology, oatp1b1_inhibitors=oatp_list or None
            )
            clint_provider = self._ddi.build_clint_provider(
                victim=drug,
                base_clint_scaled_L_per_h=hepatic.clint_scaled_L_per_h,
                exposures=perp_list,
            )

        # ── Step 1: P1 PBPK solve ───────────────────────────────
        pk_result: SimulationResult = self._pk.run(
            drug,
            schedule,
            physiology,
            clint_provider=clint_provider,
            gut_wall_mbi_exposures=gut_mbi_list if gut_mbi_list else None,
            gut_wall_induction_exposures=gut_ind_list if gut_ind_list else None,
            oatp1b1_inhibitors=oatp_list if oatp_list else None,
            gut_transit=gut_transit,
            gfr_ml_per_min=gfr_ml_per_min,
        )

        # ── Step 2: Assemble DDI summary if applicable ──────────
        if any_ddi:
            auc_ddi = pk_result.pk_metrics.auc_0_inf_mg_h_per_L
            auc_ratio = (
                auc_ddi / baseline_auc
                if baseline_auc is not None and baseline_auc > 0
                else None
            )
            # Probe CLint scaling at the first dose time for diagnostics.
            clint_scale_probe: Optional[float] = None
            if clint_provider is not None:
                try:
                    first_dose_time = (
                        schedule.events[0].time_h
                        if schedule.events
                        else 0.0
                    )
                    clint_eff = clint_provider(first_dose_time)
                    base_clint = self._hepatic.compute(
                        drug, physiology
                    ).clint_scaled_L_per_h
                    clint_scale_probe = (
                        clint_eff / base_clint
                        if base_clint > 0
                        else None
                    )
                except Exception:  # pragma: no cover — diagnostic only
                    clint_scale_probe = None

            # Merge perpetrator identities across hepatic, gut-wall,
            # and OATP1B1 lists, tagging each with its mechanism. A
            # single drug co-dosed can appear on multiple lists (e.g.
            # clarithromycin is hepatic-MBI AND gut-wall-MBI, rifampicin
            # is hepatic-induction AND OATP1B1-inhibitor, etc).
            all_perp_names: List[str] = []
            all_modes: List[str] = []
            for exp in perp_list:
                all_perp_names.append(exp.perpetrator.name)
                all_modes.append(_mode_label(exp))
            for exp in gut_mbi_list:
                name = exp.perpetrator.name + " (gut-wall MBI)"
                all_perp_names.append(name)
                all_modes.append("gut_wall_mbi")
            for exp in gut_ind_list:
                name = exp.perpetrator.name + " (gut-wall induction)"
                all_perp_names.append(name)
                all_modes.append("gut_wall_induction")
            for exp in oatp_list:
                name = exp.perpetrator.name + " (OATP1B1)"
                all_perp_names.append(name)
                all_modes.append("oatp1b1_inhibition")

            ddi_summary = DDISummary(
                victim_name=drug.name,
                perpetrators=tuple(all_perp_names),
                n_perpetrators=len(all_perp_names),
                modes=tuple(all_modes),
                auc_victim_ddi_mg_h_per_L=round(auc_ddi, 6),
                auc_victim_baseline_mg_h_per_L=(
                    round(baseline_auc, 6)
                    if baseline_auc is not None
                    else None
                ),
                auc_ratio=(
                    round(auc_ratio, 4)
                    if auc_ratio is not None
                    else None
                ),
                clint_scaling_at_dose_time=clint_scale_probe,
            )

        # ── Step 3: PD layer if profile attached ────────────────
        clinical_outcome: Optional[ClinicalOutcome] = None
        if drug.pd_profile is not None and drug.pd_profile.has_any_pd_data:
            clinical_outcome = self._pd.run(
                drug_name=pk_result.drug_name,
                time_h=pk_result.time_h,
                plasma_conc_mg_per_L=pk_result.plasma_conc_mg_per_L,
                pd_profile=drug.pd_profile,
            )

        # ── Step 4: Compose V2 result ───────────────────────────
        absorption_payload = (
            pk_result.absorption_profiles
            if pk_result.absorption_profiles
            else None
        )

        return SimulationResultV2(
            pk_result=pk_result,
            clinical_outcome=clinical_outcome,
            absorption=absorption_payload,
            ddi=ddi_summary,
            engine_version=ENGINE_VERSION,
        )


def _mode_label(exposure: PerpetratorExposure) -> str:
    """Classify an exposure's mechanism label for the DDISummary."""
    if exposure.enzyme_trace is not None:
        inh = exposure.perpetrator.inhibition
        if getattr(inh, "is_mbi", False):
            return "mbi"
        return "induction"
    return "reversible"
