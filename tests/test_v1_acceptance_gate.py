"""V1 acceptance gate — end-to-end via OrchestratorV2.

Purpose
-------
This is the *cross-phase* acceptance gate that closes the V1 engine.
It does **not** re-run the physics unit tests (each phase owns those)
and it does **not** duplicate the phase-level acceptance gates
(``test_v1_p2_acceptance_gate.py``, ``test_v1_p3_acceptance_gate.py``).

What it proves
--------------

1. ``OrchestratorV2`` produces clinically-plausible PK on a diverse
   multi-drug panel (mix of oral and IV, mix of hepatic and renal
   clearance, mix of BCS classes).
2. The P2 absorption layer is honestly wired into the facade: oral
   runs carry populated ``absorption_profiles``.
3. The P3 DDI layer is honestly wired into the facade: a known
   strong-inhibitor pair produces a meaningful AUC fold-change when
   the caller supplies perpetrator exposures.
4. The P4 PD layer is honestly wired into the facade: attaching a
   ``PDProfile`` to a drug entity produces a well-formed
   ``ClinicalOutcome`` with a non-trivial effect curve.

Acceptance metric
-----------------

For the pure-PK panel (no DDI, no PD), each drug contributes **two**
data points: ``Cmax`` and ``AUC_0_inf``. A data point passes if it
falls within a 2× envelope of the published clinical range.

Gate: **≥ 60% of data points within 2×** (matches the V1 freeze target
in the plan). At the time this gate was written, the engine clears the
bar by a comfortable margin on the current panel.

References
----------

Cmax / AUC clinical ranges are drawn from published single-dose PK
studies (citations on each entry). Ranges are deliberately wide — V1
is a mechanistic prediction tool, not a population model, so the test
honors the plan's 2× tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pytest

from neogen.core.constants.reference_physiology import build_reference_male
from neogen.core.entities.pd import (
    EffectCompartmentParams,
    EmaxParams,
    PDProfile,
)
from neogen.engine import (
    DoseScheduler,
    DrugFactory,
    ENGINE_VERSION,
    OrchestratorV2,
)
from neogen.engine.ddi import DDIOrchestrator


# ════════════════════════════════════════════════════════════════
# REFERENCE PANEL — pure PK (no DDI, no PD)
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PKReference:
    """One clinical reference entry.

    ``cmax_range_mg_per_L`` and ``auc_range_mg_h_per_L`` are published
    clinical ranges (typically single-dose, healthy volunteers). The
    acceptance envelope is then a 2× expansion on each side.
    """

    drug_name: str
    route: str                    # "oral" or "iv_bolus"
    dose_mg: float
    cmax_range_mg_per_L: Tuple[float, float]
    auc_range_mg_h_per_L: Tuple[float, float]
    end_time_h: Optional[float] = None
    notes: str = ""


REFERENCE_PANEL: Tuple[PKReference, ...] = (
    # ─── Oral — covers P2 absorption path ────────────────────────
    PKReference(
        drug_name="paracetamol",
        route="oral",
        dose_mg=1000.0,
        cmax_range_mg_per_L=(8.0, 20.0),
        auc_range_mg_h_per_L=(25.0, 80.0),
        notes="Rawlins 1977; Prescott 1980",
    ),
    PKReference(
        drug_name="ibuprofen",
        route="oral",
        dose_mg=400.0,
        cmax_range_mg_per_L=(20.0, 40.0),
        auc_range_mg_h_per_L=(60.0, 180.0),
        notes="Davies 1998",
    ),
    PKReference(
        drug_name="metformin",
        route="oral",
        dose_mg=500.0,
        cmax_range_mg_per_L=(1.0, 2.0),
        auc_range_mg_h_per_L=(6.0, 15.0),
        notes="Graham 2011 (renal clearance pathway)",
    ),
    PKReference(
        drug_name="midazolam",
        route="oral",
        dose_mg=7.5,
        cmax_range_mg_per_L=(0.03, 0.08),
        auc_range_mg_h_per_L=(0.10, 0.40),
        notes="Thummel 1996 (CYP3A4 gut + hepatic)",
    ),
    PKReference(
        drug_name="caffeine",
        route="oral",
        dose_mg=100.0,
        cmax_range_mg_per_L=(1.0, 3.0),
        auc_range_mg_h_per_L=(10.0, 40.0),
        notes="Blanchard 1983; Kaplan 1997",
    ),
    PKReference(
        drug_name="theophylline",
        route="oral",
        dose_mg=200.0,
        cmax_range_mg_per_L=(2.0, 6.0),
        auc_range_mg_h_per_L=(20.0, 80.0),
        notes="Upton 1990 (narrow TI target)",
    ),
    # ─── IV — bypasses P2, isolates disposition physics ──────────
    PKReference(
        drug_name="midazolam",
        route="iv_bolus",
        dose_mg=5.0,
        cmax_range_mg_per_L=(0.08, 0.40),
        auc_range_mg_h_per_L=(0.10, 0.60),
        notes="Heizmann 1983 (IV bolus reference)",
    ),
    PKReference(
        drug_name="caffeine",
        route="iv_bolus",
        dose_mg=100.0,
        cmax_range_mg_per_L=(1.5, 6.0),
        auc_range_mg_h_per_L=(15.0, 50.0),
        notes="Kaplan 1997",
    ),
    PKReference(
        drug_name="theophylline",
        route="iv_bolus",
        dose_mg=100.0,
        cmax_range_mg_per_L=(2.0, 8.0),
        auc_range_mg_h_per_L=(15.0, 80.0),
        notes="Upton 1990",
    ),
    PKReference(
        drug_name="metformin",
        route="iv_bolus",
        dose_mg=100.0,
        cmax_range_mg_per_L=(0.5, 4.0),
        auc_range_mg_h_per_L=(2.0, 15.0),
        notes="Tucker 1981 (renal elimination, high CL)",
    ),
)


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════


def _within_2x(value: float, lo: float, hi: float) -> bool:
    """Inclusive 2× envelope around a published clinical range."""
    return (lo / 2.0) <= value <= (hi * 2.0)


@dataclass(frozen=True)
class CheckResult:
    drug_name: str
    route: str
    cmax_actual: float
    cmax_pass: bool
    auc_actual: float
    auc_pass: bool

    @property
    def label(self) -> str:
        return f"{self.drug_name}_{self.route}"


def _build_schedule(scheduler: DoseScheduler, ref: PKReference):
    if ref.route == "oral":
        kwargs = {}
        if ref.end_time_h is not None:
            kwargs["end_time_h"] = ref.end_time_h
        return scheduler.single_oral(ref.drug_name, dose_mg=ref.dose_mg, **kwargs)
    if ref.route == "iv_bolus":
        kwargs = {}
        if ref.end_time_h is not None:
            kwargs["end_time_h"] = ref.end_time_h
        return scheduler.iv_bolus(ref.drug_name, dose_mg=ref.dose_mg, **kwargs)
    raise ValueError(f"Unknown route: {ref.route}")


def _run_panel() -> Tuple[CheckResult, ...]:
    engine = OrchestratorV2()
    factory = DrugFactory()
    scheduler = DoseScheduler()
    physiology = build_reference_male()

    results = []
    for ref in REFERENCE_PANEL:
        drug = factory.create(ref.drug_name)
        schedule = _build_schedule(scheduler, ref)
        sim = engine.run(drug, schedule, physiology)
        pk = sim.pk_result.pk_metrics

        results.append(
            CheckResult(
                drug_name=ref.drug_name,
                route=ref.route,
                cmax_actual=pk.cmax_mg_per_L,
                cmax_pass=_within_2x(pk.cmax_mg_per_L, *ref.cmax_range_mg_per_L),
                auc_actual=pk.auc_0_inf_mg_h_per_L,
                auc_pass=_within_2x(pk.auc_0_inf_mg_h_per_L, *ref.auc_range_mg_h_per_L),
            )
        )
    return tuple(results)


# ════════════════════════════════════════════════════════════════
# PK GATE
# ════════════════════════════════════════════════════════════════


class TestV1AcceptanceGatePK:
    """Pure-PK panel — 10 drug/route combinations × 2 metrics = 20 points."""

    def test_panel_has_expected_size(self):
        assert len(REFERENCE_PANEL) >= 10, (
            "V1 acceptance panel should contain at least 10 drug/route combos"
        )

    def test_sixty_percent_within_2x(self, capsys):
        """V1 freeze gate: ≥60% of (Cmax, AUC) data points within 2×."""
        results = _run_panel()

        total = len(results) * 2
        passes = sum(int(r.cmax_pass) + int(r.auc_pass) for r in results)
        pct = 100.0 * passes / total

        report_lines = [
            "",
            "== V1 acceptance gate — PK panel ==",
            f"Drugs/routes: {len(results)}   Metrics: {total}   "
            f"Within 2×: {passes}/{total} ({pct:.1f}%)",
            "",
        ]
        for r in results:
            report_lines.append(
                f"  {r.label:28s}  "
                f"Cmax={r.cmax_actual:10.4f} [{'OK' if r.cmax_pass else 'FAIL'}]  "
                f"AUC={r.auc_actual:10.4f} [{'OK' if r.auc_pass else 'FAIL'}]"
            )
        report_lines.append("")
        with capsys.disabled():
            print("\n".join(report_lines))

        assert pct >= 60.0, (
            f"V1 acceptance gate failed: only {pct:.1f}% of metrics "
            f"within 2× (need ≥60%)"
        )

    def test_every_run_produced_a_positive_cmax(self):
        """No drug should yield zero or negative Cmax — that signals
        an orchestration failure (unit bug, empty profile) rather than
        a mere accuracy miss."""
        results = _run_panel()
        for r in results:
            assert r.cmax_actual > 0.0, (
                f"{r.label}: Cmax is non-positive ({r.cmax_actual})"
            )
            assert r.auc_actual > 0.0, (
                f"{r.label}: AUC is non-positive ({r.auc_actual})"
            )


# ════════════════════════════════════════════════════════════════
# CROSS-PHASE SMOKE CHECKS
# ════════════════════════════════════════════════════════════════


class TestV1AcceptanceCrossPhase:
    """Proves every phase is actually wired through OrchestratorV2."""

    def test_oral_run_carries_absorption_profile(self):
        """P2 wiring: oral runs must surface absorption profiles."""
        engine = OrchestratorV2()
        factory = DrugFactory()
        scheduler = DoseScheduler()

        drug = factory.create("paracetamol")
        schedule = scheduler.single_oral("paracetamol", dose_mg=1000.0)
        result = engine.run(drug, schedule, build_reference_male())

        assert len(result.absorption_profiles) >= 1
        first = result.absorption_profiles[0]
        assert 0.0 < first.fa <= 1.0
        assert 0.0 < first.fg <= 1.0

    def test_ddi_coupling_raises_midazolam_auc(self):
        """P3 wiring: midazolam + ketoconazole AUC > monotherapy AUC."""
        engine = OrchestratorV2()
        factory = DrugFactory()
        scheduler = DoseScheduler()
        physiology = build_reference_male()

        midazolam = factory.create("midazolam")
        ketoconazole = factory.create("ketoconazole")
        schedule = scheduler.iv_bolus("midazolam", dose_mg=5.0)

        baseline = engine.run(midazolam, schedule, physiology)
        baseline_auc = baseline.pk_result.pk_metrics.auc_0_inf_mg_h_per_L

        ddi = DDIOrchestrator()
        keto_exposure = ddi.build_exposure(
            ketoconazole,
            steady_state_conc_um=0.5,
            duration_on_h=72.0,
            include_enzyme_pool=False,
        )
        ddi_result = engine.run(
            midazolam,
            schedule,
            physiology,
            perpetrator_exposures=[keto_exposure],
        )
        ddi_auc = ddi_result.pk_result.pk_metrics.auc_0_inf_mg_h_per_L

        aucr = ddi_auc / baseline_auc
        assert aucr >= 1.5, (
            f"DDI coupling severed? midazolam + ketoconazole AUCR={aucr:.2f} "
            f"(expected ≥1.5)"
        )
        assert ddi_result.has_ddi
        assert ddi_result.ddi.n_perpetrators == 1

    def test_pd_layer_runs_with_synthetic_profile(self):
        """P4 wiring: attaching a PDProfile yields a populated ClinicalOutcome."""
        import dataclasses

        engine = OrchestratorV2()
        factory = DrugFactory()
        scheduler = DoseScheduler()

        pd_profile = PDProfile(
            effect_compartment=EffectCompartmentParams(ke0_per_h=0.5),
            emax=EmaxParams(
                emax=100.0,
                ec50_mg_per_L=0.05,
                hill=1.0,
                effect_name="Sedation",
            ),
        )
        drug = factory.create("midazolam")
        drug_with_pd = dataclasses.replace(drug, pd_profile=pd_profile)
        schedule = scheduler.iv_bolus("midazolam", dose_mg=5.0)
        result = engine.run(drug_with_pd, schedule, build_reference_male())

        assert result.has_pd_layer
        assert result.clinical_outcome is not None
        assert result.clinical_outcome.effect_course is not None
        assert result.clinical_outcome.peak_effect_fraction is not None
        assert 0.0 <= result.clinical_outcome.peak_effect_fraction <= 1.0

    def test_engine_version_is_stamped(self):
        """Reproducibility: every result must carry the current engine tag."""
        engine = OrchestratorV2()
        factory = DrugFactory()
        scheduler = DoseScheduler()

        drug = factory.create("midazolam")
        schedule = scheduler.iv_bolus("midazolam", dose_mg=5.0)
        result = engine.run(drug, schedule, build_reference_male())

        assert result.engine_version == ENGINE_VERSION
        assert ENGINE_VERSION.startswith("v1.")

    def test_clinical_report_survives_all_layers(self):
        """to_clinical_report() must produce a layered text summary."""
        import dataclasses

        engine = OrchestratorV2()
        factory = DrugFactory()
        scheduler = DoseScheduler()

        pd_profile = PDProfile(
            effect_compartment=EffectCompartmentParams(ke0_per_h=0.5),
            emax=EmaxParams(
                emax=100.0,
                ec50_mg_per_L=0.05,
                hill=1.0,
                effect_name="Sedation",
            ),
        )
        drug = factory.create("midazolam")
        drug_with_pd = dataclasses.replace(drug, pd_profile=pd_profile)
        schedule = scheduler.single_oral("midazolam", dose_mg=7.5)
        result = engine.run(drug_with_pd, schedule, build_reference_male())

        lines = result.to_clinical_report()
        joined = "\n".join(lines)
        assert "midazolam" in joined
        assert "PK:" in joined
        assert ENGINE_VERSION in joined
