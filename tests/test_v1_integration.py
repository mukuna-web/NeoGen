"""
V1 full-pipeline integration tests — Step 16 acceptance gate.

Exercises the complete P0 → P2 → P3 → P1 → P4 pipeline through
OrchestratorV2 and SimulationOrchestrator, verifying that all phases
compose correctly for representative drug scenarios:

1. **Oral monotherapy** — P0 + P2 + P1 (no DDI, no PD)
2. **IV monotherapy with PD** — P0 + P1 + P4
3. **DDI pair** — P0 + P2 + P3 + P1
4. **Full pipeline** — P0 + P2 + P3 + P1 + P4
5. **Bioavailability decomposition** — P2 standalone
6. **DDI result contract** — P3 frozen dataclass round-trip
7. **Mass balance invariants** — cross-phase conservation

These are smoke / invariant tests, not clinical accuracy gates.
The accuracy gates live in test_v1_p2_acceptance_gate.py and the
per-phase test files.
"""
from __future__ import annotations

import numpy as np
import pytest

from neogen.core.constants.reference_physiology import build_reference_male
from neogen.core.entities.drug import BCSClass
from neogen.core.entities.formulation import Route
from neogen.core.entities.pd import (
    EffectCompartmentParams,
    EffectOrgan,
    EmaxParams,
    PDProfile,
    ToxicitySeverity,
    ToxicityThreshold,
)
from neogen.engine.absorption import AbsorptionProfile, PermeabilityEngine
from neogen.engine.absorption.bioavailability import (
    BioavailabilityEngine,
    BioavailabilityReport,
    classify_bcs,
)
from neogen.engine.ddi import DDIOrchestrator, PerpetratorExposure
from neogen.engine.ddi.ddi_result import CLintBreakdown, DDISummary
from neogen.engine.dose_scheduler import DoseScheduler
from neogen.engine.drug_factory import DrugFactory
from neogen.engine.hepatic_clearance import HepaticClearanceEngine
from neogen.engine.orchestrator import SimulationOrchestrator
from neogen.engine.pbpk_solver import SolverConfig, build_portal_rate_provider


# ════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def factory():
    return DrugFactory()


@pytest.fixture(scope="module")
def scheduler():
    return DoseScheduler()


@pytest.fixture(scope="module")
def physiology():
    return build_reference_male()


@pytest.fixture(scope="module")
def orchestrator():
    return SimulationOrchestrator()


# ════════════════════════════════════════════════════════════════
# 1. ORAL MONOTHERAPY — P0 + P2 + P1
# ════════════════════════════════════════════════════════════════


class TestOralMonotherapy:
    """Full oral pipeline: drug factory → absorption → PBPK → PK metrics."""

    def test_paracetamol_oral_produces_valid_pk(
        self, factory, scheduler, physiology, orchestrator
    ):
        drug = factory.create("paracetamol")
        schedule = scheduler.single_oral("paracetamol", dose_mg=500)
        result = orchestrator.run(drug, schedule, physiology)

        assert result.pk_metrics.cmax_mg_per_L > 0
        assert result.pk_metrics.auc_0_inf_mg_h_per_L > 0
        assert result.pk_metrics.half_life_h > 0
        assert result.pk_metrics.tmax_h > 0

    def test_midazolam_oral_auc_positive(
        self, factory, scheduler, physiology, orchestrator
    ):
        drug = factory.create("midazolam")
        schedule = scheduler.single_oral("midazolam", dose_mg=7.5)
        result = orchestrator.run(drug, schedule, physiology)

        assert result.pk_metrics.auc_0_inf_mg_h_per_L > 0
        # Oral midazolam has low F — Cmax should be modest
        assert result.pk_metrics.cmax_mg_per_L < 5.0

    def test_oral_plasma_profile_rises_then_falls(
        self, factory, scheduler, physiology, orchestrator
    ):
        drug = factory.create("caffeine")
        schedule = scheduler.single_oral("caffeine", dose_mg=200)
        result = orchestrator.run(drug, schedule, physiology)

        plasma = result.plasma_conc_mg_per_L
        # Should rise then fall for oral
        peak_idx = np.argmax(plasma)
        assert peak_idx > 0, "Oral profile should not peak at t=0"
        assert plasma[-1] < plasma[peak_idx], "Should decay after peak"


# ════════════════════════════════════════════════════════════════
# 2. IV MONOTHERAPY + PD — P0 + P1 + P4
# ════════════════════════════════════════════════════════════════


class TestIVWithPD:
    """IV bolus through PBPK + PD layer."""

    def test_midazolam_iv_with_pd_profile(
        self, factory, scheduler, physiology
    ):
        drug = factory.create("midazolam")
        schedule = scheduler.iv_bolus("midazolam", dose_mg=5.0)

        # Attach a synthetic PD profile for test purposes
        pd_profile = PDProfile(
            effect_compartment=EffectCompartmentParams(ke0_per_h=0.5),
            emax=EmaxParams(emax=100.0, ec50_mg_per_L=0.05, hill=1.5),
            toxicity_thresholds=(
                ToxicityThreshold(
                    organ=EffectOrgan.BRAIN,
                    severity=ToxicitySeverity.MODERATE,
                    concentration_mg_per_L=1.0,
                    symptom="Excessive sedation",
                ),
            ),
        )
        from dataclasses import replace
        drug_with_pd = replace(drug, pd_profile=pd_profile)

        from neogen.engine.orchestrator_v2 import OrchestratorV2

        orch_v2 = OrchestratorV2()
        result = orch_v2.run(drug_with_pd, schedule, physiology)

        # PK present
        assert result.pk_result.pk_metrics.cmax_mg_per_L > 0

        # PD present (has_pd_layer should be True)
        assert result.has_pd_layer
        outcome = result.clinical_outcome
        assert outcome is not None
        assert outcome.effect_course is not None

    def test_iv_bolus_no_spike(
        self, factory, scheduler, physiology, orchestrator
    ):
        """IV bolus should start from Vss-distributed equilibrium, not spike."""
        drug = factory.create("midazolam")
        schedule = scheduler.iv_bolus("midazolam", dose_mg=5.0)
        result = orchestrator.run(drug, schedule, physiology)

        plasma = result.plasma_conc_mg_per_L
        # With mass-balance mode, C(0) should be reasonable (< 10 mg/L for 5mg dose)
        assert plasma[0] < 10.0, f"IV C(0)={plasma[0]:.1f} too high — spike bug"


# ════════════════════════════════════════════════════════════════
# 3. DDI PAIR — P0 + P2 + P3 + P1
# ════════════════════════════════════════════════════════════════


class TestDDIPair:
    """DDI simulation: ketoconazole + midazolam."""

    def test_ketoconazole_midazolam_auc_increases(
        self, factory, scheduler, physiology
    ):
        victim = factory.create("midazolam")
        perpetrator = factory.create("ketoconazole")

        # Baseline (no DDI)
        schedule = scheduler.single_oral("midazolam", dose_mg=7.5)
        orch = SimulationOrchestrator()
        baseline = orch.run(victim, schedule, physiology)

        # Get base CLint for the DDI provider
        hep = HepaticClearanceEngine()
        hep_result = hep.compute(victim, physiology)

        # With DDI
        ddi_orch = DDIOrchestrator()
        exposure = PerpetratorExposure(
            perpetrator=perpetrator,
            steady_state_conc_um=2.0,
            duration_on_h=24.0,
        )
        provider = ddi_orch.build_clint_provider(
            victim=victim,
            base_clint_scaled_L_per_h=hep_result.clint_scaled_L_per_h,
            exposures=[exposure],
        )
        ddi_result = orch.run(
            victim, schedule, physiology,
            clint_provider=provider,
        )

        # AUC should increase under CYP3A4 inhibition
        auc_base = baseline.pk_metrics.auc_0_inf_mg_h_per_L
        auc_ddi = ddi_result.pk_metrics.auc_0_inf_mg_h_per_L
        ratio = auc_ddi / auc_base if auc_base > 0 else 0

        assert ratio > 1.5, (
            f"Expected AUC increase, got ratio={ratio:.2f}"
        )

    def test_rifampicin_midazolam_auc_decreases(
        self, factory, scheduler, physiology
    ):
        victim = factory.create("midazolam")
        rifampicin = factory.create("rifampicin")

        schedule = scheduler.single_oral("midazolam", dose_mg=7.5)
        sim_orch = SimulationOrchestrator()
        baseline = sim_orch.run(victim, schedule, physiology)

        hep = HepaticClearanceEngine()
        hep_result = hep.compute(victim, physiology)

        ddi_orch = DDIOrchestrator()
        # build_exposure pre-computes the enzyme pool trace (induction ODE)
        exposure = ddi_orch.build_exposure(
            rifampicin,
            steady_state_conc_um=0.2,
            duration_on_h=14 * 24,
            include_enzyme_pool=True,
        )
        provider = ddi_orch.build_clint_provider(
            victim=victim,
            base_clint_scaled_L_per_h=hep_result.clint_scaled_L_per_h,
            exposures=[exposure],
        )
        # Sample CLint at plateau (day 13) and use as steady-state provider
        plateau_clint = provider(13 * 24)

        induced = sim_orch.run(
            victim, schedule, physiology,
            clint_provider=lambda _t: plateau_clint,
        )

        auc_base = baseline.pk_metrics.auc_0_inf_mg_h_per_L
        auc_ind = induced.pk_metrics.auc_0_inf_mg_h_per_L
        ratio = auc_ind / auc_base if auc_base > 0 else 999

        assert ratio < 0.85, (
            f"Expected AUC decrease from induction, got ratio={ratio:.2f}"
        )


# ════════════════════════════════════════════════════════════════
# 4. BIOAVAILABILITY MODULE
# ════════════════════════════════════════════════════════════════


class TestBioavailabilityModule:
    """Standalone bioavailability decomposition."""

    def test_classify_bcs_known_cases(self):
        # Caffeine: LogP ≈ -0.07, TPSA ≈ 58 → BCS I
        bcs, sol, perm = classify_bcs(logp=-0.07, tpsa=58.0)
        assert bcs == BCSClass.I
        assert sol == "high"
        assert perm == "high"

        # Metformin: LogP ≈ -1.4, TPSA ≈ 91 → BCS III
        bcs, sol, perm = classify_bcs(logp=-1.4, tpsa=91.0)
        assert bcs == BCSClass.III
        assert sol == "high"
        assert perm == "low"

    def test_mechanistic_report_from_profile(self, factory, physiology):
        drug = factory.create("paracetamol")

        # Run P2 absorption
        from neogen.core.entities.formulation import Formulation

        form = Formulation()
        engine = PermeabilityEngine()
        profile = engine.simulate(drug, form, dose_mg=500)

        # Get Fh from hepatic clearance
        hep = HepaticClearanceEngine()
        hep_result = hep.compute(drug, physiology)

        report = BioavailabilityEngine.from_profile(drug, profile, fh=hep_result.fh)

        assert isinstance(report, BioavailabilityReport)
        assert report.source == "mechanistic"
        assert 0 < report.fa <= 1
        assert 0 < report.fg <= 1
        assert 0 < report.fh <= 1
        assert abs(report.f_oral - report.fa * report.fg * report.fh) < 0.01

    def test_estimated_report_for_unknown(self, factory):
        drug = factory.create("midazolam")
        report = BioavailabilityEngine.estimate(drug, logp=3.0, tpsa=50.0, mw=326.0)

        assert isinstance(report, BioavailabilityReport)
        assert report.source == "estimated"
        assert 0 < report.f_oral <= 1

    def test_report_is_frozen(self, factory):
        drug = factory.create("caffeine")
        report = BioavailabilityEngine.estimate(drug)
        with pytest.raises(AttributeError):
            report.fa = 0.5  # type: ignore[misc]


# ════════════════════════════════════════════════════════════════
# 5. DDI RESULT CONTRACT
# ════════════════════════════════════════════════════════════════


class TestDDIResultContract:
    """Frozen DDI result dataclass construction and invariants."""

    def test_ddi_summary_construction(self):
        summary = DDISummary(
            victim_name="midazolam",
            perpetrator_names=("ketoconazole",),
            auc_fold_change=10.5,
            mechanism="CYP3A4 competitive inhibition",
            severity="major",
        )
        assert summary.n_perpetrators == 1
        assert summary.auc_fold_change == 10.5

    def test_ddi_summary_is_frozen(self):
        summary = DDISummary(
            victim_name="midazolam",
            perpetrator_names=("ketoconazole",),
            auc_fold_change=10.5,
            mechanism="CYP3A4",
            severity="major",
        )
        with pytest.raises(AttributeError):
            summary.auc_fold_change = 5.0  # type: ignore[misc]

    def test_clint_breakdown_construction(self):
        bd = CLintBreakdown(
            time_h=0.0,
            clint_base_L_per_h=50.0,
            clint_eff_L_per_h=5.0,
            scaling=0.1,
            cyp_scaling=0.1,
            uptake_correction=1.0,
            per_cyp_scaling=(("CYP3A4", 0.1),),
        )
        assert bd.scaling == 0.1


# ════════════════════════════════════════════════════════════════
# 6. MASS BALANCE INVARIANTS
# ════════════════════════════════════════════════════════════════


class TestMassBalanceInvariants:
    """Cross-phase conservation laws."""

    def test_auc_monotonic_in_dose(self, factory, physiology):
        """AUC should scale roughly linearly with dose (first-order kinetics)."""
        drug = factory.create("caffeine")
        orch = SimulationOrchestrator()
        scheduler = DoseScheduler()

        auc_100 = orch.run(
            drug, scheduler.single_oral("caffeine", dose_mg=100), physiology
        ).pk_metrics.auc_0_inf_mg_h_per_L

        auc_200 = orch.run(
            drug, scheduler.single_oral("caffeine", dose_mg=200), physiology
        ).pk_metrics.auc_0_inf_mg_h_per_L

        # AUC at 200mg should be roughly 2× AUC at 100mg (within 30%)
        ratio = auc_200 / auc_100 if auc_100 > 0 else 0
        assert 1.5 < ratio < 2.5, f"AUC dose proportionality broken: ratio={ratio:.2f}"

    def test_absorption_mass_balance(self, factory):
        """P2 absorption: dose = undissolved + dissolved + permeated + excreted."""
        drug = factory.create("paracetamol")
        from neogen.core.entities.formulation import Formulation

        form = Formulation()
        engine = PermeabilityEngine()
        profile = engine.simulate(drug, form, dose_mg=500)

        assert profile.mass_balance_error_mg < 1.0, (
            f"P2 mass balance error: {profile.mass_balance_error_mg:.3f} mg"
        )

    def test_pk_values_all_finite(self, factory, scheduler, physiology, orchestrator):
        """All PK metrics should be finite real numbers."""
        for name in ("paracetamol", "midazolam", "caffeine", "metformin"):
            drug = factory.create(name)
            schedule = scheduler.single_oral(name, dose_mg=100)
            result = orchestrator.run(drug, schedule, physiology)
            pk = result.pk_metrics

            assert np.isfinite(pk.cmax_mg_per_L), f"{name} Cmax not finite"
            assert np.isfinite(pk.auc_0_inf_mg_h_per_L), f"{name} AUC not finite"
            assert np.isfinite(pk.half_life_h), f"{name} t½ not finite"
            assert pk.cmax_mg_per_L > 0, f"{name} Cmax ≤ 0"
            assert pk.auc_0_inf_mg_h_per_L > 0, f"{name} AUC ≤ 0"
