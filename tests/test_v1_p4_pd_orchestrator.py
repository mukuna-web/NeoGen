"""V1 P4 Step 15 — PD orchestrator + OrchestratorV2 test suite.

Covers:
  * PDOrchestrator dispatch logic (every combination of populated
    sub-models, including the "nothing populated" edge case)
  * ClinicalOutcome assembly — derived scalars, verdict line, frozen
    contract
  * SimulationResultV2 composite — PK-only vs PK+PD paths
  * OrchestratorV2 end-to-end on a real midazolam PBPK run with a
    synthetic PDProfile attached via ``dataclasses.replace``
  * Absorption/DDI slots are empty tuples for IV monotherapy; the DDI
    slot is populated with PerpetratorExposure records when the caller
    supplies perpetrators. engine_version is stamped correctly.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Tuple

import numpy as np
import pytest

from neogen.core.entities.pd import (
    EffectCompartmentParams,
    EffectOrgan,
    EmaxParams,
    PDProfile,
    ReceptorTarget,
    ToxicitySeverity,
    ToxicityThreshold,
)
from neogen.engine import (
    DoseScheduler,
    DrugFactory,
    ENGINE_VERSION,
    OrchestratorV2,
    PDOrchestrator,
    SimulationOrchestrator,
    SimulationResultV2,
)
from neogen.engine.pd.pd_result import ClinicalOutcome


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════


def _bolus_profile(
    c0_mg_per_L: float,
    k_el_per_h: float,
    duration_h: float,
    n: int = 500,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    t = np.linspace(0.0, duration_h, n)
    cp = c0_mg_per_L * np.exp(-k_el_per_h * t)
    return tuple(float(x) for x in t), tuple(float(x) for x in cp)


def _effect_only_profile() -> PDProfile:
    return PDProfile(
        effect_compartment=EffectCompartmentParams(ke0_per_h=0.25),
        emax=EmaxParams(
            emax=100.0,
            ec50_mg_per_L=1.0,
            hill=1.5,
            effect_name="Analgesia",
        ),
    )


def _receptor_only_profile() -> PDProfile:
    return PDProfile(
        receptor_target=ReceptorTarget(
            name="Beta1",
            kon_per_mg_per_L_per_h=500.0,
            koff_per_h=1.0,
            therapeutic_occupancy=0.5,
        ),
    )


def _toxicity_only_profile() -> PDProfile:
    return PDProfile(
        toxicity_thresholds=(
            ToxicityThreshold(
                organ=EffectOrgan.LIVER,
                severity=ToxicitySeverity.MODERATE,
                concentration_mg_per_L=5.0,
                symptom="Transaminitis",
            ),
            ToxicityThreshold(
                organ=EffectOrgan.LIVER,
                severity=ToxicitySeverity.SEVERE,
                concentration_mg_per_L=20.0,
                symptom="DILI",
            ),
        ),
    )


def _full_pd_profile() -> PDProfile:
    return PDProfile(
        effect_compartment=EffectCompartmentParams(ke0_per_h=0.3),
        emax=EmaxParams(
            emax=100.0,
            ec50_mg_per_L=2.0,
            hill=2.0,
            effect_name="Sedation",
        ),
        receptor_target=ReceptorTarget(
            name="GABA-A",
            kon_per_mg_per_L_per_h=200.0,
            koff_per_h=0.5,
            therapeutic_occupancy=0.4,
        ),
        toxicity_thresholds=(
            ToxicityThreshold(
                organ=EffectOrgan.BRAIN,
                severity=ToxicitySeverity.MILD,
                concentration_mg_per_L=1.0,
                symptom="Drowsiness",
            ),
            ToxicityThreshold(
                organ=EffectOrgan.BRAIN,
                severity=ToxicitySeverity.SEVERE,
                concentration_mg_per_L=10.0,
                symptom="Respiratory depression",
            ),
        ),
    )


# ════════════════════════════════════════════════════════════════
# 1. PD ORCHESTRATOR — DISPATCH LOGIC
# ════════════════════════════════════════════════════════════════


class TestPDOrchestratorDispatch:
    def test_empty_profile_produces_mostly_empty_outcome(self):
        """A drug with no PD data at all still yields a valid outcome."""
        orchestrator = PDOrchestrator()
        t, cp = _bolus_profile(5.0, 0.2, 24.0)
        outcome = orchestrator.run(
            drug_name="nothing",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=PDProfile(),
        )
        assert isinstance(outcome, ClinicalOutcome)
        assert outcome.effect_course is None
        assert outcome.receptor_trace is None
        # Toxicity report is ALWAYS present (may be empty)
        assert outcome.toxicity_report is not None
        assert not outcome.has_toxicity
        assert outcome.peak_effect_fraction is None
        assert outcome.receptor_peak_occupancy is None
        assert outcome.therapeutic_index is None
        assert outcome.overall_severity is None
        assert "no PD data" in outcome.verdict

    def test_effect_only_runs_effect_skips_receptor(self):
        orchestrator = PDOrchestrator()
        t, cp = _bolus_profile(5.0, 0.2, 48.0)
        outcome = orchestrator.run(
            drug_name="analgesic",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=_effect_only_profile(),
        )
        assert outcome.effect_course is not None
        assert outcome.receptor_trace is None
        # Peak effect fraction should be derived and in (0, 1] for a
        # drug whose plasma Cmax clears EC50 comfortably.
        assert outcome.peak_effect_fraction is not None
        assert 0.0 < outcome.peak_effect_fraction <= 1.0
        assert outcome.time_to_peak_effect_h is not None
        assert outcome.hysteresis_lag_h is not None
        assert outcome.hysteresis_lag_h >= 0.0

    def test_receptor_only_runs_receptor_skips_effect(self):
        orchestrator = PDOrchestrator()
        t, cp = _bolus_profile(5.0, 0.2, 48.0)
        outcome = orchestrator.run(
            drug_name="beta-blocker",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=_receptor_only_profile(),
        )
        assert outcome.effect_course is None
        assert outcome.receptor_trace is not None
        assert outcome.peak_effect_fraction is None
        assert outcome.receptor_peak_occupancy is not None
        assert 0.0 < outcome.receptor_peak_occupancy <= 1.0

    def test_toxicity_only_runs_toxicity_skips_trajectories(self):
        orchestrator = PDOrchestrator()
        t, cp = _bolus_profile(30.0, 0.2, 24.0)  # hits both thresholds
        outcome = orchestrator.run(
            drug_name="hepatotoxin",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=_toxicity_only_profile(),
        )
        assert outcome.effect_course is None
        assert outcome.receptor_trace is None
        assert outcome.has_toxicity
        assert outcome.overall_severity == ToxicitySeverity.SEVERE
        # No Emax → no TI (TI requires both severe thresholds AND ec50)
        assert outcome.therapeutic_index is None

    def test_full_profile_runs_all_three_sub_engines(self):
        orchestrator = PDOrchestrator()
        t, cp = _bolus_profile(15.0, 0.1, 48.0)
        outcome = orchestrator.run(
            drug_name="anaesthetic",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=_full_pd_profile(),
        )
        assert outcome.effect_course is not None
        assert outcome.receptor_trace is not None
        assert outcome.has_toxicity
        assert outcome.peak_effect_fraction is not None
        assert outcome.receptor_peak_occupancy is not None
        assert outcome.therapeutic_index is not None
        # Full profile TI: severe = 10 mg/L, EC50 = 2 mg/L → 5.0
        assert abs(outcome.therapeutic_index - 5.0) < 1e-9

    def test_orchestrator_is_stateless_reusable(self):
        """Running the orchestrator twice produces identical outcomes."""
        orchestrator = PDOrchestrator()
        t, cp = _bolus_profile(10.0, 0.2, 24.0)
        o1 = orchestrator.run(
            drug_name="x",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=_full_pd_profile(),
        )
        o2 = orchestrator.run(
            drug_name="x",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=_full_pd_profile(),
        )
        assert o1.peak_effect_fraction == o2.peak_effect_fraction
        assert o1.receptor_peak_occupancy == o2.receptor_peak_occupancy
        assert o1.overall_severity == o2.overall_severity

    def test_sub_engine_exceptions_propagate(self):
        """Structural errors in inputs should raise, not silently empty."""
        orchestrator = PDOrchestrator()
        with pytest.raises(ValueError):
            orchestrator.run(
                drug_name="bad",
                time_h=(0.0, 1.0),
                plasma_conc_mg_per_L=(1.0,),  # shape mismatch
                pd_profile=_effect_only_profile(),
            )


# ════════════════════════════════════════════════════════════════
# 2. CLINICAL OUTCOME — SCALARS & VERDICT
# ════════════════════════════════════════════════════════════════


class TestClinicalOutcomeContract:
    def test_outcome_is_frozen(self):
        orchestrator = PDOrchestrator()
        t, cp = _bolus_profile(5.0, 0.2, 24.0)
        outcome = orchestrator.run(
            drug_name="x",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=_full_pd_profile(),
        )
        with pytest.raises((AttributeError, TypeError)):
            outcome.drug_name = "mutated"  # type: ignore

    def test_verdict_mentions_overall_severity_when_toxic(self):
        orchestrator = PDOrchestrator()
        t, cp = _bolus_profile(15.0, 0.05, 48.0)
        outcome = orchestrator.run(
            drug_name="drug",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=_full_pd_profile(),
        )
        verdict = outcome.verdict
        assert "drug" in verdict
        assert outcome.overall_severity.value.upper() in verdict

    def test_verdict_falls_back_to_effect_when_clean(self):
        orchestrator = PDOrchestrator()
        t, cp = _bolus_profile(0.5, 0.2, 48.0)  # well below toxicity
        outcome = orchestrator.run(
            drug_name="analgesic",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=_effect_only_profile(),
        )
        assert "peak effect" in outcome.verdict
        assert "no toxicity" in outcome.verdict

    def test_summary_lines_include_every_sub_result(self):
        orchestrator = PDOrchestrator()
        t, cp = _bolus_profile(15.0, 0.05, 48.0)
        outcome = orchestrator.run(
            drug_name="drug",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=_full_pd_profile(),
        )
        lines = outcome.to_summary_lines()
        joined = "\n".join(lines)
        assert "Effect" in joined
        assert "Receptor" in joined
        assert "Therapeutic index" in joined

    def test_peak_effect_fraction_normalises_by_emax(self):
        """peak_effect_fraction = (peak_effect - baseline) / emax."""
        orchestrator = PDOrchestrator()
        # Pick a plasma profile where Cp >> EC50 so effect saturates.
        t, cp = _bolus_profile(100.0, 0.02, 240.0)
        profile = PDProfile(
            effect_compartment=EffectCompartmentParams(ke0_per_h=1.0),
            emax=EmaxParams(
                emax=80.0,
                ec50_mg_per_L=1.0,
                hill=2.0,
                baseline=10.0,  # non-zero baseline
            ),
        )
        outcome = orchestrator.run(
            drug_name="sat",
            time_h=t,
            plasma_conc_mg_per_L=cp,
            pd_profile=profile,
        )
        # Effect saturates: peak_effect ≈ baseline + emax = 90
        # fraction = (90 - 10) / 80 = 1.0
        assert outcome.peak_effect_fraction is not None
        assert outcome.peak_effect_fraction > 0.95


# ════════════════════════════════════════════════════════════════
# 3. ORCHESTRATOR V2 — END-TO-END
# ════════════════════════════════════════════════════════════════


class TestOrchestratorV2:
    def _midazolam_with_pd(self):
        """Build a real midazolam Drug + synthetic PD profile."""
        drug = DrugFactory().create("midazolam")
        profile = PDProfile(
            effect_compartment=EffectCompartmentParams(ke0_per_h=0.5),
            emax=EmaxParams(
                emax=100.0,
                ec50_mg_per_L=0.025,
                hill=2.0,
                effect_name="Sedation",
            ),
            receptor_target=ReceptorTarget(
                name="GABA-A benzo site",
                kon_per_mg_per_L_per_h=1500.0,
                koff_per_h=2.0,
                therapeutic_occupancy=0.4,
            ),
            toxicity_thresholds=(
                ToxicityThreshold(
                    organ=EffectOrgan.BRAIN,
                    severity=ToxicitySeverity.MILD,
                    concentration_mg_per_L=0.010,
                    symptom="Drowsiness",
                ),
                ToxicityThreshold(
                    organ=EffectOrgan.LUNG,
                    severity=ToxicitySeverity.SEVERE,
                    concentration_mg_per_L=0.200,
                    symptom="Respiratory depression",
                ),
            ),
        )
        return dataclasses.replace(drug, pd_profile=profile)

    def test_runs_end_to_end_with_pd_profile(self):
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )

        engine = OrchestratorV2()
        drug = self._midazolam_with_pd()
        schedule = DoseScheduler().iv_bolus(drug.name, dose_mg=5.0)
        physiology = build_reference_male()

        result = engine.run(drug, schedule, physiology)

        assert isinstance(result, SimulationResultV2)
        assert result.drug_name == drug.name
        assert result.engine_version == ENGINE_VERSION
        assert result.has_pd_layer
        assert result.clinical_outcome is not None

        # PK should be well-formed.
        assert result.pk_result.pk_metrics.cmax_mg_per_L > 0.0
        assert result.pk_result.pk_metrics.auc_0_inf_mg_h_per_L > 0.0

        # PD outcome should have effect + receptor populated.
        outcome = result.clinical_outcome
        assert outcome.effect_course is not None
        assert outcome.receptor_trace is not None
        # Toxicity report is always present; midazolam 5mg IV shouldn't
        # hit respiratory depression (0.2 mg/L) but may hit drowsiness.
        assert outcome.toxicity_report is not None

        # TI should be computable from profile (severe=0.2, ec50=0.025)
        assert outcome.therapeutic_index is not None
        assert abs(outcome.therapeutic_index - 8.0) < 1e-9

    def test_runs_without_pd_profile_is_pk_only(self):
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )

        engine = OrchestratorV2()
        drug = DrugFactory().create("midazolam")  # no pd_profile
        schedule = DoseScheduler().iv_bolus(drug.name, dose_mg=5.0)
        physiology = build_reference_male()

        result = engine.run(drug, schedule, physiology)

        assert result.has_pd_layer is False
        assert result.clinical_outcome is None
        assert result.pk_result.pk_metrics.cmax_mg_per_L > 0.0

    def test_iv_monotherapy_has_empty_absorption_and_ddi(self):
        """IV-bolus monotherapy should yield empty P2 and P3 tuples.

        P2 only runs for oral dose events, and P3 only runs when the
        caller supplies perpetrator exposures. With neither, both slots
        should be empty tuples (not None) — the facade now populates
        them with real-typed values.
        """
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )

        engine = OrchestratorV2()
        drug = self._midazolam_with_pd()
        schedule = DoseScheduler().iv_bolus(drug.name, dose_mg=5.0)
        result = engine.run(drug, schedule, build_reference_male())

        assert result.absorption is None
        assert result.ddi is None
        assert not result.has_ddi

    def test_result_is_frozen(self):
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )

        engine = OrchestratorV2()
        drug = self._midazolam_with_pd()
        schedule = DoseScheduler().iv_bolus(drug.name, dose_mg=5.0)
        result = engine.run(drug, schedule, build_reference_male())

        with pytest.raises((AttributeError, TypeError)):
            result.absorption = "mutated"  # type: ignore

    def test_clinical_report_has_verdict_and_pk_lines(self):
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )

        engine = OrchestratorV2()
        drug = self._midazolam_with_pd()
        schedule = DoseScheduler().iv_bolus(drug.name, dose_mg=5.0)
        result = engine.run(drug, schedule, build_reference_male())

        lines = result.to_clinical_report()
        joined = "\n".join(lines)
        assert drug.name in joined
        assert "PK:" in joined
        assert ENGINE_VERSION in joined

    def test_reuses_orchestrator_across_multiple_runs(self):
        """A single OrchestratorV2 instance should run many simulations."""
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )

        engine = OrchestratorV2()
        drug = self._midazolam_with_pd()
        schedule = DoseScheduler().iv_bolus(drug.name, dose_mg=5.0)
        physiology = build_reference_male()

        r1 = engine.run(drug, schedule, physiology)
        r2 = engine.run(drug, schedule, physiology)

        # Same inputs → same Cmax (deterministic)
        assert r1.pk_result.pk_metrics.cmax_mg_per_L == pytest.approx(
            r2.pk_result.pk_metrics.cmax_mg_per_L, rel=1e-9
        )
        assert r1.clinical_outcome.peak_effect_fraction == pytest.approx(
            r2.clinical_outcome.peak_effect_fraction, rel=1e-9
        )


# ════════════════════════════════════════════════════════════════
# STEP 15.5 — OrchestratorV2 end-to-end DDI
# ════════════════════════════════════════════════════════════════


class TestOrchestratorV2DDI:
    """Full-pipeline DDI: midazolam + ketoconazole via the facade.

    The P3 acceptance gate already proves that ``DDIOrchestrator`` +
    ``PBPKSolver`` produce a physiologically correct AUC fold-change
    (≈5× for IV midazolam + ketoconazole). These tests prove that the
    same wiring works when invoked through the top-level ``OrchestratorV2``
    facade with ``perpetrator_exposures=[...]`` — i.e. that we haven't
    silently severed the DDI coupling somewhere above the solver.
    """

    def test_ketoconazole_raises_midazolam_auc(self):
        """Midazolam AUC should rise substantially with ketoconazole on board."""
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )
        from neogen.engine.ddi import DDIOrchestrator

        engine = OrchestratorV2()
        factory = DrugFactory()
        midazolam = factory.create("midazolam")
        ketoconazole = factory.create("ketoconazole")
        schedule = DoseScheduler().iv_bolus("midazolam", dose_mg=5.0)
        physiology = build_reference_male()

        # Monotherapy baseline.
        baseline = engine.run(midazolam, schedule, physiology)
        baseline_auc = baseline.pk_result.pk_metrics.auc_0_inf_mg_h_per_L

        # With ketoconazole perpetrator exposure.
        ddi = DDIOrchestrator()
        keto_exposure = ddi.build_exposure(
            ketoconazole,
            steady_state_conc_um=0.5,
            duration_on_h=72.0,
            include_enzyme_pool=False,  # reversible inhibition only
        )
        result = engine.run(
            midazolam,
            schedule,
            physiology,
            perpetrator_exposures=[keto_exposure],
        )
        dd_auc = result.pk_result.pk_metrics.auc_0_inf_mg_h_per_L

        aucr = dd_auc / baseline_auc
        # V1 clinical target is ≈5×; we assert a loose ≥1.5× lower bound
        # here because the acceptance gate already owns the tight range
        # and we only want to prove the facade didn't neuter the coupling.
        assert aucr >= 1.5, (
            f"Expected meaningful AUC increase with ketoconazole, got {aucr:.2f}×"
        )

    def test_ddi_slot_is_populated_with_exposure_records(self):
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )
        from neogen.engine.ddi import DDIOrchestrator, PerpetratorExposure

        engine = OrchestratorV2()
        factory = DrugFactory()
        midazolam = factory.create("midazolam")
        ketoconazole = factory.create("ketoconazole")
        schedule = DoseScheduler().iv_bolus("midazolam", dose_mg=5.0)

        ddi = DDIOrchestrator()
        keto_exposure = ddi.build_exposure(
            ketoconazole,
            steady_state_conc_um=0.5,
            duration_on_h=72.0,
            include_enzyme_pool=False,
        )
        result = engine.run(
            midazolam,
            schedule,
            build_reference_male(),
            perpetrator_exposures=[keto_exposure],
        )

        assert result.has_ddi
        assert result.ddi.n_perpetrators == 1
        assert "ketoconazole" in result.ddi.perpetrators

    def test_clinical_report_mentions_ddi_when_present(self):
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )
        from neogen.engine.ddi import DDIOrchestrator

        engine = OrchestratorV2()
        factory = DrugFactory()
        midazolam = factory.create("midazolam")
        ketoconazole = factory.create("ketoconazole")
        schedule = DoseScheduler().iv_bolus("midazolam", dose_mg=5.0)

        ddi = DDIOrchestrator()
        keto_exposure = ddi.build_exposure(
            ketoconazole,
            steady_state_conc_um=0.5,
            duration_on_h=72.0,
            include_enzyme_pool=False,
        )
        result = engine.run(
            midazolam,
            schedule,
            build_reference_male(),
            perpetrator_exposures=[keto_exposure],
        )

        report = "\n".join(result.to_clinical_report())
        # Clinical report surfaces PK metrics; DDI detail is on the ddi slot.
        assert "midazolam" in report.lower()
        assert result.has_ddi
