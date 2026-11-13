"""
P5 — Population & Patient Variability Acceptance Gate
========================================================
Hard V1 quality gate for the P5 patient layer (CYP phenotype, hepatic
impairment, renal impairment, smoker induction, age ontogeny) and the
Monte Carlo population sampler / runner.

This file stays **integration-focused**: every test goes through the
real :class:`OrchestratorV2` pipeline with a patient supplied. That is
the only way to catch wiring mistakes like the pre-P5.6 bug where the
orchestrator computed renal clearance with the patient GFR but the PBPK
solver silently recomputed it at the reference 120 mL/min.

Gate criteria
-------------

1. **Patient adapter pure-function sanity** — the four scale factors
   behave as documented (no-op on ``patient=None``, correct direction
   on each axis).
2. **Directional integration checks** — each axis must move the AUC
   the right way by a clinically meaningful amount:

   * CYP3A4 PM × midazolam         → AUCR ≥ 3
   * CYP3A4 UM × midazolam         → AUCR ≤ 0.9
   * CYP2C19 PM × omeprazole       → AUCR ≥ 2
   * Child-Pugh C × midazolam      → AUCR ≥ 1.5
   * CKD4 (GFR ≈ 30) × metformin   → AUCR ≥ 1.8
   * Smoker × caffeine             → AUCR ≤ 0.9

The upper bounds (UM, smoker) are looser than their lower-bound
counterparts because the V1 simplified plasma mixing pool buffers fold
*decreases* more than fold *increases* for the same reason the P3 gate
calls out for high-extraction victims: tissue → plasma redistribution
becomes the rate-limiting step. This is a real engine limitation that
should flip when P6 adds a mass-balance venous compartment.

3. **Backward compatibility** — ``patient=None`` must reproduce legacy
   reference-male behavior bit-for-bit.
4. **Monte Carlo** — seeded reproducibility, non-degenerate AUC
   envelope when CYP3A4 variability is enabled, frozen result objects.
5. **Engine version** — tag stamped as ``v1.0.0-p5``.

Why these victims
-----------------

Each victim is chosen so that one and only one axis dominates the
prediction, which makes the directional test a clean signal of whether
that axis is wired correctly:

* **midazolam** is the canonical CYP3A4 probe (fm_3A4 ≈ 0.96).
* **omeprazole** is 80% CYP2C19 — the classic 2C19 phenotype probe.
* **metformin** is 100% renal, 0% CYP — isolates the GFR path.
* **caffeine** is 95% CYP1A2 — isolates the smoker induction path.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Tuple

import pytest

from neogen.core.constants.reference_physiology import build_reference_male
from neogen.core.entities.patient import (
    CYPPhenotype,
    GeneticProfile,
    OrganFunction,
    PatientProfile,
    Sex,
)
from neogen.engine.dose_scheduler import DoseScheduler
from neogen.engine.drug_factory import DrugFactory
from neogen.engine.orchestrator_v2 import ENGINE_VERSION, OrchestratorV2
from neogen.engine.population import (
    age_cyp_maturation_factor,
    apply_patient_to_drug,
    effective_cyp_activity_factor,
    gfr_for_patient,
    hepatic_impairment_factor,
    smoker_cyp1a2_factor,
    total_clint_scale,
)
from neogen.engine.population.population_sampler import (
    PopulationRunner,
    PopulationSampler,
    PopulationSpec,
)


# ════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def engine() -> OrchestratorV2:
    return OrchestratorV2()


@pytest.fixture(scope="module")
def factory() -> DrugFactory:
    return DrugFactory()


@pytest.fixture(scope="module")
def physiology():
    return build_reference_male()


@pytest.fixture(scope="module")
def scheduler() -> DoseScheduler:
    return DoseScheduler()


def _reference_patient(**overrides) -> PatientProfile:
    """Healthy 35-year-old reference male, minus whatever is overridden."""
    base = dict(
        weight_kg=70.0,
        height_cm=176.0,
        age_years=35.0,
        sex=Sex.MALE,
        organ_function=OrganFunction(),
        genetics=GeneticProfile(),
        is_smoker=False,
    )
    base.update(overrides)
    return PatientProfile(**base)


def _auc(result) -> float:
    return result.pk_result.pk_metrics.auc_0_inf_mg_h_per_L


# ════════════════════════════════════════════════════════════════
# UNIT — patient adapter pure functions
# ════════════════════════════════════════════════════════════════


class TestPatientAdapterPureFunctions:
    """Every factor must be a no-op on ``None`` and move the right way."""

    def test_none_patient_is_identity(self, factory: DrugFactory) -> None:
        drug = factory.create("midazolam")
        assert total_clint_scale(drug, None) == 1.0
        assert effective_cyp_activity_factor(drug, None) == 1.0
        assert hepatic_impairment_factor(None) == 1.0
        assert smoker_cyp1a2_factor(drug, None) == 1.0
        assert age_cyp_maturation_factor(drug, None) == 1.0
        assert gfr_for_patient(None) == pytest.approx(120.0)

    def test_cyp3a4_pm_on_midazolam(self, factory: DrugFactory) -> None:
        drug = factory.create("midazolam")  # fm_3A4 ≈ 0.96
        patient = _reference_patient(
            genetics=GeneticProfile(cyp3a4=CYPPhenotype.POOR)
        )
        factor = effective_cyp_activity_factor(drug, patient)
        # factor = 0.96·0 + 0.04 ≈ 0.04
        assert factor < 0.1, f"PM 3A4 on midazolam should crater CLint, got {factor}"

    def test_cyp3a4_um_on_midazolam(self, factory: DrugFactory) -> None:
        drug = factory.create("midazolam")
        patient = _reference_patient(
            genetics=GeneticProfile(cyp3a4=CYPPhenotype.ULTRA_RAPID)
        )
        factor = effective_cyp_activity_factor(drug, patient)
        # factor = 0.96·2 + 0.04 ≈ 1.96
        assert factor > 1.8

    def test_hepatic_impairment_monotonic(self) -> None:
        child_a = _reference_patient(organ_function=OrganFunction(hepatic=0.7))
        child_b = _reference_patient(organ_function=OrganFunction(hepatic=0.5))
        child_c = _reference_patient(organ_function=OrganFunction(hepatic=0.3))
        assert (
            hepatic_impairment_factor(child_a)
            > hepatic_impairment_factor(child_b)
            > hepatic_impairment_factor(child_c)
        )
        assert hepatic_impairment_factor(child_c) == pytest.approx(0.3)

    def test_smoker_cyp1a2_induction_caffeine(
        self, factory: DrugFactory
    ) -> None:
        caffeine = factory.create("caffeine")  # fm_1A2 ≈ 0.95
        smoker = _reference_patient(is_smoker=True)
        non_smoker = _reference_patient(is_smoker=False)
        assert smoker_cyp1a2_factor(caffeine, smoker) > 1.5
        assert smoker_cyp1a2_factor(caffeine, non_smoker) == 1.0

    def test_age_maturation_envelope(self, factory: DrugFactory) -> None:
        drug = factory.create("midazolam")
        infant = _reference_patient(age_years=0.5)
        toddler = _reference_patient(age_years=3.0)
        adult = _reference_patient(age_years=35.0)
        elderly = _reference_patient(age_years=80.0)
        assert age_cyp_maturation_factor(drug, infant) == pytest.approx(0.40)
        assert age_cyp_maturation_factor(drug, toddler) == pytest.approx(0.70)
        assert age_cyp_maturation_factor(drug, adult) == pytest.approx(1.00)
        assert age_cyp_maturation_factor(drug, elderly) == pytest.approx(0.70)

    def test_gfr_derived_from_renal_score(self) -> None:
        ckd4 = _reference_patient(
            organ_function=OrganFunction(renal=0.25)  # → 30 mL/min
        )
        assert gfr_for_patient(ckd4) == pytest.approx(30.0)

    def test_apply_patient_preserves_other_fields(
        self, factory: DrugFactory
    ) -> None:
        drug = factory.create("midazolam")
        patient = _reference_patient(
            genetics=GeneticProfile(cyp3a4=CYPPhenotype.POOR)
        )
        scaled = apply_patient_to_drug(drug, patient)
        # CLint scaled, everything else unchanged
        assert scaled.name == drug.name
        assert scaled.fu_plasma == drug.fu_plasma
        assert scaled.molecular_weight == drug.molecular_weight
        assert scaled.clint_microsomal < drug.clint_microsomal


# ════════════════════════════════════════════════════════════════
# INTEGRATION — full pipeline directional tests
# ════════════════════════════════════════════════════════════════


class TestPatientIntegration:
    """Each axis must move AUC the right way through the real engine."""

    def test_reference_patient_matches_none(
        self,
        engine: OrchestratorV2,
        factory: DrugFactory,
        scheduler: DoseScheduler,
        physiology,
    ) -> None:
        """patient=None and a default reference patient give the same AUC."""
        drug = factory.create("midazolam")
        schedule = scheduler.iv_bolus("midazolam", dose_mg=5.0)
        none_res = engine.run(drug, schedule, physiology)
        ref = _reference_patient()
        ref_res = engine.run(drug, schedule, physiology, patient=ref)
        assert _auc(ref_res) == pytest.approx(_auc(none_res), rel=1e-6)

    def test_cyp3a4_pm_midazolam_aucr(
        self,
        engine: OrchestratorV2,
        factory: DrugFactory,
        scheduler: DoseScheduler,
        physiology,
    ) -> None:
        drug = factory.create("midazolam")
        schedule = scheduler.iv_bolus("midazolam", dose_mg=5.0)
        ref = engine.run(drug, schedule, physiology)
        pm = engine.run(
            drug, schedule, physiology,
            patient=_reference_patient(
                genetics=GeneticProfile(cyp3a4=CYPPhenotype.POOR)
            ),
        )
        aucr = _auc(pm) / _auc(ref)
        assert aucr >= 3.0, f"CYP3A4 PM × midazolam AUCR {aucr:.2f} < 3.0"

    def test_cyp3a4_um_midazolam_aucr(
        self,
        engine: OrchestratorV2,
        factory: DrugFactory,
        scheduler: DoseScheduler,
        physiology,
    ) -> None:
        drug = factory.create("midazolam")
        schedule = scheduler.iv_bolus("midazolam", dose_mg=5.0)
        ref = engine.run(drug, schedule, physiology)
        um = engine.run(
            drug, schedule, physiology,
            patient=_reference_patient(
                genetics=GeneticProfile(cyp3a4=CYPPhenotype.ULTRA_RAPID)
            ),
        )
        aucr = _auc(um) / _auc(ref)
        assert aucr <= 0.9, f"CYP3A4 UM × midazolam AUCR {aucr:.2f} > 0.9"

    def test_cyp2c19_pm_omeprazole_aucr(
        self,
        engine: OrchestratorV2,
        factory: DrugFactory,
        scheduler: DoseScheduler,
        physiology,
    ) -> None:
        drug = factory.create("omeprazole")  # fm_2C19 ≈ 0.80
        schedule = scheduler.iv_bolus("omeprazole", dose_mg=40.0)
        ref = engine.run(drug, schedule, physiology)
        pm = engine.run(
            drug, schedule, physiology,
            patient=_reference_patient(
                genetics=GeneticProfile(cyp2c19=CYPPhenotype.POOR)
            ),
        )
        aucr = _auc(pm) / _auc(ref)
        assert aucr >= 2.0, f"CYP2C19 PM × omeprazole AUCR {aucr:.2f} < 2.0"

    def test_child_pugh_c_midazolam_aucr(
        self,
        engine: OrchestratorV2,
        factory: DrugFactory,
        scheduler: DoseScheduler,
        physiology,
    ) -> None:
        drug = factory.create("midazolam")
        schedule = scheduler.iv_bolus("midazolam", dose_mg=5.0)
        ref = engine.run(drug, schedule, physiology)
        cpc = engine.run(
            drug, schedule, physiology,
            patient=_reference_patient(
                organ_function=OrganFunction(hepatic=0.3)
            ),
        )
        aucr = _auc(cpc) / _auc(ref)
        assert aucr >= 1.5, f"Child-Pugh C × midazolam AUCR {aucr:.2f} < 1.5"

    def test_ckd4_metformin_aucr(
        self,
        engine: OrchestratorV2,
        factory: DrugFactory,
        scheduler: DoseScheduler,
        physiology,
    ) -> None:
        drug = factory.create("metformin")
        schedule = scheduler.iv_bolus("metformin", dose_mg=500.0)
        ref = engine.run(drug, schedule, physiology)
        ckd = engine.run(
            drug, schedule, physiology,
            patient=_reference_patient(
                organ_function=OrganFunction(renal=0.25)  # GFR ≈ 30
            ),
        )
        aucr = _auc(ckd) / _auc(ref)
        assert aucr >= 1.8, f"CKD4 × metformin AUCR {aucr:.2f} < 1.8"

    def test_smoker_caffeine_aucr(
        self,
        engine: OrchestratorV2,
        factory: DrugFactory,
        scheduler: DoseScheduler,
        physiology,
    ) -> None:
        drug = factory.create("caffeine")
        schedule = scheduler.iv_bolus("caffeine", dose_mg=200.0)
        ref = engine.run(drug, schedule, physiology)
        smoker = engine.run(
            drug, schedule, physiology,
            patient=_reference_patient(is_smoker=True),
        )
        aucr = _auc(smoker) / _auc(ref)
        assert aucr <= 0.9, f"Smoker × caffeine AUCR {aucr:.2f} > 0.9"


# ════════════════════════════════════════════════════════════════
# MONTE CARLO — sampler + runner
# ════════════════════════════════════════════════════════════════


class TestPopulationSampler:
    def test_seed_reproducibility(self) -> None:
        sampler = PopulationSampler()
        spec = PopulationSpec(n_patients=20, seed=1234)
        a = sampler.sample(spec)
        b = sampler.sample(spec)
        assert len(a) == len(b) == 20
        for pa, pb in zip(a, b):
            assert pa.age_years == pb.age_years
            assert pa.weight_kg == pb.weight_kg
            assert pa.sex == pb.sex
            assert pa.is_smoker == pb.is_smoker
            assert pa.genetics.cyp2d6 == pb.genetics.cyp2d6
            assert pa.genetics.cyp2c19 == pb.genetics.cyp2c19

    def test_different_seeds_give_different_cohorts(self) -> None:
        sampler = PopulationSampler()
        a = sampler.sample(PopulationSpec(n_patients=20, seed=1))
        b = sampler.sample(PopulationSpec(n_patients=20, seed=2))
        # At least one patient should differ (probabilistically near-certain).
        ages_a = tuple(p.age_years for p in a)
        ages_b = tuple(p.age_years for p in b)
        assert ages_a != ages_b

    def test_age_clipping(self) -> None:
        sampler = PopulationSampler()
        spec = PopulationSpec(
            n_patients=100,
            mean_age_years=40.0,
            age_sd_years=100.0,  # massive SD to force clipping
            min_age_years=18.0,
            max_age_years=85.0,
            seed=7,
        )
        patients = sampler.sample(spec)
        for p in patients:
            assert 18.0 <= p.age_years <= 85.0

    def test_weight_within_physiological_bounds(self) -> None:
        sampler = PopulationSampler()
        spec = PopulationSpec(n_patients=50, seed=99)
        for p in sampler.sample(spec):
            assert 40.0 <= p.weight_kg <= 200.0

    def test_smoker_prevalence_rough(self) -> None:
        sampler = PopulationSampler()
        spec = PopulationSpec(
            n_patients=500, fraction_smoker=0.20, seed=42
        )
        smokers = sum(1 for p in sampler.sample(spec) if p.is_smoker)
        # 500 × 0.2 = 100 expected. Allow ±5%.
        assert 75 <= smokers <= 125


class TestPopulationRunner:
    """End-to-end Monte Carlo through the full engine."""

    def test_runner_returns_frozen_result(
        self,
        factory: DrugFactory,
        scheduler: DoseScheduler,
        physiology,
    ) -> None:
        drug = factory.create("midazolam")
        schedule = scheduler.iv_bolus("midazolam", dose_mg=5.0)
        runner = PopulationRunner()
        pop = runner.run(
            drug, schedule, physiology,
            spec=PopulationSpec(n_patients=10, seed=0),
        )
        assert pop.n == 10
        assert len(pop.patients) == 10
        assert len(pop.results) == 10
        assert pop.cmax.median > 0
        assert pop.auc.median > 0
        # Frozen
        with pytest.raises(Exception):
            pop.cmax.median = 0.0  # type: ignore[misc]

    def test_runner_produces_nondegenerate_envelope_under_cyp3a4_variability(
        self,
        factory: DrugFactory,
        scheduler: DoseScheduler,
        physiology,
    ) -> None:
        """With CYP3A4 phenotype variability enabled, midazolam AUC
        fold span (P95/P5) should be > 3 across a 40-patient trial.
        Without variability the default table leaves 3A4 at NORMAL,
        which is what ``test_runner_returns_frozen_result`` exercises.
        """
        drug = factory.create("midazolam")
        schedule = scheduler.iv_bolus("midazolam", dose_mg=5.0)
        spec = PopulationSpec(
            n_patients=40,
            seed=2026,
            phenotype_freqs={
                "CYP3A4": {
                    CYPPhenotype.POOR: 0.10,
                    CYPPhenotype.INTERMEDIATE: 0.30,
                    CYPPhenotype.NORMAL: 0.50,
                    CYPPhenotype.ULTRA_RAPID: 0.10,
                },
            },
        )
        pop = PopulationRunner().run(drug, schedule, physiology, spec=spec)
        assert pop.auc.range_fold >= 3.0, (
            f"AUC P95/P5 = {pop.auc.range_fold:.2f}× "
            f"— expected ≥ 3× under CYP3A4 variability"
        )

    def test_runner_seed_reproducibility_end_to_end(
        self,
        factory: DrugFactory,
        scheduler: DoseScheduler,
        physiology,
    ) -> None:
        drug = factory.create("midazolam")
        schedule = scheduler.iv_bolus("midazolam", dose_mg=5.0)
        spec = PopulationSpec(n_patients=15, seed=777)
        runner = PopulationRunner()
        a = runner.run(drug, schedule, physiology, spec=spec)
        b = runner.run(drug, schedule, physiology, spec=spec)
        assert a.auc.median == pytest.approx(b.auc.median, rel=1e-12)
        assert a.cmax.median == pytest.approx(b.cmax.median, rel=1e-12)


# ════════════════════════════════════════════════════════════════
# ENGINE VERSION TAG
# ════════════════════════════════════════════════════════════════


class TestEngineVersionTag:
    def test_version_bumped_to_p5(
        self,
        engine: OrchestratorV2,
        factory: DrugFactory,
        scheduler: DoseScheduler,
        physiology,
    ) -> None:
        assert ENGINE_VERSION == "v1.2.0"
        drug = factory.create("midazolam")
        schedule = scheduler.iv_bolus("midazolam", dose_mg=5.0)
        result = engine.run(drug, schedule, physiology)
        assert result.engine_version == "v1.2.0"
