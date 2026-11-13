"""Tests for the P5 population Monte Carlo harness.

Covers:
  * Distribution validation (frequencies sum to 1.0, probabilities bounded)
  * Sampler reproducibility (same seed → same patients)
  * Scaling determinism (reference patient → identity scaling)
  * Allometric sanity (100 kg patient has larger organs than 70 kg)
  * Genotype scaling (all NM → factor=1; PM on dominant CYP → factor<1)
  * Population runner smoke test (5-patient Monte Carlo)
  * PercentileSummary math
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ════════════════════════════════════════════════════════════════
# DISTRIBUTIONS
# ════════════════════════════════════════════════════════════════


class TestDistributions:
    def test_default_frequencies_sum_to_one(self) -> None:
        from neogen.population.distributions import (
            CYP2C19_FREQUENCIES_US,
            CYP2C9_FREQUENCIES_US,
            CYP2D6_FREQUENCIES_US,
        )

        for freqs in (
            CYP2D6_FREQUENCIES_US,
            CYP2C19_FREQUENCIES_US,
            CYP2C9_FREQUENCIES_US,
        ):
            assert abs(sum(freqs.values()) - 1.0) < 1.0e-3

    def test_distributions_dataclass_validates(self) -> None:
        from neogen.core.entities.patient import CYPPhenotype
        from neogen.population.distributions import (
            PopulationDistributions,
        )

        # Default is valid
        PopulationDistributions()

        # Broken frequencies should raise
        bad = {CYPPhenotype.NORMAL: 0.3, CYPPhenotype.POOR: 0.2}
        with pytest.raises(ValueError):
            PopulationDistributions(cyp2d6_frequencies=bad)

    def test_sex_probability_bounded(self) -> None:
        from neogen.population.distributions import PopulationDistributions

        with pytest.raises(ValueError):
            PopulationDistributions(sex_male_probability=1.5)

    def test_weight_distribution_params(self) -> None:
        from neogen.population.distributions import WeightDistribution

        w = WeightDistribution(mean_kg=70.0, cv=0.15)
        # sigma_log > 0, mu_log such that exp(mu_log + sigma^2/2) ≈ mean
        mean_approx = math.exp(w.mu_log + 0.5 * w.sigma_log ** 2)
        assert mean_approx == pytest.approx(70.0, rel=1e-6)


# ════════════════════════════════════════════════════════════════
# SAMPLER REPRODUCIBILITY
# ════════════════════════════════════════════════════════════════


class TestSampler:
    def test_seed_reproducibility(self) -> None:
        from neogen.population.sampler import PopulationSampler

        pop1 = PopulationSampler().sample(n=20, seed=12345)
        pop2 = PopulationSampler().sample(n=20, seed=12345)

        assert len(pop1) == len(pop2) == 20
        for a, b in zip(pop1, pop2):
            assert a.weight_kg == b.weight_kg
            assert a.sex == b.sex
            assert a.genetics.cyp2d6 == b.genetics.cyp2d6
            assert a.age_years == b.age_years

    def test_different_seeds_different_populations(self) -> None:
        from neogen.population.sampler import PopulationSampler

        pop1 = PopulationSampler().sample(n=20, seed=1)
        pop2 = PopulationSampler().sample(n=20, seed=2)

        # At least some patients should differ
        diffs = sum(
            1
            for a, b in zip(pop1, pop2)
            if (a.weight_kg, a.sex) != (b.weight_kg, b.sex)
        )
        assert diffs > 0

    def test_weights_physiological(self) -> None:
        from neogen.population.sampler import PopulationSampler

        pop = PopulationSampler().sample(n=200, seed=99)
        weights = [p.weight_kg for p in pop]
        assert all(35.0 <= w <= 200.0 for w in weights)
        # Mean should be near 70 kg within a broad band
        mean_w = sum(weights) / len(weights)
        assert 60.0 < mean_w < 85.0

    def test_demographic_summary(self) -> None:
        from neogen.population.sampler import PopulationSampler

        pop = PopulationSampler().sample(n=100, seed=7)
        summary = pop.demographic_summary()
        assert summary["n"] == 100
        assert 0.3 <= summary["male_fraction"] <= 0.7
        assert 0.0 <= summary["cyp2d6_pm_fraction"] <= 0.25

    def test_invalid_n_raises(self) -> None:
        from neogen.population.sampler import PopulationSampler

        with pytest.raises(ValueError):
            PopulationSampler().sample(n=0)


# ════════════════════════════════════════════════════════════════
# SCALING — GENOTYPE
# ════════════════════════════════════════════════════════════════


class TestGenotypeScaling:
    def _mk_drug(self, fm_cyp3a4: float = 1.0, clint: float = 100.0):
        from neogen.core.entities.drug import (
            BCSClass,
            CYPMetabolismProfile,
            Drug,
            InductionProfile,
            InhibitionProfile,
            Ionization,
            TransporterAffinities,
        )

        return Drug(
            name="test_drug",
            smiles="C",
            molecular_weight=300.0,
            logp=2.0,
            tpsa=60.0,
            hbd=1,
            hba=3,
            rotatable_bonds=3,
            aromatic_rings=1,
            fu_plasma=0.1,
            blood_to_plasma_ratio=1.0,
            fraction_excreted_unchanged=0.0,
            bcs_class=BCSClass.I,
            ionization_class=Ionization.NEUTRAL,
            metabolism=CYPMetabolismProfile(cyp3a4=fm_cyp3a4),
            clint_microsomal=clint,
            inhibition=InhibitionProfile(),
            induction=InductionProfile(),
            transporters=TransporterAffinities(),
        )

    def _mk_patient(self, **genetics_overrides):
        from neogen.core.entities.patient import (
            CYPPhenotype,
            GeneticProfile,
            PatientProfile,
            Sex,
        )

        defaults = dict(
            cyp2d6=CYPPhenotype.NORMAL,
            cyp2c19=CYPPhenotype.NORMAL,
            cyp2c9=CYPPhenotype.NORMAL,
            cyp2b6=CYPPhenotype.NORMAL,
            cyp1a2=CYPPhenotype.NORMAL,
            cyp3a4=CYPPhenotype.NORMAL,
        )
        defaults.update(genetics_overrides)
        return PatientProfile(genetics=GeneticProfile(**defaults))

    def test_reference_patient_is_identity(self) -> None:
        from neogen.population.scaling import scale_drug_to_genotype

        drug = self._mk_drug(fm_cyp3a4=1.0, clint=100.0)
        patient = self._mk_patient()  # all NM
        scaled = scale_drug_to_genotype(drug, patient)
        # Returns the same object (no-op)
        assert scaled is drug

    def test_poor_metabolizer_scales_to_zero(self) -> None:
        from neogen.core.entities.patient import CYPPhenotype
        from neogen.population.scaling import scale_drug_to_genotype

        # fm_cyp3a4=1.0, non-CYP=0 → factor=0 → CLint drops to 0
        drug = self._mk_drug(fm_cyp3a4=1.0, clint=100.0)
        patient = self._mk_patient(cyp3a4=CYPPhenotype.POOR)
        scaled = scale_drug_to_genotype(drug, patient)
        assert scaled is not drug  # new object
        assert scaled.clint_microsomal == pytest.approx(0.0, abs=1e-9)

    def test_ultra_rapid_doubles_clint(self) -> None:
        from neogen.core.entities.patient import CYPPhenotype
        from neogen.population.scaling import (
            genotype_clint_multiplier,
            scale_drug_to_genotype,
        )

        drug = self._mk_drug(fm_cyp3a4=1.0, clint=100.0)
        patient = self._mk_patient(cyp3a4=CYPPhenotype.ULTRA_RAPID)
        factor = genotype_clint_multiplier(drug.metabolism, patient.genetics)
        assert factor == pytest.approx(2.0)
        scaled = scale_drug_to_genotype(drug, patient)
        assert scaled.clint_microsomal == pytest.approx(200.0)

    def test_mixed_metabolism_partial_effect(self) -> None:
        """fm_cyp3a4=0.7, fm_cyp2d6=0.3, cyp2d6=PM → factor=0.7×1 + 0.3×0 = 0.7."""
        from neogen.core.entities.drug import CYPMetabolismProfile
        from neogen.core.entities.patient import CYPPhenotype
        from neogen.population.scaling import genotype_clint_multiplier

        # Non-CYP contributes 0 since fm sums to 1.0
        meta = CYPMetabolismProfile(cyp3a4=0.7, cyp2d6=0.3)
        patient = self._mk_patient(cyp2d6=CYPPhenotype.POOR)
        factor = genotype_clint_multiplier(meta, patient.genetics)
        assert factor == pytest.approx(0.7, abs=1e-6)


# ════════════════════════════════════════════════════════════════
# SCALING — PHYSIOLOGY
# ════════════════════════════════════════════════════════════════


class TestPhysiologyScaling:
    def test_reference_patient_preserves_organ_volumes(self) -> None:
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )
        from neogen.core.entities.patient import PatientProfile
        from neogen.population.scaling import (
            scale_physiology_to_patient,
        )

        base = build_reference_male()
        # 70 kg, 30 y, male, normal organ function → identity
        patient = PatientProfile(weight_kg=70.0, age_years=30.0)
        scaled = scale_physiology_to_patient(base, patient)

        for o1, o2 in zip(base.all_organs, scaled.all_organs):
            assert o1.volume_L == pytest.approx(o2.volume_L, rel=1e-3)

    def test_heavier_patient_has_larger_organs(self) -> None:
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )
        from neogen.core.entities.patient import PatientProfile
        from neogen.population.scaling import (
            scale_physiology_to_patient,
        )

        base = build_reference_male()
        patient = PatientProfile(weight_kg=100.0, age_years=30.0)
        scaled = scale_physiology_to_patient(base, patient)

        # Liver volume should scale by 100/70 ≈ 1.43
        ratio = scaled.liver.volume_L / base.liver.volume_L
        assert ratio == pytest.approx(100.0 / 70.0, rel=1e-3)

        # Flows should scale by (100/70)^0.75 ≈ 1.31
        flow_ratio = scaled.liver.blood_flow_L_per_h / base.liver.blood_flow_L_per_h
        expected = (100.0 / 70.0) ** 0.75
        assert flow_ratio == pytest.approx(expected, rel=1e-3)

    def test_hepatic_impairment_reduces_liver_flow(self) -> None:
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )
        from neogen.core.entities.patient import (
            OrganFunction,
            PatientProfile,
        )
        from neogen.population.scaling import (
            scale_physiology_to_patient,
        )

        base = build_reference_male()
        patient = PatientProfile(
            weight_kg=70.0,
            age_years=30.0,
            organ_function=OrganFunction(hepatic=0.5),
        )
        scaled = scale_physiology_to_patient(base, patient)
        assert scaled.liver.blood_flow_L_per_h < base.liver.blood_flow_L_per_h * 0.6
        assert scaled.liver.volume_L < base.liver.volume_L * 0.6

    def test_elderly_patient_cardiac_output_declines(self) -> None:
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )
        from neogen.core.entities.patient import PatientProfile
        from neogen.population.scaling import (
            scale_physiology_to_patient,
        )

        base = build_reference_male()
        young = PatientProfile(weight_kg=70.0, age_years=30.0)
        old = PatientProfile(weight_kg=70.0, age_years=80.0)

        scaled_young = scale_physiology_to_patient(base, young)
        scaled_old = scale_physiology_to_patient(base, old)

        assert (
            scaled_old.liver.blood_flow_L_per_h
            < scaled_young.liver.blood_flow_L_per_h
        )

    def test_effective_gfr_declines_with_age_and_impairment(self) -> None:
        from neogen.core.entities.patient import (
            OrganFunction,
            PatientProfile,
        )
        from neogen.population.scaling import effective_gfr_ml_per_min

        young_normal = PatientProfile(age_years=30.0)
        old_impaired = PatientProfile(
            age_years=80.0,
            organ_function=OrganFunction(renal=0.5),
        )
        assert effective_gfr_ml_per_min(young_normal) == pytest.approx(120.0)
        gfr_old = effective_gfr_ml_per_min(old_impaired)
        assert gfr_old < 40.0  # 120 × 0.5 × age_factor
        assert gfr_old >= 5.0


# ════════════════════════════════════════════════════════════════
# PERCENTILE SUMMARY
# ════════════════════════════════════════════════════════════════


class TestPercentileSummary:
    def test_from_values_basic(self) -> None:
        from neogen.population.result import PercentileSummary

        vs = list(range(1, 101))  # 1..100
        s = PercentileSummary.from_values(vs)
        assert s.n == 100
        assert s.min_value == 1
        assert s.max_value == 100
        assert s.median == pytest.approx(50.5)

    def test_empty_cohort(self) -> None:
        from neogen.population.result import PercentileSummary

        s = PercentileSummary.from_values([])
        assert s.n == 0
        assert math.isnan(s.median)
        assert math.isnan(s.geomean)

    def test_filters_nonpositive_and_nan(self) -> None:
        from neogen.population.result import PercentileSummary

        s = PercentileSummary.from_values(
            [10.0, 20.0, 30.0, 0.0, -5.0, float("nan")]
        )
        assert s.n == 3


# ════════════════════════════════════════════════════════════════
# RUNNER SMOKE TESTS (require engine)
# ════════════════════════════════════════════════════════════════


def _engine_available() -> bool:
    try:
        from neogen.engine.orchestrator import (  # noqa: F401
            SimulationOrchestrator,
        )
        from neogen.engine.drug_factory import DrugFactory  # noqa: F401
    except Exception:
        return False
    return True


pytestmark_engine = pytest.mark.skipif(
    not _engine_available(),
    reason="V1 engine modules not importable",
)


@pytestmark_engine
class TestRunnerSmoke:
    def test_five_patient_iv_monte_carlo(self) -> None:
        """Smoke: 5-patient MC on midazolam 5mg IV should complete."""
        from neogen.population.runner import PopulationRunner
        from neogen.population.sampler import PopulationSampler

        pop = PopulationSampler().sample(n=5, seed=101)
        result = PopulationRunner().simulate(
            drug_name="midazolam",
            dose_mg=5.0,
            route="iv_bolus",
            population=pop,
        )
        assert result.n_requested == 5
        # At least some should complete cleanly
        assert result.n_completed >= 3
        # Summaries should compute
        cmax_summary = result.summary_for("cmax")
        assert cmax_summary.n >= 3
        assert cmax_summary.median > 0

    def test_population_result_markdown(self) -> None:
        from neogen.population.runner import PopulationRunner
        from neogen.population.sampler import PopulationSampler

        pop = PopulationSampler().sample(n=3, seed=202)
        result = PopulationRunner().simulate(
            drug_name="caffeine",
            dose_mg=100.0,
            route="iv_bolus",
            population=pop,
        )
        md = result.to_markdown()
        assert "Population Simulation" in md
        assert "caffeine" in md

    def test_invalid_route_raises(self) -> None:
        from neogen.population.runner import PopulationRunner
        from neogen.population.sampler import PopulationSampler

        pop = PopulationSampler().sample(n=1, seed=303)
        with pytest.raises(ValueError):
            PopulationRunner().simulate(
                drug_name="midazolam",
                dose_mg=5.0,
                route="intrathecal",
                population=pop,
            )
