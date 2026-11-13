"""V1 P4 — Core PD entity tests.

Covers the frozen dataclasses defined in ``neogen/core/entities/pd.py``:
construction, validation, immutability, severity ordering, the
``PDProfile`` aggregate container, and the ``Drug.pd_profile`` extension.

These are parameter-only entities — no physics is exercised here. Physics
tests live alongside the engine modules (P4 step 12+).
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from neogen.core.entities import (
    Drug,
    BCSClass,
    ClinicalEffect,
    EffectCompartmentParams,
    EffectOrgan,
    EmaxParams,
    PDProfile,
    ReceptorTarget,
    ToxicityThreshold,
    ToxicitySeverity,
)


# ────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────


@pytest.fixture
def morphine_effect_comp() -> EffectCompartmentParams:
    return EffectCompartmentParams(
        ke0_per_h=0.12,
        t_peak_lag_h=0.75,
        description="Opioid receptor equilibration lag",
    )


@pytest.fixture
def morphine_emax() -> EmaxParams:
    return EmaxParams(
        emax=100.0,
        ec50_mg_per_L=0.025,
        hill=2.5,
        baseline=0.0,
        effect_name="Pain relief",
        effect_unit="%",
    )


@pytest.fixture
def aspirin_receptor() -> ReceptorTarget:
    return ReceptorTarget(
        name="COX-1",
        kon_per_mg_per_L_per_h=50.0,
        koff_per_h=0.0,
        irreversible=True,
        recovery_half_life_h=240.0,  # ~10-day platelet turnover
        therapeutic_occupancy=0.95,
    )


@pytest.fixture
def theophylline_thresholds() -> tuple[ToxicityThreshold, ...]:
    return (
        ToxicityThreshold(
            organ=EffectOrgan.GI,
            severity=ToxicitySeverity.MILD,
            concentration_mg_per_L=15.0,
            symptom="Nausea",
        ),
        ToxicityThreshold(
            organ=EffectOrgan.BRAIN,
            severity=ToxicitySeverity.MODERATE,
            concentration_mg_per_L=20.0,
            symptom="Tremor",
        ),
        ToxicityThreshold(
            organ=EffectOrgan.HEART,
            severity=ToxicitySeverity.SEVERE,
            concentration_mg_per_L=30.0,
            symptom="Tachyarrhythmia",
        ),
        ToxicityThreshold(
            organ=EffectOrgan.BRAIN,
            severity=ToxicitySeverity.CRITICAL,
            concentration_mg_per_L=50.0,
            symptom="Seizure",
        ),
    )


# ────────────────────────────────────────────────────────────────
# ToxicitySeverity
# ────────────────────────────────────────────────────────────────


class TestToxicitySeverity:
    def test_rank_ordering_is_monotonic(self) -> None:
        ranks = [
            ToxicitySeverity.MILD.rank,
            ToxicitySeverity.MODERATE.rank,
            ToxicitySeverity.SEVERE.rank,
            ToxicitySeverity.CRITICAL.rank,
        ]
        assert ranks == sorted(ranks)
        assert len(set(ranks)) == 4

    def test_all_severities_have_ranks(self) -> None:
        for s in ToxicitySeverity:
            assert isinstance(s.rank, int)
            assert s.rank >= 1


# ────────────────────────────────────────────────────────────────
# EffectCompartmentParams
# ────────────────────────────────────────────────────────────────


class TestEffectCompartmentParams:
    def test_construction_and_fields(
        self, morphine_effect_comp: EffectCompartmentParams
    ) -> None:
        assert morphine_effect_comp.ke0_per_h == pytest.approx(0.12)
        assert morphine_effect_comp.t_peak_lag_h == pytest.approx(0.75)
        assert "Opioid" in morphine_effect_comp.description

    def test_frozen(
        self, morphine_effect_comp: EffectCompartmentParams
    ) -> None:
        with pytest.raises(FrozenInstanceError):
            morphine_effect_comp.ke0_per_h = 0.5  # type: ignore[misc]

    def test_rejects_nonpositive_ke0(self) -> None:
        with pytest.raises(ValueError, match="ke0_per_h"):
            EffectCompartmentParams(ke0_per_h=0.0)
        with pytest.raises(ValueError, match="ke0_per_h"):
            EffectCompartmentParams(ke0_per_h=-0.1)


# ────────────────────────────────────────────────────────────────
# EmaxParams
# ────────────────────────────────────────────────────────────────


class TestEmaxParams:
    def test_construction(self, morphine_emax: EmaxParams) -> None:
        assert morphine_emax.emax == 100.0
        assert morphine_emax.ec50_mg_per_L == pytest.approx(0.025)
        assert morphine_emax.hill == pytest.approx(2.5)
        assert morphine_emax.effect_unit == "%"

    def test_frozen(self, morphine_emax: EmaxParams) -> None:
        with pytest.raises(FrozenInstanceError):
            morphine_emax.emax = 50.0  # type: ignore[misc]

    def test_rejects_nonpositive_ec50(self) -> None:
        with pytest.raises(ValueError, match="ec50_mg_per_L"):
            EmaxParams(emax=100.0, ec50_mg_per_L=0.0)

    def test_rejects_nonpositive_hill(self) -> None:
        with pytest.raises(ValueError, match="hill"):
            EmaxParams(emax=100.0, ec50_mg_per_L=1.0, hill=0.0)

    def test_default_baseline_and_hill(self) -> None:
        p = EmaxParams(emax=80.0, ec50_mg_per_L=2.0)
        assert p.baseline == 0.0
        assert p.hill == 1.0


# ────────────────────────────────────────────────────────────────
# ReceptorTarget
# ────────────────────────────────────────────────────────────────


class TestReceptorTarget:
    def test_reversible_kd_computed(self) -> None:
        t = ReceptorTarget(
            name="beta1",
            kon_per_mg_per_L_per_h=10.0,
            koff_per_h=2.0,
        )
        assert t.kd_mg_per_L == pytest.approx(0.2)

    def test_irreversible_requires_recovery(self) -> None:
        with pytest.raises(ValueError, match="recovery_half_life_h"):
            ReceptorTarget(
                name="COX-1",
                kon_per_mg_per_L_per_h=50.0,
                koff_per_h=0.0,
                irreversible=True,
            )

    def test_irreversible_ok_with_recovery(
        self, aspirin_receptor: ReceptorTarget
    ) -> None:
        assert aspirin_receptor.irreversible is True
        assert aspirin_receptor.recovery_half_life_h == pytest.approx(240.0)
        # kd is None because koff == 0
        assert aspirin_receptor.kd_mg_per_L is None

    def test_therapeutic_occupancy_bounds(self) -> None:
        with pytest.raises(ValueError, match="therapeutic_occupancy"):
            ReceptorTarget(
                name="x",
                kon_per_mg_per_L_per_h=1.0,
                koff_per_h=1.0,
                therapeutic_occupancy=1.5,
            )

    def test_frozen(self, aspirin_receptor: ReceptorTarget) -> None:
        with pytest.raises(FrozenInstanceError):
            aspirin_receptor.koff_per_h = 0.5  # type: ignore[misc]


# ────────────────────────────────────────────────────────────────
# ToxicityThreshold / ClinicalEffect
# ────────────────────────────────────────────────────────────────


class TestToxicityThreshold:
    def test_construction(self) -> None:
        th = ToxicityThreshold(
            organ=EffectOrgan.LIVER,
            severity=ToxicitySeverity.SEVERE,
            concentration_mg_per_L=12.5,
            symptom="Hepatocyte injury",
        )
        assert th.organ is EffectOrgan.LIVER
        assert th.severity.rank == 3

    def test_rejects_negative_concentration(self) -> None:
        with pytest.raises(ValueError, match="concentration_mg_per_L"):
            ToxicityThreshold(
                organ=EffectOrgan.BRAIN,
                severity=ToxicitySeverity.MILD,
                concentration_mg_per_L=-1.0,
                symptom="x",
            )


class TestClinicalEffect:
    def test_construction(self) -> None:
        e = ClinicalEffect(
            organ=EffectOrgan.HEART,
            severity=ToxicitySeverity.MODERATE,
            symptom="Palpitations",
            triggered_at_h=2.5,
            triggered_concentration_mg_per_L=8.1,
        )
        assert e.triggered_at_h == pytest.approx(2.5)
        assert e.severity is ToxicitySeverity.MODERATE


# ────────────────────────────────────────────────────────────────
# PDProfile aggregate
# ────────────────────────────────────────────────────────────────


class TestPDProfile:
    def test_empty_profile_has_no_data(self) -> None:
        p = PDProfile()
        assert p.has_any_pd_data is False
        assert p.has_effect_compartment is False
        assert p.has_emax_model is False
        assert p.has_receptor_kinetics is False
        assert p.has_toxicity_model is False

    def test_full_profile_flags(
        self,
        morphine_effect_comp: EffectCompartmentParams,
        morphine_emax: EmaxParams,
        aspirin_receptor: ReceptorTarget,
        theophylline_thresholds: tuple[ToxicityThreshold, ...],
    ) -> None:
        p = PDProfile(
            effect_compartment=morphine_effect_comp,
            emax=morphine_emax,
            receptor_target=aspirin_receptor,
            toxicity_thresholds=theophylline_thresholds,
            therapeutic_range_low_mg_per_L=10.0,
            therapeutic_range_high_mg_per_L=20.0,
        )
        assert p.has_any_pd_data is True
        assert p.has_effect_compartment is True
        assert p.has_emax_model is True
        assert p.has_receptor_kinetics is True
        assert p.has_toxicity_model is True

    def test_toxicity_threshold_tuple_is_immutable(
        self, theophylline_thresholds: tuple[ToxicityThreshold, ...]
    ) -> None:
        p = PDProfile(toxicity_thresholds=theophylline_thresholds)
        assert isinstance(p.toxicity_thresholds, tuple)
        with pytest.raises(FrozenInstanceError):
            p.toxicity_thresholds = ()  # type: ignore[misc]

    def test_max_severity_walks_threshold_list(
        self, theophylline_thresholds: tuple[ToxicityThreshold, ...]
    ) -> None:
        p = PDProfile(toxicity_thresholds=theophylline_thresholds)
        assert p.max_severity_at(5.0) is None
        assert p.max_severity_at(15.0) is ToxicitySeverity.MILD
        assert p.max_severity_at(22.0) is ToxicitySeverity.MODERATE
        assert p.max_severity_at(35.0) is ToxicitySeverity.SEVERE
        assert p.max_severity_at(100.0) is ToxicitySeverity.CRITICAL

    def test_therapeutic_range_validation(self) -> None:
        with pytest.raises(ValueError, match="therapeutic_range_low"):
            PDProfile(
                therapeutic_range_low_mg_per_L=20.0,
                therapeutic_range_high_mg_per_L=10.0,
            )

    def test_therapeutic_range_accepts_one_sided(self) -> None:
        # Setting only low or only high is OK (no comparison possible).
        PDProfile(therapeutic_range_low_mg_per_L=1.0)
        PDProfile(therapeutic_range_high_mg_per_L=100.0)

    def test_replace_creates_new_profile(
        self, morphine_emax: EmaxParams
    ) -> None:
        original = PDProfile(emax=morphine_emax)
        updated = replace(
            original,
            emax=EmaxParams(emax=80.0, ec50_mg_per_L=0.1),
        )
        assert original.emax.emax == 100.0
        assert updated.emax.emax == 80.0


# ────────────────────────────────────────────────────────────────
# Drug.pd_profile extension
# ────────────────────────────────────────────────────────────────


class TestDrugPDProfileField:
    @pytest.fixture
    def bare_drug(self) -> Drug:
        return Drug(
            name="TestDrug",
            smiles="CCO",
            molecular_weight=46.07,
            logp=-0.31,
            bcs_class=BCSClass.I,
        )

    def test_default_pd_profile_is_none(self, bare_drug: Drug) -> None:
        assert bare_drug.pd_profile is None
        assert bare_drug.has_pd_profile is False

    def test_with_overrides_sets_pd_profile(
        self,
        bare_drug: Drug,
        morphine_effect_comp: EffectCompartmentParams,
        morphine_emax: EmaxParams,
    ) -> None:
        pd = PDProfile(
            effect_compartment=morphine_effect_comp,
            emax=morphine_emax,
        )
        drug_with_pd = bare_drug.with_overrides(pd_profile=pd)
        assert drug_with_pd.pd_profile is pd
        assert drug_with_pd.has_pd_profile is True
        # Original remains unchanged (immutability).
        assert bare_drug.pd_profile is None

    def test_empty_pd_profile_does_not_count_as_pd_data(
        self, bare_drug: Drug
    ) -> None:
        drug_with_empty = bare_drug.with_overrides(pd_profile=PDProfile())
        assert drug_with_empty.pd_profile is not None
        assert drug_with_empty.has_pd_profile is False
