"""Tests for NeoGen V1 Core Entity System (Phase 0).

Validates that:
1. All entity dataclasses are constructible and immutable
2. DrugFactory produces correct Drug entities from literature
3. Enzyme database has scientifically valid parameters
4. PhysiologyModel has correct reference values
5. DoseScheduler produces valid schedules
6. DDI study design is correctly structured

Run: python -m pytest tests/test_v1_core.py -v
"""

import pytest
import math


# ═══════════════════════════════════════════════════════════════════
# ENTITY TESTS
# ═══════════════════════════════════════════════════════════════════


class TestDrugEntity:
    """Test Drug dataclass construction and properties."""

    def test_drug_creation_minimal(self):
        from neogen.core.entities.drug import Drug

        drug = Drug(name="test", smiles="C", molecular_weight=16.0, logp=0.0)
        assert drug.name == "test"
        assert drug.molecular_weight == 16.0

    def test_drug_is_immutable(self):
        from neogen.core.entities.drug import Drug

        drug = Drug(name="test", smiles="C", molecular_weight=16.0, logp=0.0)
        with pytest.raises(AttributeError):
            drug.name = "modified"

    def test_drug_with_overrides(self):
        from neogen.core.entities.drug import Drug

        original = Drug(
            name="test", smiles="C", molecular_weight=16.0,
            logp=0.0, fu_plasma=0.5,
        )
        modified = original.with_overrides(fu_plasma=0.05)

        assert original.fu_plasma == 0.5  # Original unchanged
        assert modified.fu_plasma == 0.05  # New copy modified
        assert modified.name == "test"  # Other fields preserved

    def test_fu_plasma_safe_never_zero(self):
        from neogen.core.entities.drug import Drug

        drug = Drug(
            name="test", smiles="C", molecular_weight=16.0,
            logp=0.0, fu_plasma=0.0,
        )
        assert drug.fu_plasma_safe >= 0.001

    def test_cyp_metabolism_profile(self):
        from neogen.core.entities.drug import CYPMetabolismProfile

        fm = CYPMetabolismProfile(cyp3a4=0.90, cyp2d6=0.05)
        assert fm.total_cyp_fm == pytest.approx(0.95)
        assert fm.non_cyp_fraction == pytest.approx(0.05)
        assert fm.dominant_enzyme == "CYP3A4"
        assert fm.fm_for("CYP3A4") == 0.90

    def test_inhibition_profile(self):
        from neogen.core.entities.drug import InhibitionProfile

        inh = InhibitionProfile(ki_cyp3a4=0.015)
        assert inh.is_inhibitor is True
        assert inh.ki_for("CYP3A4") == 0.015
        assert inh.ki_for("CYP2D6") is None

    def test_mbi_profile(self):
        from neogen.core.entities.drug import InhibitionProfile

        mbi = InhibitionProfile(
            ki_cyp3a4=3.3,
            kinact_cyp3a4=0.054,
            k_i_mbi_cyp3a4=5.49,
        )
        assert mbi.is_mbi is True


class TestEnzymeEntity:
    """Test Enzyme entity and EnzymePool."""

    def test_enzyme_creation(self):
        from neogen.core.entities.enzyme import Enzyme, EnzymeFamily

        e = Enzyme(
            name="CYP3A4",
            family=EnzymeFamily.CYP,
            abundance_liver_pmol_per_mg=137.0,
            kdeg_liver=0.0193,
        )
        assert e.name == "CYP3A4"
        assert e.half_life_liver_h == pytest.approx(0.693 / 0.0193, rel=0.01)

    def test_enzyme_pool_initialization(self):
        from neogen.core.entities.enzyme import (
            Enzyme, EnzymePool, EnzymeLocation,
        )

        enzyme = Enzyme(
            name="CYP3A4",
            abundance_liver_pmol_per_mg=137.0,
        )
        pool = EnzymePool(enzyme=enzyme, location=EnzymeLocation.LIVER)

        assert pool.current_amount_pmol_per_mg == 137.0
        assert pool.baseline_amount == 137.0
        assert pool.fraction_remaining == 1.0

    def test_enzyme_pool_inhibition(self):
        from neogen.core.entities.enzyme import (
            Enzyme, EnzymePool, EnzymeLocation,
        )

        enzyme = Enzyme(name="CYP3A4", abundance_liver_pmol_per_mg=100.0)
        pool = EnzymePool(enzyme=enzyme, location=EnzymeLocation.LIVER)
        pool.inhibition_factor = 0.5  # 50% competitively inhibited

        assert pool.effective_activity == 50.0


class TestPatientEntity:
    """Test PatientProfile construction and derived properties."""

    def test_default_patient(self):
        from neogen.core.entities.patient import PatientProfile, Sex

        p = PatientProfile()
        assert p.weight_kg == 70.0
        assert p.sex == Sex.MALE
        assert not p.is_pediatric
        assert not p.is_elderly

    def test_bmi_calculation(self):
        from neogen.core.entities.patient import PatientProfile

        p = PatientProfile(weight_kg=70.0, height_cm=176.0)
        expected_bmi = 70.0 / (1.76 ** 2)
        assert p.bmi == pytest.approx(expected_bmi, rel=0.01)

    def test_organ_function_validation(self):
        from neogen.core.entities.patient import OrganFunction

        with pytest.raises(ValueError):
            OrganFunction(hepatic=1.5)  # Out of range

    def test_genetic_profile(self):
        from neogen.core.entities.patient import GeneticProfile, CYPPhenotype

        g = GeneticProfile(cyp2d6=CYPPhenotype.POOR)
        assert g.activity_factor_for("CYP2D6") == 0.0
        assert g.activity_factor_for("CYP3A4") == 1.0  # Default normal


class TestPhysiologyModel:
    """Test PhysiologyModel reference values."""

    def test_reference_model_has_14_compartments(self):
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )

        model = build_reference_male()
        assert len(model.all_organs) == 14

    def test_cardiac_output_reasonable(self):
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )

        model = build_reference_male()
        co = model.cardiac_output_L_per_h
        # Cardiac output should be ~350-420 L/h (5.8-7.0 L/min)
        assert 300 < co < 450

    def test_portal_vein_anatomy(self):
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )

        model = build_reference_male()
        # Portal vein carries blood from gut + spleen + pancreas
        pv_organs = [o.name for o in model.portal_vein_organs]
        assert "gut" in pv_organs
        assert "spleen" in pv_organs
        assert "pancreas" in pv_organs

        # Hepatic artery flow = liver flow - portal vein flow
        ha = model.hepatic_artery_flow_L_per_h
        pv = model.portal_vein_flow_L_per_h
        assert ha + pv == pytest.approx(
            model.liver.blood_flow_L_per_h, rel=0.01
        )

    def test_liver_has_metabolic_capacity(self):
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )

        model = build_reference_male()
        assert model.liver.has_metabolic_capacity is True
        assert model.liver.mppgl == 40.0  # mg/g
        assert model.liver.hpgl == 120.0  # million/g

    def test_female_model_differs(self):
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
            build_reference_female,
        )

        male = build_reference_male()
        female = build_reference_female()

        # Female has more fat, less muscle
        assert female.fat.volume_L > male.fat.volume_L
        assert female.muscle.volume_L < male.muscle.volume_L
        # Lower cardiac output
        assert female.cardiac_output_L_per_h < male.cardiac_output_L_per_h


# ═══════════════════════════════════════════════════════════════════
# ENZYME DATABASE TESTS
# ═══════════════════════════════════════════════════════════════════


class TestEnzymeDatabase:
    """Validate curated enzyme parameters are scientifically correct."""

    def test_cyp3a4_is_most_abundant(self):
        from neogen.core.constants.enzyme_database import CYP3A4, CYP2D6

        assert CYP3A4.abundance_liver_pmol_per_mg > CYP2D6.abundance_liver_pmol_per_mg

    def test_cyp2d6_not_inducible(self):
        from neogen.core.constants.enzyme_database import CYP2D6

        assert CYP2D6.is_inducible is False
        assert CYP2D6.induction_receptor is None

    def test_cyp3a4_in_gut_and_liver(self):
        from neogen.core.constants.enzyme_database import CYP3A4
        from neogen.core.entities.enzyme import EnzymeLocation

        assert EnzymeLocation.LIVER in CYP3A4.locations
        assert EnzymeLocation.GUT_WALL in CYP3A4.locations
        assert CYP3A4.abundance_gut_pmol_per_mg > 0

    def test_enzyme_registry_complete(self):
        from neogen.core.constants.enzyme_database import ENZYME_REGISTRY

        required = ["CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2"]
        for name in required:
            assert name in ENZYME_REGISTRY

    def test_fm_values_sum_to_at_most_one(self):
        from neogen.core.constants.enzyme_database import FDA_INDEX_SUBSTRATES

        for drug, fm_dict in FDA_INDEX_SUBSTRATES.items():
            total = sum(fm_dict.values())
            assert total <= 1.01, (
                f"{drug} fm values sum to {total:.2f} (must be ≤ 1.0)"
            )

    def test_midazolam_is_pure_3a4(self):
        from neogen.core.constants.enzyme_database import FDA_INDEX_SUBSTRATES

        fm = FDA_INDEX_SUBSTRATES["midazolam"]
        assert fm["CYP3A4"] >= 0.90

    def test_ketoconazole_ki_is_potent(self):
        from neogen.core.constants.enzyme_database import FDA_INDEX_INHIBITORS

        ki = FDA_INDEX_INHIBITORS["ketoconazole"]["CYP3A4"]
        assert ki < 0.1  # Very potent (Ki < 0.1 µM)


# ═══════════════════════════════════════════════════════════════════
# DRUG FACTORY TESTS
# ═══════════════════════════════════════════════════════════════════


class TestDrugFactory:
    """Test DrugFactory produces correct Drug entities."""

    def test_create_from_literature(self):
        from neogen.engine.drug_factory import DrugFactory

        factory = DrugFactory()
        drug = factory.create("midazolam")

        assert drug.name == "midazolam"
        assert drug.molecular_weight == pytest.approx(325.8, rel=0.01)
        assert drug.fu_plasma == pytest.approx(0.032, rel=0.01)
        assert drug.source == "literature"

    def test_create_ketoconazole_is_inhibitor(self):
        from neogen.engine.drug_factory import DrugFactory

        factory = DrugFactory()
        drug = factory.create("ketoconazole")

        assert drug.inhibition.is_inhibitor is True
        assert drug.inhibition.ki_for("CYP3A4") == pytest.approx(0.015)

    def test_create_midazolam_is_3a4_substrate(self):
        from neogen.engine.drug_factory import DrugFactory

        factory = DrugFactory()
        drug = factory.create("midazolam")

        assert drug.metabolism.cyp3a4 == pytest.approx(0.96)
        assert drug.is_cyp3a4_substrate is True

    def test_create_with_overrides(self):
        from neogen.engine.drug_factory import DrugFactory

        factory = DrugFactory()
        drug = factory.create("midazolam", overrides={"fu_plasma": 0.10})

        assert drug.fu_plasma == 0.10
        assert drug.source == "measured"

    def test_create_pair(self):
        from neogen.engine.drug_factory import DrugFactory

        factory = DrugFactory()
        perp, victim = factory.create_pair("ketoconazole", "midazolam")

        assert perp.inhibition.is_inhibitor is True
        assert victim.metabolism.cyp3a4 > 0.9

    def test_create_unknown_drug_requires_smiles(self):
        from neogen.engine.drug_factory import DrugFactory

        factory = DrugFactory()
        with pytest.raises(ValueError, match="No SMILES"):
            factory.create("totally_novel_compound_xyz")


# ═══════════════════════════════════════════════════════════════════
# DOSE SCHEDULER TESTS
# ═══════════════════════════════════════════════════════════════════


class TestDoseScheduler:
    """Test DoseScheduler produces valid dose schedules."""

    def test_single_oral_dose(self):
        from neogen.engine.dose_scheduler import DoseScheduler
        from neogen.core.entities.formulation import Route

        scheduler = DoseScheduler()
        schedule = scheduler.single_oral("midazolam", 5.0)

        assert len(schedule.events) == 1
        assert schedule.events[0].dose_mg == 5.0
        assert schedule.events[0].route == Route.ORAL
        assert schedule.events[0].time_h == 0.0

    def test_iv_bolus(self):
        from neogen.engine.dose_scheduler import DoseScheduler
        from neogen.core.entities.formulation import Route

        scheduler = DoseScheduler()
        schedule = scheduler.iv_bolus("midazolam", 2.0)

        assert schedule.events[0].route == Route.IV_BOLUS
        assert schedule.events[0].is_iv is True

    def test_repeated_dosing_bid(self):
        from neogen.engine.dose_scheduler import DoseScheduler

        scheduler = DoseScheduler()
        schedule = scheduler.repeated("ibuprofen", 400.0, "BID", days=3)

        # BID for 3 days = 6 doses
        assert len(schedule.events) == 6
        # First dose at t=0, last at t=60h (5 × 12h)
        assert schedule.events[0].time_h == 0.0
        assert schedule.events[-1].time_h == 60.0

    def test_ddi_study_design(self):
        from neogen.engine.dose_scheduler import DoseScheduler

        scheduler = DoseScheduler()
        schedule = scheduler.ddi_study(
            perpetrator="ketoconazole",
            perp_dose_mg=400.0,
            perp_regimen="QD",
            victim="midazolam",
            victim_dose_mg=5.0,
            pretreatment_days=3,
        )

        assert schedule.is_multi_drug is True
        assert "ketoconazole" in schedule.drugs_involved
        assert "midazolam" in schedule.drugs_involved

        # 3 days QD = 3 perpetrator doses + 1 victim dose
        assert len(schedule.events) == 4

        # Victim should be dosed on day 3 (t=48h)
        victim_events = schedule.events_for_drug("midazolam")
        assert len(victim_events) == 1
        assert victim_events[0].time_h == 48.0  # Day 3

    def test_combine_schedules(self):
        from neogen.engine.dose_scheduler import DoseScheduler

        scheduler = DoseScheduler()
        s1 = scheduler.single_oral("drug_a", 100.0)
        s2 = scheduler.single_oral("drug_b", 200.0)
        combined = scheduler.combine(s1, s2)

        assert combined.n_drugs == 2
        assert len(combined.events) == 2


# ═══════════════════════════════════════════════════════════════════
# FORMULATION TESTS
# ═══════════════════════════════════════════════════════════════════


class TestFormulation:
    """Test Formulation and DoseSchedule helpers."""

    def test_iv_bypasses_first_pass(self):
        from neogen.core.entities.formulation import Route

        assert Route.IV_BOLUS.bypasses_first_pass is True
        assert Route.ORAL.bypasses_first_pass is False

    def test_dose_schedule_convenience_methods(self):
        from neogen.core.entities.formulation import DoseSchedule, Route

        # Single dose
        s = DoseSchedule.single_dose("aspirin", 500.0)
        assert len(s.events) == 1

        # Multiple dose
        s = DoseSchedule.multiple_dose(
            "ibuprofen", 400.0, interval_h=8.0, n_doses=9,
        )
        assert len(s.events) == 9
        assert s.events[-1].time_h == 64.0  # 8 × 8h

    def test_ddi_scenario(self):
        from neogen.core.entities.formulation import DoseSchedule

        s = DoseSchedule.ddi_scenario(
            perpetrator_name="ketoconazole",
            perpetrator_dose_mg=400,
            perpetrator_interval_h=24,
            perpetrator_n_doses=5,
            victim_name="midazolam",
            victim_dose_mg=5,
            victim_time_h=96.0,
        )
        assert s.is_multi_drug is True
        assert len(s.events) == 6  # 5 perp + 1 victim


# ═══════════════════════════════════════════════════════════════════
# INTERACTION ENTITY TESTS
# ═══════════════════════════════════════════════════════════════════


class TestInteractionEntities:
    """Test DDI result entities."""

    def test_severity_from_auc_ratio(self):
        from neogen.core.entities.interaction import DDISeverity

        assert DDISeverity.from_auc_ratio(10.0) == DDISeverity.CONTRAINDICATED
        assert DDISeverity.from_auc_ratio(3.0) == DDISeverity.MAJOR
        assert DDISeverity.from_auc_ratio(1.5) == DDISeverity.MODERATE
        assert DDISeverity.from_auc_ratio(1.1) == DDISeverity.MINOR

    def test_inhibition_event(self):
        from neogen.core.entities.interaction import InhibitionEvent

        event = InhibitionEvent(
            perpetrator="ketoconazole",
            victim="midazolam",
            enzyme="CYP3A4",
            fm=0.96,
            ki_um=0.015,
            auc_ratio=10.0,
        )
        assert event.is_clinically_significant is True

    def test_induction_event_time_to_ss(self):
        from neogen.core.entities.interaction import InductionEvent

        event = InductionEvent(
            perpetrator="rifampicin",
            victim="midazolam",
            enzyme="CYP3A4",
            kdeg_per_h=0.0193,
        )
        # Time to SS ≈ 4 × (0.693/0.0193) ≈ 144h ≈ 6 days
        assert 100 < event.time_to_steady_state_h < 200


# ═══════════════════════════════════════════════════════════════════
# SCIENTIFIC INVARIANT TESTS
# These verify that the data is scientifically self-consistent.
# ═══════════════════════════════════════════════════════════════════


class TestScientificInvariants:
    """Tests that verify scientific correctness of the parameter system."""

    def test_fm_rule_ketoconazole_midazolam(self):
        """Verify the fm rule gives expected AUC ratio for a known DDI.

        Ketoconazole + midazolam: expected ~10-16 fold AUC increase.
        fm_midazolam_3A4 = 0.96
        Ki_ketoconazole = 0.015 µM
        Typical [I]u ~ 0.1 µM (gut) to 0.01 µM (liver)
        """
        from neogen.core.constants.enzyme_database import (
            FDA_INDEX_SUBSTRATES,
            FDA_INDEX_INHIBITORS,
        )

        fm = FDA_INDEX_SUBSTRATES["midazolam"]["CYP3A4"]
        ki = FDA_INDEX_INHIBITORS["ketoconazole"]["CYP3A4"]

        # At hepatic [I]u = 0.1 µM
        i_conc = 0.1
        auc_ratio = 1.0 / (1.0 - fm + fm / (1.0 + i_conc / ki))

        # Should be in ~5-15x range for ketoconazole + midazolam
        assert auc_ratio > 3.0, f"AUC ratio too low: {auc_ratio:.1f}"
        assert auc_ratio < 25.0, f"AUC ratio too high: {auc_ratio:.1f}"

    def test_enzyme_mass_balance(self):
        """At steady-state: synthesis rate = degradation rate."""
        from neogen.core.constants.enzyme_database import CYP3A4

        # ksyn = kdeg × abundance_baseline
        ksyn = CYP3A4.synthesis_rate_liver
        kdeg_times_amount = CYP3A4.kdeg_liver * CYP3A4.abundance_liver_pmol_per_mg

        assert ksyn == pytest.approx(kdeg_times_amount, rel=0.001)

    def test_total_cyp_abundance_reasonable(self):
        """Total CYP content should be ~300-500 pmol/mg in human liver."""
        from neogen.core.constants.enzyme_database import ENZYME_REGISTRY

        total = sum(
            e.abundance_liver_pmol_per_mg
            for e in ENZYME_REGISTRY.values()
        )
        assert 200 < total < 600, (
            f"Total CYP abundance = {total:.0f} pmol/mg "
            f"(expected 300-500)"
        )

    def test_cardiac_output_equals_lung_flow(self):
        """All blood passes through lungs. CO = lung blood flow."""
        from neogen.core.constants.reference_physiology import (
            build_reference_male,
        )

        model = build_reference_male()
        # Lung flow represents total cardiac output
        assert model.lung.blood_flow_L_per_h == pytest.approx(
            model.cardiac_output_L_per_h, rel=0.05
        )
