"""
P3 Step 8 — CYP Induction Unit Tests
======================================
Exercises ``neogen.engine.ddi.cyp_induction`` in isolation.

Scope (matches the V1 execution plan):
  * Static, steady-state induction via the Emax model.
  * No time course (that is Step 9).
  * No autoinduction (V2+).

Plan gate for Step 8:
    rifampicin pre-treatment → CYP3A4 × 5 at steady state.

We test this at a clinically realistic portal unbound concentration
for 600 mg oral rifampicin ([I]u ≈ 0.2 µM) and also at saturation.
"""
from __future__ import annotations

import math

import pytest

from neogen.core.entities.drug import (
    CYPMetabolismProfile,
    Drug,
    InductionProfile,
)
from neogen.engine.ddi import (
    INDUCIBLE_CYP_NAMES,
    CYPInductionContribution,
    CYPInductionEngine,
    InductionState,
)
from neogen.engine.drug_factory import DrugFactory


# ════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def engine() -> CYPInductionEngine:
    return CYPInductionEngine()


@pytest.fixture(scope="module")
def factory() -> DrugFactory:
    return DrugFactory()


@pytest.fixture(scope="module")
def rifampicin(factory: DrugFactory) -> Drug:
    return factory.create("rifampicin")


@pytest.fixture(scope="module")
def midazolam(factory: DrugFactory) -> Drug:
    return factory.create("midazolam")


@pytest.fixture(scope="module")
def ketoconazole(factory: DrugFactory) -> Drug:
    """Non-inducer; used for negative controls."""
    return factory.create("ketoconazole")


@pytest.fixture
def synthetic_inducer() -> Drug:
    """Toy inducer with Indmax = 10, IndC50 = 1 µM on CYP3A4 only."""
    return Drug(
        name="synthetic_inducer",
        smiles="C",
        molecular_weight=400.0,
        logp=2.0,
        fu_plasma=0.5,
        metabolism=CYPMetabolismProfile(),
        induction=InductionProfile(
            indmax_cyp3a4=10.0,
            indc50_cyp3a4=1.0,
        ),
    )


# ════════════════════════════════════════════════════════════════
# STEP 8 HARD GATE
# ════════════════════════════════════════════════════════════════


class TestStep8AcceptanceGate:
    """The assertion required by the V1 plan for Step 8."""

    def test_rifampicin_cyp3a4_fold_above_5_at_clinical_concentration(
        self,
        engine: CYPInductionEngine,
        rifampicin: Drug,
    ):
        """Rifampicin 600 mg PO daily → unbound portal [I]u ≈ 0.2 µM.

        At this concentration the Emax model gives:
            fold = 1 + 12 · 0.2 / (0.31 + 0.2) ≈ 5.7
        which matches the plan's "CYP3A4 × 5" steady-state target.
        """
        state = engine.compute(rifampicin, inducer_conc_um=0.2, site="liver")
        assert state.fold_for("CYP3A4") >= 5.0, (
            f"Step 8 gate failed: CYP3A4 fold = "
            f"{state.fold_for('CYP3A4'):.3f}, expected ≥ 5 at [I]u=0.2 µM."
        )

    def test_rifampicin_cyp3a4_fold_at_saturation_matches_indmax_plus_one(
        self,
        engine: CYPInductionEngine,
        rifampicin: Drug,
    ):
        """At [I]u >> IndC50 the fold should approach 1 + Indmax.

        Rifampicin Indmax_CYP3A4 = 12, so fold should sit near 13 at
        high concentration (allowing for the defensive ceiling)."""
        state = engine.compute(rifampicin, inducer_conc_um=1000.0)
        fold = state.fold_for("CYP3A4")
        assert 12.5 <= fold <= 14.0, (
            f"Saturation fold = {fold:.3f}, expected close to 1 + Indmax."
        )

    def test_rifampicin_midazolam_aucr_in_clinical_window(
        self,
        engine: CYPInductionEngine,
        rifampicin: Drug,
        midazolam: Drug,
    ):
        """Observed clinical AUCR for rifampicin + oral midazolam is
        ~0.04–0.10 (Backman 1996). The FDA static model gives a
        slightly higher number because it ignores the independent
        gut-wall CYP3A4 induction. Accept 0.05–0.25 at [I]u = 0.2 µM.
        """
        state = engine.compute(rifampicin, inducer_conc_um=0.2)
        aucr = engine.victim_auc_ratio(state, midazolam)
        assert 0.05 <= aucr <= 0.25, (
            f"Rifampicin+midazolam static AUCR = {aucr:.3f} "
            f"outside clinical window (target 0.05–0.25)."
        )


# ════════════════════════════════════════════════════════════════
# UNIT TESTS — ENGINE BEHAVIOUR
# ════════════════════════════════════════════════════════════════


class TestCYPInductionEngineContract:
    """Pure-function behaviour and mathematical invariants."""

    def test_zero_concentration_yields_fold_equal_one(
        self, engine: CYPInductionEngine, rifampicin: Drug
    ):
        state = engine.compute(rifampicin, inducer_conc_um=0.0)
        for c in state.contributions:
            assert c.fold == 1.0
        assert state.max_fold == 1.0
        assert state.is_meaningful is False

    def test_non_inducer_all_folds_one(
        self, engine: CYPInductionEngine, ketoconazole: Drug
    ):
        """Ketoconazole is a classic inhibitor with no induction data.
        All folds must be 1 at any concentration."""
        state = engine.compute(ketoconazole, inducer_conc_um=10.0)
        assert state.max_fold == 1.0
        assert state.is_meaningful is False

    def test_fold_is_monotonic_in_concentration(
        self, engine: CYPInductionEngine, rifampicin: Drug
    ):
        """Doubling [I]u must not decrease the fold."""
        low = engine.compute(rifampicin, inducer_conc_um=0.01)
        mid = engine.compute(rifampicin, inducer_conc_um=0.5)
        high = engine.compute(rifampicin, inducer_conc_um=50.0)
        assert (
            low.fold_for("CYP3A4")
            < mid.fold_for("CYP3A4")
            < high.fold_for("CYP3A4")
        )

    def test_fold_is_always_at_least_one(
        self, engine: CYPInductionEngine, rifampicin: Drug
    ):
        for conc in (0.0, 0.001, 0.1, 1.0, 100.0):
            state = engine.compute(rifampicin, inducer_conc_um=conc)
            for c in state.contributions:
                assert c.fold >= 1.0

    def test_emax_formula_matches_fda_basic(
        self,
        engine: CYPInductionEngine,
        synthetic_inducer: Drug,
    ):
        """Synthetic inducer with Indmax = 10, IndC50 = 1 µM at
        [I]u = 1 µM should give fold = 1 + 10 · 0.5 = 6.0 exactly.
        """
        state = engine.compute(synthetic_inducer, inducer_conc_um=1.0)
        assert math.isclose(state.fold_for("CYP3A4"), 6.0, rel_tol=1e-9)
        assert state.fold_for("CYP2B6") == 1.0  # Not induced by this toy

    def test_half_max_at_indc50(
        self, engine: CYPInductionEngine, synthetic_inducer: Drug
    ):
        """At [I]u = IndC50, induction is at half-maximum:
        fold = 1 + Indmax/2."""
        state = engine.compute(synthetic_inducer, inducer_conc_um=1.0)
        expected_half = 1.0 + 10.0 / 2.0
        assert math.isclose(
            state.fold_for("CYP3A4"), expected_half, rel_tol=1e-9
        )

    def test_saturation_approaches_indmax_plus_one(
        self, engine: CYPInductionEngine, synthetic_inducer: Drug
    ):
        """[I]u → ∞ (or >> IndC50) → fold → 1 + Indmax."""
        state = engine.compute(synthetic_inducer, inducer_conc_um=1.0e6)
        fold = state.fold_for("CYP3A4")
        assert 10.9 <= fold <= 11.0 + 1e-6

    def test_negative_concentration_raises(
        self, engine: CYPInductionEngine, rifampicin: Drug
    ):
        with pytest.raises(ValueError):
            engine.compute(rifampicin, inducer_conc_um=-0.1)

    def test_invalid_site_raises(
        self, engine: CYPInductionEngine, rifampicin: Drug
    ):
        with pytest.raises(ValueError):
            engine.compute(rifampicin, inducer_conc_um=1.0, site="brain")

    def test_fold_ceiling_prevents_pathological_values(
        self, engine: CYPInductionEngine
    ):
        """A drug with an absurd Indmax must still return a finite,
        clipped fold."""
        absurd = Drug(
            name="absurd",
            smiles="C",
            molecular_weight=300.0,
            logp=2.0,
            fu_plasma=0.5,
            induction=InductionProfile(
                indmax_cyp3a4=1.0e9,
                indc50_cyp3a4=0.1,
            ),
        )
        state = engine.compute(absurd, inducer_conc_um=1.0)
        assert math.isfinite(state.fold_for("CYP3A4"))
        assert state.fold_for("CYP3A4") <= 50.0 + 1e-6


# ════════════════════════════════════════════════════════════════
# UNIT TESTS — RESULT DATACLASS
# ════════════════════════════════════════════════════════════════


class TestInductionStateImmutability:
    def test_is_frozen(
        self, engine: CYPInductionEngine, rifampicin: Drug
    ):
        state = engine.compute(rifampicin, inducer_conc_um=0.2)
        with pytest.raises((AttributeError, TypeError)):
            state.perpetrator_name = "other"  # type: ignore[misc]

    def test_contribution_is_frozen(
        self, engine: CYPInductionEngine, rifampicin: Drug
    ):
        state = engine.compute(rifampicin, inducer_conc_um=0.2)
        with pytest.raises((AttributeError, TypeError)):
            state.contributions[0].fold = 999.0  # type: ignore[misc]

    def test_contributions_cover_inducible_cyps(
        self, engine: CYPInductionEngine, rifampicin: Drug
    ):
        state = engine.compute(rifampicin, inducer_conc_um=0.2)
        assert len(state.contributions) == len(INDUCIBLE_CYP_NAMES)
        assert tuple(c.cyp for c in state.contributions) == INDUCIBLE_CYP_NAMES

    def test_cyp2d6_not_inducible(
        self, engine: CYPInductionEngine, rifampicin: Drug
    ):
        """CYP2D6 is not inducible in humans — it must NOT appear in the
        state contributions, and fold_for('CYP2D6') must return 1.0."""
        state = engine.compute(rifampicin, inducer_conc_um=0.2)
        assert "CYP2D6" not in state.as_fold_map()
        assert state.fold_for("CYP2D6") == 1.0

    def test_fold_for_is_case_insensitive(
        self, engine: CYPInductionEngine, rifampicin: Drug
    ):
        state = engine.compute(rifampicin, inducer_conc_um=0.2)
        assert state.fold_for("cyp3a4") == state.fold_for("CYP3A4")


# ════════════════════════════════════════════════════════════════
# UNIT TESTS — VICTIM AUC RATIO
# ════════════════════════════════════════════════════════════════


class TestVictimAucRatio:
    def test_zero_induction_gives_unit_aucr(
        self,
        engine: CYPInductionEngine,
        rifampicin: Drug,
        midazolam: Drug,
    ):
        state = engine.compute(rifampicin, inducer_conc_um=0.0)
        assert engine.victim_auc_ratio(state, midazolam) == pytest.approx(1.0)

    def test_victim_with_zero_cyp_fm_is_unaffected(
        self, engine: CYPInductionEngine, rifampicin: Drug
    ):
        non_cyp_victim = Drug(
            name="non_cyp",
            smiles="C",
            molecular_weight=200.0,
            logp=1.0,
            fu_plasma=0.5,
            metabolism=CYPMetabolismProfile(),
            fraction_excreted_unchanged=1.0,
        )
        state = engine.compute(rifampicin, inducer_conc_um=100.0)
        assert engine.victim_auc_ratio(state, non_cyp_victim) == pytest.approx(1.0)

    def test_fully_3a4_victim_aucr_equals_inverse_fold(
        self, engine: CYPInductionEngine, rifampicin: Drug
    ):
        """fm_CYP3A4 = 1 victim should give AUCR = 1/fold."""
        full_victim = Drug(
            name="fully_3a4",
            smiles="C",
            molecular_weight=300.0,
            logp=2.0,
            fu_plasma=0.2,
            metabolism=CYPMetabolismProfile(cyp3a4=1.0),
        )
        state = engine.compute(rifampicin, inducer_conc_um=0.2)
        fold = state.fold_for("CYP3A4")
        aucr = engine.victim_auc_ratio(state, full_victim)
        assert math.isclose(aucr, 1.0 / fold, rel_tol=1e-6)

    def test_induction_aucr_is_at_most_one(
        self,
        engine: CYPInductionEngine,
        rifampicin: Drug,
        midazolam: Drug,
    ):
        """Induction can only reduce victim exposure — AUCR ≤ 1 always."""
        for conc in (0.0, 0.01, 0.2, 1.0, 100.0):
            state = engine.compute(rifampicin, inducer_conc_um=conc)
            aucr = engine.victim_auc_ratio(state, midazolam)
            assert aucr <= 1.0 + 1e-9

    def test_partial_cyp_victim_less_affected_than_full_cyp_victim(
        self, engine: CYPInductionEngine, rifampicin: Drug
    ):
        half = Drug(
            name="half",
            smiles="C",
            molecular_weight=300.0,
            logp=2.0,
            fu_plasma=0.2,
            metabolism=CYPMetabolismProfile(cyp3a4=0.5),
        )
        full = Drug(
            name="full",
            smiles="C",
            molecular_weight=300.0,
            logp=2.0,
            fu_plasma=0.2,
            metabolism=CYPMetabolismProfile(cyp3a4=1.0),
        )
        state = engine.compute(rifampicin, inducer_conc_um=0.2)
        aucr_half = engine.victim_auc_ratio(state, half)
        aucr_full = engine.victim_auc_ratio(state, full)
        assert aucr_half > aucr_full  # Less reduction when fm is smaller
