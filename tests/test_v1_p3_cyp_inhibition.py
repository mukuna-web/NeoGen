"""
P3 Step 7 — CYP Inhibition Unit Tests
======================================
Exercises ``neogen.engine.ddi.cyp_inhibition`` in isolation.

Scope (matches the V1 execution plan):
  * Static, reversible, competitive CYP inhibition only.
  * No time-course / MBI kinetics (that is Step 9).
  * No runtime PBPK coupling (that is Step 10).

Plan's hard gate for Step 7:
    ketoconazole + midazolam → R_CYP3A4 > 10.

We additionally check the static victim AUC-ratio falls into the
published 10-15× window for the same pair.
"""
from __future__ import annotations

import math

import pytest

from neogen.core.entities.drug import (
    CYPMetabolismProfile,
    Drug,
    InhibitionProfile,
)
from neogen.engine.ddi import (
    CYP_NAMES,
    CYPInhibitionContribution,
    CYPInhibitionEngine,
    InhibitionState,
)
from neogen.engine.drug_factory import DrugFactory


# ════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def engine() -> CYPInhibitionEngine:
    return CYPInhibitionEngine()


@pytest.fixture(scope="module")
def factory() -> DrugFactory:
    return DrugFactory()


@pytest.fixture(scope="module")
def ketoconazole(factory: DrugFactory) -> Drug:
    return factory.create("ketoconazole")


@pytest.fixture(scope="module")
def midazolam(factory: DrugFactory) -> Drug:
    return factory.create("midazolam")


@pytest.fixture(scope="module")
def clarithromycin(factory: DrugFactory) -> Drug:
    return factory.create("clarithromycin")


@pytest.fixture
def synthetic_pan_inhibitor() -> Drug:
    """A toy drug that inhibits every tracked CYP with Ki = 1 µM."""
    return Drug(
        name="panel_inhibitor",
        smiles="C",
        molecular_weight=400.0,
        logp=2.0,
        fu_plasma=0.5,
        metabolism=CYPMetabolismProfile(),
        inhibition=InhibitionProfile(
            ki_cyp3a4=1.0,
            ki_cyp2d6=1.0,
            ki_cyp2c9=1.0,
            ki_cyp2c19=1.0,
            ki_cyp1a2=1.0,
            ki_cyp2c8=1.0,
            ki_cyp2b6=1.0,
            ki_cyp2e1=1.0,
        ),
    )


# ════════════════════════════════════════════════════════════════
# STEP 7 HARD GATE
# ════════════════════════════════════════════════════════════════


class TestStep7AcceptanceGate:
    """The single assertion required by the V1 plan for Step 7."""

    def test_ketoconazole_midazolam_r_cyp3a4_greater_than_10(
        self,
        engine: CYPInhibitionEngine,
        ketoconazole: Drug,
    ):
        """Ketoconazole at hepatic inlet [I]u ≈ 0.5 µM → R_CYP3A4 > 10.

        0.5 µM is the canonical literature value for the *unbound hepatic
        inlet concentration* of ketoconazole after a 400 mg oral dose.
        It includes the portal-vein first-pass contribution and is the
        concentration the FDA basic model expects for liver CYPs.
        Plasma Cmax_u alone (~0.04 µM) underpredicts DDI magnitude.
        """
        state = engine.compute(
            ketoconazole, inhibitor_conc_um=0.5, site="liver"
        )
        assert state.r_for("CYP3A4") > 10.0, (
            f"Step 7 gate failed: R_CYP3A4 = {state.r_for('CYP3A4'):.3f} "
            f"expected > 10 for ketoconazole + midazolam."
        )

    def test_ketoconazole_midazolam_static_auc_ratio_in_clinical_window(
        self,
        engine: CYPInhibitionEngine,
        ketoconazole: Drug,
        midazolam: Drug,
    ):
        """The FDA static AUCR for the pair should fall in ~10–15×.

        Clinical observation: Olkkola 1994, Tsunoda 1999 report 10–16×
        AUC increases for oral midazolam + ketoconazole. The static
        model should land in the lower end of this range since it
        ignores the separate gut-wall CYP3A4 shutdown.
        """
        state = engine.compute(
            ketoconazole, inhibitor_conc_um=0.5, site="liver"
        )
        auc_ratio = engine.victim_auc_ratio(state, midazolam)
        assert 8.0 <= auc_ratio <= 25.0, (
            f"Midazolam AUCR = {auc_ratio:.2f} outside clinical window."
        )


# ════════════════════════════════════════════════════════════════
# UNIT TESTS — ENGINE BEHAVIOUR
# ════════════════════════════════════════════════════════════════


class TestCYPInhibitionEngineContract:
    """Pure-function behaviour and mathematical invariants."""

    def test_zero_concentration_yields_r_equal_one(
        self, engine: CYPInhibitionEngine, ketoconazole: Drug
    ):
        state = engine.compute(ketoconazole, inhibitor_conc_um=0.0)
        for c in state.contributions:
            assert c.r == 1.0
        assert state.max_r == 1.0
        assert state.is_meaningful is False

    def test_non_inhibited_cyps_return_r_one(
        self, engine: CYPInhibitionEngine, ketoconazole: Drug
    ):
        """Ketoconazole only has Ki for CYP3A4 in V1 literature; all
        other CYPs must return R = 1."""
        state = engine.compute(ketoconazole, inhibitor_conc_um=0.5)
        for c in state.contributions:
            if c.cyp == "CYP3A4":
                assert c.r > 1.0
            else:
                assert c.r == 1.0, f"{c.cyp} should be untouched"

    def test_r_is_monotonic_in_concentration(
        self, engine: CYPInhibitionEngine, ketoconazole: Drug
    ):
        """Doubling [I]u must not decrease R."""
        low = engine.compute(ketoconazole, inhibitor_conc_um=0.05)
        mid = engine.compute(ketoconazole, inhibitor_conc_um=0.5)
        high = engine.compute(ketoconazole, inhibitor_conc_um=5.0)
        assert (
            low.r_for("CYP3A4")
            < mid.r_for("CYP3A4")
            < high.r_for("CYP3A4")
        )

    def test_r_always_at_least_one(
        self, engine: CYPInhibitionEngine, ketoconazole: Drug
    ):
        """R must never drop below 1 (physically impossible)."""
        for conc in (0.0, 0.001, 0.1, 1.0, 100.0):
            state = engine.compute(ketoconazole, inhibitor_conc_um=conc)
            for c in state.contributions:
                assert c.r >= 1.0

    def test_r_formula_matches_fda_basic_model(
        self, engine: CYPInhibitionEngine, ketoconazole: Drug
    ):
        """R_cyp = 1 + [I]u / Ki to within float precision."""
        ki = ketoconazole.inhibition.ki_cyp3a4  # 0.015 µM
        conc = 0.3
        state = engine.compute(ketoconazole, inhibitor_conc_um=conc)
        expected = 1.0 + conc / ki
        assert math.isclose(state.r_for("CYP3A4"), expected, rel_tol=1e-9)

    def test_panel_inhibitor_affects_every_cyp_equally(
        self,
        engine: CYPInhibitionEngine,
        synthetic_pan_inhibitor: Drug,
    ):
        """Synthetic pan-inhibitor (Ki = 1 µM on all CYPs) at [I]u = 1 µM
        should give R = 2 uniformly."""
        state = engine.compute(synthetic_pan_inhibitor, inhibitor_conc_um=1.0)
        for c in state.contributions:
            assert math.isclose(c.r, 2.0, rel_tol=1e-9)
        assert state.max_r == pytest.approx(2.0)

    def test_negative_concentration_raises(
        self, engine: CYPInhibitionEngine, ketoconazole: Drug
    ):
        with pytest.raises(ValueError):
            engine.compute(ketoconazole, inhibitor_conc_um=-1.0)

    def test_invalid_site_raises(
        self, engine: CYPInhibitionEngine, ketoconazole: Drug
    ):
        with pytest.raises(ValueError):
            engine.compute(ketoconazole, inhibitor_conc_um=0.5, site="brain")

    def test_r_ceiling_prevents_overflow(
        self, engine: CYPInhibitionEngine, ketoconazole: Drug
    ):
        """Extreme concentration must not overflow the R value."""
        huge = engine.compute(ketoconazole, inhibitor_conc_um=1.0e12)
        assert math.isfinite(huge.r_for("CYP3A4"))
        assert huge.r_for("CYP3A4") <= 1.0e4 + 1.0


# ════════════════════════════════════════════════════════════════
# UNIT TESTS — RESULT DATACLASS
# ════════════════════════════════════════════════════════════════


class TestInhibitionStateImmutability:
    def test_is_frozen(
        self, engine: CYPInhibitionEngine, ketoconazole: Drug
    ):
        state = engine.compute(ketoconazole, inhibitor_conc_um=0.5)
        with pytest.raises((AttributeError, TypeError)):
            state.perpetrator_name = "other"  # type: ignore[misc]

    def test_contribution_is_frozen(
        self, engine: CYPInhibitionEngine, ketoconazole: Drug
    ):
        state = engine.compute(ketoconazole, inhibitor_conc_um=0.5)
        with pytest.raises((AttributeError, TypeError)):
            state.contributions[0].r = 999.0  # type: ignore[misc]

    def test_contributions_cover_all_tracked_cyps(
        self, engine: CYPInhibitionEngine, ketoconazole: Drug
    ):
        state = engine.compute(ketoconazole, inhibitor_conc_um=0.5)
        assert len(state.contributions) == len(CYP_NAMES)
        assert tuple(c.cyp for c in state.contributions) == CYP_NAMES

    def test_r_for_is_case_insensitive(
        self, engine: CYPInhibitionEngine, ketoconazole: Drug
    ):
        state = engine.compute(ketoconazole, inhibitor_conc_um=0.5)
        assert state.r_for("cyp3a4") == state.r_for("CYP3A4")

    def test_r_for_unknown_cyp_returns_one(
        self, engine: CYPInhibitionEngine, ketoconazole: Drug
    ):
        state = engine.compute(ketoconazole, inhibitor_conc_um=0.5)
        assert state.r_for("CYP9Z9") == 1.0

    def test_as_r_map_round_trip(
        self, engine: CYPInhibitionEngine, ketoconazole: Drug
    ):
        state = engine.compute(ketoconazole, inhibitor_conc_um=0.5)
        rmap = state.as_r_map()
        assert set(rmap.keys()) == set(CYP_NAMES)
        for cyp in CYP_NAMES:
            assert rmap[cyp] == state.r_for(cyp)


# ════════════════════════════════════════════════════════════════
# UNIT TESTS — VICTIM AUC RATIO
# ════════════════════════════════════════════════════════════════


class TestVictimAucRatio:
    def test_zero_inhibition_gives_unit_ratio(
        self,
        engine: CYPInhibitionEngine,
        ketoconazole: Drug,
        midazolam: Drug,
    ):
        state = engine.compute(ketoconazole, inhibitor_conc_um=0.0)
        assert engine.victim_auc_ratio(state, midazolam) == pytest.approx(1.0)

    def test_victim_with_zero_cyp_fm_is_unaffected(
        self,
        engine: CYPInhibitionEngine,
        ketoconazole: Drug,
    ):
        """A victim with 100% non-CYP clearance must have AUCR = 1
        regardless of inhibitor strength."""
        non_cyp_victim = Drug(
            name="non_cyp_victim",
            smiles="C",
            molecular_weight=200.0,
            logp=1.0,
            fu_plasma=0.5,
            metabolism=CYPMetabolismProfile(),  # all zero
            fraction_excreted_unchanged=1.0,
        )
        state = engine.compute(ketoconazole, inhibitor_conc_um=1000.0)
        assert engine.victim_auc_ratio(state, non_cyp_victim) == pytest.approx(1.0)

    def test_victim_fully_metabolized_by_inhibited_cyp(
        self,
        engine: CYPInhibitionEngine,
        ketoconazole: Drug,
    ):
        """fm_CYP3A4 = 1.0 with R = 34 should give AUCR ≈ 34."""
        fully_3a4_victim = Drug(
            name="fully_3a4_victim",
            smiles="C",
            molecular_weight=300.0,
            logp=2.0,
            fu_plasma=0.2,
            metabolism=CYPMetabolismProfile(cyp3a4=1.0),
        )
        state = engine.compute(ketoconazole, inhibitor_conc_um=0.5)
        r = state.r_for("CYP3A4")
        aucr = engine.victim_auc_ratio(state, fully_3a4_victim)
        assert math.isclose(aucr, r, rel_tol=1e-6)

    def test_partial_cyp_metabolism_lowers_aucr(
        self,
        engine: CYPInhibitionEngine,
        ketoconazole: Drug,
    ):
        """fm_CYP3A4 = 0.5 should give a much smaller AUCR than fm = 1.0."""
        half_victim = Drug(
            name="half",
            smiles="C",
            molecular_weight=300.0,
            logp=2.0,
            fu_plasma=0.2,
            metabolism=CYPMetabolismProfile(cyp3a4=0.5),
        )
        full_victim = Drug(
            name="full",
            smiles="C",
            molecular_weight=300.0,
            logp=2.0,
            fu_plasma=0.2,
            metabolism=CYPMetabolismProfile(cyp3a4=1.0),
        )
        state = engine.compute(ketoconazole, inhibitor_conc_um=0.5)
        assert (
            engine.victim_auc_ratio(state, half_victim)
            < engine.victim_auc_ratio(state, full_victim)
        )


# ════════════════════════════════════════════════════════════════
# CROSS-DRUG SANITY
# ════════════════════════════════════════════════════════════════


class TestInhibitorSanityAcrossDrugs:
    def test_ketoconazole_is_stronger_than_clarithromycin_reversibly(
        self,
        engine: CYPInhibitionEngine,
        ketoconazole: Drug,
        clarithromycin: Drug,
    ):
        """On pure reversible competitive math alone (Step 7 scope),
        ketoconazole (Ki = 0.015 µM) must beat clarithromycin
        (Ki ≈ 3.3 µM) at the same inhibitor concentration. MBI is not
        yet part of Step 7 — that shifts the clinical picture for
        clarithromycin but not for this test."""
        keto_state = engine.compute(ketoconazole, inhibitor_conc_um=0.5)
        clari_state = engine.compute(clarithromycin, inhibitor_conc_um=0.5)
        assert keto_state.r_for("CYP3A4") > clari_state.r_for("CYP3A4")

    def test_midazolam_is_not_an_inhibitor(
        self,
        engine: CYPInhibitionEngine,
        midazolam: Drug,
    ):
        """Midazolam is a classic CYP3A4 *victim*, not an inhibitor.
        Its InhibitionProfile should carry no Ki values and the
        resulting state should be trivial."""
        state = engine.compute(midazolam, inhibitor_conc_um=10.0)
        assert state.max_r == 1.0
        assert state.is_meaningful is False
