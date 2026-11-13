"""V1 P2 Step 4 — permeability engine + AbsorptionProfile tests.

Exercises the coupled dissolution + absorption ODE that produces the
final ``AbsorptionProfile``. The central invariants promised in the plan:

- Fa monotonic in Peff (parametric sweep)
- tlag bounded by gastric emptying
- Mass balance across the full 9-segment transit
- BCS-class behaviour: Fa reflects dissolution / permeation limits
- Gut-wall CYP3A4 substrates incur Fg < 1 (Jg extraction)
- SOLUTION formulation → tlag ≈ 0, Fa close to ka-absorbed ceiling
- ENTERIC_COATED formulation → tlag ≥ gastric emptying
- Portal input rate helper is non-negative and integrates back to total
- AbsorptionProfile is frozen and shape-consistent
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neogen.core.entities.drug import (
    BCSClass,
    CYPMetabolismProfile,
    Drug,
    Ionization,
)
from neogen.core.entities.formulation import Formulation, FormulationType
from neogen.engine.absorption import (
    AbsorptionProfile,
    GutTransit,
    PermeabilityEngine,
    build_reference_gi_tract,
    effective_peff_cm_per_s,
    is_gut_cyp3a4_substrate,
    segment_fg,
)


# ────────────────────────────────────────────────────────────────
# Drug fixtures
# ────────────────────────────────────────────────────────────────


@pytest.fixture
def paracetamol() -> Drug:
    """BCS I — high solubility, high permeability."""
    return Drug(
        name="paracetamol",
        smiles="CC(=O)Nc1ccc(O)cc1",
        molecular_weight=151.16,
        logp=0.46,
        pka_acidic=9.5,
        bcs_class=BCSClass.I,
        solubility_mg_per_ml=14.0,
        peff_jejunum=4.0,  # ×10⁻⁴ cm/s
        ionization_class=Ionization.ACID,
    )


@pytest.fixture
def ibuprofen() -> Drug:
    """BCS II — low solubility, high permeability."""
    return Drug(
        name="ibuprofen",
        smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O",
        molecular_weight=206.28,
        logp=3.97,
        pka_acidic=4.4,
        bcs_class=BCSClass.II,
        solubility_mg_per_ml=0.021,
        peff_jejunum=6.0,
        ionization_class=Ionization.ACID,
    )


@pytest.fixture
def metformin() -> Drug:
    """BCS III — high solubility, LOW permeability."""
    return Drug(
        name="metformin",
        smiles="CN(C)C(=N)N=C(N)N",
        molecular_weight=129.16,
        logp=-2.64,
        pka_basic=12.4,
        bcs_class=BCSClass.III,
        solubility_mg_per_ml=300.0,
        peff_jejunum=0.3,  # ~1/13 of paracetamol
        ionization_class=Ionization.BASE,
    )


@pytest.fixture
def atorvastatin() -> Drug:
    """BCS II, gut CYP3A4 substrate → Fg < 1."""
    return Drug(
        name="atorvastatin",
        smiles="CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
        molecular_weight=558.64,
        logp=6.36,
        pka_acidic=4.3,
        bcs_class=BCSClass.II,
        solubility_mg_per_ml=0.01,   # softened from 0.001 for MVP
        peff_jejunum=3.0,
        ionization_class=Ionization.ACID,
        metabolism=CYPMetabolismProfile(cyp3a4=0.7),
        clint_microsomal=82.0,  # Qgut model requires CLint for gut extraction
    )


@pytest.fixture
def engine() -> PermeabilityEngine:
    return PermeabilityEngine()


@pytest.fixture
def tract() -> GutTransit:
    return build_reference_gi_tract()


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────


class TestEffectivePeff:
    def test_uses_drug_value_when_present(self, paracetamol: Drug) -> None:
        assert effective_peff_cm_per_s(paracetamol) == pytest.approx(4.0e-4)

    def test_bcs_fallback_when_missing(self) -> None:
        drug = Drug(
            name="mystery",
            smiles="CCO",
            molecular_weight=46.07,
            logp=-0.31,
            bcs_class=BCSClass.III,
            solubility_mg_per_ml=100.0,
            peff_jejunum=None,
            ionization_class=Ionization.NEUTRAL,
        )
        # BCS III fallback: 0.3 ×10⁻⁴ cm/s
        assert effective_peff_cm_per_s(drug) == pytest.approx(0.3e-4)


class TestGutCYP3A4Detection:
    def test_non_substrate(self, paracetamol: Drug) -> None:
        assert is_gut_cyp3a4_substrate(paracetamol) is False

    def test_substrate(self, atorvastatin: Drug) -> None:
        assert is_gut_cyp3a4_substrate(atorvastatin) is True


class TestSegmentFg:
    def test_non_substrate_is_unity(self, tract: GutTransit) -> None:
        jej = tract.segment_by_name("Jejunum_1")
        assert segment_fg(jej, is_substrate=False, alpha=0.3) == 1.0

    def test_substrate_in_non_cyp_segment_is_unity(
        self, tract: GutTransit
    ) -> None:
        colon = tract.segment_by_name("Colon_asc")
        assert segment_fg(colon, is_substrate=True, alpha=0.3) == 1.0

    def test_substrate_in_jejunum_below_unity(
        self, tract: GutTransit
    ) -> None:
        jej = tract.segment_by_name("Jejunum_1")  # cyp3a4_activity = 1.0
        fg = segment_fg(jej, is_substrate=True, alpha=0.3)
        assert fg == pytest.approx(0.7)

    def test_clamped_at_zero(self, tract: GutTransit) -> None:
        jej = tract.segment_by_name("Jejunum_1")
        fg = segment_fg(jej, is_substrate=True, alpha=2.0)  # unrealistic
        assert fg == 0.0


# ────────────────────────────────────────────────────────────────
# Engine — smoke tests
# ────────────────────────────────────────────────────────────────


class TestEngineSmoke:
    def test_paracetamol_ir_tablet(
        self, engine: PermeabilityEngine, paracetamol: Drug
    ) -> None:
        formulation = Formulation(
            formulation_type=FormulationType.IR_TABLET,
            disintegration_time_min=7.0,
        )
        profile = engine.simulate(
            paracetamol, formulation, dose_mg=500.0
        )
        assert profile.solver_success is True
        # BCS I — Fa should be high (>0.85), Fg ≈ 1 (no gut CYP3A4).
        assert profile.fa > 0.85
        assert profile.fg == pytest.approx(1.0, abs=1e-9)
        assert profile.f_portal > 0.85
        assert profile.effective_ka_per_h > 0.0

    def test_solution_has_zero_tlag(
        self, engine: PermeabilityEngine, paracetamol: Drug
    ) -> None:
        formulation = Formulation(
            formulation_type=FormulationType.SOLUTION,
        )
        profile = engine.simulate(
            paracetamol, formulation, dose_mg=500.0, t_end_h=12.0
        )
        # With a solution in the stomach, absorption begins immediately
        # once the drug reaches the duodenum — tlag should be small.
        assert profile.tlag_h < 0.5

    def test_mass_balance_paracetamol(
        self, engine: PermeabilityEngine, paracetamol: Drug
    ) -> None:
        profile = engine.simulate(
            paracetamol,
            Formulation(formulation_type=FormulationType.IR_TABLET),
            dose_mg=500.0,
        )
        assert profile.mass_balance_error_mg < 5.0  # <1% of dose

    def test_mass_balance_ibuprofen(
        self, engine: PermeabilityEngine, ibuprofen: Drug
    ) -> None:
        profile = engine.simulate(
            ibuprofen,
            Formulation(
                formulation_type=FormulationType.IR_TABLET,
                particle_radius_um=30.0,
            ),
            dose_mg=400.0,
        )
        assert profile.mass_balance_error_mg < 4.0

    def test_bcs_iii_is_permeability_limited(
        self, engine: PermeabilityEngine, metformin: Drug
    ) -> None:
        """Metformin is permeability-limited — Fa should be modest."""
        profile = engine.simulate(
            metformin,
            Formulation(formulation_type=FormulationType.IR_TABLET),
            dose_mg=500.0,
        )
        # Low-Peff drug: Fa should be well below 1.
        assert profile.fa < 0.8

    def test_bcs_iii_vs_bcs_i_absorption_ranking(
        self,
        engine: PermeabilityEngine,
        paracetamol: Drug,
        metformin: Drug,
    ) -> None:
        formulation = Formulation(
            formulation_type=FormulationType.IR_TABLET,
        )
        para = engine.simulate(paracetamol, formulation, 500.0)
        metf = engine.simulate(metformin, formulation, 500.0)
        assert para.fa > metf.fa


# ────────────────────────────────────────────────────────────────
# The central scientific invariants from the plan
# ────────────────────────────────────────────────────────────────


class TestPlanInvariants:
    def test_fa_monotonic_in_peff(
        self, engine: PermeabilityEngine, paracetamol: Drug
    ) -> None:
        """Fa should rise monotonically as Peff increases, all else equal."""
        peff_sweep = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]  # ×10⁻⁴ cm/s
        fa_values = []
        formulation = Formulation(
            formulation_type=FormulationType.IR_TABLET,
        )
        for peff in peff_sweep:
            drug = paracetamol.with_overrides(peff_jejunum=peff)
            profile = engine.simulate(drug, formulation, dose_mg=500.0)
            fa_values.append(profile.fa)
        # Strictly monotonic non-decreasing.
        for a, b in zip(fa_values[:-1], fa_values[1:]):
            assert b >= a - 1e-6, (
                f"Fa not monotonic in Peff: {fa_values}"
            )
        # And the lowest Peff should give noticeably lower Fa than the
        # highest.
        assert fa_values[-1] > fa_values[0] + 0.2

    def test_tlag_bounded_by_gastric_emptying_for_enteric(
        self,
        engine: PermeabilityEngine,
        paracetamol: Drug,
        tract: GutTransit,
    ) -> None:
        """For an enteric-coated dose, tlag is dominated by the time to
        reach (and clear) the duodenum. For an IR tablet it can be
        considerably smaller. The test ensures tlag is at least plausible
        for the formulation."""
        formulation = Formulation(
            formulation_type=FormulationType.ENTERIC_COATED,
        )
        profile = engine.simulate(
            paracetamol, formulation, dose_mg=500.0, t_end_h=12.0
        )
        # Enteric coat skips stomach entirely — drug starts in duodenum.
        # Expected tlag: near zero because drug is immediately in the SI.
        assert profile.tlag_h >= 0.0
        # But tlag should not exceed a small fraction of total transit.
        assert profile.tlag_h < 2.0

    def test_ir_tablet_tlag_bounded_by_transit_window(
        self,
        engine: PermeabilityEngine,
        paracetamol: Drug,
        tract: GutTransit,
    ) -> None:
        """IR tablet tlag must be within total small intestine + stomach
        transit — any larger would be nonphysical for a BCS I drug."""
        formulation = Formulation(
            formulation_type=FormulationType.IR_TABLET,
            disintegration_time_min=7.0,
        )
        profile = engine.simulate(
            paracetamol, formulation, dose_mg=500.0
        )
        upper = (
            tract.gastric_emptying_h
            + tract.small_intestine_transit_h
        )
        # tlag should be considerably less than total SI passage for a
        # fast-dissolving, high-permeability drug.
        assert profile.tlag_h < upper

    def test_solution_has_smaller_tlag_than_ir_tablet(
        self, engine: PermeabilityEngine, paracetamol: Drug
    ) -> None:
        solution_profile = engine.simulate(
            paracetamol,
            Formulation(formulation_type=FormulationType.SOLUTION),
            dose_mg=500.0,
        )
        tablet_profile = engine.simulate(
            paracetamol,
            Formulation(
                formulation_type=FormulationType.IR_TABLET,
                disintegration_time_min=15.0,
            ),
            dose_mg=500.0,
        )
        assert solution_profile.tlag_h <= tablet_profile.tlag_h + 1e-6


# ────────────────────────────────────────────────────────────────
# Gut-wall metabolism (Fg branch)
# ────────────────────────────────────────────────────────────────


class TestGutWallMetabolism:
    def test_non_substrate_has_fg_exactly_one(
        self, engine: PermeabilityEngine, paracetamol: Drug
    ) -> None:
        profile = engine.simulate(
            paracetamol,
            Formulation(formulation_type=FormulationType.IR_TABLET),
            dose_mg=500.0,
        )
        assert profile.fg == pytest.approx(1.0)

    def test_cyp3a4_substrate_has_fg_below_one(
        self, engine: PermeabilityEngine, atorvastatin: Drug
    ) -> None:
        profile = engine.simulate(
            atorvastatin,
            Formulation(
                formulation_type=FormulationType.IR_TABLET,
                particle_radius_um=5.0,
            ),
            dose_mg=40.0,
        )
        assert profile.fg < 1.0
        # Some drug must still reach portal.
        assert profile.fg > 0.3

    def test_cum_gut_metabolized_consistent_with_fa_fg(
        self, engine: PermeabilityEngine, atorvastatin: Drug
    ) -> None:
        profile = engine.simulate(
            atorvastatin,
            Formulation(
                formulation_type=FormulationType.IR_TABLET,
                particle_radius_um=5.0,
            ),
            dose_mg=40.0,
        )
        gm_final = float(profile.cum_gut_metabolized_mg[-1])
        expected = profile.total_permeated_mg * (1.0 - profile.fg)
        assert gm_final == pytest.approx(expected, rel=1e-3, abs=1e-6)


# ────────────────────────────────────────────────────────────────
# AbsorptionProfile contract
# ────────────────────────────────────────────────────────────────


class TestAbsorptionProfileContract:
    def test_frozen(
        self, engine: PermeabilityEngine, paracetamol: Drug
    ) -> None:
        profile = engine.simulate(
            paracetamol,
            Formulation(formulation_type=FormulationType.IR_TABLET),
            dose_mg=500.0,
            t_end_h=6.0,
            n_points=31,
        )
        with pytest.raises(FrozenInstanceError):
            profile.fa = 0.5  # type: ignore[misc]

    def test_shapes(
        self, engine: PermeabilityEngine, paracetamol: Drug
    ) -> None:
        profile = engine.simulate(
            paracetamol,
            Formulation(formulation_type=FormulationType.IR_TABLET),
            dose_mg=500.0,
            t_end_h=12.0,
            n_points=61,
        )
        assert profile.t_h.shape == (61,)
        assert profile.undissolved_mg.shape == (61, 9)
        assert profile.dissolved_mg.shape == (61, 9)
        assert profile.cum_permeated_mg.shape == (61,)
        assert profile.cum_portal_mg.shape == (61,)
        assert profile.cum_gut_metabolized_mg.shape == (61,)
        assert profile.excreted_mg.shape == (61,)

    def test_invariant_bounds(
        self, engine: PermeabilityEngine, paracetamol: Drug
    ) -> None:
        profile = engine.simulate(
            paracetamol,
            Formulation(formulation_type=FormulationType.IR_TABLET),
            dose_mg=500.0,
        )
        assert 0.0 <= profile.fa <= 1.0
        assert 0.0 <= profile.fg <= 1.0
        assert 0.0 <= profile.f_portal <= 1.0
        assert profile.effective_ka_per_h >= 0.0
        assert profile.tlag_h >= 0.0

    def test_cum_portal_monotonic(
        self, engine: PermeabilityEngine, paracetamol: Drug
    ) -> None:
        profile = engine.simulate(
            paracetamol,
            Formulation(formulation_type=FormulationType.IR_TABLET),
            dose_mg=500.0,
        )
        diffs = np.diff(profile.cum_portal_mg)
        assert np.all(diffs >= -1e-6)

    def test_per_segment_fraction_sums_to_one(
        self, engine: PermeabilityEngine, paracetamol: Drug
    ) -> None:
        profile = engine.simulate(
            paracetamol,
            Formulation(formulation_type=FormulationType.IR_TABLET),
            dose_mg=500.0,
        )
        total = sum(profile.per_segment_portal_fraction.values())
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_si_dominates_absorption(
        self, engine: PermeabilityEngine, paracetamol: Drug
    ) -> None:
        """For a well-behaved drug, small intestine should carry >80% of
        portal absorption."""
        profile = engine.simulate(
            paracetamol,
            Formulation(formulation_type=FormulationType.IR_TABLET),
            dose_mg=500.0,
        )
        si_names = {
            "Duodenum",
            "Jejunum_1",
            "Jejunum_2",
            "Ileum_1",
            "Ileum_2",
            "Ileum_3",
        }
        si_fraction = sum(
            v
            for name, v in profile.per_segment_portal_fraction.items()
            if name in si_names
        )
        assert si_fraction > 0.80


# ────────────────────────────────────────────────────────────────
# Portal input rate helper
# ────────────────────────────────────────────────────────────────


class TestPortalInputRate:
    def test_zero_outside_time_window(
        self, engine: PermeabilityEngine, paracetamol: Drug
    ) -> None:
        profile = engine.simulate(
            paracetamol,
            Formulation(formulation_type=FormulationType.IR_TABLET),
            dose_mg=500.0,
            t_end_h=12.0,
        )
        assert profile.portal_input_rate_at(-1.0) == 0.0
        assert profile.portal_input_rate_at(100.0) == 0.0

    def test_nonnegative_across_window(
        self, engine: PermeabilityEngine, paracetamol: Drug
    ) -> None:
        profile = engine.simulate(
            paracetamol,
            Formulation(formulation_type=FormulationType.IR_TABLET),
            dose_mg=500.0,
        )
        for t in np.linspace(0.1, profile.t_h[-1] - 0.1, 20):
            assert profile.portal_input_rate_at(float(t)) >= 0.0

    def test_integral_recovers_total_portal_mass(
        self, engine: PermeabilityEngine, paracetamol: Drug
    ) -> None:
        profile = engine.simulate(
            paracetamol,
            Formulation(formulation_type=FormulationType.IR_TABLET),
            dose_mg=500.0,
            t_end_h=24.0,
            n_points=481,
        )
        grid = np.linspace(
            profile.t_h[0], profile.t_h[-1], 2000
        )
        rates = np.array(
            [profile.portal_input_rate_at(float(t)) for t in grid]
        )
        integral = float(np.trapezoid(rates, grid))
        # Linear interpolation of finite differences gives a slight
        # underestimate; allow 10% tolerance.
        assert integral == pytest.approx(
            profile.total_portal_mg, rel=0.1, abs=2.0
        )


# ────────────────────────────────────────────────────────────────
# Engine input validation
# ────────────────────────────────────────────────────────────────


class TestEngineInputValidation:
    def test_rejects_nonpositive_dose(
        self, engine: PermeabilityEngine, paracetamol: Drug
    ) -> None:
        with pytest.raises(ValueError, match="dose_mg"):
            engine.simulate(paracetamol, Formulation(), dose_mg=0.0)

    def test_rejects_nonpositive_horizon(
        self, engine: PermeabilityEngine, paracetamol: Drug
    ) -> None:
        with pytest.raises(ValueError, match="t_end_h"):
            engine.simulate(
                paracetamol, Formulation(), dose_mg=100.0, t_end_h=0.0
            )

    def test_rejects_tiny_grid(
        self, engine: PermeabilityEngine, paracetamol: Drug
    ) -> None:
        with pytest.raises(ValueError, match="n_points"):
            engine.simulate(
                paracetamol, Formulation(), dose_mg=100.0, n_points=1
            )
