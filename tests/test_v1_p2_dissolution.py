"""V1 P2 Step 3 — dissolution engine tests.

Exercises the pH-dependent Noyes-Whitney engine end-to-end:

- Pure-helper math (Henderson-Hasselbalch, surface area, rate)
- Mass balance across the full 9-segment transit
- BCS class I drug (paracetamol) dissolves ≥90% before leaving the SI
- BCS class II drug (ibuprofen) is solubility-limited in stomach and
  only dissolves appreciably once it reaches the jejunum
- SOLUTION formulation is fully dissolved from t=0
- ENTERIC_COATED formulation skips stomach dissolution
- Disintegration delay suppresses dissolution before t_disint
- Frozen result is immutable
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neogen.core.entities.drug import BCSClass, Drug, Ionization
from neogen.core.entities.formulation import Formulation, FormulationType
from neogen.engine.absorption import (
    DissolutionEngine,
    DissolutionTrace,
    GISegment,
    GutTransit,
    bile_salt_enhancement,
    build_reference_gi_tract,
    noyes_whitney_rate_mg_per_h,
    particle_surface_area_cm2,
    solubility_at_ph,
)


# ────────────────────────────────────────────────────────────────
# Drug fixtures (mirrored from the legacy dissolution database)
# ────────────────────────────────────────────────────────────────


@pytest.fixture
def paracetamol() -> Drug:
    """BCS class I — high solubility, high permeability."""
    return Drug(
        name="paracetamol",
        smiles="CC(=O)Nc1ccc(O)cc1",
        molecular_weight=151.16,
        logp=0.46,
        pka_acidic=9.5,
        tpsa=49.33,
        bcs_class=BCSClass.I,
        solubility_mg_per_ml=14.0,
        ionization_class=Ionization.ACID,
    )


@pytest.fixture
def ibuprofen() -> Drug:
    """BCS class II — low solubility, high permeability. Weak acid pKa 4.4."""
    return Drug(
        name="ibuprofen",
        smiles="CC(C)Cc1ccc(cc1)C(C)C(O)=O",
        molecular_weight=206.28,
        logp=3.97,
        pka_acidic=4.4,
        tpsa=37.3,
        bcs_class=BCSClass.II,
        solubility_mg_per_ml=0.021,
        ionization_class=Ionization.ACID,
    )


@pytest.fixture
def metformin() -> Drug:
    """BCS class III — high solubility, low permeability. Weak base pKa 12.4."""
    return Drug(
        name="metformin",
        smiles="CN(C)C(=N)N=C(N)N",
        molecular_weight=129.16,
        logp=-2.64,
        pka_basic=12.4,
        tpsa=88.99,
        bcs_class=BCSClass.III,
        solubility_mg_per_ml=300.0,
        ionization_class=Ionization.BASE,
    )


@pytest.fixture
def engine() -> DissolutionEngine:
    return DissolutionEngine()


@pytest.fixture
def tract() -> GutTransit:
    return build_reference_gi_tract()


# ────────────────────────────────────────────────────────────────
# Pure helpers — Henderson-Hasselbalch
# ────────────────────────────────────────────────────────────────


class TestSolubilityAtPH:
    def test_neutral_drug_is_ph_independent(self) -> None:
        drug = Drug(
            name="neutral_probe",
            smiles="c1ccccc1",
            molecular_weight=78.11,
            logp=2.1,
            solubility_mg_per_ml=1.0,
            ionization_class=Ionization.NEUTRAL,
        )
        assert solubility_at_ph(drug, 1.5) == pytest.approx(1.0)
        assert solubility_at_ph(drug, 7.4) == pytest.approx(1.0)

    def test_weak_acid_rises_with_pH(self, ibuprofen: Drug) -> None:
        s_acid = solubility_at_ph(ibuprofen, 1.5)   # stomach
        s_jej = solubility_at_ph(ibuprofen, 6.2)   # jejunum
        s_col = solubility_at_ph(ibuprofen, 7.8)   # colon
        assert s_acid < s_jej < s_col

    def test_ibuprofen_jejunum_solubility_rough_magnitude(
        self, ibuprofen: Drug
    ) -> None:
        # pKa 4.4, pH 6.2 → factor (1 + 10^1.8) ≈ 64.1
        s = solubility_at_ph(ibuprofen, 6.2)
        expected = 0.021 * (1.0 + 10 ** (6.2 - 4.4))
        assert s == pytest.approx(expected, rel=1e-3)
        assert 1.0 < s < 2.0  # ~1.34 mg/mL

    def test_weak_base_falls_with_pH(self, metformin: Drug) -> None:
        s_low = solubility_at_ph(metformin, 1.5)
        s_high = solubility_at_ph(metformin, 7.8)
        assert s_low > s_high

    def test_missing_solubility_falls_back_to_bcs_default(self) -> None:
        """After Step 5, missing solubility falls back to a BCS-class
        reference rather than raising — keeps the P2 path robust for
        drugs without measured literature values."""
        from neogen.core.entities.drug import BCSClass
        drug = Drug(
            name="no_sol",
            smiles="CCO",
            molecular_weight=46.07,
            logp=-0.31,
            solubility_mg_per_ml=None,
            bcs_class=BCSClass.I,
            ionization_class=Ionization.NEUTRAL,
        )
        s = solubility_at_ph(drug, 6.5)
        # BCS I fallback is 10 mg/mL (see dissolution_engine.py).
        # Neutral drugs ignore pH, so the value should equal the fallback.
        assert s > 0.0
        assert s == pytest.approx(10.0, rel=1e-6)


class TestBileSaltEnhancement:
    def test_zero_bile_is_identity(self) -> None:
        assert bile_salt_enhancement(0.0) == 1.0

    def test_monotonic_in_bile(self) -> None:
        assert bile_salt_enhancement(10.0) > bile_salt_enhancement(1.0)

    def test_clamps_negative(self) -> None:
        assert bile_salt_enhancement(-5.0) == 1.0


class TestParticleSurfaceArea:
    def test_zero_mass_gives_zero_sa(self) -> None:
        assert particle_surface_area_cm2(0.0, 25.0) == 0.0

    def test_sa_linear_in_mass(self) -> None:
        sa_1 = particle_surface_area_cm2(100.0, 25.0)
        sa_2 = particle_surface_area_cm2(200.0, 25.0)
        assert sa_2 == pytest.approx(2.0 * sa_1)

    def test_sa_grows_as_radius_shrinks(self) -> None:
        """Same mass, smaller particles → more surface area."""
        sa_big = particle_surface_area_cm2(100.0, 50.0)
        sa_small = particle_surface_area_cm2(100.0, 5.0)
        assert sa_small > sa_big

    def test_sa_rejects_bad_geometry(self) -> None:
        assert particle_surface_area_cm2(100.0, 0.0) == 0.0


class TestNoyesWhitneyRate:
    def test_zero_undissolved_zero_rate(
        self, paracetamol: Drug, tract: GutTransit
    ) -> None:
        formulation = Formulation()
        stomach = tract.segment_by_name("Stomach")
        rate = noyes_whitney_rate_mg_per_h(
            0.0, 0.0, stomach, paracetamol, formulation
        )
        assert rate == 0.0

    def test_saturation_halts_dissolution(
        self, ibuprofen: Drug, tract: GutTransit
    ) -> None:
        formulation = Formulation()
        stomach = tract.segment_by_name("Stomach")
        # Stomach volume 250 mL, ibuprofen stomach solubility ~0.021 mg/mL
        # → saturation at ~5.25 mg dissolved.
        rate_below = noyes_whitney_rate_mg_per_h(
            100.0, 1.0, stomach, ibuprofen, formulation
        )
        rate_at_saturation = noyes_whitney_rate_mg_per_h(
            100.0, 10.0, stomach, ibuprofen, formulation
        )
        assert rate_below > 0.0
        assert rate_at_saturation == 0.0

    def test_rate_higher_in_jejunum_for_weak_acid(
        self, ibuprofen: Drug, tract: GutTransit
    ) -> None:
        formulation = Formulation()
        stomach = tract.segment_by_name("Stomach")
        jejunum = tract.segment_by_name("Jejunum_1")
        rate_stomach = noyes_whitney_rate_mg_per_h(
            100.0, 0.0, stomach, ibuprofen, formulation
        )
        rate_jejunum = noyes_whitney_rate_mg_per_h(
            100.0, 0.0, jejunum, ibuprofen, formulation
        )
        assert rate_jejunum > rate_stomach * 10.0  # orders of magnitude


# ────────────────────────────────────────────────────────────────
# DissolutionEngine — end-to-end integration
# ────────────────────────────────────────────────────────────────


class TestDissolutionEngineSmoke:
    def test_solution_is_pre_dissolved(
        self, engine: DissolutionEngine, paracetamol: Drug
    ) -> None:
        formulation = Formulation(
            formulation_type=FormulationType.SOLUTION,
        )
        trace = engine.simulate(
            paracetamol, formulation, dose_mg=500.0, t_end_h=6.0, n_points=31
        )
        assert trace.solver_success is True
        # At t=0 everything is already dissolved.
        assert trace.total_dissolved_mg[0] == pytest.approx(500.0, rel=1e-6)
        assert trace.total_undissolved_mg[0] == pytest.approx(0.0, abs=1e-6)
        assert trace.fraction_dissolved_cumulative[0] == pytest.approx(1.0)

    def test_mass_balance_conserved_paracetamol(
        self, engine: DissolutionEngine, paracetamol: Drug
    ) -> None:
        formulation = Formulation(formulation_type=FormulationType.IR_TABLET)
        trace = engine.simulate(
            paracetamol, formulation, dose_mg=500.0, t_end_h=24.0
        )
        # Mass balance within 1% of dose.
        assert trace.mass_balance_error < 5.0  # mg (out of 500)

    def test_mass_balance_conserved_ibuprofen(
        self, engine: DissolutionEngine, ibuprofen: Drug
    ) -> None:
        formulation = Formulation(
            formulation_type=FormulationType.IR_TABLET,
            particle_radius_um=30.0,
        )
        trace = engine.simulate(
            ibuprofen, formulation, dose_mg=400.0, t_end_h=24.0
        )
        assert trace.mass_balance_error < 4.0  # mg (out of 400)

    def test_trace_shapes(
        self, engine: DissolutionEngine, paracetamol: Drug
    ) -> None:
        formulation = Formulation()
        trace = engine.simulate(
            paracetamol, formulation, dose_mg=500.0, t_end_h=10.0, n_points=51
        )
        assert trace.t_h.shape == (51,)
        assert trace.undissolved_mg.shape == (51, 9)
        assert trace.dissolved_mg.shape == (51, 9)
        assert trace.cumulative_dissolved_mg.shape == (51,)
        assert trace.excreted_mg.shape == (51,)
        assert trace.n_segments == 9


class TestBCSClassBehaviour:
    """The central scientific invariants promised in Step 3 of the plan."""

    def test_paracetamol_mostly_dissolved_by_end_of_small_intestine(
        self, engine: DissolutionEngine, paracetamol: Drug, tract: GutTransit
    ) -> None:
        formulation = Formulation(
            formulation_type=FormulationType.IR_TABLET,
            particle_radius_um=20.0,
            disintegration_time_min=7.0,
        )
        trace = engine.simulate(
            paracetamol,
            formulation,
            dose_mg=500.0,
            gut_transit=tract,
            t_end_h=24.0,
            n_points=241,
        )
        # Small intestine exit ≈ gastric + SI transit.
        t_si_exit = tract.gastric_emptying_h + tract.small_intestine_transit_h
        idx = int(np.searchsorted(trace.t_h, t_si_exit))
        idx = min(idx, len(trace.t_h) - 1)
        frac_by_si_exit = trace.fraction_dissolved_cumulative[idx]
        assert frac_by_si_exit >= 0.90, (
            f"Paracetamol (BCS I) only {frac_by_si_exit:.1%} dissolved "
            f"at SI exit (expected ≥90%)"
        )
        # And by 24h essentially all of it has dissolved.
        assert trace.final_cumulative_fraction >= 0.95

    def test_ibuprofen_solubility_limited_in_stomach(
        self, engine: DissolutionEngine, ibuprofen: Drug, tract: GutTransit
    ) -> None:
        formulation = Formulation(
            formulation_type=FormulationType.IR_TABLET,
            particle_radius_um=30.0,
            disintegration_time_min=12.0,
        )
        trace = engine.simulate(
            ibuprofen,
            formulation,
            dose_mg=400.0,
            gut_transit=tract,
            t_end_h=24.0,
            n_points=241,
        )
        # While the dose is still in the stomach (before gastric emptying
        # finishes), the dissolved mass is clamped by solubility.
        t_idx_in_stomach = int(
            np.searchsorted(trace.t_h, tract.gastric_emptying_h)
        )
        t_idx_in_stomach = max(t_idx_in_stomach - 1, 1)
        stomach_dissolved_peak = float(
            np.max(trace.dissolved_fraction_in_segment("Stomach"))
        )
        # Stomach cs ≈ 0.021 mg/mL × 250 mL ≈ 5.25 mg — capped at ~1-2%.
        assert stomach_dissolved_peak < 20.0, (
            f"Ibuprofen stomach dissolved peak {stomach_dissolved_peak:.2f} mg "
            f"exceeds saturation expectation"
        )

    def test_ibuprofen_dissolves_more_in_jejunum_than_stomach(
        self, engine: DissolutionEngine, ibuprofen: Drug, tract: GutTransit
    ) -> None:
        formulation = Formulation(
            formulation_type=FormulationType.IR_TABLET,
            particle_radius_um=30.0,
            disintegration_time_min=12.0,
        )
        trace = engine.simulate(
            ibuprofen,
            formulation,
            dose_mg=400.0,
            gut_transit=tract,
            t_end_h=24.0,
            n_points=241,
        )
        stomach_peak = float(
            np.max(trace.dissolved_fraction_in_segment("Stomach"))
        )
        jejunum_peak = float(
            np.max(trace.dissolved_fraction_in_segment("Jejunum_1"))
        )
        assert jejunum_peak > stomach_peak

    def test_paracetamol_dissolves_faster_than_ibuprofen(
        self,
        engine: DissolutionEngine,
        paracetamol: Drug,
        ibuprofen: Drug,
    ) -> None:
        """BCS I drug should out-dissolve BCS II drug at matched dose."""
        paracetamol_trace = engine.simulate(
            paracetamol,
            Formulation(formulation_type=FormulationType.IR_TABLET),
            dose_mg=500.0,
            t_end_h=6.0,
            n_points=61,
        )
        ibuprofen_trace = engine.simulate(
            ibuprofen,
            Formulation(
                formulation_type=FormulationType.IR_TABLET,
                particle_radius_um=30.0,
            ),
            dose_mg=500.0,
            t_end_h=6.0,
            n_points=61,
        )
        assert (
            paracetamol_trace.final_cumulative_fraction
            > ibuprofen_trace.final_cumulative_fraction
        )


class TestFormulationBranches:
    def test_enteric_coated_skips_stomach(
        self, engine: DissolutionEngine, paracetamol: Drug
    ) -> None:
        formulation = Formulation(
            formulation_type=FormulationType.ENTERIC_COATED,
            particle_radius_um=20.0,
        )
        trace = engine.simulate(
            paracetamol, formulation, dose_mg=500.0, t_end_h=12.0, n_points=121
        )
        # Drug begins life in the duodenum — stomach column stays empty.
        assert np.all(trace.dissolved_fraction_in_segment("Stomach") == 0.0)
        # And by 12h a meaningful amount has dissolved.
        assert trace.final_cumulative_fraction > 0.5

    def test_disintegration_delay_suppresses_early_dissolution(
        self, engine: DissolutionEngine, paracetamol: Drug
    ) -> None:
        # 30-minute disintegration — no drug should dissolve in the
        # first ~0.5h.
        formulation = Formulation(
            formulation_type=FormulationType.IR_TABLET,
            disintegration_time_min=30.0,
        )
        trace = engine.simulate(
            paracetamol, formulation, dose_mg=500.0, t_end_h=6.0, n_points=121
        )
        # At 10 minutes (0.167 h) nothing should have dissolved.
        idx_10min = int(np.searchsorted(trace.t_h, 0.15))
        assert trace.cumulative_dissolved_mg[idx_10min] < 1.0


# ────────────────────────────────────────────────────────────────
# Result dataclass — shape & immutability
# ────────────────────────────────────────────────────────────────


class TestDissolutionTraceContract:
    def test_frozen(
        self, engine: DissolutionEngine, paracetamol: Drug
    ) -> None:
        trace = engine.simulate(
            paracetamol, Formulation(), dose_mg=100.0, t_end_h=2.0, n_points=11
        )
        with pytest.raises(FrozenInstanceError):
            trace.dose_mg = 200.0  # type: ignore[misc]

    def test_segment_names_match_reference_tract(
        self, engine: DissolutionEngine, paracetamol: Drug
    ) -> None:
        trace = engine.simulate(
            paracetamol, Formulation(), dose_mg=100.0, t_end_h=2.0, n_points=11
        )
        assert trace.segment_names[0] == "Stomach"
        assert trace.segment_names[-1] == "Colon_trans"
        assert len(trace.segment_names) == 9

    def test_cumulative_dissolved_monotonic(
        self, engine: DissolutionEngine, paracetamol: Drug
    ) -> None:
        trace = engine.simulate(
            paracetamol, Formulation(), dose_mg=500.0, t_end_h=12.0, n_points=121
        )
        diffs = np.diff(trace.cumulative_dissolved_mg)
        # Allow tiny numerical noise (~rtol·dose).
        assert np.all(diffs >= -1e-6)

    def test_fraction_dissolved_in_unit_interval(
        self, engine: DissolutionEngine, paracetamol: Drug
    ) -> None:
        trace = engine.simulate(
            paracetamol, Formulation(), dose_mg=500.0, t_end_h=12.0, n_points=61
        )
        frac = trace.fraction_dissolved_cumulative
        assert float(frac.min()) >= -1e-9
        assert float(frac.max()) <= 1.0 + 1e-6

    def test_unknown_segment_name_raises(
        self, engine: DissolutionEngine, paracetamol: Drug
    ) -> None:
        trace = engine.simulate(
            paracetamol, Formulation(), dose_mg=100.0, t_end_h=2.0, n_points=11
        )
        with pytest.raises(KeyError):
            trace.dissolved_fraction_in_segment("Esophagus")


# ────────────────────────────────────────────────────────────────
# Validation — engine input guards
# ────────────────────────────────────────────────────────────────


class TestEngineInputValidation:
    def test_rejects_nonpositive_dose(
        self, engine: DissolutionEngine, paracetamol: Drug
    ) -> None:
        with pytest.raises(ValueError, match="dose_mg"):
            engine.simulate(paracetamol, Formulation(), dose_mg=0.0)

    def test_rejects_nonpositive_horizon(
        self, engine: DissolutionEngine, paracetamol: Drug
    ) -> None:
        with pytest.raises(ValueError, match="t_end_h"):
            engine.simulate(
                paracetamol, Formulation(), dose_mg=100.0, t_end_h=0.0
            )

    def test_rejects_tiny_grid(
        self, engine: DissolutionEngine, paracetamol: Drug
    ) -> None:
        with pytest.raises(ValueError, match="n_points"):
            engine.simulate(
                paracetamol, Formulation(), dose_mg=100.0, n_points=1
            )
