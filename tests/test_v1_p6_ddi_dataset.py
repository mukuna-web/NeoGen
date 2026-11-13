"""P6.3 — integrity tests for the V1 DDI benchmark dataset.

These tests validate the *dataset*, not the *engine*. They ensure:

* The factory returns a non-empty, frozen tuple of valid pairs.
* Every victim and every perpetrator is resolvable by
  :class:`DrugFactory`.
* Every case has a source citation, positive AUCR, and the V1
  tolerance (2.0-fold).
* Each perpetrator carries the inhibition / induction parameters the
  pair relies on (Ki for reversible, kinact/K_I for MBI, Emax/EC50
  for induction).
* Each victim has non-zero fm on at least one CYP the perpetrator
  can modulate — so every DDI row is at least mechanistically
  plausible, not just a typing exercise.
* Coverage across CYP isoforms and mechanisms is broad enough to
  exercise the DDI orchestrator's major code paths.
* The dataset runs end-to-end through :class:`BenchmarkRunner`
  without any pair raising an unhandled exception.

Engine accuracy (% within 2-fold) is **not** asserted here — that is
the job of ``test_v1_p6_acceptance_gate.py`` (P6.5).
"""
from __future__ import annotations

import pytest

from neogen.engine.drug_factory import DrugFactory
from neogen.validation import (
    BenchmarkRunner,
    DDIBenchmarkCase,
    DDIBenchmarkResult,
    build_ddi_benchmark_v1,
)


# Which CYPs each perpetrator in this dataset is allowed to touch.
# Derived from the drug factory's literature profiles, not invented:
#   ketoconazole/itraconazole   — reversible CYP3A4
#   clarithromycin              — MBI CYP3A4
#   rifampicin                  — induction CYP3A4 + CYP2C9
#   fluoxetine                  — reversible CYP2D6
#   omeprazole                  — reversible CYP2C19
_PERPETRATOR_TARGETS = {
    "ketoconazole": ("cyp3a4",),
    "itraconazole": ("cyp3a4",),
    "clarithromycin": ("cyp3a4",),
    "rifampicin": ("cyp3a4", "cyp2c9"),
    "fluoxetine": ("cyp2d6",),
    "omeprazole": ("cyp2c19",),
}

# Victims that are intentionally mismatched to their perpetrator
# (negative controls). Excluded from the victim-fm plausibility test.
_NEGATIVE_CONTROLS = {
    "ketoconazole+warfarin-oral-10mg",
    "ketoconazole+ibuprofen-oral-400mg",
}


# ════════════════════════════════════════════════════════════════
# DATASET SHAPE
# ════════════════════════════════════════════════════════════════


class TestDatasetShape:
    def test_returns_nonempty_tuple(self) -> None:
        cases = build_ddi_benchmark_v1()
        assert isinstance(cases, tuple)
        assert len(cases) >= 20, (
            f"dataset too small: got {len(cases)}, expected ≥ 20"
        )

    def test_target_size_at_least_thirty(self) -> None:
        """V1 target is 30 DDI pairs."""
        cases = build_ddi_benchmark_v1()
        assert len(cases) >= 30

    def test_all_cases_are_ddi_benchmark_case(self) -> None:
        cases = build_ddi_benchmark_v1()
        for c in cases:
            assert isinstance(c, DDIBenchmarkCase)

    def test_unique_case_names(self) -> None:
        cases = build_ddi_benchmark_v1()
        names = [c.name for c in cases]
        assert len(names) == len(set(names)), (
            "duplicate case names detected"
        )


# ════════════════════════════════════════════════════════════════
# CASE-LEVEL INVARIANTS
# ════════════════════════════════════════════════════════════════


class TestCaseInvariants:
    def test_every_case_has_source(self) -> None:
        cases = build_ddi_benchmark_v1()
        for c in cases:
            assert c.source.strip(), f"{c.name} missing source"

    def test_positive_aucr(self) -> None:
        cases = build_ddi_benchmark_v1()
        for c in cases:
            assert c.observed_aucr > 0, (
                f"{c.name} has non-positive AUCR"
            )

    def test_positive_victim_dose(self) -> None:
        cases = build_ddi_benchmark_v1()
        for c in cases:
            assert c.victim_dose_mg > 0, (
                f"{c.name} has non-positive victim dose"
            )

    def test_positive_perpetrator_concentration(self) -> None:
        cases = build_ddi_benchmark_v1()
        for c in cases:
            assert c.perpetrator_steady_state_conc_um > 0, (
                f"{c.name} has non-positive perpetrator Cu"
            )

    def test_tolerance_is_two_fold(self) -> None:
        cases = build_ddi_benchmark_v1()
        for c in cases:
            assert c.tolerance_fold == 2.0, (
                f"{c.name} overrides tolerance_fold — not allowed at V1"
            )

    def test_routes_are_iv_or_oral(self) -> None:
        cases = build_ddi_benchmark_v1()
        for c in cases:
            assert c.victim_route in ("IV", "ORAL"), (
                f"{c.name} has bad route {c.victim_route!r}"
            )


# ════════════════════════════════════════════════════════════════
# DRUG RESOLVABILITY
# ════════════════════════════════════════════════════════════════


class TestDrugResolvability:
    def test_every_victim_resolvable(self) -> None:
        factory = DrugFactory()
        cases = build_ddi_benchmark_v1()
        for c in cases:
            try:
                drug = factory.create(c.victim_drug_name)
            except Exception as exc:  # pragma: no cover
                pytest.fail(
                    f"{c.name}: DrugFactory cannot resolve victim "
                    f"{c.victim_drug_name!r} — {type(exc).__name__}: {exc}"
                )
            assert drug is not None

    def test_every_perpetrator_resolvable(self) -> None:
        factory = DrugFactory()
        cases = build_ddi_benchmark_v1()
        for c in cases:
            try:
                drug = factory.create(c.perpetrator_drug_name)
            except Exception as exc:  # pragma: no cover
                pytest.fail(
                    f"{c.name}: DrugFactory cannot resolve perpetrator "
                    f"{c.perpetrator_drug_name!r} — "
                    f"{type(exc).__name__}: {exc}"
                )
            assert drug is not None


# ════════════════════════════════════════════════════════════════
# MECHANISM PLAUSIBILITY — perpetrator has the machinery it needs
# ════════════════════════════════════════════════════════════════


class TestPerpetratorMechanism:
    """Every perpetrator in the dataset must actually carry the
    Ki / kinact / Emax fields its mechanism relies on. Without this
    check, it's easy to silently reference a drug whose inhibition
    profile is all-None and watch every AUCR predict 1.0.
    """

    def test_ketoconazole_has_reversible_cyp3a4(self) -> None:
        d = DrugFactory().create("ketoconazole")
        assert d.inhibition.ki_cyp3a4 is not None
        assert d.inhibition.ki_cyp3a4 > 0

    def test_itraconazole_has_reversible_cyp3a4(self) -> None:
        d = DrugFactory().create("itraconazole")
        assert d.inhibition.ki_cyp3a4 is not None
        assert d.inhibition.ki_cyp3a4 > 0

    def test_clarithromycin_has_mbi_cyp3a4(self) -> None:
        d = DrugFactory().create("clarithromycin")
        assert d.inhibition.kinact_cyp3a4 is not None
        assert d.inhibition.k_i_mbi_cyp3a4 is not None
        assert d.inhibition.kinact_cyp3a4 > 0
        assert d.inhibition.k_i_mbi_cyp3a4 > 0

    def test_rifampicin_has_cyp3a4_induction(self) -> None:
        d = DrugFactory().create("rifampicin")
        assert d.induction.indmax_cyp3a4 is not None
        assert d.induction.indc50_cyp3a4 is not None
        assert d.induction.indmax_cyp3a4 > 1.0

    def test_rifampicin_has_cyp2c9_induction(self) -> None:
        d = DrugFactory().create("rifampicin")
        assert d.induction.indmax_cyp2c9 is not None
        assert d.induction.indc50_cyp2c9 is not None

    def test_fluoxetine_has_reversible_cyp2d6(self) -> None:
        d = DrugFactory().create("fluoxetine")
        assert d.inhibition.ki_cyp2d6 is not None
        assert d.inhibition.ki_cyp2d6 > 0

    def test_omeprazole_has_reversible_cyp2c19(self) -> None:
        d = DrugFactory().create("omeprazole")
        assert d.inhibition.ki_cyp2c19 is not None
        assert d.inhibition.ki_cyp2c19 > 0


# ════════════════════════════════════════════════════════════════
# MECHANISM PLAUSIBILITY — victim has non-zero fm on a targeted CYP
# ════════════════════════════════════════════════════════════════


class TestVictimMechanism:
    """For every non-control pair, the victim must be metabolized
    (fm > 0) by at least one CYP that the perpetrator modulates.
    Negative controls are explicitly exempt.
    """

    def test_victim_has_fm_on_perpetrator_target(self) -> None:
        factory = DrugFactory()
        cases = build_ddi_benchmark_v1()
        for c in cases:
            if c.name in _NEGATIVE_CONTROLS:
                continue
            targets = _PERPETRATOR_TARGETS[c.perpetrator_drug_name]
            victim = factory.create(c.victim_drug_name)
            fms = {
                cyp: getattr(victim.metabolism, cyp)
                for cyp in targets
            }
            assert any(v > 0.0 for v in fms.values()), (
                f"{c.name}: victim {c.victim_drug_name!r} has zero fm "
                f"on every perpetrator target CYP {targets} "
                f"(got {fms}) — the pair is mechanistically silent"
            )


# ════════════════════════════════════════════════════════════════
# COVERAGE DIMENSIONS
# ════════════════════════════════════════════════════════════════


class TestCoverage:
    def test_perpetrator_diversity(self) -> None:
        cases = build_ddi_benchmark_v1()
        perps = {c.perpetrator_drug_name for c in cases}
        assert len(perps) >= 5, (
            f"need ≥ 5 unique perpetrators, got {len(perps)}"
        )

    def test_victim_diversity(self) -> None:
        cases = build_ddi_benchmark_v1()
        victims = {c.victim_drug_name for c in cases}
        assert len(victims) >= 6, (
            f"need ≥ 6 unique victims, got {len(victims)}"
        )

    def test_all_three_mechanisms_present(self) -> None:
        """Dataset must exercise reversible + MBI + induction."""
        cases = build_ddi_benchmark_v1()
        perps = {c.perpetrator_drug_name for c in cases}
        # Reversible
        assert perps & {"ketoconazole", "itraconazole", "fluoxetine",
                        "omeprazole"}, "no reversible inhibitor in set"
        # MBI
        assert "clarithromycin" in perps, "no MBI perpetrator in set"
        # Induction
        assert "rifampicin" in perps, "no inducer in set"

    def test_multiple_cyps_targeted(self) -> None:
        """At least three CYP isoforms represented across pairs."""
        cases = build_ddi_benchmark_v1()
        touched: set[str] = set()
        for c in cases:
            if c.name in _NEGATIVE_CONTROLS:
                continue
            touched.update(_PERPETRATOR_TARGETS[c.perpetrator_drug_name])
        assert len(touched) >= 3, (
            f"need ≥ 3 CYP isoforms, got {sorted(touched)}"
        )

    def test_both_routes_present(self) -> None:
        cases = build_ddi_benchmark_v1()
        routes = {c.victim_route for c in cases}
        assert routes == {"IV", "ORAL"}, (
            f"dataset should cover both IV and oral victims, got {routes}"
        )

    def test_enzyme_pool_engaged_for_time_dependent_cases(self) -> None:
        """MBI + induction cases must opt into the enzyme-pool evolver
        (otherwise their mechanism is silent at t=0). Reversible cases
        should leave it off to keep the ODE cheap."""
        cases = build_ddi_benchmark_v1()
        for c in cases:
            if c.perpetrator_drug_name in ("clarithromycin", "rifampicin"):
                assert c.include_enzyme_pool, (
                    f"{c.name}: MBI/induction case must engage enzyme pool"
                )
            elif c.perpetrator_drug_name in (
                "ketoconazole", "itraconazole", "fluoxetine", "omeprazole"
            ):
                assert not c.include_enzyme_pool, (
                    f"{c.name}: reversible inhibitor should not engage "
                    f"enzyme pool evolver"
                )

    def test_induction_and_inhibition_aucr_directions(self) -> None:
        """Sanity: every rifampicin pair should report AUCR < 1;
        every ketoconazole/itraconazole/clarithromycin pair targeting
        a CYP3A4-dominant victim should report AUCR > 1."""
        cases = build_ddi_benchmark_v1()
        for c in cases:
            if c.perpetrator_drug_name == "rifampicin":
                assert c.observed_aucr < 1.0, (
                    f"{c.name}: inducer should reduce AUC "
                    f"(AUCR={c.observed_aucr})"
                )
            if c.name in _NEGATIVE_CONTROLS:
                continue
            if c.perpetrator_drug_name in ("ketoconazole", "itraconazole",
                                            "clarithromycin"):
                assert c.observed_aucr >= 1.0, (
                    f"{c.name}: CYP3A4 inhibitor should not reduce AUC "
                    f"(AUCR={c.observed_aucr})"
                )


# ════════════════════════════════════════════════════════════════
# END-TO-END RUN — no unhandled exceptions
# ════════════════════════════════════════════════════════════════


class TestEndToEndRun:
    """The dataset must run through BenchmarkRunner.run_all_ddi
    without any pair raising. Engine accuracy is NOT asserted here.
    """

    def test_all_cases_produce_results(self) -> None:
        cases = build_ddi_benchmark_v1()
        runner = BenchmarkRunner()
        results = runner.run_all_ddi(cases)
        assert len(results) == len(cases)
        for r in results:
            assert isinstance(r, DDIBenchmarkResult)

    def test_no_case_fails_with_exception(self) -> None:
        cases = build_ddi_benchmark_v1()
        runner = BenchmarkRunner()
        results = runner.run_all_ddi(cases)
        failures = [r for r in results if not r.success]
        assert not failures, (
            "engine raised on DDI cases: "
            + ", ".join(
                f"{r.case_name} ({r.error_message})" for r in failures
            )
        )

    def test_every_successful_case_produces_aucr(self) -> None:
        """Every pair that runs successfully must produce a scored
        AUCR — otherwise the dataset is unscoreable at P6.5 time."""
        cases = build_ddi_benchmark_v1()
        runner = BenchmarkRunner()
        results = runner.run_all_ddi(cases)
        for r in results:
            if r.success:
                assert r.aucr.fold_error is not None, (
                    f"{r.case_name}: successful run but no AUCR score"
                )

    def test_engine_version_consistent_across_batch(self) -> None:
        cases = build_ddi_benchmark_v1()
        runner = BenchmarkRunner()
        results = runner.run_all_ddi(cases)
        versions = {r.engine_version for r in results if r.success}
        assert len(versions) == 1, (
            f"mixed engine versions in one batch: {versions}"
        )
