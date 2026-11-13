"""P6.2 — integrity tests for the V1 single-drug PK benchmark dataset.

These tests validate the *dataset*, not the *engine*. They ensure:

* The factory returns a non-empty, frozen tuple of valid cases.
* Every drug in the dataset is resolvable by :class:`DrugFactory`
  (so no dangling case will blow up the runner at P6.5 time).
* Every case has a source citation and at least one scorable metric.
* Coverage across route, ionization, and clearance mechanism is
  broad enough to exercise the engine's major code paths.
* The dataset runs end-to-end through :class:`BenchmarkRunner`
  without any case raising an unhandled exception.

Engine accuracy (% within 2-fold) is **not** asserted here — that is
the job of ``test_v1_p6_acceptance_gate.py`` (P6.5). A dataset can
be perfectly well-formed and still score poorly.
"""
from __future__ import annotations

import pytest

from neogen.engine.drug_factory import DrugFactory
from neogen.validation import (
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkRunner,
    build_pk_benchmark_v1,
)


# ════════════════════════════════════════════════════════════════
# DATASET SHAPE
# ════════════════════════════════════════════════════════════════


class TestDatasetShape:
    def test_returns_nonempty_tuple(self) -> None:
        cases = build_pk_benchmark_v1()
        assert isinstance(cases, tuple)
        assert len(cases) >= 20, (
            f"dataset too small: got {len(cases)}, expected ≥ 20"
        )

    def test_target_size_at_least_thirty(self) -> None:
        """V1 target is ~30 single-drug cases."""
        cases = build_pk_benchmark_v1()
        assert len(cases) >= 30

    def test_all_cases_are_benchmark_case(self) -> None:
        cases = build_pk_benchmark_v1()
        for c in cases:
            assert isinstance(c, BenchmarkCase)

    def test_unique_case_names(self) -> None:
        cases = build_pk_benchmark_v1()
        names = [c.name for c in cases]
        assert len(names) == len(set(names)), (
            "duplicate case names detected"
        )


# ════════════════════════════════════════════════════════════════
# CASE-LEVEL INVARIANTS
# ════════════════════════════════════════════════════════════════


class TestCaseInvariants:
    def test_every_case_has_source(self) -> None:
        cases = build_pk_benchmark_v1()
        for c in cases:
            assert c.source.strip(), f"{c.name} missing source"

    def test_every_case_has_at_least_one_reference(self) -> None:
        cases = build_pk_benchmark_v1()
        for c in cases:
            assert c.has_any_reference, f"{c.name} has no scorable metric"

    def test_positive_dose(self) -> None:
        cases = build_pk_benchmark_v1()
        for c in cases:
            assert c.dose_mg > 0, f"{c.name} has non-positive dose"

    def test_tolerance_is_two_fold(self) -> None:
        """V1 PBPK convention is 2-fold. Dataset should not silently change it."""
        cases = build_pk_benchmark_v1()
        for c in cases:
            assert c.tolerance_fold == 2.0, (
                f"{c.name} overrides tolerance_fold — not allowed at V1"
            )

    def test_observed_values_are_positive(self) -> None:
        cases = build_pk_benchmark_v1()
        for c in cases:
            for field in (
                "observed_cmax_mg_per_L",
                "observed_auc_mg_h_per_L",
                "observed_half_life_h",
            ):
                val = getattr(c, field)
                if val is not None:
                    assert val > 0, f"{c.name}.{field} must be positive"


# ════════════════════════════════════════════════════════════════
# DRUG RESOLVABILITY — catches dangling cases early
# ════════════════════════════════════════════════════════════════


class TestDrugResolvability:
    def test_every_drug_resolvable_by_factory(self) -> None:
        factory = DrugFactory()
        cases = build_pk_benchmark_v1()
        for c in cases:
            try:
                drug = factory.create(c.drug_name)
            except Exception as exc:  # pragma: no cover - explicit failure
                pytest.fail(
                    f"{c.name}: DrugFactory cannot resolve "
                    f"{c.drug_name!r} — {type(exc).__name__}: {exc}"
                )
            assert drug is not None


# ════════════════════════════════════════════════════════════════
# COVERAGE DIMENSIONS
# ════════════════════════════════════════════════════════════════


class TestCoverage:
    def test_both_routes_present(self) -> None:
        cases = build_pk_benchmark_v1()
        routes = {c.route for c in cases}
        assert routes == {"IV", "ORAL"}

    def test_meaningful_iv_coverage(self) -> None:
        """At least 5 IV cases — the engine's IV path must be exercised."""
        cases = build_pk_benchmark_v1()
        iv = [c for c in cases if c.route == "IV"]
        assert len(iv) >= 5

    def test_meaningful_oral_coverage(self) -> None:
        """At least 15 oral cases — oral is the dominant clinical route."""
        cases = build_pk_benchmark_v1()
        oral = [c for c in cases if c.route == "ORAL"]
        assert len(oral) >= 15

    def test_minimum_drug_diversity(self) -> None:
        cases = build_pk_benchmark_v1()
        drugs = {c.drug_name for c in cases}
        assert len(drugs) >= 15, (
            f"need ≥ 15 unique drugs, got {len(drugs)}"
        )

    def test_ionization_class_spread(self) -> None:
        """Acids, bases, and neutrals must all appear in the set."""
        factory = DrugFactory()
        cases = build_pk_benchmark_v1()
        classes = set()
        for c in cases:
            classes.add(factory.create(c.drug_name).ionization_class.name)
        assert "ACID" in classes
        assert "BASE" in classes
        assert "NEUTRAL" in classes

    def test_half_life_spread(self) -> None:
        """Short (< 3h) and long (> 20h) t½ drugs both represented."""
        cases = build_pk_benchmark_v1()
        short = [
            c
            for c in cases
            if c.observed_half_life_h is not None
            and c.observed_half_life_h < 3.0
        ]
        long_ = [
            c
            for c in cases
            if c.observed_half_life_h is not None
            and c.observed_half_life_h > 20.0
        ]
        assert len(short) >= 2, "need short-t½ drugs (e.g. midazolam)"
        assert len(long_) >= 2, "need long-t½ drugs (e.g. warfarin, diazepam)"

    def test_cmax_auc_half_life_all_exercised(self) -> None:
        cases = build_pk_benchmark_v1()
        n_cmax = sum(1 for c in cases if c.observed_cmax_mg_per_L is not None)
        n_auc = sum(1 for c in cases if c.observed_auc_mg_h_per_L is not None)
        n_hl = sum(1 for c in cases if c.observed_half_life_h is not None)
        assert n_cmax >= 20
        assert n_auc >= 20
        assert n_hl >= 20


# ════════════════════════════════════════════════════════════════
# END-TO-END RUN — no unhandled exceptions
# ════════════════════════════════════════════════════════════════


class TestEndToEndRun:
    """The dataset must run through BenchmarkRunner without any case raising.

    Engine accuracy is NOT asserted here — that's P6.5's job. We only
    check that every row produces a ``BenchmarkResult`` (success or
    caught failure) rather than crashing the batch.
    """

    def test_all_cases_produce_results(self) -> None:
        cases = build_pk_benchmark_v1()
        runner = BenchmarkRunner()
        results = runner.run_all(cases)
        assert len(results) == len(cases)
        for r in results:
            assert isinstance(r, BenchmarkResult)

    def test_no_case_fails_with_exception(self) -> None:
        """No case should end up as success=False (that would mean the
        simulation raised). Individual metrics may be out-of-tolerance,
        but the engine must complete every case."""
        cases = build_pk_benchmark_v1()
        runner = BenchmarkRunner()
        results = runner.run_all(cases)
        failures = [r for r in results if not r.success]
        assert not failures, (
            "engine raised on cases: "
            + ", ".join(f"{r.case_name} ({r.error_message})" for r in failures)
        )

    def test_engine_version_consistent_across_batch(self) -> None:
        cases = build_pk_benchmark_v1()
        runner = BenchmarkRunner()
        results = runner.run_all(cases)
        versions = {r.engine_version for r in results if r.success}
        assert len(versions) == 1, (
            f"mixed engine versions in one batch: {versions}"
        )
