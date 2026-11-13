"""P6.4 — tests for the combined benchmark report writer.

These tests use **synthetic** scorecards (not a full engine run) so
they are fast and deterministic. The report writer is a pure
composition layer — its job is to take PK + DDI scorecards plus
metadata and emit markdown / JSON. No engine math is exercised here.

A separate end-to-end test (``test_end_to_end_real_data``) runs the
full pipeline against the real datasets to confirm wiring.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import pytest

from neogen.validation import (
    BenchmarkReport,
    BenchmarkRunner,
    DDIScorecard,
    RunMetadata,
    Scorecard,
    build_ddi_benchmark_v1,
    build_pk_benchmark_v1,
)
from neogen.validation.benchmark_runner import (
    BenchmarkResult,
    DDIBenchmarkResult,
    MetricScore,
)


# ════════════════════════════════════════════════════════════════
# SYNTHETIC-RESULT HELPERS
# ════════════════════════════════════════════════════════════════


def _pk_result(
    name: str,
    drug: str,
    cmax_obs: Optional[float] = 1.0,
    cmax_pred: Optional[float] = 1.1,
    auc_obs: Optional[float] = 10.0,
    auc_pred: Optional[float] = 12.0,
    hl_obs: Optional[float] = 5.0,
    hl_pred: Optional[float] = 4.5,
    tolerance: float = 2.0,
    success: bool = True,
    error: Optional[str] = None,
) -> BenchmarkResult:
    def _mk(name: str, obs: Optional[float], pred: Optional[float]) -> MetricScore:
        if obs is None or pred is None:
            return MetricScore(
                name=name,
                observed=obs,
                predicted=pred,
                fold_error=None,
                within_tolerance=None,
            )
        fold = max(pred / obs, obs / pred)
        return MetricScore(
            name=name,
            observed=obs,
            predicted=pred,
            fold_error=fold,
            within_tolerance=fold <= tolerance,
        )

    return BenchmarkResult(
        case_name=name,
        drug_name=drug,
        success=success,
        error_message=error,
        cmax=_mk("Cmax", cmax_obs, cmax_pred),
        auc=_mk("AUC", auc_obs, auc_pred),
        half_life=_mk("t½", hl_obs, hl_pred),
        engine_version="v1.0.0-test",
    )


def _ddi_result(
    name: str,
    victim: str,
    perpetrator: str,
    obs_aucr: float = 5.0,
    pred_aucr: Optional[float] = 5.5,
    tolerance: float = 2.0,
    success: bool = True,
    error: Optional[str] = None,
) -> DDIBenchmarkResult:
    if pred_aucr is None:
        aucr = MetricScore(
            name="AUCR",
            observed=obs_aucr,
            predicted=None,
            fold_error=None,
            within_tolerance=None,
        )
    else:
        fold = max(pred_aucr / obs_aucr, obs_aucr / pred_aucr)
        aucr = MetricScore(
            name="AUCR",
            observed=obs_aucr,
            predicted=pred_aucr,
            fold_error=fold,
            within_tolerance=fold <= tolerance,
        )
    return DDIBenchmarkResult(
        case_name=name,
        victim=victim,
        perpetrator=perpetrator,
        success=success,
        error_message=error,
        aucr=aucr,
        engine_version="v1.0.0-test",
    )


def _synthetic_pk_scorecard() -> Scorecard:
    return Scorecard.build(
        [
            _pk_result("drug-a", "a", 1.0, 1.1, 10.0, 11.0, 5.0, 5.2),
            _pk_result("drug-b", "b", 1.0, 3.5, 10.0, 35.0, 5.0, 18.0),  # all out
            _pk_result("drug-c", "c", 1.0, 1.4, 10.0, 14.0, 5.0, 6.0),
        ]
    )


def _synthetic_ddi_scorecard() -> DDIScorecard:
    return DDIScorecard.build(
        [
            _ddi_result("keto+mida", "midazolam", "ketoconazole",
                        obs_aucr=5.0, pred_aucr=5.2),
            _ddi_result("rif+mida", "midazolam", "rifampicin",
                        obs_aucr=0.04, pred_aucr=0.5),  # out
        ]
    )


# ════════════════════════════════════════════════════════════════
# RUN METADATA
# ════════════════════════════════════════════════════════════════


class TestRunMetadata:
    def test_capture_has_required_fields(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        assert meta.engine_version == "v1.0.0"
        assert "T" in meta.timestamp_utc  # ISO 8601
        assert "@" in meta.host
        assert meta.git_sha is None
        assert meta.notes == ""

    def test_capture_accepts_git_sha_and_notes(self) -> None:
        meta = RunMetadata.capture(
            engine_version="v1.0.0",
            git_sha="abc1234",
            notes="calibration run",
        )
        assert meta.git_sha == "abc1234"
        assert meta.notes == "calibration run"

    def test_metadata_is_frozen(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        with pytest.raises(Exception):
            meta.engine_version = "other"  # type: ignore[misc]


# ════════════════════════════════════════════════════════════════
# BUILD / VALIDATION
# ════════════════════════════════════════════════════════════════


class TestReportBuild:
    def test_rejects_empty_build(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        with pytest.raises(ValueError):
            BenchmarkReport.build(meta, None, None)

    def test_accepts_pk_only(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        report = BenchmarkReport.build(meta, _synthetic_pk_scorecard(), None)
        assert report.pk_scorecard is not None
        assert report.ddi_scorecard is None

    def test_accepts_ddi_only(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        report = BenchmarkReport.build(meta, None, _synthetic_ddi_scorecard())
        assert report.pk_scorecard is None
        assert report.ddi_scorecard is not None

    def test_accepts_both_halves(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        report = BenchmarkReport.build(
            meta, _synthetic_pk_scorecard(), _synthetic_ddi_scorecard()
        )
        assert report.pk_scorecard is not None
        assert report.ddi_scorecard is not None


# ════════════════════════════════════════════════════════════════
# HEADLINE AGGREGATION
# ════════════════════════════════════════════════════════════════


class TestHeadlineAggregation:
    def test_total_scored_combines_halves(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        pk = _synthetic_pk_scorecard()
        ddi = _synthetic_ddi_scorecard()
        report = BenchmarkReport.build(meta, pk, ddi)
        assert report.total_scored == pk.total_scored + ddi.aucr_summary.n_scored

    def test_within_tolerance_combines_halves(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        pk = _synthetic_pk_scorecard()
        ddi = _synthetic_ddi_scorecard()
        report = BenchmarkReport.build(meta, pk, ddi)
        expected = (
            pk.total_within_tolerance
            + ddi.aucr_summary.n_within_tolerance
        )
        assert report.total_within_tolerance == expected

    def test_pk_only_headline_matches_pk(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        pk = _synthetic_pk_scorecard()
        report = BenchmarkReport.build(meta, pk, None)
        assert report.total_scored == pk.total_scored
        assert report.total_within_tolerance == pk.total_within_tolerance

    def test_v1_gate_both_halves_must_pass(self) -> None:
        """Both halves must individually clear 60% — either failing is FAIL."""
        meta = RunMetadata.capture(engine_version="v1.0.0")
        # Synthetic PK: 6/9 within 2× = 66.7% (passes)
        # Synthetic DDI: 1/2 within 2× = 50% (fails)
        report = BenchmarkReport.build(
            meta, _synthetic_pk_scorecard(), _synthetic_ddi_scorecard()
        )
        assert report.pk_scorecard.passes_v1_gate is True
        assert report.ddi_scorecard.passes_v1_gate is False
        assert report.passes_v1_gate is False

    def test_v1_gate_pk_only_pass(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        report = BenchmarkReport.build(meta, _synthetic_pk_scorecard(), None)
        assert report.passes_v1_gate is True

    def test_v1_gate_ddi_only_fail(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        report = BenchmarkReport.build(meta, None, _synthetic_ddi_scorecard())
        assert report.passes_v1_gate is False


# ════════════════════════════════════════════════════════════════
# MARKDOWN RENDERING
# ════════════════════════════════════════════════════════════════


class TestMarkdownRendering:
    def test_markdown_has_required_sections(self) -> None:
        meta = RunMetadata.capture(
            engine_version="v1.0.0", git_sha="abc123", notes="test run"
        )
        report = BenchmarkReport.build(
            meta, _synthetic_pk_scorecard(), _synthetic_ddi_scorecard()
        )
        md = report.to_markdown()
        assert "# NeoGen V1 Benchmark Report" in md
        assert "## Headline" in md
        assert "## PK benchmark" in md
        assert "## DDI benchmark" in md
        assert "abc123" in md
        assert "test run" in md
        assert "v1.0.0" in md

    def test_markdown_pk_only_shows_ddi_placeholder(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        report = BenchmarkReport.build(meta, _synthetic_pk_scorecard(), None)
        md = report.to_markdown()
        assert "## DDI benchmark — not run" in md
        assert "## PK benchmark" in md

    def test_markdown_ddi_only_shows_pk_placeholder(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        report = BenchmarkReport.build(meta, None, _synthetic_ddi_scorecard())
        md = report.to_markdown()
        assert "## PK benchmark — not run" in md
        assert "## DDI benchmark" in md

    def test_failing_section_surfaces_out_of_tolerance_cases(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        report = BenchmarkReport.build(
            meta, _synthetic_pk_scorecard(), _synthetic_ddi_scorecard()
        )
        md = report.to_markdown()
        assert "## Failing" in md
        # The synthetic bad PK row was named "drug-b"
        assert "drug-b" in md
        # And the rifampicin DDI pair was named "rif+mida"
        assert "rif+mida" in md

    def test_failing_section_hidden_when_everything_passes(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        # Build a scorecard where every metric is in tolerance
        clean = Scorecard.build(
            [_pk_result("clean", "c", 1.0, 1.05, 10.0, 10.5, 5.0, 5.1)]
        )
        report = BenchmarkReport.build(meta, clean, None)
        md = report.to_markdown()
        assert "## Failing" not in md

    def test_failed_run_case_appears_in_failing_section(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        failed = BenchmarkResult(
            case_name="boom",
            drug_name="mystery",
            success=False,
            error_message="ValueError: bad Kp",
            cmax=MetricScore("Cmax", None, None, None, None),
            auc=MetricScore("AUC", None, None, None, None),
            half_life=MetricScore("t½", None, None, None, None),
            engine_version="v1.0.0-test",
        )
        sc = Scorecard.build([failed])
        report = BenchmarkReport.build(meta, sc, None)
        md = report.to_markdown()
        assert "boom" in md
        assert "ValueError" in md


# ════════════════════════════════════════════════════════════════
# JSON RENDERING
# ════════════════════════════════════════════════════════════════


class TestJSONRendering:
    def test_json_parses_as_valid_json(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        report = BenchmarkReport.build(
            meta, _synthetic_pk_scorecard(), _synthetic_ddi_scorecard()
        )
        payload = json.loads(report.to_json())
        assert "metadata" in payload
        assert "headline" in payload
        assert "pk" in payload
        assert "ddi" in payload

    def test_json_headline_fields(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        report = BenchmarkReport.build(
            meta, _synthetic_pk_scorecard(), _synthetic_ddi_scorecard()
        )
        headline = json.loads(report.to_json())["headline"]
        assert headline["total_scored"] == report.total_scored
        assert headline["total_within_tolerance"] == report.total_within_tolerance
        assert headline["passes_v1_gate"] is report.passes_v1_gate

    def test_json_pk_only_has_null_ddi(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        report = BenchmarkReport.build(meta, _synthetic_pk_scorecard(), None)
        payload = json.loads(report.to_json())
        assert payload["pk"] is not None
        assert payload["ddi"] is None

    def test_json_ddi_only_has_null_pk(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        report = BenchmarkReport.build(meta, None, _synthetic_ddi_scorecard())
        payload = json.loads(report.to_json())
        assert payload["pk"] is None
        assert payload["ddi"] is not None

    def test_json_cases_roundtrip(self) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        pk = _synthetic_pk_scorecard()
        report = BenchmarkReport.build(meta, pk, None)
        cases = json.loads(report.to_json())["pk"]["cases"]
        assert len(cases) == pk.n_cases
        names = {c["name"] for c in cases}
        assert "drug-a" in names
        assert "drug-b" in names

    def test_json_within_tolerance_is_bool_not_float(self) -> None:
        """Regression: ``np.True_`` used to fall through the default
        encoder and serialize as ``1.0``, making a passed metric
        indistinguishable from an unscored one."""
        import numpy as np

        np_true = np.True_
        np_false = np.False_
        result = BenchmarkResult(
            case_name="np-bool-case",
            drug_name="mystery",
            success=True,
            error_message=None,
            cmax=MetricScore("Cmax", 1.0, 1.05, 1.05, np_true),
            auc=MetricScore("AUC", 10.0, 35.0, 3.5, np_false),
            half_life=MetricScore("t½", None, None, None, None),
            engine_version="v1.0.0-test",
        )
        sc = Scorecard.build([result])
        meta = RunMetadata.capture(engine_version="v1.0.0")
        report = BenchmarkReport.build(meta, sc, None)
        payload = json.loads(report.to_json())
        case = payload["pk"]["cases"][0]
        assert case["cmax"]["within_tolerance"] is True
        assert case["auc"]["within_tolerance"] is False
        assert case["half_life"]["within_tolerance"] is None

    def test_json_handles_nan_as_null(self) -> None:
        """An empty scorecard produces NaN aggregates; JSON must not
        emit literal 'NaN' (invalid JSON) — should be null instead."""
        meta = RunMetadata.capture(engine_version="v1.0.0")
        empty = Scorecard.build([])
        report = BenchmarkReport.build(meta, empty, None)
        text = report.to_json()
        # json.loads rejects raw NaN by default, so this parsing
        # succeeding is itself the assertion.
        payload = json.loads(text)
        assert payload["pk"]["overall_pct_within_tolerance"] is None
        assert payload["pk"]["overall_gmfe"] is None


# ════════════════════════════════════════════════════════════════
# FILE WRITING
# ════════════════════════════════════════════════════════════════


class TestFileWriting:
    def test_write_markdown_creates_file(self, tmp_path) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        report = BenchmarkReport.build(meta, _synthetic_pk_scorecard(), None)
        out = tmp_path / "report.md"
        returned = report.write_markdown(str(out))
        assert returned == str(out)
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "# NeoGen V1 Benchmark Report" in text

    def test_write_json_creates_file(self, tmp_path) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        report = BenchmarkReport.build(meta, _synthetic_pk_scorecard(), None)
        out = tmp_path / "report.json"
        report.write_json(str(out))
        assert out.exists()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["metadata"]["engine_version"] == "v1.0.0"

    def test_write_creates_parent_dirs(self, tmp_path) -> None:
        meta = RunMetadata.capture(engine_version="v1.0.0")
        report = BenchmarkReport.build(meta, _synthetic_pk_scorecard(), None)
        nested = tmp_path / "a" / "b" / "c" / "report.md"
        report.write_markdown(str(nested))
        assert nested.exists()


# ════════════════════════════════════════════════════════════════
# END-TO-END WITH REAL DATA
# ════════════════════════════════════════════════════════════════


class TestEndToEndRealData:
    """Run the actual benchmark through the actual report writer.

    This is the smoke test that proves P6.2 + P6.3 + P6.4 compose.
    Accuracy is NOT asserted — the gate itself is P6.5's job.
    """

    def test_full_pipeline_produces_markdown_and_json(self, tmp_path) -> None:
        runner = BenchmarkRunner()
        pk_sc = Scorecard.build(runner.run_all(build_pk_benchmark_v1()))
        ddi_sc = DDIScorecard.build(
            runner.run_all_ddi(build_ddi_benchmark_v1())
        )
        meta = RunMetadata.capture(
            engine_version="v1-test",
            notes="P6.4 end-to-end smoke",
        )
        report = BenchmarkReport.build(meta, pk_sc, ddi_sc)

        md_path = tmp_path / "real.md"
        json_path = tmp_path / "real.json"
        report.write_markdown(str(md_path))
        report.write_json(str(json_path))

        assert md_path.exists() and os.path.getsize(md_path) > 500
        payload = json.loads(json_path.read_text())
        assert payload["pk"]["n_cases"] >= 30
        assert payload["ddi"]["n_cases"] >= 30
        assert payload["headline"]["total_scored"] > 0
