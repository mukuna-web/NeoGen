"""Validation report aggregation and serialization.

Collects Tier 1 / 2 / 3 / 4 reports into a single ValidationReport
that can be serialized to JSON or rendered to Markdown. The
ValidationReport is the output contract for the P6 validation
harness — the API layer (P7) will serialize it for clients and
the CLI (scripts/run_validation.py) prints it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from neogen.validation.tier1 import Tier1CaseResult, Tier1Report
from neogen.validation.tier2 import Tier2CaseResult, Tier2Report
from neogen.validation.tier3 import Tier3CaseResult, Tier3Report
from neogen.validation.tier4 import InvariantResult, Tier4Report


# ════════════════════════════════════════════════════════════════
# TOP-LEVEL REPORT
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ValidationReport:
    """Composite report for the full V1 validation harness."""

    engine_version: str
    run_timestamp_utc: str
    tier1: Optional[Tier1Report] = None
    tier2: Optional[Tier2Report] = None
    tier3: Optional[Tier3Report] = None
    tier4: Optional[Tier4Report] = None
    notes: tuple = field(default_factory=tuple)

    @property
    def overall_pass(self) -> bool:
        """Green only if every run tier passes its gate."""
        gates = []
        if self.tier1 is not None:
            gates.append(self.tier1.gate_pass)
        if self.tier2 is not None:
            gates.append(self.tier2.gate_pass)
        if self.tier3 is not None:
            gates.append(self.tier3.gate_pass)
        if self.tier4 is not None:
            gates.append(self.tier4.gate_pass)
        return bool(gates) and all(gates)

    # ── serialization ───────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_version": self.engine_version,
            "run_timestamp_utc": self.run_timestamp_utc,
            "overall_pass": self.overall_pass,
            "notes": list(self.notes),
            "tier1": _tier1_to_dict(self.tier1),
            "tier2": _tier2_to_dict(self.tier2),
            "tier3": _tier3_to_dict(self.tier3),
            "tier4": _tier4_to_dict(self.tier4),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            default=_json_safe,
            sort_keys=False,
        )

    def to_markdown(self) -> str:
        """Render as a Markdown report suitable for CI output."""
        lines = [
            f"# NeoGen V1 Validation Report",
            "",
            f"- **Engine version:** `{self.engine_version}`",
            f"- **Run (UTC):** {self.run_timestamp_utc}",
            f"- **Overall gate:** "
            f"{'PASS' if self.overall_pass else 'FAIL'}",
            "",
        ]
        if self.notes:
            lines.append("## Notes")
            for note in self.notes:
                lines.append(f"- {note}")
            lines.append("")

        if self.tier4 is not None:
            lines.extend(_tier4_markdown(self.tier4))
        if self.tier1 is not None:
            lines.extend(_tier1_markdown(self.tier1))
        if self.tier2 is not None:
            lines.extend(_tier2_markdown(self.tier2))
        if self.tier3 is not None:
            lines.extend(_tier3_markdown(self.tier3))

        return "\n".join(lines) + "\n"

    def save(self, out_dir: Path) -> tuple[Path, Path]:
        """Write JSON and Markdown to out_dir. Returns both paths."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = self.run_timestamp_utc.replace(":", "-").replace("+", "_")
        json_path = out_dir / f"validation_{ts}.json"
        md_path = out_dir / f"validation_{ts}.md"
        json_path.write_text(self.to_json())
        md_path.write_text(self.to_markdown())
        return json_path, md_path

    @staticmethod
    def now_timestamp() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ════════════════════════════════════════════════════════════════
# PER-TIER SERIALIZATION HELPERS
# ════════════════════════════════════════════════════════════════


def _json_safe(obj: Any) -> Any:
    """Convert non-JSON-native objects."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def _tier1_to_dict(report: Optional[Tier1Report]) -> Optional[dict]:
    if report is None:
        return None
    return {
        "threshold_pct": report.acceptance_threshold_pct,
        "n_total": report.n_total,
        "n_scorable": report.n_scorable,
        "n_passed": report.n_passed,
        "n_errored": report.n_errored,
        "pct_passed": round(report.pct_passed, 2),
        "gate_pass": report.gate_pass,
        "metric_summaries": {
            "half_life_h": _safe_summary(report.metric_summary("half_life")),
            "cl_L_per_h": _safe_summary(report.metric_summary("cl")),
            "vd_L": _safe_summary(report.metric_summary("vd")),
        },
        "pk_consistency": {
            "n_cases_with_qc": report.n_cases_with_consistency,
            "n_cases_consistent": report.n_cases_consistent,
            "n_cases_borderline": report.n_cases_borderline,
            "n_cases_inconsistent": report.n_cases_inconsistent,
        },
        "cases": [_tier1_case_to_dict(r) for r in report.results],
    }


def _tier1_case_to_dict(r: Tier1CaseResult) -> dict:
    return {
        "name": r.case.name,
        "literature_available": r.case.literature_available,
        "observed": {
            "half_life_h": r.case.observed_half_life_h,
            "cl_L_per_h": r.case.observed_cl_L_per_h,
            "vd_L": r.case.observed_vd_L,
        },
        "predicted": {
            "half_life_h": r.predicted_half_life_h,
            "cl_L_per_h": r.predicted_cl_L_per_h,
            "vd_L": r.predicted_vd_L,
        },
        "fold_error": {
            "half_life": _nan_none(r.half_life_fold_error),
            "cl": _nan_none(r.cl_fold_error),
            "vd": _nan_none(r.vd_fold_error),
        },
        "within_2_fold": {
            "half_life": r.half_life_within_2_fold,
            "cl": r.cl_within_2_fold,
            "vd": r.vd_within_2_fold,
        },
        "n_scorable": r.n_scorable,
        "n_passed": r.n_passed,
        "case_passed": r.case_passed,
        "pk_consistency": r.pk_consistency,
        "error": r.error or None,
    }


def _tier2_to_dict(report: Optional[Tier2Report]) -> Optional[dict]:
    if report is None:
        return None
    return {
        "threshold_pct": report.acceptance_threshold_pct,
        "n_total": report.n_total,
        "n_passed": report.n_passed,
        "n_errored": report.n_errored,
        "pct_passed": round(report.pct_passed, 2),
        "gate_pass": report.gate_pass,
        "pairs": [_tier2_case_to_dict(r) for r in report.results],
    }


def _tier2_case_to_dict(r: Tier2CaseResult) -> dict:
    return {
        "label": r.case.label,
        "victim": r.case.victim_name,
        "perpetrator": r.case.perpetrator_name,
        "mode": r.case.mode,
        "clinical_aucr": r.case.clinical_aucr,
        "predicted_aucr": _nan_none(r.aucr_predicted),
        "auc_baseline": _nan_none(r.auc_baseline),
        "auc_ddi": _nan_none(r.auc_ddi),
        "fold_error": _nan_none(r.fold_error_vs_clinical),
        "within_2_fold": r.within_2_fold,
        "citation": r.case.citation,
        "error": r.error or None,
    }


def _tier3_to_dict(report: Optional[Tier3Report]) -> Optional[dict]:
    if report is None:
        return None
    return {
        "acceptance_band_pct": report.acceptance_band_pct,
        "n_total": report.n_total,
        "n_passed": report.n_passed,
        "n_errored": report.n_errored,
        "pct_passed": round(report.pct_passed, 2),
        "gate_pass": report.gate_pass,
        "cases": [_tier3_case_to_dict(r) for r in report.results],
    }


def _tier3_case_to_dict(r: Tier3CaseResult) -> dict:
    return {
        "drug_name": r.case.drug_name,
        "observed_f_oral": r.case.observed_f_oral,
        "predicted_f_empirical": _nan_none(r.f_empirical),
        "predicted_f_decomposed": _nan_none(r.f_decomposed),
        "pct_error_empirical": _nan_none(r.pct_err_empirical),
        "within_acceptance": r.within_acceptance,
        "auc_oral": _nan_none(r.auc_oral),
        "auc_iv": _nan_none(r.auc_iv),
        "citation": r.case.citation,
        "error": r.error or None,
    }


def _tier4_to_dict(report: Optional[Tier4Report]) -> Optional[dict]:
    if report is None:
        return None
    return {
        "n_total": report.n_total,
        "n_passed": report.n_passed,
        "n_failed": report.n_failed,
        "n_gating": report.n_gating,
        "n_gating_failed": report.n_gating_failed,
        "n_advisory_failed": report.n_advisory_failed,
        "gate_pass": report.gate_pass,
        "invariants": [
            {
                "name": r.name,
                "description": r.description,
                "passed": r.passed,
                "gating": r.gating,
                "detail": r.detail,
                "error": r.error or None,
            }
            for r in report.results
        ],
    }


def _safe_summary(d: dict) -> dict:
    """Convert NaN floats in summary dict to None for JSON."""
    return {
        k: (None if isinstance(v, float) and math.isnan(v) else v)
        for k, v in d.items()
    }


def _nan_none(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


# ════════════════════════════════════════════════════════════════
# MARKDOWN RENDERING
# ════════════════════════════════════════════════════════════════


def _check_mark(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _tier4_markdown(t4: Tier4Report) -> list[str]:
    lines = [
        "## Tier 4 — Mechanistic Invariants",
        "",
        f"- **Gate:** {_check_mark(t4.gate_pass)} "
        f"(gating {t4.n_gating - t4.n_gating_failed}/{t4.n_gating} "
        f"passed; advisory failures: {t4.n_advisory_failed})",
        "",
        "| Invariant | Gating | Status | Detail |",
        "|---|---|---|---|",
    ]
    for r in t4.results:
        status = _check_mark(r.passed)
        gating = "yes" if r.gating else "advisory"
        detail = (r.detail or r.error or "").replace("|", "\\|")
        lines.append(
            f"| `{r.name}` | {gating} | {status} | {detail} |"
        )
    if t4.n_advisory_failed:
        lines.append("")
        lines.append(
            "_Advisory failures do not block the Tier 4 gate — "
            "they track known V1 architectural limitations._"
        )
    lines.append("")
    return lines


def _tier1_markdown(t1: Tier1Report) -> list[str]:
    lines = [
        "## Tier 1 — Single-Drug PK",
        "",
        f"- **Acceptance:** "
        f"≥{t1.acceptance_threshold_pct:.0f}% of scorable cases "
        f"within 2-fold on majority of {{t½, CL, Vd}}",
        f"- **Result:** "
        f"{t1.n_passed}/{t1.n_scorable} scorable cases passed "
        f"({t1.pct_passed:.1f}%) — "
        f"{_check_mark(t1.gate_pass)}",
        f"- **Errors:** {t1.n_errored}",
        "",
        "### Metric summaries (GMFE / %-within-2-fold)",
        "",
        "| Metric | N valid | GMFE | AFE | % within 2-fold |",
        "|---|---|---|---|---|",
    ]
    for metric_name, label in (
        ("half_life", "t½"),
        ("cl", "CL"),
        ("vd", "Vd"),
    ):
        s = t1.metric_summary(metric_name)
        gmfe = f"{s['gmfe']:.2f}" if not math.isnan(s["gmfe"]) else "-"
        afe = f"{s['afe']:.2f}" if not math.isnan(s["afe"]) else "-"
        lines.append(
            f"| {label} | {s['n_valid']} | {gmfe} | {afe} | "
            f"{s['pct_within_2_fold']:.1f}% |"
        )
    lines.extend(["", "### First 10 case details", ""])
    lines.append(
        "| Drug | t½ pred/obs | CL pred/obs | Vd pred/obs | Pass |"
    )
    lines.append("|---|---|---|---|---|")
    for r in t1.results[:10]:
        name = r.case.name
        t_pred = _fmt(r.predicted_half_life_h)
        t_obs = _fmt(r.case.observed_half_life_h)
        cl_pred = _fmt(r.predicted_cl_L_per_h)
        cl_obs = _fmt(r.case.observed_cl_L_per_h)
        vd_pred = _fmt(r.predicted_vd_L)
        vd_obs = _fmt(r.case.observed_vd_L)
        pass_str = _check_mark(r.case_passed) if not r.error else "ERR"
        lines.append(
            f"| {name} | {t_pred}/{t_obs} | {cl_pred}/{cl_obs} | "
            f"{vd_pred}/{vd_obs} | {pass_str} |"
        )
    lines.append("")
    return lines


def _tier2_markdown(t2: Tier2Report) -> list[str]:
    lines = [
        "## Tier 2 — DDI Pair AUCR",
        "",
        f"- **Acceptance:** "
        f"≥{t2.acceptance_threshold_pct:.0f}% of pairs within 2-fold",
        f"- **Result:** "
        f"{t2.n_passed}/{t2.n_total} pairs passed "
        f"({t2.pct_passed:.1f}%) — "
        f"{_check_mark(t2.gate_pass)}",
        "",
        "| Pair | Mode | Clinical AUCR | Predicted | Fold err | Pass |",
        "|---|---|---|---|---|---|",
    ]
    for r in t2.results:
        pred = _fmt(r.aucr_predicted)
        fe = _fmt(r.fold_error_vs_clinical)
        pass_str = _check_mark(r.within_2_fold) if not r.error else "ERR"
        lines.append(
            f"| {r.case.label} | {r.case.mode} | "
            f"{r.case.clinical_aucr:.2f} | {pred} | {fe} | {pass_str} |"
        )
    lines.append("")
    return lines


def _tier3_markdown(t3: Tier3Report) -> list[str]:
    lines = [
        "## Tier 3 — Bioavailability",
        "",
        f"- **Acceptance:** "
        f"per-case ±{t3.acceptance_band_pct:.0f}% of observed F. "
        f"Advisory gate at ≥50% cohort pass.",
        f"- **Result:** "
        f"{t3.n_passed}/{t3.n_total} cases within band "
        f"({t3.pct_passed:.1f}%) — "
        f"{_check_mark(t3.gate_pass)}",
        "",
        "| Drug | Observed F | Predicted F_emp | %err | Pass |",
        "|---|---|---|---|---|",
    ]
    for r in t3.results:
        obs = f"{r.case.observed_f_oral:.3f}"
        pred = _fmt(r.f_empirical)
        pct_err = _fmt(r.pct_err_empirical)
        pass_str = _check_mark(r.within_acceptance) if not r.error else "ERR"
        lines.append(
            f"| {r.case.drug_name} | {obs} | {pred} | {pct_err} | "
            f"{pass_str} |"
        )
    lines.append("")
    return lines


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "-"
    return f"{v:.3f}"
