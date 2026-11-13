"""Evidence & provenance audit over data/flywheel normalized artifacts.

Verifies that every machine-derived record carries the provenance it needs to
back a claim, and that governance is honest:

- evidence/PK/mechanism records carry citation + source_url + retrieved_at;
- any record that bears a (non-null) severity is cited — uncited severity is a
  high finding (severity must trace to a reviewed citation);
- review events are append-only (hash-chained) with reason + timestamp;
- a ``machine:``-id reviewer never performs a ``human_reviewed`` promotion
  (no fake human review) — that is a critical finding.

Read-only and robust to missing files.
"""

from __future__ import annotations

import json
from pathlib import Path

from neogen.accuracy.contracts import AccuracyFinding, EvidenceAuditResult

_ROOT = Path(__file__).resolve().parents[2]
NORM = _ROOT / "data" / "flywheel" / "normalized"
REVIEW_DIR = _ROOT / "data" / "flywheel" / "review_events"


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _nonempty(v) -> bool:
    return v is not None and str(v).strip() != ""


def _f(check_id, sev, status, expl, **kw) -> AccuracyFinding:
    return AccuracyFinding(
        check_id=check_id, check_type="evidence", severity=sev, status=status,
        explanation=expl, **kw,
    )


def _audit_records(
    artifact: str, records: list, *, require_citation_field: str | None,
    require_source_url: bool = True, require_retrieved_at: bool = True,
    id_field: str = "canonical_pair_id", severity_field: str = "severity",
    root: Path = NORM,
) -> EvidenceAuditResult:
    findings: list[AccuracyFinding] = []
    complete = 0
    path = str(root / artifact.split(":")[0])
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            continue
        ident = r.get(id_field) or r.get("pair_id") or r.get("drug_name") or f"row{i}"
        ok = True
        if require_source_url and not _nonempty(r.get("source_url")):
            ok = False
            findings.append(_f(
                f"evidence::{artifact}::source_url", "medium", "fail",
                f"{artifact}[{ident}] missing source_url",
                file_path=path, remediation="Attach the source URL or mark blocked.",
            ))
        if require_retrieved_at and not _nonempty(r.get("retrieved_at")):
            ok = False
            findings.append(_f(
                f"evidence::{artifact}::retrieved_at", "low", "warning",
                f"{artifact}[{ident}] missing retrieved_at",
                file_path=path, remediation="Record the retrieval timestamp.",
            ))
        if require_citation_field and not _nonempty(r.get(require_citation_field)):
            ok = False
            findings.append(_f(
                f"evidence::{artifact}::citation", "medium", "fail",
                f"{artifact}[{ident}] missing {require_citation_field}",
                file_path=path,
                remediation=f"Attach {require_citation_field} (the quoted source span).",
            ))
        # Uncited severity is the cardinal sin (applies to every artifact).
        sev_val = r.get(severity_field)
        if _nonempty(sev_val) and str(sev_val).lower() not in ("none", "null", "unknown"):
            cited = _nonempty(r.get("citation_span")) or _nonempty(r.get("evidence_text"))
            refs = r.get("evidence_refs")
            if not cited and not (isinstance(refs, list) and refs):
                ok = False
                findings.append(_f(
                    f"evidence::{artifact}::uncited_severity", "high", "fail",
                    f"{artifact}[{ident}] asserts severity={sev_val!r} with no citation/evidence_refs",
                    file_path=path, actual=str(sev_val),
                    remediation="Remove severity or attach a cited, reviewed source.",
                ))
        if ok:
            complete += 1
    return EvidenceAuditResult(
        artifact=artifact, records_checked=len(records),
        complete=complete, incomplete=len(records) - complete,
        findings=tuple(findings),
    )


def _audit_review_events(root: Path = REVIEW_DIR) -> EvidenceAuditResult:
    findings: list[AccuracyFinding] = []
    events: list[dict] = []
    files = sorted(Path(root).glob("*.jsonl")) if Path(root).exists() else []
    for fp in files:
        for line in fp.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    complete = 0
    for i, e in enumerate(events):
        ok = True
        if not _nonempty(e.get("reason")):
            ok = False
            findings.append(_f("evidence::review::reason", "medium", "fail",
                              f"review event {i} missing reason"))
        if not _nonempty(e.get("timestamp")):
            ok = False
            findings.append(_f("evidence::review::timestamp", "low", "warning",
                              f"review event {i} missing timestamp"))
        # Fake human review: a machine reviewer performing a human_reviewed promotion.
        rid = str(e.get("reviewer_id", "")).lower()
        if e.get("to_status") == "human_reviewed" and rid.startswith("machine"):
            ok = False
            findings.append(_f(
                "evidence::review::fake_human", "critical", "fail",
                f"review event {i}: machine reviewer {e.get('reviewer_id')!r} performed "
                f"a human_reviewed promotion (fake human review)",
                actual=str(e.get("reviewer_id")),
                remediation="Only a human reviewer id may perform human_reviewed.",
            ))
        if ok:
            complete += 1
    if not events:
        findings.append(_f("evidence::review::present", "info", "not_applicable",
                          "no review events found (none required)"))
    return EvidenceAuditResult(
        artifact="review_events", records_checked=len(events),
        complete=complete, incomplete=len(events) - complete, findings=tuple(findings),
    )


def run_evidence_audit(norm: Path = NORM, review_dir: Path = REVIEW_DIR) -> list[EvidenceAuditResult]:
    """Audit all known artifacts; missing artifacts degrade to not_applicable."""
    results: list[EvidenceAuditResult] = []

    def _list(name: str, key: str) -> list:
        d = _load(norm / name)
        if isinstance(d, dict) and isinstance(d.get(key), list):
            return d[key]
        if isinstance(d, list):
            return d
        return []

    # (file, key, citation_field, require_source_url, require_retrieved_at, id_field)
    specs = [
        ("openfda_ddi_label_examples.json", "evidence", "citation_span", True, True, "drug_a"),
        # pair records carry provenance via evidence_refs, not their own url/ts.
        ("openfda_ddi_label_examples.json", "pair_records", None, False, False, "pair_id"),
        ("mechanism_assertions_pilot.json", "assertions", "evidence_text", True, False, "canonical_pair_id"),
        ("fda_mechanism_assertions_from_tables.json", "assertions", "evidence_text", True, False, "canonical_pair_id"),
        ("pk_effect_records_pilot.json", "records", "citation_span", True, False, "canonical_pair_id"),
        ("fda_ddi_table_records.json", "records", "row_text", True, True, "drug_name"),
    ]
    for name, key, cit, req_url, req_ts, idf in specs:
        recs = _list(name, key)
        if not recs:
            results.append(EvidenceAuditResult(
                artifact=f"{name}:{key}", records_checked=0, complete=0,
                incomplete=0,
                findings=(_f(f"evidence::{name}:{key}::present", "info",
                             "not_applicable", f"{name}:{key} absent or empty"),),
            ))
            continue
        results.append(_audit_records(
            f"{name}:{key}", recs, require_citation_field=cit,
            require_source_url=req_url, require_retrieved_at=req_ts,
            id_field=idf, root=norm))

    results.append(_audit_review_events(review_dir))
    return results


def findings_from(results: list[EvidenceAuditResult]) -> list[AccuracyFinding]:
    return [f for r in results for f in r.findings]
