"""Aggregate data-flywheel status for operator visibility (GAP-010).

Counts artifacts across the flywheel (sources, evidence, pairs, identity,
structure, mechanism, PK, gaps, review events) into a single summary that the
VORTEXT terminal / API can surface. Read-only; robust to missing files.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
NORM = _REPO_ROOT / "data" / "flywheel" / "normalized"
GAPS = _REPO_ROOT / "data" / "flywheel" / "gap_registry.json"
REVIEW_EVENTS = _REPO_ROOT / "data" / "flywheel" / "review_events" / "review_events_2026-05-29.jsonl"
SUMMARY_OUT = NORM / "flywheel_status_summary.json"


def _jload(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _count_list(path: Path, key: str) -> int:
    d = _jload(path)
    if isinstance(d, dict) and isinstance(d.get(key), list):
        return len(d[key])
    return 0


def compute_status() -> dict:
    # Source registry (best-effort).
    try:
        from neogen.data_flywheel.source_registry import load_registry

        sources = len(load_registry().sources)
    except Exception:
        sources = 0

    ev = _jload(NORM / "openfda_ddi_label_examples.json") or {}
    evidence = len(ev.get("evidence", [])) if isinstance(ev, dict) else 0
    pairs = len(ev.get("pair_records", [])) if isinstance(ev, dict) else 0
    verified_spl = sum(
        1 for e in ev.get("evidence", []) if isinstance(e, dict) and e.get("verified_against_spl")
    ) if isinstance(ev, dict) else 0

    idn = _jload(NORM / "identity_spine_2026-05-29.json") or {}
    identity = idn.get("resolved", 0) if isinstance(idn, dict) else 0

    structure = _count_list(NORM / "structure_stereo_cache_2026-05-29.json", "records")
    isomer_cf = _count_list(NORM / "isomer_counterfactuals_pilot.json", "counterfactuals")
    mechanism = _count_list(NORM / "mechanism_assertions_pilot.json", "assertions")
    pk = _count_list(NORM / "pk_effect_records_pilot.json", "records")

    gaps = _jload(GAPS) or {}
    glist = gaps.get("gaps", []) if isinstance(gaps, dict) else []
    open_gaps = sum(1 for g in glist if g.get("status") == "open")
    closed_gaps = sum(1 for g in glist if g.get("status") == "closed")

    # FDA structured-table cycle (GAP-014).
    fda_tbl = _jload(NORM / "fda_ddi_table_records.json") or {}
    fda_table_records = fda_tbl.get("clinical_record_count", 0) if isinstance(fda_tbl, dict) else 0
    fda_invitro_inhibitors = fda_tbl.get("invitro_record_count", 0) if isinstance(fda_tbl, dict) else 0
    fda_asserts = _jload(NORM / "fda_mechanism_assertions_from_tables.json") or {}
    fda_mechanism_assertions = fda_asserts.get("assertion_count", 0) if isinstance(fda_asserts, dict) else 0
    gres = _jload(NORM / "fda_table_directionality_gap_updates.json") or {}
    directionality_gaps_resolved = gres.get("pairs_resolved", 0) if isinstance(gres, dict) else 0
    directionality_gaps_remaining = gres.get("residual_gap_count", 0) if isinstance(gres, dict) else 0

    reviewed = blocked = 0
    if REVIEW_EVENTS.exists():
        for line in REVIEW_EVENTS.read_text().splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            to = e.get("to_status")
            if to in ("machine_checked", "human_reviewed"):
                reviewed += 1
            elif to in ("rejected", "blocked_license", "blocked_ambiguity"):
                blocked += 1

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task_id": "moat-next5-adversarial-2026-05-29",
        "sources": sources,
        "evidence": evidence,
        "evidence_verified_against_spl": verified_spl,
        "pairs": pairs,
        "identity_mappings": identity,
        "structure_records": structure,
        "isomer_counterfactuals": isomer_cf,
        "mechanism_assertions": mechanism,
        "pk_records": pk,
        "open_gaps": open_gaps,
        "open_gap_count": open_gaps,
        "closed_gaps": closed_gaps,
        "reviewed_count": reviewed,
        "blocked_count": blocked,
        "fda_table_records_count": fda_table_records,
        "fda_invitro_inhibitor_count": fda_invitro_inhibitors,
        "fda_mechanism_assertions_count": fda_mechanism_assertions,
        "directionality_gaps_resolved": directionality_gaps_resolved,
        "directionality_gaps_remaining": directionality_gaps_remaining,
    }


def write_summary() -> str:
    summary = compute_status()
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return str(SUMMARY_OUT)


if __name__ == "__main__":  # pragma: no cover
    print(write_summary())
