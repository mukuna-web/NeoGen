"""Determinism / replay audit.

Confirms the moat is reproducible: provenance hashes recompute to the stored
values, canonical pair ids are stable, there are no duplicate records, and the
flywheel status projection is reproducible across calls. A non-deterministic
pipeline cannot be trusted to replay, so drift here is a finding.

Read-only; robust to missing artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

from neogen.accuracy.contracts import AccuracyFinding
from neogen.data_flywheel.provenance import content_hash
from neogen.engine.ddi.pair_record import make_pair_id

_ROOT = Path(__file__).resolve().parents[2]
NORM = _ROOT / "data" / "flywheel" / "normalized"


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _rf(check_id, sev, status, expl, **kw) -> AccuracyFinding:
    return AccuracyFinding(
        check_id=check_id, check_type="replay", severity=sev, status=status,
        explanation=expl, **kw,
    )


def run_replay_audit(norm: Path = NORM) -> list[AccuracyFinding]:
    findings: list[AccuracyFinding] = []

    # 1. FDA mechanism assertion provenance hashes recompute + are unique.
    asr = _load(norm / "fda_mechanism_assertions_from_tables.json") or {}
    assertions = asr.get("assertions", []) if isinstance(asr, dict) else []
    if assertions:
        mism = 0
        bad_pair = 0
        hashes = []
        for a in assertions:
            pid = a.get("canonical_pair_id")
            recomputed = content_hash(
                f"{pid}|{a.get('perpetrator')}|{a.get('victim')}|"
                f"{a.get('mechanism')}|{a.get('enzyme_or_transporter')}|fda_table"
            )
            if recomputed != a.get("provenance_hash"):
                mism += 1
            if pid != make_pair_id(a.get("drug_a", ""), a.get("drug_b", "")):
                bad_pair += 1
            hashes.append(a.get("provenance_hash"))
        if mism:
            findings.append(_rf("replay::fda_hash", "high", "fail",
                                f"{mism}/{len(assertions)} FDA assertion provenance hashes do not recompute"))
        else:
            findings.append(_rf("replay::fda_hash", "info", "pass",
                                f"all {len(assertions)} FDA assertion hashes recompute deterministically"))
        if bad_pair:
            findings.append(_rf("replay::pair_id", "high", "fail",
                                f"{bad_pair} assertion(s) have a non-canonical pair id"))
        if len(hashes) != len(set(hashes)):
            findings.append(_rf("replay::fda_dupes", "high", "fail",
                                f"duplicate FDA assertion hashes: {len(hashes) - len(set(hashes))}"))
        else:
            findings.append(_rf("replay::fda_dupes", "info", "pass",
                                "FDA assertion hashes are unique (no duplicates)"))
    else:
        findings.append(_rf("replay::fda_hash", "info", "not_applicable",
                            "no FDA assertions to replay"))

    # 2. isomer counterfactual + structure hashes are present and unique.
    for name, key, hkey in (
        ("isomer_counterfactuals_pilot.json", "counterfactuals", "content_hash"),
        ("structure_stereo_cache_2026-05-29.json", "records", "content_hash"),
    ):
        d = _load(norm / name) or {}
        recs = d.get(key, []) if isinstance(d, dict) else []
        if not recs:
            continue
        hs = [r.get(hkey) for r in recs if r.get(hkey)]
        if len(hs) != len(recs):
            findings.append(_rf(f"replay::{name}::hash_present", "medium", "fail",
                                f"{name}: {len(recs) - len(hs)} record(s) missing {hkey}"))
        elif len(hs) != len(set(hs)):
            findings.append(_rf(f"replay::{name}::hash_unique", "medium", "warning",
                                f"{name}: duplicate {hkey} values"))
        else:
            findings.append(_rf(f"replay::{name}::hash", "info", "pass",
                                f"{name}: {len(recs)} content hashes present and unique"))

    # 3. flywheel status projection is reproducible across calls.
    try:
        from neogen.data_flywheel.flywheel_status import compute_status

        a = compute_status()
        b = compute_status()
        a.pop("generated_at", None)
        b.pop("generated_at", None)
        if a == b:
            findings.append(_rf("replay::status_reproducible", "info", "pass",
                                "compute_status() is reproducible across calls (modulo timestamp)"))
        else:
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            findings.append(_rf("replay::status_reproducible", "medium", "fail",
                                f"compute_status() not reproducible; differing keys: {diff}"))
    except Exception as exc:  # pragma: no cover - defensive
        findings.append(_rf("replay::status_reproducible", "low", "warning",
                            f"could not evaluate compute_status reproducibility: {str(exc)[:120]}"))

    return findings
