"""Tests for the determinism / replay audit (offline tmp fixtures)."""

from __future__ import annotations

import json

import pytest

from neogen.accuracy.replay_audit import run_replay_audit
from neogen.data_flywheel.provenance import content_hash


def _assertion(pid="amiodarone|warfarin", perp="amiodarone", vic="warfarin",
               mech="cyp_inhibition", ent="CYP2C9", tamper=False):
    good = content_hash(f"{pid}|{perp}|{vic}|{mech}|{ent}|fda_table")
    return {
        "canonical_pair_id": pid, "drug_a": "warfarin", "drug_b": "amiodarone",
        "perpetrator": perp, "victim": vic, "mechanism": mech,
        "enzyme_or_transporter": ent,
        "provenance_hash": "0" * 64 if tamper else good,
    }


def _write(norm, assertions):
    norm.mkdir(parents=True, exist_ok=True)
    (norm / "fda_mechanism_assertions_from_tables.json").write_text(
        json.dumps({"assertions": assertions}))


def _by(findings, frag):
    return [f for f in findings if frag in f.check_id]


@pytest.mark.unit
def test_matching_hash_is_deterministic(tmp_path):
    norm = tmp_path / "n"
    _write(norm, [_assertion()])
    findings = run_replay_audit(norm)
    h = _by(findings, "fda_hash")
    assert h and h[0].status == "pass"


@pytest.mark.unit
def test_tampered_hash_flagged(tmp_path):
    norm = tmp_path / "n"
    _write(norm, [_assertion(tamper=True)])
    findings = run_replay_audit(norm)
    h = _by(findings, "fda_hash")
    assert h and h[0].status == "fail"


@pytest.mark.unit
def test_noncanonical_pair_id_flagged(tmp_path):
    norm = tmp_path / "n"
    a = _assertion()
    a["canonical_pair_id"] = "warfarin|amiodarone"  # not sorted-canonical
    a["provenance_hash"] = content_hash(
        "warfarin|amiodarone|amiodarone|warfarin|cyp_inhibition|CYP2C9|fda_table")
    _write(norm, [a])
    findings = run_replay_audit(norm)
    assert _by(findings, "pair_id") and _by(findings, "pair_id")[0].status == "fail"


@pytest.mark.unit
def test_repeated_run_is_stable(tmp_path):
    norm = tmp_path / "n"
    _write(norm, [_assertion()])
    a = [f.to_dict() for f in run_replay_audit(norm) if "fda" in f.check_id]
    b = [f.to_dict() for f in run_replay_audit(norm) if "fda" in f.check_id]
    assert a == b
