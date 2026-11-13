"""Tests for the blind-cohort leakage auditor.

All fixtures are SYNTHETIC toy molecules. This suite never reads or tunes against
any real sealed evaluation labels -- it only verifies the leakage-detection
mechanics on hand-constructed identity overlaps.
"""

import json

import pytest

pytest.importorskip("rdkit")  # structural checks require RDKit (pinned in reqs)

from neogen.validation.blind_harness.leakage_audit import (  # noqa: E402
    LeakageAuditor,
    Record,
    canonical_smiles,
    inchikey_of,
    load_records,
    run_audit,
)

# Toy molecules
ETHANOL_A = "CCO"
ETHANOL_B = "OCC"          # same molecule, different SMILES ordering
L_ALANINE = "C[C@H](N)C(=O)O"
D_ALANINE = "C[C@@H](N)C(=O)O"
BENZENE = "c1ccccc1"
METHANE = "C"
NOVEL = "CCCCCCCCCCCCN"     # dodecylamine, far from benzene/methane


def _auditor(refs, threshold=0.95):
    return LeakageAuditor([Record(**r) for r in refs], fp_threshold=threshold)


def test_name_overlap_flags_leak():
    a = _auditor([{"name": "aspirin", "smiles": None}])
    res = a.audit_record(Record(name="aspirin", smiles=None))
    assert res["flags"]["name_overlap"] is True
    assert res["leaked"] is True


def test_canonical_smiles_overlap_flags_leak_across_smiles_spelling():
    a = _auditor([{"name": "ref", "smiles": ETHANOL_A}])
    res = a.audit_record(Record(name="different-name", smiles=ETHANOL_B))
    assert res["flags"]["name_overlap"] is False
    assert res["flags"]["canonical_smiles_overlap"] is True
    assert res["leaked"] is True


def test_parent_block_catches_enantiomer_but_isomeric_distinguishes():
    # Reference has L-alanine; blind has D-alanine (mirror image).
    a = _auditor([{"name": "l-ala", "smiles": L_ALANINE}])
    res = a.audit_record(Record(name="d-ala", smiles=D_ALANINE))
    # Same skeleton -> parent/connectivity leak; but full InChIKey + isomeric differ.
    assert res["flags"]["parent_block_overlap"] is True
    assert res["flags"]["inchikey_overlap"] is False
    assert res["flags"]["isomeric_smiles_overlap"] is False
    assert res["leaked"] is True  # fail-closed: parent overlap counts as leakage


def test_isomeric_overlap_true_for_same_enantiomer():
    a = _auditor([{"name": "l-ala", "smiles": L_ALANINE}])
    res = a.audit_record(Record(name="l-ala-2", smiles=L_ALANINE))
    assert res["flags"]["isomeric_smiles_overlap"] is True
    assert res["flags"]["inchikey_overlap"] is True


def test_near_duplicate_fires_at_identity_threshold():
    a = _auditor([{"name": "ref", "smiles": BENZENE}], threshold=1.0)
    res = a.audit_record(Record(name="novel", smiles=BENZENE))
    assert res["max_tanimoto"] == pytest.approx(1.0)
    assert res["flags"]["near_duplicate"] is True


def test_dissimilar_molecule_not_near_duplicate():
    a = _auditor([{"name": "ref", "smiles": BENZENE}], threshold=0.95)
    res = a.audit_record(Record(name="methane", smiles=METHANE))
    assert res["flags"]["near_duplicate"] is False


def test_clean_cohort_passes_verdict():
    a = _auditor([{"name": "benzene", "smiles": BENZENE}])
    blind = [Record(name="novel-amine", smiles=NOVEL)]
    result = a.audit(blind, "blind", ["ref"], {})
    assert result.n_leaked == 0
    assert result.verdict == "BLIND_INTEGRITY_PASS"


def test_any_leak_fails_closed():
    a = _auditor([{"name": "benzene", "smiles": BENZENE}])
    blind = [
        Record(name="novel-amine", smiles=NOVEL),
        Record(name="benzene", smiles=BENZENE),  # exact leak
    ]
    result = a.audit(blind, "blind", ["ref"], {})
    assert result.n_leaked == 1
    assert result.leak_rate == pytest.approx(0.5)
    assert result.verdict == "BLIND_INTEGRITY_FAIL"


def test_canonical_and_inchikey_helpers():
    assert canonical_smiles(ETHANOL_A) == canonical_smiles(ETHANOL_B)
    assert inchikey_of(BENZENE) is not None
    assert canonical_smiles("not-a-smiles") is None


def test_load_records_list_and_dict(tmp_path):
    lst = tmp_path / "list.json"
    lst.write_text(json.dumps([{"name": "A", "smiles": "CCO"}, {"drug_name": "B", "SMILES": "C"}]))
    recs = load_records(lst)
    assert len(recs) == 2
    assert recs[0].name == "a"  # normalized lower-case

    dct = tmp_path / "dict.json"
    dct.write_text(json.dumps({"pairs": [{"name": "C", "smiles": "c1ccccc1"}], "meta": 1}))
    recs2 = load_records(dct)
    assert len(recs2) == 1 and recs2[0].name == "c"


def test_run_audit_end_to_end(tmp_path):
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps([{"name": "novel", "smiles": NOVEL}]))
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps([{"name": "benzene", "smiles": BENZENE}]))
    result = run_audit(cohort, [ref])
    assert result.verdict == "BLIND_INTEGRITY_PASS"
    assert result.n_blind == 1
    assert result.input_hashes  # hashes recorded for replay
