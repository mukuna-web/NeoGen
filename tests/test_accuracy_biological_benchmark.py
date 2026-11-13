"""Tests for the biological benchmark (offline tmp fixtures)."""

from __future__ import annotations

import json

import pytest

from neogen.accuracy.biological_benchmark import run_biological_benchmark


def _setup(norm, records=None, invitro=None, assertions=None, structures=None, counterfactuals=None):
    norm.mkdir(parents=True, exist_ok=True)
    (norm / "fda_ddi_table_records.json").write_text(json.dumps({
        "records": records or [], "invitro_records": invitro or []}))
    (norm / "fda_mechanism_assertions_from_tables.json").write_text(json.dumps({
        "assertions": assertions or []}))
    if structures is not None:
        (norm / "structure_stereo_cache_2026-05-29.json").write_text(json.dumps({"records": structures}))
    if counterfactuals is not None:
        (norm / "isomer_counterfactuals_pilot.json").write_text(json.dumps({"counterfactuals": counterfactuals}))


_REC = [
    {"normalized_drug_name": "amiodarone", "entity": "CYP2C9", "role_class": "inhibitor", "strength": "moderate"},
    {"normalized_drug_name": "warfarin", "entity": "CYP2C9", "role_class": "substrate", "strength": "moderate_sensitive"},
    {"normalized_drug_name": "rifampin", "entity": "CYP3A", "role_class": "inducer", "strength": "strong"},
    {"normalized_drug_name": "midazolam", "entity": "CYP3A", "role_class": "substrate", "strength": "sensitive"},
]


def _dim(results, name):
    return next(r for r in results if r.dimension == name)


@pytest.mark.unit
def test_correct_cyp_direction_passes(tmp_path):
    norm = tmp_path / "n"
    _setup(norm, records=_REC, assertions=[
        {"canonical_pair_id": "amiodarone|warfarin", "drug_a": "warfarin", "drug_b": "amiodarone",
         "perpetrator": "amiodarone", "victim": "warfarin", "mechanism": "cyp_inhibition",
         "enzyme_or_transporter": "CYP2C9"}])
    assert _dim(run_biological_benchmark(norm), "cyp_directionality").status == "pass"


@pytest.mark.unit
def test_backwards_cyp_direction_fails(tmp_path):
    norm = tmp_path / "n"
    # warfarin is only a substrate; asserting it as the inhibitor is unsupported
    _setup(norm, records=_REC, assertions=[
        {"canonical_pair_id": "amiodarone|warfarin", "drug_a": "warfarin", "drug_b": "amiodarone",
         "perpetrator": "warfarin", "victim": "amiodarone", "mechanism": "cyp_inhibition",
         "enzyme_or_transporter": "CYP2C9"}])
    assert _dim(run_biological_benchmark(norm), "cyp_directionality").status == "fail"


@pytest.mark.unit
def test_inducer_direction_passes(tmp_path):
    norm = tmp_path / "n"
    _setup(norm, records=_REC, assertions=[
        {"canonical_pair_id": "midazolam|rifampin", "drug_a": "midazolam", "drug_b": "rifampin",
         "perpetrator": "rifampin", "victim": "midazolam", "mechanism": "cyp_induction",
         "enzyme_or_transporter": "CYP3A"}])
    assert _dim(run_biological_benchmark(norm), "cyp_directionality").status == "pass"


@pytest.mark.unit
def test_marker_reaction_not_promoted(tmp_path):
    norm = tmp_path / "n"
    bad = _REC + [{"normalized_drug_name": "s-warfarin 7-hydroxylation", "entity": "CYP2C9",
                   "role_class": "marker_reaction", "strength": "unknown"}]
    _setup(norm, records=bad, assertions=[])
    assert _dim(run_biological_benchmark(norm), "marker_reaction_excluded").status == "fail"


@pytest.mark.unit
def test_in_vitro_not_promoted_to_clinical(tmp_path):
    norm = tmp_path / "n"
    # ketoconazole's inhibitor role exists ONLY in vitro but is used in an assertion
    inv = [{"normalized_drug_name": "ketoconazole", "entity": "CYP3A", "role_class": "inhibitor", "strength": "unknown"}]
    _setup(norm, records=_REC, invitro=inv, assertions=[
        {"canonical_pair_id": "ketoconazole|midazolam", "drug_a": "ketoconazole", "drug_b": "midazolam",
         "perpetrator": "ketoconazole", "victim": "midazolam", "mechanism": "cyp_inhibition",
         "enzyme_or_transporter": "CYP3A"}])
    assert _dim(run_biological_benchmark(norm), "in_vitro_not_promoted").status == "fail"


@pytest.mark.unit
def test_strength_is_not_severity(tmp_path):
    norm = tmp_path / "n"
    _setup(norm, records=_REC, assertions=[
        {"canonical_pair_id": "amiodarone|warfarin", "drug_a": "warfarin", "drug_b": "amiodarone",
         "perpetrator": "amiodarone", "victim": "warfarin", "mechanism": "cyp_inhibition",
         "enzyme_or_transporter": "CYP2C9", "severity": "major"}])  # illegal severity field
    crit = [f for f in _dim(run_biological_benchmark(norm), "strength_not_severity").findings
            if f.severity == "critical"]
    assert crit


@pytest.mark.unit
def test_enantiomer_sensitive_legacy_schema_warns(tmp_path):
    norm = tmp_path / "n"
    _setup(norm, records=_REC, assertions=[], structures=[
        {"drug": "warfarin", "stereo_relevance": "enantiomer_sensitive",
         "stereochemistry_status": "unspecified_parent", "required_feature_schema": "legacy_2d_v1"}])
    dim = _dim(run_biological_benchmark(norm), "stereochemistry_handling")
    assert dim.status in ("warning", "fail")
