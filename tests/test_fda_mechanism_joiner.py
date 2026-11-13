"""Tests for the FDA mechanism joiner (GAP-014).

The joiner converts directionality gaps into cited mechanism assertions using
FDA table roles. These tests encode the adversarial traps the cycle must not
fall into: direction must come from roles (not ordering), dual roles are
ambiguous, two perpetrators don't pair, transporters map to transporter
mechanisms, missing drugs stay gaps, duplicates collapse, in-vitro evidence is
excluded from clinical assertions, and strength is never severity.
"""

from __future__ import annotations

import pytest

from neogen.data_flywheel.fda_ddi_table_parser import parse_clinical_matrix_html
from neogen.data_flywheel.fda_mechanism_joiner import (
    build_role_index,
    join_gaps_to_fda_roles,
)
from neogen.data_flywheel.mechanism_assertion import DirectionalityGap, MechanismAssertion

URL = "https://www.fda.gov/x"

# Fixture matrix: perpetrators, victims, a dual-role drug, transporter roles.
_HTML = """
<table>
<tr><th>Drug</th><th>c1</th><th>c2</th><th>c3</th></tr>
<tr><td>amiodarone</td><td>2C9 moderate inhibitor</td><td></td><td>P-gp inhibitor</td></tr>
<tr><td>warfarin7</td><td>2C9 moderate sensitive substrate</td><td></td><td></td></tr>
<tr><td>itraconazole</td><td>3A strong inhibitor</td><td></td><td></td></tr>
<tr><td>clarithromycin</td><td>3A strong inhibitor</td><td></td><td></td></tr>
<tr><td>midazolam</td><td>3A sensitive substrate</td><td></td><td></td></tr>
<tr><td>rifampin</td><td></td><td>3A strong inducer</td><td></td></tr>
<tr><td>digoxin</td><td>P-gp substrate</td><td></td><td></td></tr>
<tr><td>ritonavir</td><td>3A strong inhibitor</td><td>3A sensitive substrate</td><td></td></tr>
</table>
"""


def _records():
    return parse_clinical_matrix_html(_HTML, source_url=URL, retrieved_at="t")


def _join(gaps):
    return join_gaps_to_fda_roles(gaps=gaps, fda_records=_records(), source_url=URL)


@pytest.mark.unit
def test_inhibitor_plus_substrate_yields_assertion() -> None:
    res = _join([{"drug_a": "warfarin", "drug_b": "amiodarone"}])
    assert len(res.assertions) == 1
    a = res.assertions[0]
    assert a.perpetrator == "amiodarone" and a.victim == "warfarin"
    assert a.mechanism == "cyp_inhibition"
    assert a.enzyme_or_transporter == "CYP2C9"
    assert a.confidence == "fda_table"


@pytest.mark.unit
def test_pair_order_trap_direction_from_roles_not_ordering() -> None:
    # Same pair, both ordering conventions -> identical perpetrator/victim/hash.
    a = _join([{"drug_a": "warfarin", "drug_b": "amiodarone"}]).assertions[0]
    b = _join([{"drug_a": "amiodarone", "drug_b": "warfarin"}]).assertions[0]
    assert (a.perpetrator, a.victim) == (b.perpetrator, b.victim) == ("amiodarone", "warfarin")
    assert a.provenance_hash == b.provenance_hash


@pytest.mark.unit
def test_inducer_plus_substrate_yields_induction() -> None:
    a = _join([{"drug_a": "midazolam", "drug_b": "rifampin"}]).assertions[0]
    assert a.perpetrator == "rifampin" and a.victim == "midazolam"
    assert a.mechanism == "cyp_induction"


@pytest.mark.unit
def test_transporter_trap_maps_to_transporter_mechanism() -> None:
    a = _join([{"drug_a": "digoxin", "drug_b": "amiodarone"}]).assertions[0]
    assert a.perpetrator == "amiodarone" and a.victim == "digoxin"
    assert a.mechanism == "transporter_inhibition"
    assert a.enzyme_or_transporter == "P-gp"


@pytest.mark.unit
def test_role_trap_dual_role_yields_gap_not_assertion() -> None:
    # ritonavir is BOTH a 3A inhibitor and a 3A substrate -> ambiguous direction.
    res = _join([{"drug_a": "ritonavir", "drug_b": "midazolam"}])
    assert res.assertions == ()
    assert len(res.gaps) == 1
    assert "dual" in res.gaps[0].reason


@pytest.mark.unit
def test_no_complement_two_perpetrators_yields_gap() -> None:
    # itraconazole and clarithromycin are both 3A inhibitors -> no victim.
    res = _join([{"drug_a": "itraconazole", "drug_b": "clarithromycin"}])
    assert res.assertions == ()
    assert len(res.gaps) == 1
    assert "complement" in res.gaps[0].reason


@pytest.mark.unit
def test_citation_trap_missing_drug_stays_gap() -> None:
    res = _join([{"drug_a": "warfarin", "drug_b": "aspirin"}])  # aspirin absent
    assert res.assertions == ()
    assert len(res.gaps) == 1
    assert "no FDA clinical-table role for aspirin" in res.gaps[0].reason


@pytest.mark.unit
def test_duplicate_trap_collapses_repeated_pair() -> None:
    res = _join([
        {"drug_a": "warfarin", "drug_b": "amiodarone"},
        {"drug_a": "amiodarone", "drug_b": "warfarin"},
    ])
    assert len(res.assertions) == 1  # deduped by (pair, entity, perp, mechanism)


@pytest.mark.unit
def test_synonym_rifampicin_resolves_against_rifampin() -> None:
    # The gap names "rifampicin"; the FDA table names "rifampin" (same RxCUI).
    # The synonym must let the inducer role resolve against a substrate.
    res = _join([{"drug_a": "midazolam", "drug_b": "rifampicin"}])
    assert len(res.assertions) == 1
    a = res.assertions[0]
    assert a.perpetrator == "rifampin" and a.victim == "midazolam"
    assert a.mechanism == "cyp_induction"


@pytest.mark.unit
def test_in_vitro_trap_excluded_from_clinical_index() -> None:
    from neogen.data_flywheel.fda_ddi_table_parser import parse_invitro_inhibitors
    invitro = parse_invitro_inhibitors(
        "| Enzyme | Inhibitor |\n| --- | --- |\n| CYP3A | ketoconazole |\n",
        source_url=URL, retrieved_at="t",
    )
    idx_clinical = build_role_index(invitro, clinical_only=True)
    idx_all = build_role_index(invitro, clinical_only=False)
    assert "ketoconazole" not in idx_clinical   # in_vitro excluded by default
    assert "ketoconazole" in idx_all


@pytest.mark.unit
def test_strength_is_not_severity_but_is_cited() -> None:
    a = _join([{"drug_a": "warfarin", "drug_b": "amiodarone"}]).assertions[0]
    assert not hasattr(a, "severity")
    # strength survives as classification text in the citation span, not severity
    assert "moderate" in a.evidence_text
    assert "amiodarone" in a.evidence_text and "warfarin" in a.evidence_text


@pytest.mark.unit
def test_gaps_are_directionality_gaps_and_assertions_typed() -> None:
    res = _join([
        {"drug_a": "warfarin", "drug_b": "amiodarone"},
        {"drug_a": "itraconazole", "drug_b": "clarithromycin"},
    ])
    assert all(isinstance(a, MechanismAssertion) for a in res.assertions)
    assert all(isinstance(g, DirectionalityGap) for g in res.gaps)
    assert res.stats["assertions_created"] == 1
    assert res.stats["pairs_resolved"] == 1


# --- cycle orchestration + 9 adversarial checks ---------------------------

_DESIGN_HTML = (
    "| Enzyme | Inhibitor |\n| --- | --- |\n"
    "| CYP3A | ketoconazole |\n| CYP2D6 | quinidine |\n"
)


@pytest.mark.unit
def test_run_cycle_all_adversarial_checks_pass() -> None:
    from neogen.data_flywheel.fda_table_cycle import all_checks_passed, run_cycle
    gaps = [
        {"drug_a": "warfarin", "drug_b": "amiodarone"},
        {"drug_a": "digoxin", "drug_b": "amiodarone"},
        {"drug_a": "midazolam", "drug_b": "rifampin"},
        {"drug_a": "ritonavir", "drug_b": "midazolam"},          # dual-role gap
        {"drug_a": "itraconazole", "drug_b": "clarithromycin"},  # no-complement gap
        {"drug_a": "warfarin", "drug_b": "aspirin"},             # missing-drug gap
    ]
    res = run_cycle(
        clinical_html=_HTML, design_html=_DESIGN_HTML, gaps=gaps,
        name_to_rxcui={"warfarin": "11289", "amiodarone": "703"},
        source_url_clinical=URL, source_url_design=URL, retrieved_at="t",
    )
    assert all_checks_passed(res.adversarial), [c for c in res.adversarial if not c["passed"]]
    assert len(res.adversarial) == 9
    # 3 resolvable pairs (warfarin/amio, digoxin/amio, midazolam/rifampin)
    assert len(res.resolved_pair_ids) == 3
    # in-vitro ketoconazole must NOT have leaked into a clinical assertion
    assert all(a.confidence == "fda_table" for a in res.assertions)


@pytest.mark.unit
def test_adversarial_checks_catch_injected_backwards_assertion() -> None:
    # A deliberately backwards assertion (warfarin as perpetrator) must trip the
    # backwards-direction trap.
    from neogen.data_flywheel.fda_mechanism_joiner import join_gaps_to_fda_roles
    from neogen.data_flywheel.fda_table_cycle import run_adversarial_checks
    from neogen.data_flywheel.mechanism_assertion import MechanismAssertion
    from neogen.data_flywheel.provenance import content_hash
    from neogen.engine.ddi.pair_record import make_pair_id

    recs = _records()
    good = list(join_gaps_to_fda_roles(
        gaps=[{"drug_a": "warfarin", "drug_b": "amiodarone"}],
        fda_records=recs, source_url=URL,
    ).assertions)
    pid = make_pair_id("warfarin", "amiodarone")
    bogus = MechanismAssertion(
        drug_a="warfarin", drug_b="amiodarone", canonical_pair_id=pid,
        perpetrator="warfarin", victim="amiodarone", mechanism="cyp_inhibition",
        enzyme_or_transporter="CYP2C9",
        evidence_text="injected backwards warfarin amiodarone",
        source_url=URL, confidence="fda_table",
        provenance_hash=content_hash("bogus"),
    )
    checks = run_adversarial_checks(
        records=recs, invitro_records=[], assertions=good + [bogus],
        gaps=[{"drug_a": "warfarin", "drug_b": "amiodarone"}], source_url=URL,
    )
    by = {c["check"]: c for c in checks}
    assert by["backwards_direction_trap"]["passed"] is False
