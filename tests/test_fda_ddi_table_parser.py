"""Tests for the FDA DDI table parser (GAP-014).

Covers cell parsing, entity normalization (incl. footnote contamination),
drug-name footnote stripping, section-header skipping, and HTML/markdown
matrix parsing. All offline (inline fixtures).
"""

from __future__ import annotations

import pytest

from neogen.data_flywheel.fda_ddi_table_parser import (
    _split_footnote,
    normalize_entity,
    parse_cell,
    parse_clinical_matrix,
    parse_clinical_matrix_html,
    parse_invitro_inhibitors,
)

# Minimal HTML mirroring the real clinical matrix structure: a header row, drug
# rows (some with footnote digits), and an enzyme section-header row.
_HTML = """
<table>
<tr><th>Drug or Other Substance</th><th>CYP INH</th><th>CYP SUB</th><th>CYP IND</th><th>TRNSP</th></tr>
<tr><td>amiodarone</td><td>2C9 moderate inhibitor</td><td></td><td></td><td>P-gp inhibitor</td></tr>
<tr><td>warfarin19</td><td></td><td>2C9 moderate sensitive substrate</td><td></td><td></td></tr>
<tr><td>clarithromycin5</td><td>3A strong inhibitor</td><td></td><td></td><td>OATP1B1;OATP1B3 inhibitor</td></tr>
<tr><td>rifampin</td><td></td><td></td><td>3A strong inducer</td><td></td></tr>
<tr><td>P-gp</td><td></td><td></td><td></td><td></td></tr>
<tr><td>OATP1B1/OATP1B3</td><td></td><td></td><td></td><td></td></tr>
</table>
"""


@pytest.mark.unit
def test_parse_cell_inhibitor_with_strength() -> None:
    assert parse_cell("2C9 moderate inhibitor") == [("CYP2C9", "moderate", "inhibitor")]


@pytest.mark.unit
def test_parse_cell_sensitive_substrate() -> None:
    assert parse_cell("3A sensitive substrate") == [("CYP3A", "sensitive", "substrate")]


@pytest.mark.unit
def test_parse_cell_inhibitor_unknown_strength() -> None:
    assert parse_cell("P-gp inhibitor") == [("P-gp", "unknown", "inhibitor")]


@pytest.mark.unit
def test_parse_cell_multi_entity() -> None:
    out = parse_cell("OATP1B1;OATP1B3 inhibitor")
    assert ("OATP1B1", "unknown", "inhibitor") in out
    assert ("OATP1B3", "unknown", "inhibitor") in out


@pytest.mark.unit
def test_parse_cell_empty_or_no_role() -> None:
    assert parse_cell("") == []
    assert parse_cell("   ") == []
    assert parse_cell("see footnote 3") == []  # no role keyword -> nothing guessed


@pytest.mark.unit
def test_normalize_entity_cyp_and_transporter() -> None:
    assert normalize_entity("3A") == "CYP3A"
    assert normalize_entity("2C9") == "CYP2C9"
    assert normalize_entity("P-gp") == "P-gp"
    assert normalize_entity("PGP") == "P-gp"
    assert normalize_entity("ABCB1") == "P-gp"


@pytest.mark.unit
def test_normalize_entity_strips_footnote_digits() -> None:
    # Trailing footnote digits must not create a bogus transporter token.
    assert normalize_entity("OATP1B113") == "OATP1B1"
    assert normalize_entity("OATP1B1/OATP1B3") == "OATP1B1"


@pytest.mark.unit
def test_normalize_entity_strips_cyp_footnote_digits() -> None:
    # "1A220" = CYP1A2 + superscript footnote 20 -> must resolve to CYP1A2,
    # NOT a bogus CYP1A220. Valid sub-families must survive unchanged.
    assert normalize_entity("1A220") == "CYP1A2"
    assert normalize_entity("2C19") == "CYP2C19"   # not CYP2C1 / CYP2C
    assert normalize_entity("3A") == "CYP3A"
    assert normalize_entity("CYP3A4") == "CYP3A4"


@pytest.mark.unit
def test_normalize_entity_unknown_returns_none() -> None:
    assert normalize_entity("aspirin") is None
    assert normalize_entity("") is None


@pytest.mark.unit
def test_split_footnote() -> None:
    assert _split_footnote("warfarin19") == ("warfarin", "19")
    assert _split_footnote("diltiazem7") == ("diltiazem", "7")
    assert _split_footnote("amiodarone") == ("amiodarone", None)


@pytest.mark.unit
def test_parse_clinical_matrix_html_full() -> None:
    recs = parse_clinical_matrix_html(
        _HTML, source_url="https://www.fda.gov/x", retrieved_at="2026-05-29T00:00:00Z",
    )
    by = {(r.normalized_drug_name, r.entity, r.role_class) for r in recs}
    assert ("amiodarone", "CYP2C9", "inhibitor") in by
    assert ("amiodarone", "P-gp", "inhibitor") in by
    assert ("warfarin", "CYP2C9", "substrate") in by      # footnote stripped
    assert ("clarithromycin", "CYP3A", "inhibitor") in by  # footnote stripped
    assert ("rifampin", "CYP3A", "inducer") in by


@pytest.mark.unit
def test_parse_clinical_matrix_html_skips_section_headers() -> None:
    recs = parse_clinical_matrix_html(
        _HTML, source_url="u", retrieved_at="t",
    )
    drugs = {r.normalized_drug_name for r in recs}
    # "P-gp" and "OATP1B1/OATP1B3" are enzyme/transporter section headers,
    # never drugs.
    assert "p-gp" not in drugs
    assert "oatp1b1/oatp1b3" not in drugs
    assert "drug or other substance" not in drugs


@pytest.mark.unit
def test_parse_clinical_matrix_html_captures_footnote() -> None:
    recs = parse_clinical_matrix_html(_HTML, source_url="u", retrieved_at="t")
    wf = next(r for r in recs if r.normalized_drug_name == "warfarin")
    assert wf.footnotes == ("footnote:19",)  # never dropped


@pytest.mark.unit
def test_parse_clinical_matrix_markdown() -> None:
    md = (
        "| Drug | CYP |\n"
        "| --- | --- |\n"
        "| amiodarone | 2C9 moderate inhibitor |\n"
        "| midazolam | 3A sensitive substrate |\n"
    )
    recs = parse_clinical_matrix(md, source_url="u", retrieved_at="t")
    by = {(r.normalized_drug_name, r.entity, r.role_class) for r in recs}
    assert ("amiodarone", "CYP2C9", "inhibitor") in by
    assert ("midazolam", "CYP3A", "substrate") in by


@pytest.mark.unit
def test_parse_invitro_inhibitors_markdown() -> None:
    md = (
        "| Enzyme | Inhibitor |\n"
        "| --- | --- |\n"
        "| CYP3A | ketoconazole; itraconazole |\n"
        "| CYP2D6 | quinidine |\n"
    )
    recs = parse_invitro_inhibitors(md, source_url="u", retrieved_at="t")
    assert all(r.evidence_context == "in_vitro" for r in recs)
    by = {(r.normalized_drug_name, r.entity) for r in recs}
    assert ("ketoconazole", "CYP3A") in by
    assert ("itraconazole", "CYP3A") in by
    assert ("quinidine", "CYP2D6") in by


@pytest.mark.unit
def test_parse_invitro_inhibitors_html_excludes_marker_reactions() -> None:
    # Stacked sections like the real FDA design page: a marker-reaction table
    # must NOT be misread as inhibitors (marker-reaction trap).
    html = """
    <table>
    <tr><th>Enzyme</th><th>Marker reaction</th></tr>
    <tr><td>CYP2C9</td><td>S-warfarin 7-hydroxylation</td></tr>
    <tr><th>Enzyme</th><th>Inhibitor</th></tr>
    <tr><td>CYP3A4</td><td>ketoconazole, itraconazole(a)</td></tr>
    <tr><td>CYP2D6</td><td>quinidine</td></tr>
    </table>
    """
    recs = parse_invitro_inhibitors(html, source_url="u", retrieved_at="t")
    drugs = {r.normalized_drug_name for r in recs}
    assert "ketoconazole" in drugs
    assert "itraconazole" in drugs            # "(a)" footnote stripped
    assert "quinidine" in drugs
    # the marker reaction must never appear as an inhibitor
    assert not any("warfarin" in d or "hydroxylation" in d for d in drugs)
    assert all(r.evidence_context == "in_vitro" for r in recs)


@pytest.mark.unit
def test_strength_is_not_severity() -> None:
    recs = parse_clinical_matrix_html(_HTML, source_url="u", retrieved_at="t")
    assert recs, "fixture should yield records"
    assert all(not hasattr(r, "severity") for r in recs)
