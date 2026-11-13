"""Tests for the hallucination / risky-claim sweep.

Proves the sweep flags unattributed regulatory/medical over-claims and fake
citations, while NOT flagging disclaimers, quoted source spans, the DDI engine's
modeling vocabulary, or cardiac "beats".
"""

from __future__ import annotations

import pytest

from neogen.accuracy.code_sweep import scan_text


def _flagged(text, path="x.md"):
    return [h for h in scan_text(text, path) if not h.allowed]


def _allowed(text, path="x.md"):
    return [h for h in scan_text(text, path) if h.allowed]


@pytest.mark.unit
def test_uncited_fda_approval_flagged():
    f = _flagged("NeoGen is FDA approved for clinical use.")
    assert any(h.pattern == "fda approved" for h in f)


@pytest.mark.unit
def test_disclaimer_fda_language_allowed():
    text = "No FDA approval, acceptance, clearance, or clinical validity is claimed."
    assert _flagged(text) == []


@pytest.mark.unit
def test_quoted_fda_text_in_json_citation_allowed():
    line = '  "citation_span": "These drugs are contraindicated per the FDA label.",'
    assert _flagged(line, "data/x.json") == []


@pytest.mark.unit
def test_clinically_validated_flagged_but_validation_noun_allowed():
    assert any(h.pattern == "clinically validated"
               for h in _flagged("This model is clinically validated."))
    # "clinical validation" (the activity / dashboard name) is NOT a claim
    assert _flagged("Clinical Validation Dashboard shows 19-drug results.") == []


@pytest.mark.unit
def test_tier_b_vocab_allowed_in_code_flagged_in_prose():
    code = 'CONTRAINDICATED = "contraindicated"  # enum member'
    assert _flagged(code, "engine.py") == []           # modeling vocab in code
    prose = "These two drugs are contraindicated together."
    assert any(h.pattern == "contraindicated" for h in _flagged(prose, "report.md"))


@pytest.mark.unit
def test_fake_citation_flagged():
    assert _flagged("See [citation needed] and http://example.com/source")


@pytest.mark.unit
def test_cardiac_beats_not_flagged_as_performance():
    assert _flagged("The heart beats faster when the rhythm changes.", "cardiac.py") == []
    assert _flagged("for beat in range(n_beats):  # per beat", "wave.py") == []


@pytest.mark.unit
def test_unbenchmarked_accuracy_claim_flagged_low_in_code():
    f = _flagged("# Renal classifier reaches 91.8% accuracy", "model.py")
    assert any(h.pattern == "unbenchmarked performance claim" and h.severity == "low" for h in f)


@pytest.mark.unit
def test_accuracy_with_eval_set_allowed():
    # naming the eval set ("on holdout") is a benchmark reference
    assert _flagged("# 91.8% accuracy on holdout set", "model.py") == []


@pytest.mark.unit
def test_fda_approved_drugs_noun_allowed():
    # describing the universe of approved drugs is not a NeoGen claim
    assert _flagged("Discover FDA-approved drugs via the openFDA API.", "crawler.py") == []
