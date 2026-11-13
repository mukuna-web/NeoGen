"""Tests for explicit-only directionality extraction (offline)."""

from __future__ import annotations

import pytest

from neogen.data_flywheel.directionality_extractor import extract_direction


@pytest.mark.unit
def test_explicit_perpetrator_victim() -> None:
    d = extract_direction(
        "Amiodarone inhibits the metabolism of warfarin, increasing INR.",
        "amiodarone", "warfarin",
    )
    assert d.perpetrator == "amiodarone"
    assert d.victim == "warfarin"
    assert d.confidence == "explicit_label"


@pytest.mark.unit
def test_direction_is_order_independent() -> None:
    # Same text, drug args swapped -> SAME chemistry conclusion (amiodarone perp).
    text = "Amiodarone inhibits the metabolism of warfarin."
    d1 = extract_direction(text, "amiodarone", "warfarin")
    d2 = extract_direction(text, "warfarin", "amiodarone")
    assert d1.perpetrator == "amiodarone" == d2.perpetrator
    assert d1.victim == "warfarin" == d2.victim


@pytest.mark.unit
def test_no_direction_inferred_from_mere_coadministration() -> None:
    # "coadministration of X with Y" has no directional verb -> unknown.
    d = extract_direction(
        "Coadministration of warfarin with amiodarone is common.",
        "warfarin", "amiodarone",
    )
    assert d.confidence == "unknown"
    assert d.perpetrator is None and d.victim is None


@pytest.mark.unit
def test_bidirectional_flagged() -> None:
    d = extract_direction(
        "Drug A increases drug B exposure; drug B also increases drug A exposure.",
        "drug A", "drug B",
    )
    assert d.confidence == "bidirectional_needs_review"
    assert d.perpetrator is None


@pytest.mark.unit
def test_empty_text_unknown() -> None:
    d = extract_direction("", "a", "b")
    assert d.confidence == "unknown"


@pytest.mark.unit
def test_flattened_inhibitor_table_does_not_assert_direction() -> None:
    # Adversarial: a flattened CYP450 table header is NOT a directional sentence.
    # "Inhibitors" is a plural NOUN here, not the verb "inhibits".
    span = (
        "Examples of CYP450 Interactions with Warfarin Enzyme Inhibitors "
        "Inducers CYP2C9 amiodarone, capecitabine, fluconazole, metronidazole"
    )
    d = extract_direction(span, "warfarin", "amiodarone")
    assert d.confidence == "unknown"
    assert d.perpetrator is None and d.victim is None
