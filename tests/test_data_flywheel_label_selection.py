"""Tests for formulation classification + systemic label selection (offline)."""

from __future__ import annotations

import pytest

from neogen.data_flywheel.formulation_classifier import classify_formulation
from neogen.data_flywheel.label_selection import (
    rank_labels,
    score_label,
    select_best_label,
)

_SHAMPOO = {
    "set_id": "set-shampoo",
    "openfda": {"generic_name": ["KETOCONAZOLE"], "brand_name": ["KETOCONAZOLE SHAMPOO"],
                "route": ["TOPICAL"], "dosage_form": ["SHAMPOO"]},
    # no drug_interactions section
}
_ORAL = {
    "set_id": "set-oral",
    "openfda": {"generic_name": ["KETOCONAZOLE"], "brand_name": ["KETOCONAZOLE TABLET"],
                "route": ["ORAL"], "dosage_form": ["TABLET"]},
    "drug_interactions": [
        "Coadministration with warfarin may increase anticoagulant effect. "
        "Ketoconazole is a strong inhibitor of CYP3A4."
    ],
    "clinical_pharmacology": ["Ketoconazole is metabolized in the liver."],
}


@pytest.mark.unit
def test_classify_shampoo_is_topical() -> None:
    fc = classify_formulation(_SHAMPOO)
    assert fc.klass == "topical"


@pytest.mark.unit
def test_classify_oral_tablet_is_systemic() -> None:
    fc = classify_formulation(_ORAL)
    assert fc.klass == "systemic"


@pytest.mark.unit
def test_systemic_with_ddi_outranks_topical() -> None:
    oral = score_label(_ORAL, drug="ketoconazole")
    shampoo = score_label(_SHAMPOO, drug="ketoconazole")
    assert oral.score > shampoo.score
    assert oral.has_drug_interactions is True
    assert shampoo.has_drug_interactions is False


@pytest.mark.unit
def test_rank_is_deterministic() -> None:
    a = rank_labels([_SHAMPOO, _ORAL], drug="ketoconazole")
    b = rank_labels([_ORAL, _SHAMPOO], drug="ketoconazole")
    assert [c.set_id for c in a] == [c.set_id for c in b]
    assert a[0].set_id == "set-oral"  # systemic+DDI wins


@pytest.mark.unit
def test_select_best_picks_systemic_with_ddi() -> None:
    best, ranked = select_best_label([_SHAMPOO, _ORAL], drug="ketoconazole")
    assert best is not None
    assert best.set_id == "set-oral"


@pytest.mark.unit
def test_select_best_returns_none_when_only_topical() -> None:
    # Adversarial: only a topical label -> refuse to select, signal a gap.
    best, ranked = select_best_label([_SHAMPOO], drug="ketoconazole")
    assert best is None
    assert ranked  # ranking still returned for transparency
