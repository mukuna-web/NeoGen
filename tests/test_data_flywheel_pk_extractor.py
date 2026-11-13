"""Tests for AUC/Cmax PK extraction (offline)."""

from __future__ import annotations

import pytest

from neogen.data_flywheel.pk_extractor import extract_pk_effects


@pytest.mark.unit
def test_fold_change_auc() -> None:
    eff = extract_pk_effects("Midazolam AUC was increased 3.5-fold when coadministered.")
    assert len(eff) == 1
    assert eff[0].metric == "AUC"
    assert eff[0].kind == "fold"
    assert eff[0].magnitude == 3.5
    assert eff[0].direction == "increase"
    assert eff[0].ratio == 3.5


@pytest.mark.unit
def test_percent_change_distinct_from_fold() -> None:
    eff = extract_pk_effects("The AUC increased by 50% in healthy subjects.")
    assert len(eff) == 1
    assert eff[0].kind == "percent"
    assert eff[0].magnitude == 50
    assert abs(eff[0].ratio - 1.5) < 1e-9  # +50% -> 1.5x, NOT 50-fold


@pytest.mark.unit
def test_cmax_distinguished_from_auc() -> None:
    eff = extract_pk_effects("Cmax increased 2-fold.")
    assert len(eff) == 1
    assert eff[0].metric == "Cmax"


@pytest.mark.unit
def test_no_number_no_effect() -> None:
    assert extract_pk_effects("Coadministration may increase exposure.") == []


@pytest.mark.unit
def test_random_percent_not_near_metric_ignored() -> None:
    # 50% is an adverse-event rate, NOT an AUC/Cmax change -> ignored.
    assert extract_pk_effects("Nausea was reported in 50% of patients.") == []


@pytest.mark.unit
def test_decrease_ratio_below_one() -> None:
    eff = extract_pk_effects("AUC was decreased by 75% with the inducer.")
    assert eff[0].direction == "decrease"
    assert abs(eff[0].ratio - 0.25) < 1e-9
