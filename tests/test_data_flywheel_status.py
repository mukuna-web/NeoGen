"""Tests for the flywheel status aggregator."""

from __future__ import annotations

import pytest

from neogen.data_flywheel.flywheel_status import compute_status


@pytest.mark.unit
def test_status_has_expected_keys() -> None:
    s = compute_status()
    for k in (
        "sources", "evidence", "pairs", "identity_mappings", "structure_records",
        "mechanism_assertions", "pk_records", "open_gaps", "closed_gaps",
        "reviewed_count", "blocked_count", "isomer_counterfactuals",
    ):
        assert k in s, f"missing status key {k}"
        assert isinstance(s[k], int)


@pytest.mark.unit
def test_counts_are_nonnegative() -> None:
    s = compute_status()
    assert all(
        s[k] >= 0 for k in s if isinstance(s[k], int)
    )


@pytest.mark.unit
def test_status_reflects_prior_cycles() -> None:
    # These artifacts were produced by earlier cycles in this repo.
    s = compute_status()
    assert s["sources"] >= 70          # 75-source registry
    assert s["evidence"] >= 70         # 74 openFDA evidence records
    assert s["identity_mappings"] >= 20
    assert s["closed_gaps"] >= 1
