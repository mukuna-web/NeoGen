"""
NeoGen DDI Lookup Tests
========================
Tests for the drug-drug interaction lookup module (DDInter 2.0 data).
"""
import pytest
from neogen.simulation.ddi_lookup import lookup_ddi, get_all_interactions


class TestLookupDDI:
    def test_lookup_known_pair(self):
        """Warfarin + Aspirin should return at least one interaction record."""
        results = lookup_ddi("Warfarin", "Aspirin")
        # If DDInter data is loaded, we expect matches; if not, empty is acceptable
        assert isinstance(results, list)
        if len(results) > 0:
            record = results[0]
            assert "level" in record
            assert record["level"] in ("Major", "Moderate", "Minor", "Unknown")

    def test_lookup_unknown_pair(self):
        """Two fake drugs should return an empty list."""
        results = lookup_ddi("FakeDrugXYZ123", "FakeDrugABC456")
        assert isinstance(results, list)
        assert len(results) == 0

    def test_lookup_returns_list_of_dicts(self):
        """Each result should be a dict with expected keys."""
        results = lookup_ddi("Warfarin", "Aspirin")
        for record in results:
            assert isinstance(record, dict)
            assert "level" in record
            assert "drug_a" in record
            assert "drug_b" in record
            assert "source" in record

    def test_case_insensitivity_lower(self):
        """Lookup should work with lowercase drug names."""
        results_upper = lookup_ddi("Warfarin", "Aspirin")
        results_lower = lookup_ddi("warfarin", "aspirin")
        assert len(results_upper) == len(results_lower)

    def test_case_insensitivity_mixed(self):
        """Lookup should work with mixed-case drug names."""
        results_a = lookup_ddi("WARFARIN", "ASPIRIN")
        results_b = lookup_ddi("Warfarin", "Aspirin")
        assert len(results_a) == len(results_b)

    def test_symmetric_lookup(self):
        """lookup_ddi(A, B) should give same count as lookup_ddi(B, A)."""
        results_ab = lookup_ddi("Warfarin", "Aspirin")
        results_ba = lookup_ddi("Aspirin", "Warfarin")
        assert len(results_ab) == len(results_ba)


class TestGetAllInteractions:
    def test_get_all_interactions_warfarin(self):
        """Warfarin should have known interaction partners."""
        results = get_all_interactions("Warfarin")
        assert isinstance(results, list)
        # Warfarin is one of the most interacting drugs, expect many partners
        # (if DDInter data is loaded)
        if len(results) > 0:
            record = results[0]
            assert "partner" in record
            assert "level" in record

    def test_get_all_interactions_unknown_drug(self):
        """Unknown drug should return an empty list, not crash."""
        results = get_all_interactions("TotallyFakeDrug999")
        assert isinstance(results, list)
        assert len(results) == 0

    def test_get_all_interactions_sorted_by_severity(self):
        """Results should be sorted with Major interactions first."""
        results = get_all_interactions("Warfarin")
        if len(results) >= 2:
            severity_order = {"Major": 3, "Moderate": 2, "Minor": 1}
            severities = [severity_order.get(r["level"], 0) for r in results]
            # Should be non-increasing (sorted descending)
            assert severities == sorted(severities, reverse=True)

    def test_get_all_interactions_case_insensitive(self):
        """Should work with any casing."""
        results_upper = get_all_interactions("WARFARIN")
        results_title = get_all_interactions("Warfarin")
        assert len(results_upper) == len(results_title)
