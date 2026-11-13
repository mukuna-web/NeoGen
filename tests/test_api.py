"""
NeoGen API Endpoint Tests
==========================
Tests for all primary FastAPI endpoints: health, predict, DDI lookup, examples.

Requires RDKit (the legacy ``neogen.api.predict`` module imports RDKit at
module load time). The whole file is skipped when RDKit is unavailable so
CI environments without the cheminformatics toolchain don't see spurious
failures. The V1 PBPK API lives in :mod:`neogen.api.v1` and has its own
tests in ``test_v1_p7_api.py`` — those run unconditionally.
"""
import pytest

pytest.importorskip(
    "rdkit",
    reason="RDKit not installed — legacy /api/predict tests require it.",
)


# ── Health Check ────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_check(self, client):
        """GET /api/health returns 200 with status field."""
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_has_model_info(self, client):
        """Health response includes model metadata."""
        resp = client.get("/api/health")
        data = resp.json()
        assert "prediction_engine" in data


# ── Predict Endpoint ────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_predict_valid_smiles(self, client, aspirin_smiles):
        """POST /api/predict with aspirin SMILES returns 200 and predictions."""
        resp = client.post("/api/predict", json={"smiles": aspirin_smiles})
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert len(data["predictions"]) > 0

    def test_predict_invalid_smiles(self, client, invalid_smiles):
        """POST /api/predict with garbage SMILES returns 400 error."""
        resp = client.post("/api/predict", json={"smiles": invalid_smiles})
        # The API should return a 4xx error for invalid molecules
        assert resp.status_code in (400, 422)

    def test_predict_response_structure(self, client, aspirin_smiles):
        """Response contains all expected top-level fields."""
        resp = client.post("/api/predict", json={"smiles": aspirin_smiles})
        assert resp.status_code == 200
        data = resp.json()
        # PredictionResponse fields
        assert "smiles" in data
        assert "predictions" in data
        assert "time_course" in data
        assert "organ_activity" in data
        assert "dose_mg" in data
        assert "route" in data

    def test_predict_predictions_have_required_fields(self, client, aspirin_smiles):
        """Each PKPrediction entry has parameter, value, unit, confidence."""
        resp = client.post("/api/predict", json={"smiles": aspirin_smiles})
        data = resp.json()
        for pred in data["predictions"]:
            assert "parameter" in pred
            assert "value" in pred
            assert "unit" in pred
            assert "confidence" in pred

    def test_predict_time_course_non_empty(self, client, aspirin_smiles):
        """Time course should have multiple time points for animation."""
        resp = client.post("/api/predict", json={"smiles": aspirin_smiles})
        data = resp.json()
        assert isinstance(data["time_course"], list)
        assert len(data["time_course"]) > 0

    def test_predict_with_custom_dose(self, client, aspirin_smiles):
        """Specifying a custom dose should be accepted."""
        resp = client.post("/api/predict", json={
            "smiles": aspirin_smiles,
            "dose_mg": 500.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["dose_mg"] == 500.0

    def test_predict_ibuprofen(self, client, ibuprofen_smiles):
        """Another valid drug should also return successfully."""
        resp = client.post("/api/predict", json={"smiles": ibuprofen_smiles})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) > 0


# ── DDI Lookup Endpoint ─────────────────────────────────────────────

class TestDDILookupEndpoint:
    def test_ddi_lookup_known_drug(self, client):
        """GET /api/ddi-lookup/warfarin returns interactions."""
        resp = client.get("/api/ddi-lookup/warfarin")
        assert resp.status_code == 200
        data = resp.json()
        # Should return a dict or list with interaction data
        assert data is not None

    def test_ddi_lookup_unknown_drug(self, client):
        """Unknown drug returns 200 with empty or minimal results (not a crash)."""
        resp = client.get("/api/ddi-lookup/FakeDrugXYZ999")
        # Should not crash — either 200 with empty or 404
        assert resp.status_code in (200, 404)


# ── Examples Endpoint ───────────────────────────────────────────────

class TestExamplesEndpoint:
    def test_examples_returns_list(self, client):
        """GET /api/examples returns a non-empty list."""
        resp = client.get("/api/examples")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_examples_have_required_fields(self, client):
        """Each example has name, smiles, and use."""
        resp = client.get("/api/examples")
        data = resp.json()
        for example in data:
            assert "name" in example
            assert "smiles" in example
            assert "use" in example

    def test_examples_smiles_are_strings(self, client):
        """SMILES in examples should be non-empty strings."""
        resp = client.get("/api/examples")
        data = resp.json()
        for example in data:
            assert isinstance(example["smiles"], str)
            assert len(example["smiles"]) > 0
