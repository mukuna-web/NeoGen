"""
NeoGen ADME Pipeline Unit Tests
=================================
Tests for the ModularADMEPredictor class: initialization, prediction
outputs, value ranges, and reproducibility.

Requires RDKit (the ADME pipeline calls RDKit for SMILES sanitation and
descriptor computation). The whole file is skipped when RDKit is
unavailable.
"""
import pytest

pytest.importorskip(
    "rdkit",
    reason="RDKit not installed — legacy ADME pipeline tests require it.",
)

# Known drug SMILES (duplicated here since conftest is auto-loaded, not importable)
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(O)=O"
IBUPROFEN_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
METFORMIN_SMILES = "CN(C)C(=N)NC(=N)N"


class TestPredictorInitialization:
    def test_predictor_loads(self, adme_predictor):
        """ModularADMEPredictor initializes and loads models without error."""
        assert adme_predictor is not None
        assert adme_predictor.loaded is True

    def test_models_are_loaded(self, adme_predictor):
        """Critical sub-models (fu, cl, vd, F) should be loaded."""
        assert adme_predictor.fu_model is not None
        assert adme_predictor.cl_model is not None
        assert adme_predictor.vd_model is not None
        assert adme_predictor.f_model is not None


class TestPredictAspirin:
    def test_predict_returns_dict(self, adme_predictor):
        """predict() should return a dictionary."""
        result = adme_predictor.predict(ASPIRIN_SMILES)
        assert isinstance(result, dict)

    def test_predict_has_predictions_key(self, adme_predictor):
        """Result must contain a 'predictions' dict with PK params."""
        result = adme_predictor.predict(ASPIRIN_SMILES)
        assert "predictions" in result
        preds = result["predictions"]
        assert isinstance(preds, dict)

    def test_predict_has_expected_pk_params(self, adme_predictor):
        """predictions dict should contain cmax, auc, half_life, clearance, vd."""
        result = adme_predictor.predict(ASPIRIN_SMILES)
        preds = result["predictions"]
        expected_keys = {"cmax", "auc", "half_life", "clearance", "vd"}
        assert expected_keys.issubset(set(preds.keys())), (
            f"Missing keys: {expected_keys - set(preds.keys())}"
        )

    def test_predict_has_adme_details(self, adme_predictor):
        """Result should include adme_details with fu, F, etc."""
        result = adme_predictor.predict(ASPIRIN_SMILES)
        assert "adme_details" in result
        details = result["adme_details"]
        assert "fu" in details
        assert "F" in details


class TestPredictPositiveValues:
    """All PK parameters must be positive (physically meaningful)."""

    @pytest.fixture
    def aspirin_preds(self, adme_predictor):
        result = adme_predictor.predict(ASPIRIN_SMILES)
        return result["predictions"]

    def test_cmax_positive(self, aspirin_preds):
        assert aspirin_preds["cmax"] > 0

    def test_auc_positive(self, aspirin_preds):
        assert aspirin_preds["auc"] > 0

    def test_half_life_positive(self, aspirin_preds):
        assert aspirin_preds["half_life"] > 0

    def test_clearance_positive(self, aspirin_preds):
        assert aspirin_preds["clearance"] > 0

    def test_vd_positive(self, aspirin_preds):
        assert aspirin_preds["vd"] > 0


class TestPredictValueRanges:
    """PK values should fall within pharmacologically reasonable ranges."""

    def test_fu_in_range(self, adme_predictor):
        """Fraction unbound must be between 0 and 1."""
        result = adme_predictor.predict(ASPIRIN_SMILES)
        fu = result["adme_details"]["fu"]
        assert 0 < fu <= 1.0, f"fu={fu} is out of range (0, 1]"

    def test_half_life_reasonable(self, adme_predictor):
        """Half-life should be between 0.1 and 1000 hours for real drugs."""
        result = adme_predictor.predict(ASPIRIN_SMILES)
        hl = result["predictions"]["half_life"]
        assert 0.1 < hl < 1000, f"half_life={hl} is outside [0.1, 1000] hours"

    def test_bioavailability_in_range(self, adme_predictor):
        """Oral bioavailability F should be between 0 and 1."""
        result = adme_predictor.predict(ASPIRIN_SMILES)
        F = result["adme_details"]["F"]
        assert 0 < F <= 1.0, f"F={F} is out of range (0, 1]"

    def test_clearance_reasonable(self, adme_predictor):
        """Clearance should be between 0.001 and 500 L/h for a 70 kg human."""
        result = adme_predictor.predict(ASPIRIN_SMILES)
        cl = result["predictions"]["clearance"]
        assert 0.001 < cl < 500, f"clearance={cl} L/h is outside reasonable range"

    def test_vd_reasonable(self, adme_predictor):
        """Volume of distribution: 0.01 to 5000 L."""
        result = adme_predictor.predict(ASPIRIN_SMILES)
        vd = result["predictions"]["vd"]
        assert 0.01 < vd < 5000, f"vd={vd} L is outside reasonable range"


class TestPredictMultipleDrugs:
    """Ensure predictions work for different drug classes."""

    def test_predict_ibuprofen(self, adme_predictor):
        result = adme_predictor.predict(IBUPROFEN_SMILES)
        assert "error" not in result
        assert result["predictions"]["cmax"] > 0

    def test_predict_metformin(self, adme_predictor):
        result = adme_predictor.predict(METFORMIN_SMILES)
        assert "error" not in result
        assert result["predictions"]["cmax"] > 0


class TestPredictBatchConsistency:
    def test_same_smiles_gives_same_result(self, adme_predictor):
        """Running predict() twice with the same SMILES should give identical results."""
        result1 = adme_predictor.predict(ASPIRIN_SMILES)
        result2 = adme_predictor.predict(ASPIRIN_SMILES)
        preds1 = result1["predictions"]
        preds2 = result2["predictions"]
        for key in preds1:
            assert abs(preds1[key] - preds2[key]) < 1e-9, (
                f"{key} differs: {preds1[key]} vs {preds2[key]}"
            )


class TestPredictInvalidInput:
    def test_invalid_smiles_returns_error(self, adme_predictor):
        """Invalid SMILES should return an error dict, not crash."""
        result = adme_predictor.predict("NOT_A_REAL_MOLECULE")
        assert "error" in result

    def test_garbage_smiles_returns_error(self, adme_predictor):
        """Completely invalid SMILES string should return an error dict."""
        result = adme_predictor.predict("XYZXYZ[[[]]]]")
        assert "error" in result
