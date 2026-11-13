"""
NeoGen Feature Engineering Tests
=================================
Tests for molecular fingerprint and descriptor computation.

Requires RDKit (the feature module calls ``Chem.MolFromSmiles`` and the
Morgan fingerprint generator directly). The whole file is skipped when
RDKit is unavailable.
"""
import pytest
import numpy as np

pytest.importorskip(
    "rdkit",
    reason="RDKit not installed — legacy feature engineering tests require it.",
)

# Known drug SMILES (duplicated here since conftest is auto-loaded, not importable)
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(O)=O"
IBUPROFEN_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
INVALID_SMILES = "THIS_IS_NOT_A_MOLECULE"


class TestComputeFeatures:
    """Test the _compute_features function from adme_pipeline."""

    @pytest.fixture(autouse=True)
    def _import_compute_features(self):
        """Import the feature computation function."""
        from neogen.simulation.adme_pipeline import _compute_features
        self.compute_features = _compute_features

    def test_valid_smiles_returns_array(self):
        """Valid SMILES should return a numpy array."""
        result = self.compute_features(ASPIRIN_SMILES)
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_morgan_fingerprint_length(self):
        """Feature vector should be 1044 dims (1024-bit Morgan FP + 20 descriptors)."""
        result = self.compute_features(ASPIRIN_SMILES)
        assert result is not None
        assert result.shape[0] == 1044, (
            f"Expected 1044 features, got {result.shape[0]}"
        )

    def test_descriptor_count(self):
        """Last 20 elements are molecular descriptors."""
        result = self.compute_features(ASPIRIN_SMILES)
        assert result is not None
        # First 1024 are fingerprint bits (0 or 1), last 20 are descriptors
        fp_bits = result[:1024]
        descriptors = result[1024:]
        assert len(descriptors) == 20
        # FP bits should only be 0 or 1
        assert set(fp_bits).issubset({0, 1, 0.0, 1.0})

    def test_invalid_smiles_returns_none(self):
        """Invalid SMILES should return None, not crash."""
        result = self.compute_features(INVALID_SMILES)
        assert result is None

    def test_garbage_smiles_returns_none(self):
        """Garbage SMILES with invalid syntax should return None."""
        result = self.compute_features("XYZXYZ[[[]]]]")
        assert result is None

    def test_different_drugs_give_different_features(self):
        """Aspirin and Ibuprofen should produce different feature vectors."""
        feat_asp = self.compute_features(ASPIRIN_SMILES)
        feat_ibu = self.compute_features(IBUPROFEN_SMILES)
        assert feat_asp is not None
        assert feat_ibu is not None
        # They should not be identical
        assert not np.array_equal(feat_asp, feat_ibu)

    def test_same_smiles_gives_same_features(self):
        """Same SMILES should always produce identical features."""
        feat1 = self.compute_features(ASPIRIN_SMILES)
        feat2 = self.compute_features(ASPIRIN_SMILES)
        assert feat1 is not None
        assert np.array_equal(feat1, feat2)

    def test_molecular_weight_positive(self):
        """First descriptor (index 1024) is MolWt — should be positive."""
        result = self.compute_features(ASPIRIN_SMILES)
        assert result is not None
        mol_wt = result[1024]  # First descriptor is MolWt
        assert mol_wt > 0, f"MolWt should be positive, got {mol_wt}"

    def test_features_are_finite(self):
        """All features should be finite (no NaN or Inf)."""
        result = self.compute_features(ASPIRIN_SMILES)
        assert result is not None
        assert np.all(np.isfinite(result)), "Feature vector contains NaN or Inf"
