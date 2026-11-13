"""Legacy 1044-dimensional 2D feature schema."""
from __future__ import annotations

import numpy as np

from neogen.chem.descriptors_2d import (
    legacy_2d_descriptor_values_in_original_order,
    mol_from_smiles,
    morgan_bits,
)

SCHEMA_VERSION = "legacy_2d_v1"
DIMENSIONS = 1044
FEATURE_NAMES = tuple(sorted(f"legacy_2d_{idx:04d}" for idx in range(DIMENSIONS)))


def compute(smiles: str) -> np.ndarray | None:
    """Return the byte-identical pre-refactor legacy vector.

    The public legacy function historically returned float64, and existing
    sklearn/joblib predictors depend on that dtype.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    fp = morgan_bits(mol, n_bits=1024, use_chirality=False)
    descriptors = legacy_2d_descriptor_values_in_original_order(mol)
    return np.concatenate([fp, descriptors]).astype(np.float64)
