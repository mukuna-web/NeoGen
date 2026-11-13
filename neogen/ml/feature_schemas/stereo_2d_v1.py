"""Stereo-aware 1054-dimensional 2D feature schema."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from neogen.chem.descriptors_2d import (
    legacy_2d_descriptor_values_in_original_order,
    mol_from_smiles,
    morgan_bits,
)
from neogen.chem.identity import Stereoisomer
from neogen.chem.stereo import stereoisomer_from_mol

SCHEMA_VERSION = "stereo_2d_v1"
DIMENSIONS = 1054
FEATURE_NAMES = tuple(sorted(f"stereo_2d_{idx:04d}" for idx in range(DIMENSIONS)))

_AMBIGUITY = {"fully_specified": 0.0, "partial": 1.0, "unknown": 2.0}


def stereo_features(isomer: Stereoisomer, n_heavy_atoms: int) -> np.ndarray:
    centers = list(isomer.stereo_centers_json)
    bonds = list(isomer.ez_bonds_json)
    n_total = len(centers)
    n_r = sum(1 for center in centers if center.get("chirality") == "R")
    n_s = sum(1 for center in centers if center.get("chirality") == "S")
    n_unspecified = sum(1 for center in centers if center.get("chirality") == "unspecified")
    n_specified = n_r + n_s
    n_e = sum(1 for bond in bonds if bond.get("stereo") == "E")
    n_z = sum(1 for bond in bonds if bond.get("stereo") == "Z")
    return np.array(
        [
            n_total,
            n_specified,
            n_unspecified,
            n_r,
            n_s,
            len(bonds),
            n_e,
            n_z,
            n_specified / max(n_heavy_atoms, 1),
            _AMBIGUITY[isomer.stereo_ambiguity_level],
        ],
        dtype=np.float64,
    )


def compute(isomer: Stereoisomer | dict) -> np.ndarray | None:
    if isinstance(isomer, dict):
        isomer = Stereoisomer(**isomer)
    mol = mol_from_smiles(isomer.isomeric_smiles)
    if mol is None:
        return None
    fp = morgan_bits(mol, n_bits=1024, use_chirality=True)
    descriptors = legacy_2d_descriptor_values_in_original_order(mol)
    stereo_record = stereoisomer_from_mol(mol, isomer.parent_id, isomer.identity_id)
    # Preserve repository metadata while refreshing stereochemical labels from RDKit.
    refreshed = Stereoisomer(**{**asdict(isomer), **asdict(stereo_record)})
    stereo = stereo_features(refreshed, mol.GetNumHeavyAtoms())
    return np.concatenate([fp, descriptors, stereo]).astype(np.float64)
