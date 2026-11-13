"""Deterministic RDKit 2D descriptor helpers for feature schemas."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors


LEGACY_2D_DESCRIPTOR_NAMES = tuple(
    sorted(
        [
            "mol_wt",
            "mol_logp",
            "tpsa",
            "num_h_donors",
            "num_h_acceptors",
            "num_rotatable_bonds",
            "ring_count",
            "fraction_csp3",
            "heavy_atom_count",
            "num_aromatic_rings",
            "num_aliphatic_rings",
            "num_heteroatoms",
            "aromatic_atom_fraction",
            "mol_mr",
            "num_oxygen",
            "num_nitrogen",
            "num_sulfur",
            "num_fluorine",
            "num_chlorine",
            "max_partial_charge",
        ]
    )
)


def is_nan(value: object) -> bool:
    try:
        return bool(value != value)
    except Exception:
        return False


def nan_to_zero(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    arr[np.isnan(arr)] = 0.0
    return arr


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def morgan_bits(mol: Chem.Mol, *, n_bits: int = 1024, use_chirality: bool = False) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        2,
        nBits=n_bits,
        useChirality=use_chirality,
    )
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def legacy_2d_descriptor_values_in_original_order(mol: Chem.Mol) -> np.ndarray:
    """Return the 20 legacy descriptor values in the pre-refactor order.

    NaN or missing descriptor values are normalized to 0.0, matching the
    existing policy for partial-charge descriptors.
    """
    n_heavy = Descriptors.HeavyAtomCount(mol)
    max_pc = Descriptors.MaxPartialCharge(mol)
    values = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        Descriptors.FractionCSP3(mol),
        n_heavy,
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumHeteroatoms(mol),
        sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()) / max(n_heavy, 1),
        Descriptors.MolMR(mol),
        sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8),
        sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7),
        sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16),
        sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9),
        sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 17),
        max_pc if max_pc is not None and not is_nan(max_pc) else 0.0,
    ]
    return nan_to_zero(values)


def sorted_legacy_feature_names(prefix: str = "legacy2d") -> tuple[str, ...]:
    fp_names = [f"{prefix}_morgan_{i:04d}" for i in range(1024)]
    return tuple(sorted([*fp_names, *LEGACY_2D_DESCRIPTOR_NAMES]))
