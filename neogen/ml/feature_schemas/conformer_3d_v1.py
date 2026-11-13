"""Conformer ensemble 1086-dimensional 3D feature schema."""
from __future__ import annotations

import math

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from neogen.chem.conformers import ConformerRecord, ConformerSet
from neogen.chem.geometry import (
    DESCRIPTOR_NAMES,
    boltzmann_weights,
    conformer_descriptor_values,
    forcefield_version_encoded,
    signed_chiral_volume,
    weighted_mean_std,
)
from neogen.chem.identity import Stereoisomer
from neogen.ml.feature_schemas import stereo_2d_v1

SCHEMA_VERSION = "conformer_3d_v1"
DIMENSIONS = 1086
FEATURE_NAMES = tuple(sorted(f"conformer_3d_{idx:04d}" for idx in range(DIMENSIONS)))

RAW_3D_FEATURE_NAMES = tuple(
    name
    for descriptor in DESCRIPTOR_NAMES
    for name in (f"{descriptor}_boltzmann_mean", f"{descriptor}_boltzmann_std")
) + (
    "n_conformers",
    "energy_min_kcal_mol",
    "energy_max_kcal_mol",
    "energy_spread_kcal_mol",
    "rmsd_diversity_mean",
    "rmsd_diversity_max",
    "chiral_signed_volume_mean",
    "chiral_signed_volume_std",
    "forcefield_version_encoded",
    "seed_used",
)


def _as_conformer_set(value: ConformerSet | dict) -> ConformerSet:
    if isinstance(value, ConformerSet):
        return value
    data = dict(value)
    data["conformers"] = tuple(
        item if isinstance(item, ConformerRecord) else ConformerRecord(**item)
        for item in data["conformers"]
    )
    data.setdefault("cache_hit", True)
    data.setdefault("warnings", ())
    return ConformerSet(**data)


def _as_isomer(value: Stereoisomer | dict) -> Stereoisomer:
    return value if isinstance(value, Stereoisomer) else Stereoisomer(**value)


def _load_mol(record: ConformerRecord) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(record.coordinates_uri, removeHs=False)
    mol = supplier[0] if supplier and len(supplier) else None
    if mol is None:
        raise FileNotFoundError(record.coordinates_uri)
    return mol


def _rmsd(mols: list[Chem.Mol]) -> tuple[float, float]:
    if len(mols) < 2:
        return 0.0, 0.0
    values: list[float] = []
    for left, mol_i in enumerate(mols):
        for mol_j in mols[left + 1 :]:
            values.append(float(rdMolAlign.GetBestRMS(mol_i, mol_j)))
    return float(np.mean(values)), float(np.max(values))


def compute_3d_tail(conformer_set: ConformerSet | dict, isomer: Stereoisomer | dict) -> np.ndarray:
    conf_set = _as_conformer_set(conformer_set)
    iso = _as_isomer(isomer)
    ordered = sorted(conf_set.conformers, key=lambda item: (float(item.energy_kcal_mol), item.coordinate_hash))
    mols = [_load_mol(record) for record in ordered]
    energies = [float(record.energy_kcal_mol) for record in ordered]
    weights = boltzmann_weights(energies)
    raw = np.vstack([conformer_descriptor_values(mol, 0) for mol in mols]) if mols else np.zeros((0, len(DESCRIPTOR_NAMES)))
    descriptor_tail = weighted_mean_std(raw, weights)
    rmsd_mean, rmsd_max = _rmsd(mols)
    signed = np.asarray([signed_chiral_volume(mol, iso, 0) for mol in mols], dtype=np.float64)
    energy_min = min(energies) if energies else 0.0
    energy_max = max(energies) if energies else 0.0
    ensemble_tail = np.asarray(
        [
            len(mols),
            energy_min,
            energy_max,
            energy_max - energy_min,
            rmsd_mean,
            rmsd_max,
            float(np.mean(signed)) if signed.size else 0.0,
            float(np.std(signed)) if signed.size else 0.0,
            forcefield_version_encoded(conf_set.forcefield),
            float(conf_set.seed),
        ],
        dtype=np.float64,
    )
    tail = np.concatenate([descriptor_tail, ensemble_tail]).astype(np.float64)
    tail[np.isnan(tail)] = 0.0
    tail[np.isinf(tail)] = 0.0
    return tail


def compute(conformer_set: ConformerSet | dict, isomer: Stereoisomer | dict) -> np.ndarray | None:
    iso = _as_isomer(isomer)
    stereo_values = stereo_2d_v1.compute(iso)
    if stereo_values is None:
        return None
    tail = compute_3d_tail(conformer_set, iso)
    values = np.concatenate([stereo_values, tail]).astype(np.float64)
    if values.shape[0] != DIMENSIONS:
        raise ValueError(f"{SCHEMA_VERSION} produced {values.shape[0]} dims; expected {DIMENSIONS}")
    values[np.isnan(values)] = 0.0
    values[np.isinf(values)] = 0.0
    return values
