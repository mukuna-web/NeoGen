"""3D geometry descriptors and deterministic ensemble aggregation."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdMolDescriptors

from neogen.chem.identity import Stereoisomer

KCAL_PER_MOL_TO_J_PER_MOL = 4184.0
GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
TEMPERATURE_K = 298.15

DESCRIPTOR_NAMES = (
    "asphericity",
    "eccentricity",
    "inertial_shape_factor",
    "npr1",
    "npr2",
    "pmi1",
    "pmi2",
    "pmi3",
    "radius_of_gyration",
    "psa_3d",
    "sasa",
)


@dataclass(frozen=True)
class EnsembleGeometry:
    descriptor_values: np.ndarray
    ensemble_values: np.ndarray


def _safe_float(fn, default: float = 0.0) -> float:
    try:
        value = float(fn())
    except Exception:
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def conformer_descriptor_values(mol: Chem.Mol, conf_id: int) -> np.ndarray:
    """Return the 11 raw descriptor values for one conformer."""
    return np.array(
        [
            _safe_float(lambda: rdMolDescriptors.CalcAsphericity(mol, confId=conf_id)),
            _safe_float(lambda: rdMolDescriptors.CalcEccentricity(mol, confId=conf_id)),
            _safe_float(lambda: rdMolDescriptors.CalcInertialShapeFactor(mol, confId=conf_id)),
            _safe_float(lambda: rdMolDescriptors.CalcNPR1(mol, confId=conf_id)),
            _safe_float(lambda: rdMolDescriptors.CalcNPR2(mol, confId=conf_id)),
            _safe_float(lambda: rdMolDescriptors.CalcPMI1(mol, confId=conf_id)),
            _safe_float(lambda: rdMolDescriptors.CalcPMI2(mol, confId=conf_id)),
            _safe_float(lambda: rdMolDescriptors.CalcPMI3(mol, confId=conf_id)),
            _safe_float(lambda: rdMolDescriptors.CalcRadiusOfGyration(mol, confId=conf_id)),
            _safe_float(lambda: rdMolDescriptors.CalcTPSA(mol, includeSandP=True)),
            _safe_float(lambda: rdMolDescriptors.CalcLabuteASA(mol)),
        ],
        dtype=np.float64,
    )


def boltzmann_weights(energies_kcal_mol: list[float]) -> np.ndarray:
    energies = np.asarray(energies_kcal_mol, dtype=np.float64)
    if energies.size == 0:
        return np.zeros(0, dtype=np.float64)
    shifted = energies - float(np.min(energies))
    factors = np.exp(-(shifted * KCAL_PER_MOL_TO_J_PER_MOL) / (GAS_CONSTANT_J_PER_MOL_K * TEMPERATURE_K))
    total = float(np.sum(factors))
    if total <= 0.0:
        return np.full(energies.shape, 1.0 / energies.size, dtype=np.float64)
    return factors / total


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(len(DESCRIPTOR_NAMES) * 2, dtype=np.float64)
    means = np.sum(values * weights[:, None], axis=0)
    variances = np.sum(weights[:, None] * np.square(values - means), axis=0)
    stds = np.sqrt(np.maximum(variances, 0.0))
    out: list[float] = []
    for mean, std in zip(means, stds, strict=True):
        out.extend([float(mean), float(std)])
    return np.asarray(out, dtype=np.float64)


def signed_chiral_volume(mol: Chem.Mol, isomer: Stereoisomer, conf_id: int) -> float:
    centers = [c for c in isomer.stereo_centers_json if c.get("chirality") in {"R", "S"}]
    if not centers:
        return 0.0
    center_idx = int(centers[0]["atom_index"])
    atom = mol.GetAtomWithIdx(center_idx)
    neighbors = sorted(n.GetIdx() for n in atom.GetNeighbors())
    if len(neighbors) < 3:
        return 0.0
    conf = mol.GetConformer(conf_id)
    center = np.asarray(conf.GetAtomPosition(center_idx), dtype=np.float64)
    vectors = [np.asarray(conf.GetAtomPosition(idx), dtype=np.float64) - center for idx in neighbors[:3]]
    return float(np.dot(np.cross(vectors[0], vectors[1]), vectors[2]))


def rmsd_diversity(mol: Chem.Mol, conf_ids: list[int]) -> tuple[float, float]:
    if len(conf_ids) < 2:
        return 0.0, 0.0
    values: list[float] = []
    for left, cid_i in enumerate(conf_ids):
        for cid_j in conf_ids[left + 1 :]:
            values.append(float(rdMolAlign.GetBestRMS(mol, mol, prbId=cid_i, refId=cid_j)))
    return float(np.mean(values)), float(np.max(values))


def forcefield_version_encoded(forcefield: str) -> float:
    return {"mmff94s": 1.0, "mmff94": 2.0, "uff": 3.0, "none": 0.0}.get(forcefield, 0.0)


def ensemble_geometry_features(
    *,
    mol: Chem.Mol,
    conformers: list[dict],
    isomer: Stereoisomer,
    forcefield: str,
    seed: int,
) -> EnsembleGeometry:
    ordered = sorted(conformers, key=lambda item: (float(item["energy_kcal_mol"]), item["coordinate_hash"]))
    conf_ids = [int(item["source_conf_id"]) if "source_conf_id" in item else int(item.get("rank", 0)) for item in ordered]
    energies = [float(item["energy_kcal_mol"]) for item in ordered]
    weights = boltzmann_weights(energies)
    raw = np.vstack([conformer_descriptor_values(mol, conf_id) for conf_id in conf_ids]) if conf_ids else np.zeros((0, len(DESCRIPTOR_NAMES)))
    descriptors = weighted_mean_std(raw, weights)
    rmsd_mean, rmsd_max = rmsd_diversity(mol, conf_ids)
    signed = np.asarray([signed_chiral_volume(mol, isomer, conf_id) for conf_id in conf_ids], dtype=np.float64)
    energy_min = min(energies) if energies else 0.0
    energy_max = max(energies) if energies else 0.0
    ensemble = np.asarray(
        [
            len(conf_ids),
            energy_min,
            energy_max,
            energy_max - energy_min,
            rmsd_mean,
            rmsd_max,
            float(np.mean(signed)) if signed.size else 0.0,
            float(np.std(signed)) if signed.size else 0.0,
            forcefield_version_encoded(forcefield),
            float(seed),
        ],
        dtype=np.float64,
    )
    return EnsembleGeometry(descriptor_values=descriptors, ensemble_values=ensemble)
