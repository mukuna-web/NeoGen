"""Stereochemistry detection, enumeration, and racemate handling."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

from neogen.chem.identity import (
    MolecularIdentity,
    StereoMixture,
    Stereoisomer,
    mixture_id_for,
    sorted_ez_bonds,
    sorted_stereo_centers,
    stereo_hash_for,
    stereoisomer_id_for,
)
from neogen.chem.standardize import (
    WARFARIN_FLAT_SMILES,
    isomeric_smiles,
    mol_from_input,
    molecule_parent_from_mol,
    normalize_known_name,
)


def stereo_centers_for_mol(mol: Chem.Mol) -> list[dict[str, Any]]:
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    centers = Chem.FindMolChiralCenters(
        mol,
        includeUnassigned=True,
        useLegacyImplementation=False,
    )
    mapped = [
        {
            "atom_index": int(atom_idx),
            "chirality": label if label in {"R", "S"} else "unspecified",
        }
        for atom_idx, label in centers
    ]
    return sorted_stereo_centers(mapped)


def ez_bonds_for_mol(mol: Chem.Mol) -> list[dict[str, Any]]:
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    bonds: list[dict[str, Any]] = []
    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo not in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ}:
            continue
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        atom_1, atom_2 = sorted((begin, end))
        bonds.append(
            {
                "bond_atom_1": atom_1,
                "bond_atom_2": atom_2,
                "stereo": "E" if stereo == Chem.BondStereo.STEREOE else "Z",
            }
        )
    return sorted_ez_bonds(bonds)


def ambiguity_and_class(
    stereo_centers: list[dict[str, Any]],
    ez_bonds: list[dict[str, Any]],
) -> tuple[bool, str, str]:
    labels = [center["chirality"] for center in stereo_centers]
    has_stereo_features = bool(stereo_centers or ez_bonds)
    has_unspecified = any(label == "unspecified" for label in labels)
    has_defined = any(label in {"R", "S"} for label in labels) or bool(ez_bonds)

    if not has_stereo_features:
        return True, "fully_specified", "achiral"
    if has_unspecified and has_defined:
        return False, "partial", "chiral_unresolved"
    if has_unspecified:
        return False, "unknown", "chiral_unresolved"
    return True, "fully_specified", "chiral_resolved"


def stereoisomer_from_mol(
    mol: Chem.Mol,
    parent_id: str,
    identity_id: str | None,
) -> Stereoisomer:
    centers = stereo_centers_for_mol(mol)
    bonds = ez_bonds_for_mol(mol)
    stereo_hash = stereo_hash_for(centers, bonds)
    defined, ambiguity, chirality_class = ambiguity_and_class(centers, bonds)
    return Stereoisomer(
        stereoisomer_id=stereoisomer_id_for(parent_id, stereo_hash),
        parent_id=parent_id,
        identity_id=identity_id,
        isomeric_smiles=isomeric_smiles(mol),
        stereo_hash=stereo_hash,
        stereo_centers_json=centers,
        ez_bonds_json=bonds,
        stereo_defined_bool=defined,
        stereo_ambiguity_level=ambiguity,
        chirality_class=chirality_class,
    )


def resolve_stereoisomer(identity: MolecularIdentity) -> Stereoisomer:
    mol = mol_from_input(identity.isomeric_smiles or identity.canonical_smiles, "isomeric_smiles")
    return stereoisomer_from_mol(mol, identity.parent_id, identity.identity_id)


def enumerate_stereoisomers(
    identity: MolecularIdentity,
    max_isomers: int = 64,
) -> list[Stereoisomer]:
    if max_isomers < 1:
        raise ValueError("max_isomers must be >= 1")
    mol = mol_from_input(identity.isomeric_smiles or identity.canonical_smiles, "isomeric_smiles")
    opts = StereoEnumerationOptions(
        tryEmbedding=False,
        unique=True,
        maxIsomers=max_isomers,
        onlyUnassigned=True,
    )
    isomers = tuple(EnumerateStereoisomers(mol, options=opts))
    if not isomers:
        isomers = (mol,)
    records = [
        stereoisomer_from_mol(isomer, identity.parent_id, identity.identity_id)
        for isomer in isomers[:max_isomers]
    ]
    deduped = {record.stereoisomer_id: record for record in records}
    return sorted(deduped.values(), key=lambda item: item.stereoisomer_id)


def is_default_racemic_warfarin(input_string: str, input_type: str) -> bool:
    return input_type in {"name", "known_drug_id"} and normalize_known_name(input_string) == "warfarin"


def racemic_warfarin_mixture(identity: MolecularIdentity) -> tuple[StereoMixture, list[Stereoisomer]]:
    parent = molecule_parent_from_mol(mol_from_input(WARFARIN_FLAT_SMILES, "smiles"))
    flat_identity = MolecularIdentity(
        **{
            **asdict(identity),
            "parent_id": parent.parent_id,
            "canonical_smiles": parent.canonical_smiles_no_stereo,
        }
    )
    components = enumerate_stereoisomers(flat_identity, max_isomers=2)
    component_ids = [component.stereoisomer_id for component in components]
    mixture = StereoMixture(
        mixture_id=mixture_id_for(parent.parent_id, "racemic", component_ids),
        parent_id=parent.parent_id,
        mixture_type="racemic",
        component_stereoisomer_ids=component_ids,
        component_fractions=[0.5, 0.5],
        evidence_id="known_drug:warfarin",
        uncertainty=0.0,
    )
    return mixture, components
