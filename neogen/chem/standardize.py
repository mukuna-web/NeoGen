"""RDKit-backed deterministic molecule normalization."""
from __future__ import annotations

import hashlib

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from neogen.chem.identity import (
    STANDARDIZATION_VERSION,
    InputType,
    MolecularIdentity,
    MoleculeParent,
    identity_id_for,
    parent_id_for,
)

MAX_HEAVY_ATOMS = 80

WARFARIN_R_SMILES = "CC(=O)C[C@H](c1ccccc1)c1c(O)c2ccccc2oc1=O"
WARFARIN_S_SMILES = "CC(=O)C[C@@H](c1ccccc1)c1c(O)c2ccccc2oc1=O"
WARFARIN_FLAT_SMILES = "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"

KNOWN_DRUG_SMILES: dict[str, str] = {
    "warfarin": WARFARIN_FLAT_SMILES,
    "(r)-warfarin": WARFARIN_R_SMILES,
    "r-warfarin": WARFARIN_R_SMILES,
    "(s)-warfarin": WARFARIN_S_SMILES,
    "s-warfarin": WARFARIN_S_SMILES,
}


class MoleculeNormalizationError(ValueError):
    """Raised when an input cannot be converted to an RDKit molecule."""


class HeavyAtomLimitError(MoleculeNormalizationError):
    """Raised when a molecule exceeds the Week 1 heavy atom cap."""


def normalize_known_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def smiles_for_input(input_string: str, input_type: InputType) -> str:
    if input_type in {"smiles", "isomeric_smiles"}:
        return input_string.strip()
    if input_type in {"name", "known_drug_id"}:
        key = normalize_known_name(input_string)
        if key in KNOWN_DRUG_SMILES:
            return KNOWN_DRUG_SMILES[key]
        raise MoleculeNormalizationError(f"unknown known-drug/name input: {input_string!r}")
    raise MoleculeNormalizationError(f"unsupported input_type for Week 1: {input_type!r}")


def mol_from_input(input_string: str, input_type: InputType) -> Chem.Mol:
    smiles = smiles_for_input(input_string, input_type)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise MoleculeNormalizationError(f"invalid molecular input: {input_string!r}")
    if mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
        raise HeavyAtomLimitError(
            f"molecule has {mol.GetNumHeavyAtoms()} heavy atoms; cap is {MAX_HEAVY_ATOMS}"
        )
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def canonical_smiles_no_stereo(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def isomeric_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def molecule_parent_from_mol(mol: Chem.Mol) -> MoleculeParent:
    canonical_no_stereo = canonical_smiles_no_stereo(mol)
    inchi_key = Chem.MolToInchiKey(mol)
    skeleton = inchi_key.split("-", 1)[0] if inchi_key else ""
    return MoleculeParent(
        parent_id=parent_id_for(canonical_no_stereo),
        canonical_smiles_no_stereo=canonical_no_stereo,
        formula=rdMolDescriptors.CalcMolFormula(mol),
        charge=Chem.GetFormalCharge(mol),
        inchikey_skeleton=skeleton,
        standardization_version=STANDARDIZATION_VERSION,
    )


def molecular_identity_from_input(
    input_string: str,
    input_type: InputType,
) -> tuple[MoleculeParent, MolecularIdentity, Chem.Mol]:
    mol = mol_from_input(input_string, input_type)
    parent = molecule_parent_from_mol(mol)
    canonical = canonical_smiles_no_stereo(mol)
    iso = isomeric_smiles(mol)
    inchi = Chem.MolToInchi(mol)
    inchikey = Chem.MolToInchiKey(mol)
    rdkit_mol_hash = hashlib.sha256(
        Chem.MolToMolBlock(mol, includeStereo=True).encode("utf-8")
    ).hexdigest()
    identity = MolecularIdentity(
        identity_id=identity_id_for(input_type, input_string),
        parent_id=parent.parent_id,
        input_string=input_string,
        input_type=input_type,
        canonical_smiles=canonical,
        isomeric_smiles=iso,
        inchi=inchi,
        inchikey=inchikey,
        rdkit_mol_hash=rdkit_mol_hash,
        standardization_version=STANDARDIZATION_VERSION,
    )
    return parent, identity, mol
