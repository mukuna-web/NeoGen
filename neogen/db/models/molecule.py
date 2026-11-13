"""Pydantic mirrors for the Week 1 molecule identity tables."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from neogen.chem.identity import STANDARDIZATION_VERSION


class MoleculeParentRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    parent_id: str
    canonical_smiles_no_stereo: str
    formula: str
    charge: int
    inchikey_skeleton: str
    standardization_version: str = STANDARDIZATION_VERSION


class MolecularIdentityRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    identity_id: str
    parent_id: str
    input_string: str
    input_type: Literal["smiles", "isomeric_smiles", "inchi", "sdf", "mol", "name", "known_drug_id"]
    canonical_smiles: str
    isomeric_smiles: str | None
    inchi: str | None
    inchikey: str
    rdkit_mol_hash: str
    standardization_version: str = STANDARDIZATION_VERSION


class StereoisomerRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stereoisomer_id: str
    parent_id: str
    identity_id: str | None
    isomeric_smiles: str
    stereo_hash: str
    stereo_centers_json: list[dict]
    ez_bonds_json: list[dict]
    stereo_defined_bool: bool
    stereo_ambiguity_level: Literal["fully_specified", "partial", "unknown"]
    chirality_class: Literal["achiral", "chiral_resolved", "chiral_unresolved", "racemic_intent"]


class StereoMixtureRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mixture_id: str
    parent_id: str
    mixture_type: Literal["racemic", "scalemic", "arbitrary"]
    component_stereoisomer_ids: list[str]
    component_fractions: list[float]
    evidence_id: str | None = None
    uncertainty: float | None = None


class DrugProductMappingRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    drug_product_id: str
    active_moiety_parent_id: str
    marketed_stereoisomer_id: str | None = None
    marketed_mixture_id: str | None = None
