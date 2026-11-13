"""Deterministic molecular identity records and hash functions."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

# Bumped per DR-003 (2026-05-13) — local RDKit moved to 2026.03.1; salt is text not assertion.
STANDARDIZATION_VERSION = "stdz_v2_rdkit_2026.03.1"
logger = logging.getLogger(__name__)


def assert_rdkit_version_alignment() -> None:
    expected = STANDARDIZATION_VERSION.split("rdkit_", 1)[1]
    try:
        from rdkit import __version__ as rdkit_version
    except Exception as exc:  # pragma: no cover - import failure covered elsewhere
        logger.warning("RDKit unavailable while checking standardization version: %s", exc)
        return
    if not rdkit_version.startswith(expected):
        message = (
            f"RDKit version drift: {rdkit_version} vs expected {expected} "
            f"(stdz={STANDARDIZATION_VERSION})"
        )
        if os.getenv("NEOGEN_STRICT_RDKIT_VERSION") == "1":
            raise RuntimeError(message)
        logger.warning(message)


assert_rdkit_version_alignment()

InputType = Literal[
    "smiles",
    "isomeric_smiles",
    "inchi",
    "sdf",
    "mol",
    "name",
    "known_drug_id",
]
StereoAmbiguityLevel = Literal["fully_specified", "partial", "unknown"]
ChiralityClass = Literal[
    "achiral",
    "chiral_resolved",
    "chiral_unresolved",
    "racemic_intent",
]
MixtureType = Literal["racemic", "scalemic", "arbitrary"]


def _sha32(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def sorted_stereo_centers(
    stereo_centers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(stereo_centers, key=lambda item: int(item["atom_index"]))


def sorted_ez_bonds(ez_bonds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        ez_bonds,
        key=lambda item: (int(item["bond_atom_1"]), int(item["bond_atom_2"])),
    )


def parent_id_for(
    canonical_smiles_no_stereo: str,
    standardization_version: str = STANDARDIZATION_VERSION,
) -> str:
    return _sha32(f"PARENT|{canonical_smiles_no_stereo}|{standardization_version}")


def identity_id_for(
    input_type: InputType,
    input_string: str,
    standardization_version: str = STANDARDIZATION_VERSION,
) -> str:
    return _sha32(f"IDENT|{input_type}|{input_string}|{standardization_version}")


def stereo_hash_for(
    stereo_centers: list[dict[str, Any]],
    ez_bonds: list[dict[str, Any]],
) -> str:
    centers_json = json.dumps(
        sorted_stereo_centers(stereo_centers),
        separators=(",", ":"),
        sort_keys=True,
    )
    bonds_json = json.dumps(
        sorted_ez_bonds(ez_bonds),
        separators=(",", ":"),
        sort_keys=True,
    )
    return _sha32(f"STEREO|{centers_json}|{bonds_json}")


def stereoisomer_id_for(parent_id: str, stereo_hash: str) -> str:
    return _sha32(f"ISOMER|{parent_id}|{stereo_hash}")


def mixture_id_for(
    parent_id: str,
    mixture_type: MixtureType,
    component_stereoisomer_ids: list[str],
) -> str:
    components = ",".join(sorted(component_stereoisomer_ids))
    return _sha32(f"MIX|{parent_id}|{mixture_type}|{components}")


@dataclass(frozen=True)
class MoleculeParent:
    parent_id: str
    canonical_smiles_no_stereo: str
    formula: str
    charge: int
    inchikey_skeleton: str
    standardization_version: str = STANDARDIZATION_VERSION


@dataclass(frozen=True)
class MolecularIdentity:
    identity_id: str
    parent_id: str
    input_string: str
    input_type: InputType
    canonical_smiles: str
    isomeric_smiles: str | None
    inchi: str | None
    inchikey: str
    rdkit_mol_hash: str
    standardization_version: str = STANDARDIZATION_VERSION


@dataclass(frozen=True)
class Stereoisomer:
    stereoisomer_id: str
    parent_id: str
    identity_id: str | None
    isomeric_smiles: str
    stereo_hash: str
    stereo_centers_json: list[dict[str, Any]]
    ez_bonds_json: list[dict[str, Any]]
    stereo_defined_bool: bool
    stereo_ambiguity_level: StereoAmbiguityLevel
    chirality_class: ChiralityClass


@dataclass(frozen=True)
class StereoMixture:
    mixture_id: str
    parent_id: str
    mixture_type: MixtureType
    component_stereoisomer_ids: list[str]
    component_fractions: list[float]
    evidence_id: str | None = None
    uncertainty: float | None = None


@dataclass(frozen=True)
class StereoRelationship:
    from_stereoisomer_id: str
    to_stereoisomer_id: str
    relation_type: Literal[
        "enantiomer_of",
        "diastereomer_of",
        "racemizes_to",
        "inverts_to",
    ]
    evidence_id: str | None = None


@dataclass(frozen=True)
class DrugProductMapping:
    drug_product_id: str
    active_moiety_parent_id: str
    marketed_stereoisomer_id: str | None
    marketed_mixture_id: str | None
