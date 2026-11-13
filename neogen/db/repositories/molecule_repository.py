"""CRUD repository for Week 1 molecular identity records."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict

from neogen.chem.identity import MolecularIdentity, MoleculeParent, StereoMixture, Stereoisomer


class MoleculeRepository:
    def __init__(self, session: sqlite3.Connection):
        self.session = session

    def upsert_parent(self, parent: MoleculeParent) -> MoleculeParent:
        self.session.execute(
            """
            INSERT INTO molecule_parent VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(parent_id) DO UPDATE SET
              canonical_smiles_no_stereo=excluded.canonical_smiles_no_stereo,
              formula=excluded.formula,
              charge=excluded.charge,
              inchikey_skeleton=excluded.inchikey_skeleton,
              standardization_version=excluded.standardization_version
            """,
            (
                parent.parent_id,
                parent.canonical_smiles_no_stereo,
                parent.formula,
                parent.charge,
                parent.inchikey_skeleton,
                parent.standardization_version,
            ),
        )
        self.session.commit()
        return parent

    def upsert_identity(self, identity: MolecularIdentity) -> MolecularIdentity:
        self.session.execute(
            """
            INSERT INTO molecular_identity VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(identity_id) DO UPDATE SET
              parent_id=excluded.parent_id,
              input_string=excluded.input_string,
              input_type=excluded.input_type,
              canonical_smiles=excluded.canonical_smiles,
              isomeric_smiles=excluded.isomeric_smiles,
              inchi=excluded.inchi,
              inchikey=excluded.inchikey,
              rdkit_mol_hash=excluded.rdkit_mol_hash,
              standardization_version=excluded.standardization_version
            """,
            (
                identity.identity_id,
                identity.parent_id,
                identity.input_string,
                identity.input_type,
                identity.canonical_smiles,
                identity.isomeric_smiles,
                identity.inchi,
                identity.inchikey,
                identity.rdkit_mol_hash,
                identity.standardization_version,
            ),
        )
        self.session.commit()
        return identity

    def upsert_stereoisomer(self, isomer: Stereoisomer) -> Stereoisomer:
        self.session.execute(
            """
            INSERT INTO stereoisomer VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(stereoisomer_id) DO UPDATE SET
              parent_id=excluded.parent_id,
              identity_id=excluded.identity_id,
              isomeric_smiles=excluded.isomeric_smiles,
              stereo_hash=excluded.stereo_hash,
              stereo_centers_json=excluded.stereo_centers_json,
              ez_bonds_json=excluded.ez_bonds_json,
              stereo_defined_bool=excluded.stereo_defined_bool,
              stereo_ambiguity_level=excluded.stereo_ambiguity_level,
              chirality_class=excluded.chirality_class
            """,
            self._stereoisomer_params(isomer),
        )
        self.session.commit()
        return isomer

    def upsert_mixture(self, mixture: StereoMixture) -> StereoMixture:
        self.session.execute(
            """
            INSERT INTO stereo_mixture VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(mixture_id) DO UPDATE SET
              parent_id=excluded.parent_id,
              mixture_type=excluded.mixture_type,
              component_stereoisomer_ids=excluded.component_stereoisomer_ids,
              component_fractions=excluded.component_fractions,
              evidence_id=excluded.evidence_id,
              uncertainty=excluded.uncertainty
            """,
            (
                mixture.mixture_id,
                mixture.parent_id,
                mixture.mixture_type,
                json.dumps(mixture.component_stereoisomer_ids, sort_keys=True),
                json.dumps(mixture.component_fractions),
                mixture.evidence_id,
                mixture.uncertainty,
            ),
        )
        self.session.commit()
        return mixture

    def get_identity_by_id(self, identity_id: str) -> MolecularIdentity | None:
        row = self.session.execute(
            "SELECT * FROM molecular_identity WHERE identity_id = ?", (identity_id,)
        ).fetchone()
        return MolecularIdentity(**dict(row)) if row else None

    def get_stereoisomer_by_id(self, stereoisomer_id: str) -> Stereoisomer | None:
        row = self.session.execute(
            "SELECT * FROM stereoisomer WHERE stereoisomer_id = ?", (stereoisomer_id,)
        ).fetchone()
        return self._row_to_stereoisomer(row) if row else None

    def get_parent_by_id(self, parent_id: str) -> MoleculeParent | None:
        row = self.session.execute(
            "SELECT * FROM molecule_parent WHERE parent_id = ?", (parent_id,)
        ).fetchone()
        return MoleculeParent(**dict(row)) if row else None

    def list_stereoisomers_for_parent(self, parent_id: str) -> list[Stereoisomer]:
        rows = self.session.execute(
            "SELECT * FROM stereoisomer WHERE parent_id = ? ORDER BY stereoisomer_id",
            (parent_id,),
        ).fetchall()
        return [self._row_to_stereoisomer(row) for row in rows]

    def clear(self) -> None:
        for table in [
            "conformer",
            "conformer_set",
            "feature_model_compatibility",
            "feature_vector",
            "stereo_mixture",
            "stereoisomer",
            "molecular_identity",
            "molecule_parent",
        ]:
            self.session.execute(f"DELETE FROM {table}")
        self.session.commit()

    @staticmethod
    def _stereoisomer_params(isomer: Stereoisomer) -> tuple:
        data = asdict(isomer)
        return (
            data["stereoisomer_id"],
            data["parent_id"],
            data["identity_id"],
            data["isomeric_smiles"],
            data["stereo_hash"],
            json.dumps(data["stereo_centers_json"], sort_keys=True),
            json.dumps(data["ez_bonds_json"], sort_keys=True),
            1 if data["stereo_defined_bool"] else 0,
            data["stereo_ambiguity_level"],
            data["chirality_class"],
        )

    @staticmethod
    def _row_to_stereoisomer(row: sqlite3.Row) -> Stereoisomer:
        data = dict(row)
        data["stereo_centers_json"] = json.loads(data["stereo_centers_json"])
        data["ez_bonds_json"] = json.loads(data["ez_bonds_json"])
        data["stereo_defined_bool"] = bool(data["stereo_defined_bool"])
        return Stereoisomer(**data)
