"""Small sqlite-backed session factory for Week 2 repositories."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Iterator

DEFAULT_SQLITE_URL = "sqlite:///./neogen_local.db"


def database_url() -> str:
    return os.getenv("NEOGEN_DB_URL", DEFAULT_SQLITE_URL)


def _sqlite_path(url: str) -> str:
    if url == "sqlite:///:memory:":
        return ":memory:"
    if url.startswith("sqlite:///"):
        return url.removeprefix("sqlite:///")
    if url.startswith("postgres"):
        raise RuntimeError(
            "Postgres NEOGEN_DB_URL requires the Week 4.5 DB layer; "
            "Week 2 local repositories use sqlite3."
        )
    return url


def get_connection(url: str | None = None) -> sqlite3.Connection:
    path = _sqlite_path(url or database_url())
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    initialize_schema(conn)
    return conn


def session_scope(url: str | None = None) -> Iterator[sqlite3.Connection]:
    conn = get_connection(url)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS molecule_parent (
          parent_id TEXT PRIMARY KEY,
          canonical_smiles_no_stereo TEXT NOT NULL,
          formula TEXT NOT NULL,
          charge INTEGER NOT NULL,
          inchikey_skeleton TEXT NOT NULL,
          standardization_version TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS molecular_identity (
          identity_id TEXT PRIMARY KEY,
          parent_id TEXT NOT NULL,
          input_string TEXT NOT NULL,
          input_type TEXT NOT NULL,
          canonical_smiles TEXT NOT NULL,
          isomeric_smiles TEXT,
          inchi TEXT,
          inchikey TEXT NOT NULL,
          rdkit_mol_hash TEXT NOT NULL,
          standardization_version TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS stereoisomer (
          stereoisomer_id TEXT PRIMARY KEY,
          parent_id TEXT NOT NULL,
          identity_id TEXT,
          isomeric_smiles TEXT NOT NULL,
          stereo_hash TEXT NOT NULL,
          stereo_centers_json TEXT NOT NULL,
          ez_bonds_json TEXT NOT NULL,
          stereo_defined_bool INTEGER NOT NULL,
          stereo_ambiguity_level TEXT NOT NULL,
          chirality_class TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ux_stereo_parent_hash
          ON stereoisomer(parent_id, stereo_hash);
        CREATE TABLE IF NOT EXISTS stereo_mixture (
          mixture_id TEXT PRIMARY KEY,
          parent_id TEXT NOT NULL,
          mixture_type TEXT NOT NULL,
          component_stereoisomer_ids TEXT NOT NULL,
          component_fractions TEXT NOT NULL,
          evidence_id TEXT,
          uncertainty REAL
        );
        CREATE TABLE IF NOT EXISTS feature_vector (
          feature_vector_id TEXT PRIMARY KEY,
          input_entity_id TEXT NOT NULL,
          input_entity_type TEXT NOT NULL,
          schema_version TEXT NOT NULL,
          mode TEXT NOT NULL,
          vector_hash TEXT NOT NULL,
          feature_names_hash TEXT NOT NULL,
          dense_values BLOB NOT NULL,
          created_by_event_id TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ux_fv_input_schema
          ON feature_vector(input_entity_id, schema_version);
        CREATE TABLE IF NOT EXISTS feature_model_compatibility (
          model_id TEXT NOT NULL,
          schema_version TEXT NOT NULL,
          PRIMARY KEY (model_id, schema_version)
        );
        CREATE TABLE IF NOT EXISTS conformer_set (
          conformer_set_id TEXT PRIMARY KEY,
          stereoisomer_id TEXT NOT NULL,
          method TEXT NOT NULL,
          seed INTEGER NOT NULL,
          rdkit_version TEXT NOT NULL,
          forcefield TEXT NOT NULL,
          forcefield_version TEXT NOT NULL,
          max_conformers INTEGER NOT NULL,
          prune_rmsd REAL NOT NULL,
          energy_window_kcal_mol REAL NOT NULL,
          config_hash TEXT NOT NULL,
          manifest_uri TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ux_confset_stereo_config
          ON conformer_set(stereoisomer_id, config_hash);
        CREATE TABLE IF NOT EXISTS conformer (
          conformer_id TEXT PRIMARY KEY,
          conformer_set_id TEXT NOT NULL,
          rank INTEGER NOT NULL,
          energy_kcal_mol REAL NOT NULL,
          rmsd_cluster_id INTEGER,
          coordinates_uri TEXT NOT NULL,
          molblock_uri TEXT,
          coordinate_hash TEXT NOT NULL,
          UNIQUE (conformer_set_id, rank)
        );
        """
    )
