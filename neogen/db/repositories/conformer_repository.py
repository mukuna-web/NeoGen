"""SQLite repository for conformer sets and conformer records."""
from __future__ import annotations

import sqlite3
from dataclasses import asdict

from neogen.chem.conformers import ConformerRecord, ConformerSet


class ConformerRepository:
    def __init__(self, session: sqlite3.Connection):
        self.session = session

    def get_set_by_id(self, conformer_set_id: str) -> ConformerSet | None:
        row = self.session.execute(
            "SELECT * FROM conformer_set WHERE conformer_set_id = ?",
            (conformer_set_id,),
        ).fetchone()
        return self._row_to_set(row) if row else None

    def get_set_by_stereo_config(self, stereoisomer_id: str, config_hash: str) -> ConformerSet | None:
        row = self.session.execute(
            "SELECT * FROM conformer_set WHERE stereoisomer_id = ? AND config_hash = ?",
            (stereoisomer_id, config_hash),
        ).fetchone()
        return self._row_to_set(row) if row else None

    def get_conformer_by_id(self, conformer_id: str) -> ConformerRecord | None:
        row = self.session.execute(
            "SELECT * FROM conformer WHERE conformer_id = ?",
            (conformer_id,),
        ).fetchone()
        return ConformerRecord(**dict(row)) if row else None

    def upsert_set(self, conformer_set: ConformerSet) -> ConformerSet:
        self.session.execute(
            """
            INSERT INTO conformer_set VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conformer_set_id) DO UPDATE SET
              stereoisomer_id=excluded.stereoisomer_id,
              method=excluded.method,
              seed=excluded.seed,
              rdkit_version=excluded.rdkit_version,
              forcefield=excluded.forcefield,
              forcefield_version=excluded.forcefield_version,
              max_conformers=excluded.max_conformers,
              prune_rmsd=excluded.prune_rmsd,
              energy_window_kcal_mol=excluded.energy_window_kcal_mol,
              config_hash=excluded.config_hash,
              manifest_uri=excluded.manifest_uri
            """,
            (
                conformer_set.conformer_set_id,
                conformer_set.stereoisomer_id,
                conformer_set.method,
                conformer_set.seed,
                conformer_set.rdkit_version,
                conformer_set.forcefield,
                conformer_set.forcefield_version,
                conformer_set.max_conformers,
                conformer_set.prune_rmsd,
                conformer_set.energy_window_kcal_mol,
                conformer_set.config_hash,
                conformer_set.manifest_uri,
            ),
        )
        for conformer in conformer_set.conformers:
            self.session.execute(
                """
                INSERT INTO conformer VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conformer_id) DO UPDATE SET
                  conformer_set_id=excluded.conformer_set_id,
                  rank=excluded.rank,
                  energy_kcal_mol=excluded.energy_kcal_mol,
                  rmsd_cluster_id=excluded.rmsd_cluster_id,
                  coordinates_uri=excluded.coordinates_uri,
                  molblock_uri=excluded.molblock_uri,
                  coordinate_hash=excluded.coordinate_hash
                """,
                (
                    conformer.conformer_id,
                    conformer.conformer_set_id,
                    conformer.rank,
                    conformer.energy_kcal_mol,
                    conformer.rmsd_cluster_id,
                    conformer.coordinates_uri,
                    conformer.molblock_uri,
                    conformer.coordinate_hash,
                ),
            )
        self.session.commit()
        return conformer_set

    def clear(self) -> None:
        self.session.execute("DELETE FROM conformer")
        self.session.execute("DELETE FROM conformer_set")
        self.session.commit()

    def _row_to_set(self, row: sqlite3.Row) -> ConformerSet:
        conformer_rows = self.session.execute(
            "SELECT * FROM conformer WHERE conformer_set_id = ? ORDER BY rank",
            (row["conformer_set_id"],),
        ).fetchall()
        conformers = tuple(ConformerRecord(**dict(item)) for item in conformer_rows)
        data = dict(row)
        data["conformers"] = conformers
        data["cache_hit"] = True
        data["warnings"] = ()
        return ConformerSet(**data)
