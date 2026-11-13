"""Repository for feature vectors and model compatibility rows."""
from __future__ import annotations

import sqlite3

import numpy as np

from neogen.ml._features import FeatureVector, serialize_vector_float32_le
from neogen.ml.feature_registry import register_model_for_schema


class FeatureRepository:
    def __init__(self, session: sqlite3.Connection):
        self.session = session

    def get_by_input_schema(self, entity_id: str, schema_version: str) -> FeatureVector | None:
        row = self.session.execute(
            "SELECT * FROM feature_vector WHERE input_entity_id = ? AND schema_version = ?",
            (entity_id, schema_version),
        ).fetchone()
        return self._row_to_feature_vector(row) if row else None

    def get_by_id(self, feature_vector_id: str) -> FeatureVector | None:
        row = self.session.execute(
            "SELECT * FROM feature_vector WHERE feature_vector_id = ?",
            (feature_vector_id,),
        ).fetchone()
        return self._row_to_feature_vector(row) if row else None

    def upsert(self, feature_vector: FeatureVector, event_id: str) -> FeatureVector:
        self.session.execute(
            """
            INSERT INTO feature_vector VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(input_entity_id, schema_version) DO UPDATE SET
              feature_vector_id=excluded.feature_vector_id,
              input_entity_type=excluded.input_entity_type,
              mode=excluded.mode,
              vector_hash=excluded.vector_hash,
              feature_names_hash=excluded.feature_names_hash,
              dense_values=excluded.dense_values,
              created_by_event_id=excluded.created_by_event_id
            """,
            (
                feature_vector.feature_vector_id,
                feature_vector.input_entity_id,
                feature_vector.input_entity_type,
                feature_vector.schema_version,
                feature_vector.mode,
                feature_vector.vector_hash,
                feature_vector.feature_names_hash,
                serialize_vector_float32_le(feature_vector.values),
                event_id,
            ),
        )
        self.session.commit()
        return feature_vector

    def register_model_for_schema(self, model_id: str, schema_version: str) -> None:
        self.session.execute(
            "INSERT OR IGNORE INTO feature_model_compatibility VALUES (?, ?)",
            (model_id, schema_version),
        )
        self.session.commit()
        register_model_for_schema(model_id, schema_version)

    def load_model_registry(self) -> None:
        for row in self.session.execute("SELECT model_id, schema_version FROM feature_model_compatibility"):
            register_model_for_schema(row["model_id"], row["schema_version"])

    @staticmethod
    def _row_to_feature_vector(row: sqlite3.Row) -> FeatureVector:
        values = np.frombuffer(row["dense_values"], dtype="<f4").astype(np.float64)
        return FeatureVector(
            feature_vector_id=row["feature_vector_id"],
            input_entity_id=row["input_entity_id"],
            input_entity_type=row["input_entity_type"],
            schema_version=row["schema_version"],
            mode=row["mode"],
            vector_hash=row["vector_hash"],
            feature_names_hash=row["feature_names_hash"],
            values=values,
        )
