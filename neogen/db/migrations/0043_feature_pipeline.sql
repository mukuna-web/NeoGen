CREATE TABLE feature_schema (
  schema_version TEXT PRIMARY KEY,
  mode TEXT NOT NULL CHECK (mode IN ('legacy_2d','stereo_2d','conformer_3d')),
  feature_names TEXT[] NOT NULL,
  dimensions INTEGER NOT NULL,
  required_inputs JSONB NOT NULL,
  compatible_model_ids TEXT[] NOT NULL DEFAULT '{}',
  registered_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE feature_vector (
  feature_vector_id TEXT PRIMARY KEY,
  input_entity_id TEXT NOT NULL,
  input_entity_type TEXT NOT NULL CHECK (input_entity_type IN ('stereoisomer','conformer_set')),
  schema_version TEXT NOT NULL REFERENCES feature_schema(schema_version),
  mode TEXT NOT NULL,
  vector_hash TEXT NOT NULL,
  feature_names_hash TEXT NOT NULL,
  dense_values_uri TEXT,
  sparse_values_uri TEXT,
  created_by_event_id TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX ux_fv_input_schema ON feature_vector(input_entity_id, schema_version);

CREATE TABLE feature_model_compatibility (
  model_id TEXT NOT NULL,
  schema_version TEXT NOT NULL REFERENCES feature_schema(schema_version),
  PRIMARY KEY (model_id, schema_version)
);
