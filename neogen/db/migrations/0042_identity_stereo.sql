CREATE TABLE molecule_parent (
  parent_id TEXT PRIMARY KEY,
  canonical_smiles_no_stereo TEXT NOT NULL,
  formula TEXT NOT NULL,
  charge INTEGER NOT NULL,
  inchikey_skeleton TEXT NOT NULL UNIQUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  standardization_version TEXT NOT NULL
);

CREATE INDEX ix_parent_inchikey ON molecule_parent(inchikey_skeleton);

CREATE TABLE molecular_identity (
  identity_id TEXT PRIMARY KEY,
  parent_id TEXT NOT NULL REFERENCES molecule_parent(parent_id),
  input_string TEXT NOT NULL,
  input_type TEXT NOT NULL CHECK (input_type IN ('smiles','isomeric_smiles','inchi','sdf','mol','name','known_drug_id')),
  canonical_smiles TEXT NOT NULL,
  isomeric_smiles TEXT,
  inchi TEXT,
  inchikey TEXT NOT NULL,
  rdkit_mol_hash TEXT NOT NULL,
  standardization_version TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX ix_identity_inchikey ON molecular_identity(inchikey);

CREATE TABLE stereoisomer (
  stereoisomer_id TEXT PRIMARY KEY,
  parent_id TEXT NOT NULL REFERENCES molecule_parent(parent_id),
  identity_id TEXT REFERENCES molecular_identity(identity_id),
  isomeric_smiles TEXT NOT NULL,
  stereo_hash TEXT NOT NULL,
  stereo_centers_json JSONB NOT NULL,
  ez_bonds_json JSONB NOT NULL,
  stereo_defined_bool BOOLEAN NOT NULL,
  stereo_ambiguity_level TEXT NOT NULL CHECK (stereo_ambiguity_level IN ('fully_specified','partial','unknown')),
  chirality_class TEXT NOT NULL CHECK (chirality_class IN ('achiral','chiral_resolved','chiral_unresolved','racemic_intent')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX ux_stereo_parent_hash ON stereoisomer(parent_id, stereo_hash);

CREATE TABLE stereo_mixture (
  mixture_id TEXT PRIMARY KEY,
  parent_id TEXT NOT NULL REFERENCES molecule_parent(parent_id),
  mixture_type TEXT NOT NULL CHECK (mixture_type IN ('racemic','scalemic','arbitrary')),
  component_stereoisomer_ids TEXT[] NOT NULL,
  component_fractions NUMERIC[] NOT NULL,
  evidence_id TEXT,
  uncertainty NUMERIC,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE stereo_relationship (
  from_stereoisomer_id TEXT NOT NULL REFERENCES stereoisomer(stereoisomer_id),
  to_stereoisomer_id TEXT NOT NULL REFERENCES stereoisomer(stereoisomer_id),
  relation_type TEXT NOT NULL CHECK (relation_type IN ('enantiomer_of','diastereomer_of','racemizes_to','inverts_to')),
  evidence_id TEXT,
  PRIMARY KEY (from_stereoisomer_id, to_stereoisomer_id, relation_type)
);

CREATE TABLE drug_product_mapping (
  drug_product_id TEXT NOT NULL,
  active_moiety_parent_id TEXT NOT NULL REFERENCES molecule_parent(parent_id),
  marketed_stereoisomer_id TEXT REFERENCES stereoisomer(stereoisomer_id),
  marketed_mixture_id TEXT REFERENCES stereo_mixture(mixture_id),
  PRIMARY KEY (drug_product_id)
);
