CREATE TABLE conformer_set (
  conformer_set_id TEXT PRIMARY KEY,
  stereoisomer_id TEXT NOT NULL REFERENCES stereoisomer(stereoisomer_id),
  method TEXT NOT NULL CHECK (method IN ('etkdgv3','etkdgv2','distance_geometry')),
  seed INTEGER NOT NULL,
  rdkit_version TEXT NOT NULL,
  forcefield TEXT NOT NULL CHECK (forcefield IN ('mmff94s','mmff94','uff','none')),
  forcefield_version TEXT NOT NULL,
  max_conformers INTEGER NOT NULL,
  prune_rmsd NUMERIC NOT NULL,
  energy_window_kcal_mol NUMERIC NOT NULL,
  config_hash TEXT NOT NULL,
  manifest_uri TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX ux_confset_stereo_config ON conformer_set(stereoisomer_id, config_hash);

CREATE TABLE conformer (
  conformer_id TEXT PRIMARY KEY,
  conformer_set_id TEXT NOT NULL REFERENCES conformer_set(conformer_set_id),
  rank INTEGER NOT NULL,
  energy_kcal_mol NUMERIC NOT NULL,
  rmsd_cluster_id INTEGER,
  coordinates_uri TEXT NOT NULL,
  molblock_uri TEXT,
  coordinate_hash TEXT NOT NULL,
  UNIQUE (conformer_set_id, rank)
);
