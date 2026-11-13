"""
Feature Engineering for NeoGen
================================
Converts raw molecular data into features that ML models can understand.

Two types of features:
1. Morgan Fingerprints — binary vectors representing molecular substructures
   (think of it as a barcode for the molecule's shape)
2. Molecular Descriptors — numeric properties like molecular weight, LogP,
   polar surface area, etc.

These features are what the model "sees" instead of raw SMILES strings.
The Random Forest and XGBoost models use these directly.
The GNN (later) will learn its own features from the molecular graph.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from tqdm import tqdm

logger = logging.getLogger(__name__)

# The molecular descriptors we compute
# These are physicochemical properties that influence PK behavior
DESCRIPTOR_LIST = [
    ("MolWt", Descriptors.MolWt),           # Molecular weight — heavier drugs clear slower
    ("LogP", Descriptors.MolLogP),           # Lipophilicity — fat-soluble drugs have higher Vd
    ("TPSA", Descriptors.TPSA),             # Polar surface area — affects absorption
    ("HBD", Descriptors.NumHDonors),         # H-bond donors — affects membrane permeability
    ("HBA", Descriptors.NumHAcceptors),      # H-bond acceptors — affects solubility
    ("RotBonds", Descriptors.NumRotatableBonds),  # Flexibility — affects binding
    ("Rings", Descriptors.RingCount),        # Ring count — affects rigidity
    ("AromaticRings", Descriptors.NumAromaticRings),
    ("HeavyAtoms", Descriptors.HeavyAtomCount),  # Non-hydrogen atoms
    ("FractionCSP3", Descriptors.FractionCSP3),   # 3D character
]


def standardize_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize units for each PK parameter to a single consistent unit.

    Why: Our clearance data has two main units:
    - mL.min-1.kg-1 (53%) — per kilogram body weight
    - mL.min-1.g-1 (41%) — per gram (used in in-vitro assays)

    We'll keep only the dominant clinical unit for each parameter.
    This ensures the model learns consistent relationships.
    """
    original_count = len(df)

    # Define the target unit for each PK param
    target_units = {
        "cmax": ["nM"],
        "auc": ["ng.hr.mL-1"],
        "half_life": ["hr"],
        "clearance": ["mL.min-1.kg-1"],
        "vd": ["L.kg-1"],
    }

    filtered_parts = []
    for param, units in target_units.items():
        subset = df[df["pk_param"] == param]
        subset = subset[subset["standard_units"].isin(units)]
        filtered_parts.append(subset)
        logger.info(
            "  %s: kept %d / %d (unit: %s)",
            param, len(subset), len(df[df["pk_param"] == param]), units,
        )

    df_std = pd.concat(filtered_parts, ignore_index=True)

    # Remove negative and zero values (physically impossible for PK)
    df_std = df_std[df_std["standard_value"] > 0]

    logger.info(
        "Unit standardization: %d -> %d records",
        original_count, len(df_std),
    )
    return df_std


def compute_morgan_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray | None:
    """
    Compute Morgan (circular) fingerprint for a molecule.

    What it does: Looks at each atom and its neighbors within a radius.
    Creates a binary vector where each bit represents a specific
    substructure pattern. Two similar molecules will share many bits.

    Args:
        smiles: SMILES string of the molecule
        radius: How far from each atom to look (2 = ECFP4)
        n_bits: Length of the fingerprint vector

    Returns:
        numpy array of 0s and 1s, or None if SMILES is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def compute_descriptors(smiles: str) -> dict | None:
    """
    Compute molecular descriptors for a molecule.

    These are interpretable numeric properties:
    - Molecular weight, LogP, polar surface area, etc.
    - Used alongside fingerprints to give the model more context.

    Returns:
        Dict of descriptor name -> value, or None if SMILES is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {name: func(mol) for name, func in DESCRIPTOR_LIST}


def build_feature_matrix(
    df: pd.DataFrame,
    morgan_radius: int = 2,
    morgan_bits: int = 2048,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Build the complete feature matrix for all molecules.

    This is the core function that turns SMILES strings into numbers
    that ML models can process.

    Steps:
    1. Get unique SMILES (avoid computing features for duplicates)
    2. Compute Morgan fingerprint for each molecule
    3. Compute molecular descriptors for each molecule
    4. Combine into one feature matrix

    Args:
        df: DataFrame with 'canonical_smiles' column
        morgan_radius: Fingerprint radius (2 = ECFP4, standard)
        morgan_bits: Fingerprint length

    Returns:
        Tuple of (metadata_df, fingerprint_array, descriptor_df)
    """
    # Get unique SMILES to avoid duplicate computation
    unique_smiles = df["canonical_smiles"].unique()
    logger.info("Computing features for %d unique molecules...", len(unique_smiles))

    fingerprints = []
    descriptors = []
    valid_smiles = []
    failed = 0

    for smiles in tqdm(unique_smiles, desc="Computing features"):
        # Morgan fingerprint
        fp = compute_morgan_fingerprint(smiles, morgan_radius, morgan_bits)
        if fp is None:
            failed += 1
            continue

        # Molecular descriptors
        desc = compute_descriptors(smiles)
        if desc is None:
            failed += 1
            continue

        fingerprints.append(fp)
        descriptors.append(desc)
        valid_smiles.append(smiles)

    logger.info(
        "Computed features: %d successful, %d failed",
        len(valid_smiles), failed,
    )

    # Convert to arrays/DataFrames
    fp_array = np.array(fingerprints)  # Shape: (n_molecules, morgan_bits)
    desc_df = pd.DataFrame(descriptors)  # Shape: (n_molecules, n_descriptors)

    # Create a lookup table: SMILES -> feature index
    smiles_to_idx = {s: i for i, s in enumerate(valid_smiles)}

    # Map features back to all records in the original DataFrame
    # (many records can share the same molecule)
    record_fp_indices = []
    valid_records = []

    for idx, row in df.iterrows():
        smiles = row["canonical_smiles"]
        if smiles in smiles_to_idx:
            record_fp_indices.append(smiles_to_idx[smiles])
            valid_records.append(idx)

    # Build the final feature matrix aligned with records
    record_fps = fp_array[record_fp_indices]
    record_descs = desc_df.iloc[record_fp_indices].reset_index(drop=True)
    record_df = df.loc[valid_records].reset_index(drop=True)

    logger.info(
        "Final feature matrix: %d records, %d FP bits + %d descriptors = %d total features",
        len(record_df), morgan_bits, len(desc_df.columns),
        morgan_bits + len(desc_df.columns),
    )

    return record_df, record_fps, record_descs


def create_ml_dataset(
    df: pd.DataFrame,
    fp_array: np.ndarray,
    desc_df: pd.DataFrame,
    output_dir: str = "data/processed",
) -> None:
    """
    Create the final ML-ready dataset by combining fingerprints + descriptors.

    Saves:
    - features.npz: Combined feature matrix (fingerprints + descriptors)
    - targets.parquet: PK parameter values + metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine fingerprints and descriptors into one matrix
    desc_array = desc_df.values.astype(np.float32)
    combined_features = np.hstack([fp_array, desc_array])

    # Save features
    np.savez_compressed(
        output_dir / "features.npz",
        features=combined_features,
        fingerprints=fp_array,
        descriptors=desc_array,
    )

    # Save targets (the PK values we want to predict)
    targets = df[["molecule_chembl_id", "canonical_smiles", "pk_param",
                   "standard_value", "standard_units"]].copy()

    # Add log-transformed values (better for ML — see distribution plots)
    targets["log_value"] = np.log10(targets["standard_value"].clip(lower=1e-10))

    targets.to_parquet(output_dir / "targets.parquet", index=False)

    # Save descriptor column names for later interpretation
    desc_names = list(desc_df.columns)
    with open(output_dir / "feature_names.txt", "w") as f:
        for name in desc_names:
            f.write(f"{name}\n")

    logger.info(
        "Saved ML dataset: features=%s, targets=%d records",
        combined_features.shape, len(targets),
    )
    print(f"\nML Dataset saved to {output_dir}/")
    print(f"  Features shape: {combined_features.shape}")
    print(f"    - Morgan FP:   {fp_array.shape[1]} bits")
    print(f"    - Descriptors:  {desc_array.shape[1]} values")
    print(f"  Targets: {len(targets)} records")
    print(f"  Descriptor names: {desc_names}")


# ── Main entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    # Load validated data
    print("=" * 60)
    print("NeoGen — Feature Engineering")
    print("=" * 60)

    df = pd.read_parquet("data/processed/chembl_pk_validated.parquet")
    print(f"\nLoaded {len(df)} validated records")

    # Step 1: Standardize units
    print("\nStep 1: Standardizing units...")
    df_std = standardize_units(df)
    print(f"After unit standardization: {len(df_std)} records")

    # Step 2: Build feature matrix
    print("\nStep 2: Computing molecular features...")
    print("This computes Morgan fingerprints + 10 molecular descriptors")
    print("for each unique molecule. May take 1-2 minutes.\n")

    record_df, fp_array, desc_df = build_feature_matrix(df_std)

    # Step 3: Save ML-ready dataset
    print("\nStep 3: Saving ML-ready dataset...")
    create_ml_dataset(record_df, fp_array, desc_df)

    # Print descriptor statistics
    print("\n" + "=" * 60)
    print("DESCRIPTOR STATISTICS")
    print("=" * 60)
    print(desc_df.describe().round(2).to_string())

    print("\nFeature engineering complete!")
