"""
ChEMBL PK Data Downloader
==========================
Downloads pharmacokinetic activity data from ChEMBL.

ChEMBL is the world's largest open drug discovery database, maintained by
the European Bioinformatics Institute (EMBL-EBI). It contains millions of
bioactivity measurements for drug-like compounds.

What we extract:
- Cmax: Peak plasma concentration (how high does the drug level get?)
- AUC: Area under the curve (total drug exposure over time)
- T1/2: Half-life (how quickly does the drug leave the body?)
- CL: Clearance (how fast the body eliminates the drug)
- Vd/Vdss: Volume of distribution (how widely does the drug spread?)

Each record includes the molecule's SMILES string — the text representation
of its chemical structure. This is what our AI model will use as input.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# The 5 PK parameter types we want to predict
# These map to ChEMBL's standard_type field
PK_TYPES = {
    "Cmax": "cmax",       # Peak concentration
    "AUC": "auc",         # Area under curve
    "T1/2": "half_life",  # Half-life
    "CL": "clearance",    # Clearance
    "Vd": "vd",           # Volume of distribution
    "Vdss": "vd",         # Vd at steady state (maps to same target)
}


def download_chembl_pk_data(
    output_dir: str | Path = "data/raw",
    pk_types: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Download PK activity data from ChEMBL for all target PK parameters.

    How it works:
    1. For each PK type (Cmax, AUC, etc.), we query ChEMBL's activity API
    2. We request: SMILES, value, units, molecule ID, assay info
    3. We combine all results into one DataFrame
    4. Save to CSV

    Args:
        output_dir: Where to save the raw data
        pk_types: Dict mapping ChEMBL standard_type -> our column name.
                  Defaults to PK_TYPES.

    Returns:
        Combined DataFrame with all PK data
    """
    from chembl_webresource_client.new_client import new_client

    if pk_types is None:
        pk_types = PK_TYPES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    activity = new_client.activity
    all_data = []

    for chembl_type, our_name in pk_types.items():
        logger.info("Downloading %s data from ChEMBL...", chembl_type)

        # Query ChEMBL for this PK type
        # .only() limits the fields we fetch (faster download)
        acts = activity.filter(
            standard_type=chembl_type,
        ).only([
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_value",
            "standard_units",
            "standard_type",
            "standard_relation",
            "assay_chembl_id",
            "assay_type",
            "assay_organism",
            "target_organism",
            "pchembl_value",
            "data_validity_comment",
        ])

        # Collect all records for this type
        records = []
        for record in tqdm(acts, desc=f"  {chembl_type}", unit=" records"):
            records.append(record)

        logger.info("  %s: downloaded %d records", chembl_type, len(records))

        if records:
            df_type = pd.DataFrame(records)
            df_type["pk_param"] = our_name  # Our standardized name
            all_data.append(df_type)

            # Save individual type file
            type_path = output_dir / f"chembl_{our_name}.parquet"
            df_type.to_parquet(type_path, index=False)
            logger.info("  Saved %s to %s", chembl_type, type_path)

        # Small delay to be nice to the API
        time.sleep(1)

    if not all_data:
        logger.warning("No data downloaded!")
        return pd.DataFrame()

    # Combine all PK types into one DataFrame
    df_combined = pd.concat(all_data, ignore_index=True)
    logger.info(
        "Combined dataset: %d records, %d unique molecules",
        len(df_combined),
        df_combined["molecule_chembl_id"].nunique(),
    )

    # Save combined file
    combined_path = output_dir / "chembl_pk_all.parquet"
    df_combined.to_parquet(combined_path, index=False)
    logger.info("Saved combined data to %s", combined_path)

    return df_combined


def clean_pk_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and filter the raw ChEMBL PK data.

    Steps:
    1. Remove records without SMILES (can't use for ML without structure)
    2. Remove records without values (nothing to predict)
    3. Filter for human data only (our model predicts human PK)
    4. Remove records with data validity issues
    5. Keep only "=" relations (not ">", "<", "~")
    6. Convert values to float

    Args:
        df: Raw ChEMBL PK DataFrame

    Returns:
        Cleaned DataFrame
    """
    original_count = len(df)
    logger.info("Cleaning PK data: %d records", original_count)

    # Step 1: Must have SMILES
    df = df.dropna(subset=["canonical_smiles"])
    df = df[df["canonical_smiles"].str.len() > 0]
    logger.info("  After SMILES filter: %d (removed %d)", len(df), original_count - len(df))

    # Step 2: Must have a value
    df = df.dropna(subset=["standard_value"])
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value"])
    logger.info("  After value filter: %d", len(df))

    # Step 3: Filter for human organism
    # ChEMBL uses 'assay_organism' and 'target_organism'
    human_mask = (
        df["assay_organism"].astype(str).str.lower().str.contains("homo sapiens", na=False)
        | df["target_organism"].astype(str).str.lower().str.contains("homo sapiens", na=False)
    )
    # Also keep records where organism is not specified (often human PK studies)
    no_organism = (
        df["assay_organism"].isna() & df["target_organism"].isna()
    )
    df = df[human_mask | no_organism]
    logger.info("  After human filter: %d", len(df))

    # Step 4: Remove flagged data
    if "data_validity_comment" in df.columns:
        df = df[df["data_validity_comment"].isna()]
        logger.info("  After validity filter: %d", len(df))

    # Step 5: Keep only exact measurements ("=")
    if "standard_relation" in df.columns:
        exact_mask = df["standard_relation"].isna() | (df["standard_relation"] == "=") | (df["standard_relation"] == "'='")
        df = df[exact_mask]
        logger.info("  After relation filter: %d", len(df))

    # Step 6: Remove extreme outliers (> 3 standard deviations per PK type)
    cleaned_parts = []
    for pk_type in df["pk_param"].unique():
        subset = df[df["pk_param"] == pk_type].copy()
        values = subset["standard_value"]
        mean_val = values.mean()
        std_val = values.std()
        if std_val > 0:
            mask = (values - mean_val).abs() <= 3 * std_val
            subset = subset[mask]
        cleaned_parts.append(subset)
        logger.info(
            "  %s: kept %d / %d after outlier removal",
            pk_type, len(subset), len(df[df["pk_param"] == pk_type]),
        )

    df = pd.concat(cleaned_parts, ignore_index=True)
    logger.info("Final cleaned dataset: %d records", len(df))

    return df


def create_pk_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table showing stats per PK parameter.

    This helps us understand:
    - How much data we have for each parameter
    - The value ranges and distributions
    - How many unique drugs we have

    Args:
        df: Cleaned PK DataFrame

    Returns:
        Summary DataFrame
    """
    summary = df.groupby("pk_param").agg(
        n_records=("standard_value", "count"),
        n_molecules=("molecule_chembl_id", "nunique"),
        mean_value=("standard_value", "mean"),
        median_value=("standard_value", "median"),
        std_value=("standard_value", "std"),
        min_value=("standard_value", "min"),
        max_value=("standard_value", "max"),
        common_unit=("standard_units", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "N/A"),
    ).reset_index()

    return summary


# ── Main entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    print("=" * 60)
    print("NeoGen — ChEMBL PK Data Download")
    print("=" * 60)

    # Step 1: Download
    print("\nStep 1: Downloading PK data from ChEMBL...")
    print("This may take 5-15 minutes depending on your connection.\n")
    df_raw = download_chembl_pk_data(output_dir="data/raw")

    if df_raw.empty:
        print("No data downloaded. Check your internet connection.")
        exit(1)

    # Step 2: Clean
    print("\n" + "=" * 60)
    print("Step 2: Cleaning data...")
    print("=" * 60)
    df_clean = clean_pk_data(df_raw)

    # Save cleaned data
    df_clean.to_parquet("data/processed/chembl_pk_clean.parquet", index=False)
    print(f"\nSaved cleaned data: {len(df_clean)} records")

    # Step 3: Summary
    print("\n" + "=" * 60)
    print("Step 3: Data Summary")
    print("=" * 60)
    summary = create_pk_summary(df_clean)
    print(summary.to_string(index=False))

    print(f"\nTotal: {len(df_clean)} records across {df_clean['molecule_chembl_id'].nunique()} unique molecules")
    print("Done!")
