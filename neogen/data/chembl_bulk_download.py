"""
ChEMBL Bulk PK Data Download (Fast Method)
===========================================
Instead of hitting the API 20 records at a time, we:
1. Download ChEMBL's pre-built SQLite database (~4 GB)
2. Query it locally for all PK data in seconds

This is MUCH faster than the web API approach.

Alternatively, we use ChEMBL's FTP to get pre-filtered activity data.
"""
from __future__ import annotations

import logging
import sqlite3
import subprocess
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ChEMBL SQLite download URL (latest version)
CHEMBL_SQLITE_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_35_sqlite.tar.gz"


def download_chembl_sqlite(output_dir: str | Path = "data/raw") -> Path:
    """
    Download the ChEMBL SQLite database.

    This is a ~4 GB download that contains ALL of ChEMBL.
    We only need to do this once.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tar_path = output_dir / "chembl_sqlite.tar.gz"
    db_path = output_dir / "chembl_35" / "chembl_35_sqlite" / "chembl_35.db"

    if db_path.exists():
        logger.info("ChEMBL SQLite already exists at %s", db_path)
        return db_path

    logger.info("Downloading ChEMBL SQLite database (~4 GB)...")
    logger.info("This is a one-time download. It will take 5-20 minutes.")

    # Use curl for reliable download with progress
    subprocess.run(
        ["curl", "-L", "-o", str(tar_path), "--progress-bar", CHEMBL_SQLITE_URL],
        check=True,
    )

    logger.info("Extracting archive...")
    subprocess.run(
        ["tar", "-xzf", str(tar_path), "-C", str(output_dir)],
        check=True,
    )

    # Clean up tar file to save space
    tar_path.unlink()
    logger.info("ChEMBL SQLite ready at %s", db_path)
    return db_path


def query_pk_data_from_sqlite(db_path: str | Path) -> pd.DataFrame:
    """
    Query all PK-relevant activity data from ChEMBL SQLite.

    This runs locally — no API rate limits, no network latency.
    A query that would take hours via API takes seconds here.
    """
    db_path = Path(db_path)
    logger.info("Querying PK data from %s", db_path)

    conn = sqlite3.connect(str(db_path))

    # This SQL query extracts all PK parameter measurements
    # with their SMILES strings and standardized values
    query = """
    SELECT
        act.molregno,
        md.chembl_id AS molecule_chembl_id,
        cs.canonical_smiles,
        act.standard_type,
        act.standard_value,
        act.standard_units,
        act.standard_relation,
        act.pchembl_value,
        act.data_validity_comment,
        ass.chembl_id AS assay_chembl_id,
        ass.assay_type,
        ass.assay_organism,
        td.organism AS target_organism
    FROM activities act
    JOIN assays ass ON act.assay_id = ass.assay_id
    JOIN molecule_dictionary md ON act.molregno = md.molregno
    LEFT JOIN compound_structures cs ON act.molregno = cs.molregno
    LEFT JOIN target_dictionary td ON ass.tid = td.tid
    WHERE act.standard_type IN ('Cmax', 'AUC', 'T1/2', 'CL', 'Vd', 'Vdss', 'Tmax')
        AND act.standard_value IS NOT NULL
        AND cs.canonical_smiles IS NOT NULL
    """

    logger.info("Executing query...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    logger.info("Query returned %d records", len(df))
    return df


def download_pk_via_api_fast(
    output_dir: str | Path = "data/raw",
    page_size: int = 1000,
) -> pd.DataFrame:
    """
    Faster API download using larger page sizes and requests library directly.

    The chembl_webresource_client fetches 20 records at a time.
    By hitting the API directly with page_size=1000, we're 50x faster.
    """
    import requests

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    pk_types = ["Cmax", "AUC", "T1/2", "CL", "Vd", "Vdss"]

    all_data = []

    for pk_type in pk_types:
        logger.info("Downloading %s...", pk_type)

        offset = 0
        type_records = []
        total = None

        while True:
            params = {
                "standard_type": pk_type,
                "limit": page_size,
                "offset": offset,
                "format": "json",
            }

            try:
                resp = requests.get(base_url, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error("Request failed at offset %d: %s", offset, e)
                time.sleep(5)
                continue

            if total is None:
                total = data.get("page_meta", {}).get("total_count", 0)
                logger.info("  %s: %d total records", pk_type, total)

            molecules = data.get("activities", [])
            if not molecules:
                break

            for mol in molecules:
                type_records.append({
                    "molecule_chembl_id": mol.get("molecule_chembl_id"),
                    "canonical_smiles": mol.get("canonical_smiles"),
                    "standard_type": mol.get("standard_type"),
                    "standard_value": mol.get("standard_value"),
                    "standard_units": mol.get("standard_units"),
                    "standard_relation": mol.get("standard_relation"),
                    "assay_chembl_id": mol.get("assay_chembl_id"),
                    "assay_type": mol.get("assay_type"),
                    "assay_organism": mol.get("assay_organism"),
                    "target_organism": mol.get("target_organism"),
                    "pchembl_value": mol.get("pchembl_value"),
                    "data_validity_comment": mol.get("data_validity_comment"),
                })

            offset += page_size
            logger.info(
                "  %s: %d / %d (%.1f%%)",
                pk_type, min(offset, total), total, min(offset / total * 100, 100),
            )

            if offset >= total:
                break

        # Map to our standardized name
        pk_name_map = {
            "Cmax": "cmax", "AUC": "auc", "T1/2": "half_life",
            "CL": "clearance", "Vd": "vd", "Vdss": "vd",
        }

        for rec in type_records:
            rec["pk_param"] = pk_name_map.get(pk_type, pk_type.lower())

        all_data.extend(type_records)
        logger.info("  %s: %d records collected", pk_type, len(type_records))

        # Save intermediate checkpoint per type
        if type_records:
            df_type = pd.DataFrame(type_records)
            type_path = output_dir / f"chembl_{pk_name_map.get(pk_type, pk_type.lower())}.parquet"
            df_type.to_parquet(type_path, index=False)
            logger.info("  Saved checkpoint: %s", type_path)

    # Combine all
    df = pd.DataFrame(all_data)
    combined_path = output_dir / "chembl_pk_all.parquet"
    df.to_parquet(combined_path, index=False)
    logger.info("Saved combined data: %d records to %s", len(df), combined_path)

    return df


# ── Main entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    print("=" * 60)
    print("NeoGen — ChEMBL PK Data Bulk Download (Fast)")
    print("=" * 60)
    print("\nUsing direct API with 1000 records/page (50x faster)\n")

    df = download_pk_via_api_fast(output_dir="data/raw", page_size=1000)

    if not df.empty:
        print(f"\nDownloaded {len(df)} total records")
        print(f"Unique molecules: {df['molecule_chembl_id'].nunique()}")
        print(f"\nPer PK type:")
        print(df["pk_param"].value_counts().to_string())

        # Quick clean
        from neogen.data.chembl_downloader import clean_pk_data, create_pk_summary
        df_clean = clean_pk_data(df)
        df_clean.to_parquet("data/processed/chembl_pk_clean.parquet", index=False)

        print(f"\nAfter cleaning: {len(df_clean)} records")
        summary = create_pk_summary(df_clean)
        print("\nSummary:")
        print(summary.to_string(index=False))
