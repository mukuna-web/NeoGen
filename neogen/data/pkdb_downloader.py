"""
PK-DB Data Downloader
=====================
Downloads pharmacokinetic data from PK-DB (pk-db.com).

PK-DB is a public database of pharmacokinetic parameters extracted from
published clinical studies. We use it to get real-world measurements of
how drugs behave in the human body.

What this script does:
1. Queries the PK-DB REST API
2. Downloads PK output data (Cmax, AUC, half-life, clearance, Vd)
3. Filters for HUMAN data only
4. Saves raw data to data/raw/
"""
from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# PK-DB API base URL
BASE_URL = "https://pk-db.com/api/v1"

# PK parameters we care about (these are what our model will predict)
TARGET_PK_PARAMS = [
    "cmax",          # Peak plasma concentration (mg/L)
    "auc",           # Area under the curve - total drug exposure
    "auc_inf",       # AUC extrapolated to infinity
    "half_life",     # Elimination half-life (hours)
    "t_half",        # Alternative name for half-life
    "clearance",     # How fast the body removes the drug (L/hr)
    "cl",            # Alternative name for clearance
    "vd",            # Volume of distribution (L/kg)
    "vd_ss",         # Vd at steady state
    "tmax",          # Time to reach peak concentration
]


def get_pkdb_statistics() -> dict:
    """
    Get overview statistics from PK-DB.

    Returns a dict with counts of studies, substances, outputs, etc.
    This is useful to understand how much data is available.
    """
    url = f"{BASE_URL}/statistics/"
    logger.info("Fetching PK-DB statistics from %s", url)

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def download_pkdb_outputs(
    output_dir: str | Path = "data/raw",
    page_size: int = 100,
    max_pages: int | None = None,
) -> pd.DataFrame:
    """
    Download all PK output data from PK-DB via the REST API.

    How the API works:
    - We call /outputs/ endpoint with pagination
    - Each page returns up to `page_size` records
    - We keep fetching until there are no more pages

    Args:
        output_dir: Directory to save raw data
        page_size: Number of records per API call
        max_pages: Limit pages for testing (None = download all)

    Returns:
        DataFrame with all PK output records
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    page = 1
    total_count = None

    logger.info("Starting PK-DB output download...")

    while True:
        # Build the API URL with pagination
        url = f"{BASE_URL}/outputs/"
        params = {
            "page": page,
            "page_size": page_size,
            "format": "json",
        }

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.error("API request failed on page %d: %s", page, e)
            break

        # First page: get total count
        if total_count is None:
            total_count = data.get("count", 0)
            total_pages = (total_count + page_size - 1) // page_size
            logger.info(
                "Total records: %d, Pages: %d (page_size=%d)",
                total_count, total_pages, page_size,
            )

        # Extract results from this page
        results = data.get("results", [])
        if not results:
            logger.info("No more results. Download complete.")
            break

        all_records.extend(results)
        logger.info(
            "Page %d: fetched %d records (total so far: %d / %d)",
            page, len(results), len(all_records), total_count,
        )

        # Check if we've reached our page limit
        if max_pages and page >= max_pages:
            logger.info("Reached max_pages limit (%d). Stopping.", max_pages)
            break

        # Check if there's a next page
        if not data.get("next"):
            break

        page += 1

    if not all_records:
        logger.warning("No records downloaded!")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.json_normalize(all_records)
    logger.info("Downloaded %d total records with %d columns", len(df), len(df.columns))

    # Save raw data
    raw_path = output_dir / "pkdb_outputs_raw.json"
    with open(raw_path, "w") as f:
        json.dump(all_records, f, indent=2)
    logger.info("Saved raw JSON to %s", raw_path)

    return df


def download_pkdb_zip(
    output_dir: str | Path = "data/raw",
    search: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Download PK-DB data as a ZIP archive containing CSV files.

    This is an alternative approach using the filter + download endpoint,
    which gives us CSVs directly.

    Args:
        output_dir: Directory to save downloaded files
        search: Optional search term to filter substances

    Returns:
        Dict mapping filename -> DataFrame
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create a filter to get a download UUID
    filter_url = f"{BASE_URL}/filter/"
    params = {"download": "true"}
    if search:
        params["search"] = search

    logger.info("Creating PK-DB filter for download...")
    response = requests.get(filter_url, params=params, timeout=120)
    response.raise_for_status()

    # The response should contain a ZIP file
    if "application/zip" in response.headers.get("Content-Type", ""):
        z = zipfile.ZipFile(io.BytesIO(response.content))
        dfs = {}
        for name in z.namelist():
            if name.endswith(".csv"):
                df = pd.read_csv(z.open(name))
                dfs[name] = df
                save_path = output_dir / name
                df.to_csv(save_path, index=False)
                logger.info("Extracted %s: %d rows, %d cols", name, len(df), len(df.columns))
        return dfs

    # If not a ZIP, save the JSON response
    data = response.json()
    logger.info("Filter response (not ZIP): %s", list(data.keys()) if isinstance(data, dict) else type(data))
    return {}


def filter_human_pk_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter PK data to keep only HUMAN pharmacokinetic measurements.

    Why? Our model predicts drug behavior in HUMANS, not mice or rats.
    Animal PK data follows different patterns (different metabolism,
    different enzymes), so mixing species would confuse the model.

    Args:
        df: Raw PK-DB output DataFrame

    Returns:
        Filtered DataFrame with human data only
    """
    if df.empty:
        return df

    # Look for species/organism columns
    species_cols = [c for c in df.columns if any(
        term in c.lower() for term in ["species", "organism", "subject"]
    )]
    logger.info("Found species-related columns: %s", species_cols)

    original_count = len(df)

    # Try filtering by organism/species column
    for col in species_cols:
        unique_vals = df[col].dropna().unique()
        logger.info("Column '%s' unique values: %s", col, unique_vals[:20])

        human_terms = ["human", "homo sapiens", "homo_sapiens", "man", "woman", "h. sapiens"]
        mask = df[col].astype(str).str.lower().isin(human_terms)
        if mask.any():
            df = df[mask]
            logger.info(
                "Filtered by '%s': %d -> %d records (kept %d human)",
                col, original_count, len(df), mask.sum(),
            )
            return df

    logger.warning(
        "Could not find species column to filter. "
        "Keeping all %d records. Columns: %s",
        len(df), list(df.columns)[:20],
    )
    return df


def extract_pk_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and standardize the 5 key PK parameters we need.

    From the raw PK-DB data, we extract:
    - Cmax: Peak plasma concentration
    - AUC: Area under the concentration-time curve
    - half_life: Time for drug concentration to halve
    - clearance: Rate the body eliminates the drug
    - vd: Apparent volume the drug distributes into

    Args:
        df: Filtered PK-DB DataFrame

    Returns:
        DataFrame with standardized PK parameter columns
    """
    if df.empty:
        return df

    logger.info("Extracting PK parameters from %d records...", len(df))
    logger.info("Available columns: %s", list(df.columns)[:30])

    # Look for measurement type / parameter name column
    param_cols = [c for c in df.columns if any(
        term in c.lower() for term in ["measurement_type", "pktype", "parameter", "output_type"]
    )]
    logger.info("Parameter type columns found: %s", param_cols)

    # Look for value columns
    value_cols = [c for c in df.columns if any(
        term in c.lower() for term in ["value", "mean", "median"]
    )]
    logger.info("Value columns found: %s", value_cols)

    # Look for substance/drug columns
    drug_cols = [c for c in df.columns if any(
        term in c.lower() for term in ["substance", "drug", "compound", "intervention"]
    )]
    logger.info("Drug/substance columns found: %s", drug_cols)

    return df


# ── Main entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    # Step 1: Check what's available
    print("\n" + "=" * 60)
    print("STEP 1: Checking PK-DB statistics")
    print("=" * 60)
    try:
        stats = get_pkdb_statistics()
        print(f"\nPK-DB Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Could not fetch statistics: {e}")

    # Step 2: Download outputs (start with a small batch to test)
    print("\n" + "=" * 60)
    print("STEP 2: Downloading PK outputs (first 3 pages as test)")
    print("=" * 60)
    df = download_pkdb_outputs(
        output_dir="data/raw",
        page_size=100,
        max_pages=3,  # Start small to test
    )

    if not df.empty:
        print(f"\nDownloaded {len(df)} records")
        print(f"Columns: {list(df.columns)[:15]}...")
        print(f"\nFirst few rows:")
        print(df.head())

        # Step 3: Explore the data structure
        print("\n" + "=" * 60)
        print("STEP 3: Exploring data structure")
        print("=" * 60)
        for col in df.columns:
            non_null = df[col].notna().sum()
            if non_null > 0:
                print(f"  {col}: {non_null} non-null, sample: {df[col].dropna().iloc[0] if non_null > 0 else 'N/A'}")
