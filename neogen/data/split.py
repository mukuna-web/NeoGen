"""
Train/Validation/Test Split
=============================
Splits data into 80/10/10 with NO DATA LEAKAGE.

Key principle: We split by MOLECULE, not by record.
If aspirin appears in training, ALL aspirin records go to training.
This prevents the model from "memorizing" specific molecules.

Why this matters: If the same molecule appears in both train and test,
the model's test accuracy will be artificially inflated — it already
"saw" that molecule during training. This would give us a false sense
of how well the model works on truly new molecules.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def create_molecule_splits(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_seed: int = 42,
    output_dir: str = "data/splits",
) -> dict[str, pd.DataFrame]:
    """
    Split data by molecule ID to prevent data leakage.

    Steps:
    1. Get unique molecule IDs
    2. Split molecules into train/val/test
    3. Assign all records for each molecule to the same split
    4. Verify no overlap

    Args:
        df: DataFrame with 'molecule_chembl_id' column
        test_size: Fraction for test set (0.1 = 10%)
        val_size: Fraction for validation set (0.1 = 10%)
        random_seed: For reproducibility
        output_dir: Where to save the splits

    Returns:
        Dict with 'train', 'val', 'test' DataFrames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get unique molecules (convert to numpy for sklearn compatibility)
    molecules = np.array(df["molecule_chembl_id"].unique())
    n_total = len(molecules)
    logger.info("Total unique molecules: %d", n_total)

    # Step 2: Split molecules (not records!)
    # First split: train+val vs test
    train_val_mols, test_mols = train_test_split(
        molecules,
        test_size=test_size,
        random_state=random_seed,
    )

    # Second split: train vs val
    val_fraction = val_size / (1 - test_size)  # Adjust fraction
    train_mols, val_mols = train_test_split(
        train_val_mols,
        test_size=val_fraction,
        random_state=random_seed,
    )

    logger.info(
        "Molecule split: train=%d, val=%d, test=%d",
        len(train_mols), len(val_mols), len(test_mols),
    )

    # Step 3: Assign records based on molecule splits
    train_df = df[df["molecule_chembl_id"].isin(train_mols)].copy()
    val_df = df[df["molecule_chembl_id"].isin(val_mols)].copy()
    test_df = df[df["molecule_chembl_id"].isin(test_mols)].copy()

    # Step 4: Verify no overlap
    train_set = set(train_df["molecule_chembl_id"])
    val_set = set(val_df["molecule_chembl_id"])
    test_set = set(test_df["molecule_chembl_id"])

    assert len(train_set & val_set) == 0, "LEAKAGE: Train/Val overlap!"
    assert len(train_set & test_set) == 0, "LEAKAGE: Train/Test overlap!"
    assert len(val_set & test_set) == 0, "LEAKAGE: Val/Test overlap!"

    # Print summary
    splits = {"train": train_df, "val": val_df, "test": test_df}

    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    for name, split_df in splits.items():
        n_mols = split_df["molecule_chembl_id"].nunique()
        print(f"\n{name.upper()}:")
        print(f"  Records:   {len(split_df):,}")
        print(f"  Molecules: {n_mols:,}")
        for param in split_df["pk_param"].unique():
            n = len(split_df[split_df["pk_param"] == param])
            print(f"    {param:12s}: {n:,}")

    # Save splits
    for name, split_df in splits.items():
        path = output_dir / f"{name}.parquet"
        split_df.to_parquet(path, index=False)
        logger.info("Saved %s split: %s", name, path)

    # Save molecule ID lists (for reproducibility)
    for name, mols in [("train", train_mols), ("val", val_mols), ("test", test_mols)]:
        mol_path = output_dir / f"{name}_molecules.txt"
        with open(mol_path, "w") as f:
            for mol_id in mols:
                f.write(f"{mol_id}\n")

    print(f"\nSplits saved to {output_dir}/")
    print("NO DATA LEAKAGE VERIFIED")

    return splits


# ── Main entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    # Load the targets (which have standardized units + log values)
    df = pd.read_parquet("data/processed/targets.parquet")
    print(f"Loaded {len(df)} records, {df['molecule_chembl_id'].nunique()} molecules")

    splits = create_molecule_splits(df, test_size=0.1, val_size=0.1, random_seed=42)
