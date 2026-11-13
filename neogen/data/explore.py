"""
Data Exploration & Validation
==============================
Explores the cleaned ChEMBL PK dataset to understand:
- How many records per PK parameter
- Value distributions (are they normal? skewed?)
- Unit consistency
- Missing data patterns
- Molecule overlap across PK types

This is a critical step before ML — we need to catch data quality
issues before they become model quality issues.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Make plots look clean
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"] = 100


def load_clean_data(path: str = "data/processed/chembl_pk_clean.parquet") -> pd.DataFrame:
    """Load the cleaned PK dataset."""
    df = pd.read_parquet(path)
    logger.info("Loaded %d records from %s", len(df), path)
    return df


def print_overview(df: pd.DataFrame) -> None:
    """Print high-level dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total records:      {len(df):,}")
    print(f"Unique molecules:   {df['molecule_chembl_id'].nunique():,}")
    print(f"Columns:            {len(df.columns)}")
    print(f"\nRecords per PK parameter:")
    for param, count in df["pk_param"].value_counts().items():
        n_mols = df[df["pk_param"] == param]["molecule_chembl_id"].nunique()
        print(f"  {param:12s}: {count:6,} records  ({n_mols:,} unique molecules)")

    # Check molecule overlap — how many drugs have multiple PK params?
    mol_params = df.groupby("molecule_chembl_id")["pk_param"].nunique()
    print(f"\nMolecule coverage:")
    for n in range(1, 6):
        count = (mol_params == n).sum()
        if count > 0:
            print(f"  Molecules with {n} PK params: {count:,}")

    # Molecules with ALL 5 PK params — these are the "gold standard"
    mols_with_all = (mol_params == 5).sum()
    print(f"\n  Gold standard (all 5 params): {mols_with_all:,} molecules")


def print_unit_analysis(df: pd.DataFrame) -> None:
    """Check unit consistency per PK parameter."""
    print("\n" + "=" * 60)
    print("UNIT ANALYSIS")
    print("=" * 60)
    for param in df["pk_param"].unique():
        subset = df[df["pk_param"] == param]
        print(f"\n{param}:")
        unit_counts = subset["standard_units"].value_counts()
        for unit, count in unit_counts.head(5).items():
            pct = count / len(subset) * 100
            print(f"  {unit:20s}: {count:6,} ({pct:.1f}%)")


def print_value_statistics(df: pd.DataFrame) -> None:
    """Print value statistics per PK parameter."""
    print("\n" + "=" * 60)
    print("VALUE STATISTICS")
    print("=" * 60)
    for param in df["pk_param"].unique():
        subset = df[df["pk_param"] == param]
        vals = subset["standard_value"]
        unit = subset["standard_units"].mode().iloc[0] if len(subset["standard_units"].mode()) > 0 else "?"
        print(f"\n{param} ({unit}):")
        print(f"  Count:   {len(vals):,}")
        print(f"  Mean:    {vals.mean():.2f}")
        print(f"  Median:  {vals.median():.2f}")
        print(f"  Std:     {vals.std():.2f}")
        print(f"  Min:     {vals.min():.4f}")
        print(f"  Max:     {vals.max():.2f}")
        print(f"  Zeros:   {(vals == 0).sum()}")
        print(f"  Negative: {(vals < 0).sum()}")


def plot_distributions(df: pd.DataFrame, output_dir: str = "reports/figures") -> None:
    """
    Plot value distributions for each PK parameter.

    We plot both raw and log-transformed distributions because
    PK values are typically log-normally distributed (most drugs
    have low values, a few have very high values).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = df["pk_param"].unique()
    fig, axes = plt.subplots(len(params), 2, figsize=(14, 4 * len(params)))

    for i, param in enumerate(params):
        subset = df[df["pk_param"] == param]
        vals = subset["standard_value"]
        unit = subset["standard_units"].mode().iloc[0] if len(subset["standard_units"].mode()) > 0 else "?"

        # Raw distribution
        ax1 = axes[i, 0]
        ax1.hist(vals, bins=100, color="steelblue", alpha=0.7, edgecolor="white")
        ax1.set_title(f"{param} — Raw Distribution")
        ax1.set_xlabel(f"Value ({unit})")
        ax1.set_ylabel("Count")

        # Log-transformed distribution (much more useful for ML)
        ax2 = axes[i, 1]
        log_vals = np.log10(vals[vals > 0] + 1e-10)
        ax2.hist(log_vals, bins=100, color="coral", alpha=0.7, edgecolor="white")
        ax2.set_title(f"{param} — Log10 Distribution")
        ax2.set_xlabel(f"log10(Value)")
        ax2.set_ylabel("Count")

    plt.tight_layout()
    fig_path = output_dir / "pk_distributions.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"\nSaved distribution plot: {fig_path}")


def plot_molecule_overlap(df: pd.DataFrame, output_dir: str = "reports/figures") -> None:
    """
    Visualize how many PK parameters each molecule has.

    This is important because our neural model will be multi-task —
    predicting all 5 PK params simultaneously. Molecules with more
    params give the model more learning signal.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a pivot: which molecules have which PK params?
    mol_params = df.groupby("molecule_chembl_id")["pk_param"].apply(set).reset_index()
    mol_params["n_params"] = mol_params["pk_param"].apply(len)

    fig, ax = plt.subplots(figsize=(8, 5))
    mol_params["n_params"].value_counts().sort_index().plot(
        kind="bar", color="steelblue", alpha=0.8, edgecolor="white", ax=ax
    )
    ax.set_xlabel("Number of PK Parameters Available")
    ax.set_ylabel("Number of Molecules")
    ax.set_title("Molecule Coverage: How Many PK Params Per Drug?")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig_path = output_dir / "molecule_overlap.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved molecule overlap plot: {fig_path}")


def plot_smiles_length_dist(df: pd.DataFrame, output_dir: str = "reports/figures") -> None:
    """
    Plot distribution of SMILES string lengths.

    Why this matters: Very long SMILES = very complex molecules.
    Very short SMILES = simple molecules. If our dataset is skewed
    toward one extreme, our model might not generalize well.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    smiles_lengths = df["canonical_smiles"].str.len()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(smiles_lengths, bins=100, color="seagreen", alpha=0.7, edgecolor="white")
    ax.set_xlabel("SMILES String Length (characters)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Molecular Complexity (SMILES Length)")
    ax.axvline(smiles_lengths.median(), color="red", linestyle="--", label=f"Median: {smiles_lengths.median():.0f}")
    ax.legend()

    plt.tight_layout()
    fig_path = output_dir / "smiles_length_dist.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved SMILES length plot: {fig_path}")


def validate_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that all SMILES strings can be parsed by RDKit.

    Why: Some SMILES from databases can be malformed. RDKit will
    return None for invalid SMILES. We need to remove these before
    generating fingerprints.
    """
    from rdkit import Chem

    print("\n" + "=" * 60)
    print("SMILES VALIDATION")
    print("=" * 60)

    valid_mask = []
    invalid_smiles = []

    for smiles in df["canonical_smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_mask.append(True)
        else:
            valid_mask.append(False)
            invalid_smiles.append(smiles)

    valid_count = sum(valid_mask)
    invalid_count = len(valid_mask) - valid_count

    print(f"Total SMILES:   {len(df):,}")
    print(f"Valid:          {valid_count:,} ({valid_count/len(df)*100:.1f}%)")
    print(f"Invalid:        {invalid_count:,} ({invalid_count/len(df)*100:.1f}%)")

    if invalid_smiles:
        print(f"\nSample invalid SMILES:")
        for s in invalid_smiles[:5]:
            print(f"  {s[:80]}")

    # Return only valid records
    df_valid = df[valid_mask].copy()
    return df_valid


# ── Main entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    # Load data
    df = load_clean_data()

    # Explore
    print_overview(df)
    print_unit_analysis(df)
    print_value_statistics(df)

    # Validate SMILES
    df_valid = validate_smiles(df)

    # Save validated data
    df_valid.to_parquet("data/processed/chembl_pk_validated.parquet", index=False)
    print(f"\nSaved validated data: {len(df_valid)} records")

    # Generate plots
    print("\nGenerating plots...")
    plot_distributions(df_valid)
    plot_molecule_overlap(df_valid)
    plot_smiles_length_dist(df_valid)

    print("\nExploration complete!")
