"""Tests for ``neogen.ml.similarity_core``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neogen.ml.similarity_core import (
    EMBED_DIM,
    INDEX_DIR,
    MORGAN_BITS,
    SimilarityCore,
    _canonical_smiles,
)

# ---------------------------------------------------------------------------
# Fixtures / module-level guards
# ---------------------------------------------------------------------------

CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
HOLDOUT_PATH = Path(__file__).resolve().parents[1] / "data" / "validation" / "blind_holdout_50.json"

INDEX_PRESENT = (INDEX_DIR / "smiles.json").exists() and (INDEX_DIR / "embeddings.npy").exists()

pytestmark = pytest.mark.skipif(
    not INDEX_PRESENT,
    reason="similarity_core index not built; run scripts/build_similarity_index.py",
)


@pytest.fixture(scope="module")
def core() -> SimilarityCore:
    return SimilarityCore.load()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_roundtrip_populates_index(core: SimilarityCore) -> None:
    """Loading the on-disk index gives us drugs + parallel labels + matrices."""
    assert core.n_drugs > 0
    assert core.n_drugs == len(core.smiles)
    assert core.n_drugs == len(core.labels)
    assert core._embeddings is not None
    assert core._embeddings.shape == (core.n_drugs, EMBED_DIM)
    assert core._morgan is not None
    assert core._morgan.shape == (core.n_drugs, MORGAN_BITS)


def test_embed_returns_768d_float_vector(core: SimilarityCore) -> None:
    vec = core.embed(ASPIRIN)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (EMBED_DIM,)
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()


def test_morgan_fp_returns_2048_bits(core: SimilarityCore) -> None:
    fp = core.morgan_fp(ASPIRIN)
    assert fp.shape == (MORGAN_BITS,)
    assert fp.dtype == np.uint8
    assert set(np.unique(fp).tolist()).issubset({0, 1})
    assert fp.sum() > 0


def test_knn_caffeine_returns_xanthine_neighbour(core: SimilarityCore) -> None:
    """A xanthine query should retrieve at least one xanthine-like neighbour."""
    neighbours = core.knn(CAFFEINE, k=10, space="hybrid")
    assert len(neighbours) >= 1
    smis = [s for s, _ in neighbours]
    # Look for a fused imidazo-pyrimidinone (xanthine) substructure or the
    # explicit caffeine analog string in the neighbour SMILES.
    from rdkit import Chem

    # The xanthine scaffold is an imidazo[4,5-d]pyrimidine-2,6-dione: a
    # five-membered imidazole fused to a pyrimidinedione. We match the fused
    # bicyclic core (allowing tautomers / N-substituents) via SMARTS.
    xanthine_smarts = [
        "O=c1[nH,n]c(=O)c2ncn([*,H])c2[nH,n]1",  # standard xanthine tautomer
        "O=c1nc(=O)c2nc[nH,n]c2n1",  # alternate H positions
    ]
    patterns = [p for p in (Chem.MolFromSmarts(s) for s in xanthine_smarts) if p is not None]
    assert patterns, "failed to build xanthine SMARTS pattern"

    hits = 0
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if any(mol.HasSubstructMatch(p) for p in patterns):
            hits += 1
    assert hits >= 1, f"no xanthine-like neighbour among caffeine k-NN: {smis}"


def test_no_holdout_drug_in_index(core: SimilarityCore) -> None:
    """Every blind-holdout SMILES (canonical) must be absent from the index."""
    raw = json.loads(HOLDOUT_PATH.read_text())
    holdout_canon = {
        c
        for c in (_canonical_smiles(e["smiles"]) for e in raw if e.get("smiles"))
        if c is not None
    }
    assert holdout_canon, "holdout list parsed empty"
    index_set = set(core.smiles)
    leak = sorted(index_set & holdout_canon)
    assert not leak, f"{len(leak)} holdout SMILES leaked into index: {leak[:5]}"


def test_knn_chemberta_only_space(core: SimilarityCore) -> None:
    out = core.knn(ASPIRIN, k=5, space="chemberta")
    assert len(out) == 5
    distances = [d for _, d in out]
    assert distances == sorted(distances)


def test_knn_morgan_only_space(core: SimilarityCore) -> None:
    out = core.knn(ASPIRIN, k=5, space="morgan")
    assert len(out) == 5
    for _, d in out:
        assert 0.0 <= d <= 1.0
