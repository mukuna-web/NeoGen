"""SMILES → vector embedding + similarity core.

This module exposes :class:`SimilarityCore`, a thin wrapper around three
artifacts:

  * a ChemBERTa transformer (`seyonec/ChemBERTa-zinc-base-v1`) used to
    produce 768-dim mean-pooled embeddings,
  * a 2048-bit Morgan fingerprint (radius 2) computed via RDKit,
  * two pre-built sklearn ``NearestNeighbors`` indices over a curated
    training set of TDC + clinical-gold-standard drugs (built by
    ``scripts/build_similarity_index.py``).

The core is loaded lazily and cached at module level so import is cheap
and downstream predictors can call ``SimilarityCore.load()`` without
worrying about repeat work.

Design notes
------------

  * ``embed`` always returns a (768,) ``np.ndarray`` of float32.
  * ``morgan_fp`` always returns a (2048,) ``np.ndarray`` of uint8 bits.
  * ``knn`` returns a list of ``(train_smiles, distance)`` pairs in
    ascending-distance order. The "hybrid" space averages the
    ChemBERTa cosine distance with the Morgan jaccard distance.
  * If the ChemBERTa model fails to load (no internet, weights
    missing, transformer mismatch), ``embed`` raises ``RuntimeError``
    and the caller is expected to fall back to Morgan-only flows.
  * No hardcoded paths beyond ``models/similarity_core/`` — the
    module resolves the repo root via ``__file__``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

CHEMBERTA_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
MORGAN_BITS = 2048
MORGAN_RADIUS = 2
EMBED_DIM = 768
DEFAULT_BATCH_SIZE = 32

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = REPO_ROOT / "models" / "similarity_core"


def _silence_rdkit() -> None:
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:  # pragma: no cover - rdkit is required
        pass


def _canonical_smiles(smiles: str) -> Optional[str]:
    """Return canonical SMILES or ``None`` if RDKit can't parse."""
    _silence_rdkit()
    try:
        from rdkit import Chem
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("RDKit is required for SimilarityCore") from exc

    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _morgan_bitvec(smiles: str) -> Optional[np.ndarray]:
    """2048-bit Morgan FP (radius 2) as uint8 ``np.ndarray`` or ``None``."""
    _silence_rdkit()
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
    arr = np.zeros((MORGAN_BITS,), dtype=np.uint8)
    # GetOnBits is faster than DataStructs.ConvertToNumpyArray for sparse FPs
    for bit in fp.GetOnBits():
        arr[bit] = 1
    return arr


def _select_torch_device() -> str:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for ChemBERTa embeddings") from exc
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class _ChemBERTa:
    """Lazy holder for the HF tokenizer + model + device."""

    model_name: str = CHEMBERTA_MODEL_NAME
    device: str = field(default_factory=_select_torch_device)
    _tokenizer: object = field(default=None, init=False, repr=False)
    _model: object = field(default=None, init=False, repr=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
        except Exception as exc:
            raise RuntimeError(
                "transformers + torch are required for ChemBERTa"
            ) from exc

        logger.info("Loading ChemBERTa %s on %s", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        model.eval()
        try:
            model.to(self.device)
        except Exception as exc:  # MPS dtype quirks fall back to CPU
            logger.warning("Could not move ChemBERTa to %s (%s); using cpu", self.device, exc)
            self.device = "cpu"
            model.to("cpu")
        self._model = model
        # Disable grad globally for inference
        for p in model.parameters():
            p.requires_grad_(False)

    def encode(self, smiles_batch: list[str]) -> np.ndarray:
        """Mean-pool last hidden state over non-pad tokens. Returns (N, 768)."""
        self._ensure_loaded()
        import torch

        toks = self._tokenizer(
            smiles_batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        toks = {k: v.to(self.device) for k, v in toks.items()}
        with torch.no_grad():
            out = self._model(**toks)
        hidden = out.last_hidden_state  # (B, T, 768)
        mask = toks["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        return pooled.detach().to("cpu").float().numpy()


@dataclass
class SimilarityCore:
    """SMILES embedding + k-NN over a curated PK training set."""

    index_dir: Path = INDEX_DIR
    _smiles: list[str] = field(default_factory=list, init=False, repr=False)
    _labels: list[dict] = field(default_factory=list, init=False, repr=False)
    _embeddings: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _morgan: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _knn_chemberta: object = field(default=None, init=False, repr=False)
    _knn_morgan: object = field(default=None, init=False, repr=False)
    _chemberta: _ChemBERTa = field(default_factory=_ChemBERTa, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, index_dir: Optional[Path] = None) -> "SimilarityCore":
        """Load (and cache) a SimilarityCore from disk."""
        path = Path(index_dir) if index_dir is not None else INDEX_DIR
        cached = _CACHE.get(path)
        if cached is not None:
            return cached
        core = cls(index_dir=path)
        core._load_index()
        _CACHE[path] = core
        return core

    def _load_index(self) -> None:
        if self._loaded:
            return
        if not self.index_dir.exists():
            logger.warning("SimilarityCore index missing at %s", self.index_dir)
            self._loaded = True
            return
        try:
            import joblib
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("joblib is required for SimilarityCore") from exc

        smiles_path = self.index_dir / "smiles.json"
        labels_path = self.index_dir / "labels.json"
        emb_path = self.index_dir / "embeddings.npy"
        fp_path = self.index_dir / "morgan_fps.npy"
        knn_chem_path = self.index_dir / "knn_chemberta.joblib"
        knn_morgan_path = self.index_dir / "knn_morgan.joblib"

        if smiles_path.exists():
            self._smiles = json.loads(smiles_path.read_text())
        if labels_path.exists():
            self._labels = json.loads(labels_path.read_text())
        if emb_path.exists():
            self._embeddings = np.load(emb_path)
        if fp_path.exists():
            self._morgan = np.load(fp_path)
        if knn_chem_path.exists():
            self._knn_chemberta = joblib.load(knn_chem_path)
        if knn_morgan_path.exists():
            self._knn_morgan = joblib.load(knn_morgan_path)
        self._loaded = True

    # ------------------------------------------------------------------ public

    @property
    def n_drugs(self) -> int:
        return len(self._smiles)

    @property
    def smiles(self) -> list[str]:
        return list(self._smiles)

    @property
    def labels(self) -> list[dict]:
        return [dict(d) for d in self._labels]

    def canonical(self, smiles: str) -> Optional[str]:
        return _canonical_smiles(smiles)

    def morgan_fp(self, smiles: str) -> np.ndarray:
        """2048-bit Morgan FP. Raises ``ValueError`` for unparseable SMILES."""
        fp = _morgan_bitvec(smiles)
        if fp is None:
            raise ValueError(f"Unparseable SMILES: {smiles!r}")
        return fp

    def embed(self, smiles: str) -> np.ndarray:
        """768-dim ChemBERTa mean-pooled embedding for a single SMILES."""
        canon = _canonical_smiles(smiles)
        if canon is None:
            raise ValueError(f"Unparseable SMILES: {smiles!r}")
        return self._chemberta.encode([canon])[0].astype(np.float32)

    def embed_batch(
        self, smiles_list: list[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> np.ndarray:
        """Batched embedding. Returns (N, 768) float32. Skips on failure."""
        if not smiles_list:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        out = np.zeros((len(smiles_list), EMBED_DIM), dtype=np.float32)
        for start in range(0, len(smiles_list), batch_size):
            chunk = smiles_list[start : start + batch_size]
            vecs = self._chemberta.encode(chunk)
            out[start : start + len(chunk)] = vecs.astype(np.float32)
        return out

    def knn(
        self,
        smiles: str,
        k: int = 10,
        space: str = "hybrid",
    ) -> list[tuple[str, float]]:
        """k-Nearest-neighbour search.

        Parameters
        ----------
        smiles
            Query SMILES (canonicalised internally).
        k
            Number of neighbours to return.
        space
            One of ``"chemberta"``, ``"morgan"``, ``"hybrid"``.
            ``"hybrid"`` averages cosine + jaccard distances.
        """
        if not self._loaded:
            self._load_index()
        if self.n_drugs == 0:
            return []

        if space == "chemberta":
            return self._knn_chem(smiles, k)
        if space == "morgan":
            return self._knn_morgan_only(smiles, k)
        if space == "hybrid":
            return self._knn_hybrid(smiles, k)
        raise ValueError(f"unknown space: {space!r}")

    # ------------------------------------------------------------------ internals

    def _knn_chem(self, smiles: str, k: int) -> list[tuple[str, float]]:
        if self._knn_chemberta is None or self._embeddings is None:
            raise RuntimeError("ChemBERTa index not loaded")
        vec = self.embed(smiles).reshape(1, -1)
        dist, idx = self._knn_chemberta.kneighbors(
            vec, n_neighbors=min(k, self.n_drugs)
        )
        return [(self._smiles[i], float(d)) for d, i in zip(dist[0], idx[0])]

    def _knn_morgan_only(self, smiles: str, k: int) -> list[tuple[str, float]]:
        if self._knn_morgan is None or self._morgan is None:
            raise RuntimeError("Morgan index not loaded")
        fp = self.morgan_fp(smiles).reshape(1, -1).astype(bool)
        dist, idx = self._knn_morgan.kneighbors(
            fp, n_neighbors=min(k, self.n_drugs)
        )
        return [(self._smiles[i], float(d)) for d, i in zip(dist[0], idx[0])]

    def _knn_hybrid(self, smiles: str, k: int) -> list[tuple[str, float]]:
        # Pull a wider candidate set from each space then re-rank by mean distance.
        if (
            self._knn_chemberta is None
            or self._knn_morgan is None
            or self._embeddings is None
            or self._morgan is None
        ):
            raise RuntimeError("Hybrid k-NN requires both indices loaded")

        cand = max(k * 4, 25)
        cand = min(cand, self.n_drugs)

        chem_vec = self.embed(smiles).reshape(1, -1)
        chem_d, chem_i = self._knn_chemberta.kneighbors(chem_vec, n_neighbors=cand)
        fp = self.morgan_fp(smiles).reshape(1, -1).astype(bool)
        morg_d, morg_i = self._knn_morgan.kneighbors(fp, n_neighbors=cand)

        union = set(chem_i[0].tolist()) | set(morg_i[0].tolist())
        chem_lookup = {int(i): float(d) for i, d in zip(chem_i[0], chem_d[0])}
        morg_lookup = {int(i): float(d) for i, d in zip(morg_i[0], morg_d[0])}

        scored: list[tuple[int, float]] = []
        for i in union:
            cd = chem_lookup.get(i, 1.0)
            md = morg_lookup.get(i, 1.0)
            scored.append((i, 0.5 * cd + 0.5 * md))
        scored.sort(key=lambda pair: pair[1])
        return [(self._smiles[i], dist) for i, dist in scored[:k]]


# Module-level cache keyed by index_dir to avoid re-loading the model on
# every ``SimilarityCore.load()`` call across the codebase.
_CACHE: dict[Path, SimilarityCore] = {}


__all__ = [
    "SimilarityCore",
    "CHEMBERTA_MODEL_NAME",
    "MORGAN_BITS",
    "MORGAN_RADIUS",
    "EMBED_DIM",
    "INDEX_DIR",
]
