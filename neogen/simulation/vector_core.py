"""SMILES → vector similarity core for NeoGen AI-first pipeline.

Zero external deps beyond MLX (already installed). Uses RDKit when available
for proper Morgan fingerprints; otherwise falls back to a deterministic
hashed-substring featurizer that still preserves rough structural similarity.

API:
    core = VectorCore()
    core.register("atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O")
    core.query_by_smiles(smiles, k=5) -> [(drug_name, cosine), ...]
    core.query_by_name("atorvastatin", k=5)
    core.pair_similarity("atorvastatin", "simvastatin")
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import mlx.core as mx

FP_BITS = 2048
EMBED_DIM = 256
MORGAN_RADIUS = 2


def _hashed_morgan_like(smiles: str, bits: int = FP_BITS, radius: int = MORGAN_RADIUS) -> list[int]:
    """RDKit-free structural fingerprint.

    Tokenises SMILES into overlapping substring shingles up to length 2*radius+1
    and hashes each into one of `bits` positions. Preserves substructure
    co-occurrence and behaves similarly to Morgan in pairwise-similarity
    comparisons on small/medium molecules (Tanimoto within ~0.05 on a random
    sample we have not yet formally benchmarked — marked assumption).
    """
    vec = [0] * bits
    s = smiles.strip()
    if not s:
        return vec
    # include character n-grams 1..(2*radius+1)
    for n in range(1, 2 * radius + 2):
        for i in range(0, max(0, len(s) - n + 1)):
            sub = s[i : i + n]
            h = int(hashlib.blake2b(sub.encode("utf-8"), digest_size=4).hexdigest(), 16)
            vec[h % bits] |= 1
    return vec


def _rdkit_morgan(smiles: str) -> list[int] | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=FP_BITS)
    return list(fp)


def featurise(smiles: str) -> list[int]:
    rd = _rdkit_morgan(smiles)
    return rd if rd is not None else _hashed_morgan_like(smiles)


@dataclass
class VectorCore:
    """In-memory SMILES → dense embedding index."""

    projection_seed: int = 20260421
    _projection: mx.array | None = field(default=None, init=False)
    _index: dict[str, mx.array] = field(default_factory=dict, init=False)
    _smiles_by_name: dict[str, str] = field(default_factory=dict, init=False)

    def _ensure_projection(self) -> mx.array:
        if self._projection is None:
            # Deterministic Johnson-Lindenstrauss projection, gaussian.
            mx.random.seed(self.projection_seed)
            self._projection = mx.random.normal((FP_BITS, EMBED_DIM)) / (EMBED_DIM**0.5)
            mx.eval(self._projection)
        return self._projection

    def embed(self, smiles: str) -> mx.array:
        fp = featurise(smiles)
        fp_arr = mx.array(fp, dtype=mx.float32)
        proj = self._ensure_projection()
        dense = fp_arr @ proj
        norm = mx.sqrt((dense * dense).sum()) + 1e-9
        out = dense / norm
        mx.eval(out)
        return out

    def register(self, name: str, smiles: str) -> None:
        key = name.strip().lower()
        self._smiles_by_name[key] = smiles
        self._index[key] = self.embed(smiles)

    def register_many(self, pairs: Iterable[tuple[str, str]]) -> int:
        count = 0
        for name, smiles in pairs:
            try:
                self.register(name, smiles)
                count += 1
            except Exception:
                continue
        return count

    def pair_similarity(self, name_a: str, name_b: str) -> float | None:
        a = self._index.get(name_a.strip().lower())
        b = self._index.get(name_b.strip().lower())
        if a is None or b is None:
            return None
        sim = (a * b).sum()
        mx.eval(sim)
        return float(sim.item())

    def query_by_smiles(self, smiles: str, k: int = 5) -> list[tuple[str, float]]:
        if not self._index:
            return []
        q = self.embed(smiles)
        names = list(self._index.keys())
        matrix = mx.stack([self._index[n] for n in names])
        sims = matrix @ q
        mx.eval(sims)
        sims_list = sims.tolist()
        ranked = sorted(zip(names, sims_list), key=lambda x: -x[1])
        return ranked[: k]

    def query_by_name(self, name: str, k: int = 5) -> list[tuple[str, float]]:
        smi = self._smiles_by_name.get(name.strip().lower())
        if smi is None:
            return []
        ranked = self.query_by_smiles(smi, k=k + 1)
        return [(n, s) for n, s in ranked if n != name.strip().lower()][:k]

    def size(self) -> int:
        return len(self._index)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "projection_seed": self.projection_seed,
            "smiles_by_name": self._smiles_by_name,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "VectorCore":
        data = json.loads(Path(path).read_text())
        core = cls(projection_seed=data.get("projection_seed", 20260421))
        for name, smi in data.get("smiles_by_name", {}).items():
            core.register(name, smi)
        return core
