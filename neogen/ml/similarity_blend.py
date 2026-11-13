"""Retrieval-augmented blend over engine PK predictions.

Given (smiles, engine_predictions) the blender:
  1. Looks up k=10 nearest analogs in the SimilarityCore index.
  2. Computes a similarity-weighted geometric-mean k-NN prediction
     for half_life_h and vd_l_per_kg (only — see notes).
  3. Blends k-NN with engine in log-space:
        alpha = max_sim ** 2  (high sim → trust k-NN, low sim → trust engine)
        log(p_blend) = alpha * log(p_knn) + (1-alpha) * log(p_engine)

Why not blend CL?
  The available CL training set (Clearance_Hepatocyte_AZ) is in
  µL/min/10⁶ cells (intrinsic clearance), not L/h systemic. Blending
  it directly into systemic CL is a unit error and worsens accuracy
  (measured: CL %w2f drops from 63% → 56% on the leak-blocked holdout
  when CL is naively blended). So we leave CL alone.

Numbers from this module are emitted only when the caller opts in.
The default behavior of the engine is unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class BlendedPK:
    half_life_h: Optional[float]
    cl_L_per_h: Optional[float]      # passthrough — never blended
    vd_L: Optional[float]
    max_similarity: float
    n_neighbors: int
    knn_t12_h: Optional[float]
    knn_vd_l_per_kg: Optional[float]
    used_blend_t12: bool
    used_blend_vd: bool


def _safe_log_blend(p_knn: Optional[float], p_engine: Optional[float], alpha: float) -> Optional[float]:
    """Log-space alpha-blend; falls back to whichever input is valid."""
    if p_knn is None or p_knn <= 0:
        return p_engine
    if p_engine is None or p_engine <= 0:
        return p_knn
    return float(math.exp(alpha * math.log(p_knn) + (1.0 - alpha) * math.log(p_engine)))


def blend_engine_with_knn(
    smiles: str,
    engine_half_life_h: Optional[float],
    engine_cl_L_per_h: Optional[float],
    engine_vd_L: Optional[float],
    *,
    body_weight_kg: float = 70.0,
    k: int = 10,
    min_similarity: float = 0.20,
) -> Optional[BlendedPK]:
    """Return a BlendedPK or None if the similarity core can't load.

    `engine_vd_L` is in litres absolute (matches engine output).  Blending
    is done in litres-per-kg space because that's what the Lombardo
    training set uses; we convert with `body_weight_kg`.
    """
    try:
        from neogen.ml.similarity_core import SimilarityCore
        core = SimilarityCore.load()
    except Exception:
        return None

    # Hybrid space (Morgan + ChemBERTa) for ranking
    try:
        neighbors = core.knn(smiles, k=k, space="hybrid")
    except Exception:
        return None
    if not neighbors:
        return None

    # neighbors: list[(train_smiles, distance)] — convert distance→similarity
    # The SimilarityCore knn returns distances in [0,1] for cosine/jaccard.
    # We treat 1 - distance as similarity, clamped to [0, 1].
    sims = []
    t12s = []
    vds = []
    labels = core.labels  # parallel array, dict per drug
    smi_to_idx = {s: i for i, s in enumerate(core.smiles)}
    for nbr_smi, dist in neighbors:
        idx = smi_to_idx.get(nbr_smi)
        if idx is None:
            continue
        sim = max(0.0, min(1.0, 1.0 - float(dist)))
        lab = labels[idx] if idx < len(labels) else {}
        t12 = lab.get("half_life_h")
        vd_kg = lab.get("vd_l_per_kg")
        if t12 is None and vd_kg is None:
            continue
        sims.append(sim)
        t12s.append(t12)
        vds.append(vd_kg)

    if not sims:
        return None
    sims_arr = np.asarray(sims)
    max_sim = float(sims_arr.max())

    def _wgmean(values, weights):
        pairs = [(v, w) for v, w in zip(values, weights)
                 if v is not None and v > 0 and w > 0]
        if not pairs:
            return None
        vs, ws = zip(*pairs)
        ws_norm = np.asarray(ws) / float(np.sum(ws))
        return float(math.exp(float(np.sum(ws_norm * np.log(vs)))))

    knn_t12 = _wgmean(t12s, sims) if any(t is not None for t in t12s) else None
    knn_vd_kg = _wgmean(vds, sims) if any(v is not None for v in vds) else None

    # Apply blend only when at least one neighbor is meaningfully similar
    apply_blend = max_sim >= min_similarity
    alpha = max_sim ** 2 if apply_blend else 0.0

    blended_t12 = _safe_log_blend(knn_t12, engine_half_life_h, alpha)

    # Vd: convert engine Vd (L) → L/kg, blend, then convert back
    engine_vd_kg = engine_vd_L / body_weight_kg if engine_vd_L else None
    blended_vd_kg = _safe_log_blend(knn_vd_kg, engine_vd_kg, alpha)
    blended_vd_L = blended_vd_kg * body_weight_kg if blended_vd_kg else engine_vd_L

    return BlendedPK(
        half_life_h=blended_t12,
        cl_L_per_h=engine_cl_L_per_h,
        vd_L=blended_vd_L,
        max_similarity=max_sim,
        n_neighbors=len(sims),
        knn_t12_h=knn_t12,
        knn_vd_l_per_kg=knn_vd_kg,
        used_blend_t12=apply_blend and (knn_t12 is not None),
        used_blend_vd=apply_blend and (knn_vd_kg is not None),
    )
