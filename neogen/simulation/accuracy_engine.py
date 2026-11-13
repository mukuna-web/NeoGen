"""
Maximum Accuracy Engine
=========================
Combines EVERY accuracy improvement into one unified prediction system.

Strategy: For ANY molecule, use ALL available information:
1. Known drug database (100% if match)
2. Molecular similarity to 232 clinical drugs (weighted neighbor predictions)
3. PK first-principles (AUC = F×Dose/CL, t½ = 0.693×Vd/CL, Cmax from AUC)
4. ML prediction (XGBoost + Chemprop ensemble)
5. Clinical calibration (GBR correction trained on 80 gold-standard drugs)
6. Sanity bounds (physiological limits)

The key insight: CL and Vd are FUNDAMENTAL. Everything else derives from them.
Focus all effort on getting CL and Vd right, then derive AUC, t½, and Cmax.
"""
from __future__ import annotations

import math
import json
import logging
from pathlib import Path

import numpy as np
import joblib

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# MOLECULAR SIMILARITY ENGINE
# ════════════════════════════════════════════════════════════════

class ClinicalDrugDatabase:
    """Database of drugs with known clinical PK values + fingerprints for similarity search."""

    def __init__(self):
        self.drugs = []
        self.fingerprints = []
        self._loaded = False

    def load(self):
        """Load all clinical drugs (gold-standard + known_drugs)."""
        if self._loaded:
            return

        from rdkit import Chem
        from rdkit.Chem import AllChem

        # Load gold-standard dataset
        gold_path = Path("data/clinical/gold_standard_pk.json")
        if gold_path.exists():
            with open(gold_path) as f:
                gold = json.load(f)
            for d in gold:
                mol = Chem.MolFromSmiles(d["smiles"])
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    self.drugs.append(d)
                    self.fingerprints.append(fp)

        # Load known_drugs database
        try:
            from neogen.simulation.known_drugs import KNOWN_DRUGS
            for smiles, drug in KNOWN_DRUGS.items():
                # Check if already in gold standard (by SMILES)
                if any(d["smiles"] == smiles for d in self.drugs):
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    self.drugs.append({
                        "drug_name": drug.name,
                        "smiles": smiles,
                        "dose_mg": drug.reference_dose_mg,
                        "cmax_mg_l": drug.cmax_mg_l,
                        "auc_mg_h_l": drug.auc_mg_h_l,
                        "half_life_h": drug.half_life_h,
                        "clearance_l_h": drug.clearance_l_h,
                        "vd_l": drug.vd_l,
                        "F": drug.F,
                        "fu": drug.fu,
                        "primary_cyp": getattr(drug, 'primary_cyp', ''),
                    })
                    self.fingerprints.append(fp)
        except Exception as e:
            logger.warning("Could not load known_drugs: %s", e)

        self._loaded = True
        logger.info("Clinical drug database loaded: %d drugs", len(self.drugs))

    def find_similar(self, smiles: str, top_k: int = 5) -> list:
        """Find the K most similar clinical drugs by Tanimoto similarity."""
        self.load()

        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return []

        query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

        similarities = []
        for i, db_fp in enumerate(self.fingerprints):
            sim = DataStructs.TanimotoSimilarity(query_fp, db_fp)
            similarities.append((sim, i))

        # Sort by similarity (highest first)
        similarities.sort(reverse=True)

        results = []
        for sim, idx in similarities[:top_k]:
            drug = self.drugs[idx].copy()
            drug["similarity"] = round(sim, 4)
            results.append(drug)

        return results

    def predict_from_neighbors(self, smiles: str, dose_mg: float, top_k: int = 5) -> dict:
        """
        Predict PK by weighted average of similar clinical drugs.

        Weight = similarity^3 (emphasize close matches).
        Dose-scale Cmax and AUC linearly.
        """
        neighbors = self.find_similar(smiles, top_k)

        if not neighbors or neighbors[0]["similarity"] < 0.15:
            return None  # Too dissimilar — ML prediction is better

        # Weighted prediction
        params = ["cmax_mg_l", "auc_mg_h_l", "half_life_h", "clearance_l_h", "vd_l"]
        weighted_preds = {p: 0.0 for p in params}
        weight_sum = 0.0

        for neighbor in neighbors:
            sim = neighbor["similarity"]
            if sim < 0.1:
                continue

            weight = sim ** 3  # Cubic weighting — close matches dominate

            # Dose scaling for Cmax and AUC
            dose_ratio = dose_mg / neighbor["dose_mg"] if neighbor["dose_mg"] > 0 else 1.0

            for p in params:
                val = neighbor.get(p, 0)
                if val and val > 0:
                    if p in ["cmax_mg_l", "auc_mg_h_l"]:
                        val *= dose_ratio  # Scale with dose
                    weighted_preds[p] += val * weight

            weight_sum += weight

        if weight_sum < 0.01:
            return None

        # Normalize
        for p in params:
            weighted_preds[p] /= weight_sum

        return {
            "cmax": weighted_preds["cmax_mg_l"],
            "auc": weighted_preds["auc_mg_h_l"],
            "half_life": weighted_preds["half_life_h"],
            "clearance": weighted_preds["clearance_l_h"],
            "vd": weighted_preds["vd_l"],
            "top_neighbor": neighbors[0]["drug_name"],
            "top_similarity": neighbors[0]["similarity"],
            "n_neighbors_used": sum(1 for n in neighbors if n["similarity"] >= 0.1),
            "confidence": min(neighbors[0]["similarity"] * 1.2, 0.95),
        }


# ════════════════════════════════════════════════════════════════
# PK FIRST-PRINCIPLES DERIVATION
# ════════════════════════════════════════════════════════════════

def derive_pk_from_fundamentals(
    cl_total: float,
    vd: float,
    dose_mg: float,
    F: float,
    fu: float,
    logp: float,
    mw: float,
    tpsa: float,
) -> dict:
    """
    Derive ALL PK parameters from CL and Vd using first principles.

    Fundamental equations:
    - AUC = F × Dose / CL
    - t½ = 0.693 × Vd / CL
    - ke = CL / Vd
    - ka estimated from BCS/molecular properties
    - Cmax = (F × Dose × ka) / (Vd × (ka - ke)) × (exp(-ke×tmax) - exp(-ka×tmax))
    - tmax = ln(ka/ke) / (ka - ke)
    """
    if cl_total <= 0 or vd <= 0:
        return None

    # AUC
    auc = F * dose_mg / cl_total

    # Half-life
    ke = cl_total / vd
    half_life = 0.693 / ke

    # ka from molecular properties
    if tpsa < 60 and logp > 1:
        ka = 2.0  # BCS I — fast absorption
    elif tpsa < 90:
        ka = 1.2  # BCS I/II — moderate
    elif tpsa < 120:
        ka = 0.6  # BCS III — slow
    else:
        ka = 0.3  # BCS IV — very slow

    if mw > 500:
        ka *= 0.7

    # Cmax
    if abs(ka - ke) > 0.001:
        tmax = math.log(ka / ke) / (ka - ke) if ka > ke else 1.0
        tmax = max(0.25, min(tmax, 12))
        cmax = (F * dose_mg * ka) / (vd * (ka - ke)) * (math.exp(-ke * tmax) - math.exp(-ka * tmax))
    else:
        tmax = 1.0 / ka
        cmax = F * dose_mg / vd * 0.632  # At tmax when ka=ke

    cmax = max(0, cmax)

    # Sanity bounds
    cmax = min(cmax, dose_mg / 3.0)  # Can't exceed dose/plasma_volume
    auc = max(0, auc)
    half_life = max(0.1, min(half_life, 1000))

    return {
        "cmax": round(cmax, 6),
        "auc": round(auc, 4),
        "half_life": round(half_life, 2),
        "clearance": round(cl_total, 3),
        "vd": round(vd, 1),
        "tmax": round(tmax, 2),
        "ka": round(ka, 2),
        "ke": round(ke, 4),
    }


# ════════════════════════════════════════════════════════════════
# SMART ENSEMBLE — COMBINES EVERYTHING
# ════════════════════════════════════════════════════════════════

# Global clinical database instance
_clinical_db = ClinicalDrugDatabase()


def maximum_accuracy_predict(
    smiles: str,
    dose_mg: float,
    weight_kg: float = 70.0,
    ml_predictions: dict = None,
    measured_fu: float = None,
    measured_clint: float = None,
) -> dict:
    """
    Maximum accuracy prediction using ALL available methods combined.

    Combines:
    1. Known drug lookup (100% if match)
    2. Neighbor-weighted prediction (similarity to 232 clinical drugs)
    3. PK first-principles (CL/Vd → AUC/t½/Cmax)
    4. ML prediction (XGBoost/Chemprop ensemble)
    5. Clinical calibration
    6. Physiological sanity bounds
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {"error": "Invalid SMILES"}

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)

    corrections = []

    # ── Step 1: Known drug lookup ──
    try:
        from neogen.simulation.known_drugs import get_known_pk_at_dose
        known = get_known_pk_at_dose(smiles, dose_mg, weight_kg)
        if known:
            return {
                "mode": "KNOWN_DRUG",
                "predictions": {
                    "cmax": known["cmax"],
                    "auc": known["auc"],
                    "half_life": known["half_life"],
                    "clearance": known["clearance"],
                    "vd": known["vd"],
                },
                "confidence": 0.99,
                "drug_name": known["drug_name"],
                "method": "Published clinical values",
            }
    except:
        pass

    # ── Step 2: Find similar clinical drugs ──
    neighbor_pred = _clinical_db.predict_from_neighbors(smiles, dose_mg)

    # ── Step 3: ML prediction ──
    ml_pred = None
    if ml_predictions:
        from neogen.simulation.accuracy_fixes import predict_fu_advanced

        # Determine fu
        if measured_fu is not None:
            fu = measured_fu
        else:
            n_ar = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
            charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
            fu_result = predict_fu_advanced(logp, mw, tpsa, hbd,
                                           Descriptors.NumHAcceptors(mol), n_ar, charge)
            fu = fu_result["fu_predicted"]

        # Convert ML raw predictions to clinical units
        raw = {}
        for p in ml_predictions:
            raw[p] = 10 ** ml_predictions[p] if isinstance(ml_predictions[p], (int, float)) else 1.0

        # Unit conversion
        ml_cmax = raw.get('cmax', 1) * mw / 1e6
        ml_auc = raw.get('auc', 1) / 1000
        ml_hl = raw.get('half_life', 1)
        ml_cl = raw.get('clearance', 1) * weight_kg * 60 / 1000
        ml_vd = raw.get('vd', 1) * weight_kg

        # Apply calibration if available
        try:
            cal_features = [logp, np.log10(max(mw,100)), tpsa/100, fu, np.log10(max(dose_mg,0.1)),
                           0.5, fu, Descriptors.NumAromaticRings(mol)]

            for param, ml_val in [('half_life', ml_hl)]:  # Only calibrate half-life (proven to help)
                try:
                    cal_model = joblib.load(f'models/baselines/calibration_{param}.joblib')
                    cal_input = np.array(cal_features + [np.log10(max(ml_val, 1e-10))]).reshape(1, -1)
                    correction = cal_model.predict(cal_input)[0]
                    if param == 'half_life':
                        ml_hl *= 10 ** correction
                except:
                    pass
        except:
            pass

        # Estimate F
        fa = max(0, min(100, 109 - 0.345 * tpsa)) / 100
        fg = max(0.3, 1.0 - 0.15 * (logp - 3)) if logp > 3 else 0.90
        fh = max(0.1, 1 - ml_cl / 87.0) if ml_cl < 87 else 0.1
        F = max(0.01, min(fa * fg * fh, 0.99))

        # IVIVE correction if measured CLint available
        if measured_clint is not None and measured_fu is not None:
            from neogen.simulation.accuracy_fixes import improved_ivive
            ivive = improved_ivive(measured_clint, measured_fu, smiles, logp, mw, tpsa)
            ml_cl = ivive["cl_hepatic"] + 7.2 * fu * 0.5  # Add renal

        # Derive PK from CL and Vd (first principles)
        derived = derive_pk_from_fundamentals(ml_cl, ml_vd, dose_mg, F, fu, logp, mw, tpsa)

        if derived:
            ml_pred = {
                "cmax": derived["cmax"],
                "auc": derived["auc"],
                "half_life": ml_hl * 1.54 if not measured_clint else derived["half_life"],
                "clearance": ml_cl,
                "vd": ml_vd,
            }

    # ── Step 4: Smart blend of neighbor + ML predictions ──
    final = {}

    if neighbor_pred and neighbor_pred["top_similarity"] > 0.3:
        # High similarity — trust neighbors more
        sim = neighbor_pred["top_similarity"]
        neighbor_weight = min(sim * 1.5, 0.8)  # Max 80% neighbor
        ml_weight = 1 - neighbor_weight

        corrections.append(f"Neighbor blend: {neighbor_pred['top_neighbor']} (sim={sim:.2f}), "
                          f"weights: neighbor={neighbor_weight:.0%} ML={ml_weight:.0%}")

        for param in ['cmax', 'auc', 'half_life', 'clearance', 'vd']:
            n_val = neighbor_pred.get(param, 0)
            m_val = ml_pred.get(param, 0) if ml_pred else 0

            if n_val > 0 and m_val > 0:
                # Blend in LOG space (geometric mean style)
                final[param] = math.exp(
                    neighbor_weight * math.log(max(n_val, 1e-10)) +
                    ml_weight * math.log(max(m_val, 1e-10))
                )
            elif n_val > 0:
                final[param] = n_val
            elif m_val > 0:
                final[param] = m_val
            else:
                final[param] = 0

        confidence = 0.3 + sim * 0.5
        mode = f"NEIGHBOR_BLEND (sim={sim:.2f})"

    elif neighbor_pred and neighbor_pred["top_similarity"] > 0.15:
        # Moderate similarity — use as sanity check
        sim = neighbor_pred["top_similarity"]
        corrections.append(f"ML primary, neighbor sanity check: {neighbor_pred['top_neighbor']} (sim={sim:.2f})")

        for param in ['cmax', 'auc', 'half_life', 'clearance', 'vd']:
            m_val = ml_pred.get(param, 0) if ml_pred else 0
            n_val = neighbor_pred.get(param, 0)

            if m_val > 0:
                # If ML is wildly off from neighbor, pull toward neighbor
                if n_val > 0 and m_val > 0:
                    ratio = m_val / n_val
                    if ratio > 10 or ratio < 0.1:
                        # ML is >10x off from neighbor — blend more toward neighbor
                        final[param] = math.exp(0.5 * math.log(max(m_val, 1e-10)) +
                                               0.5 * math.log(max(n_val, 1e-10)))
                    else:
                        final[param] = m_val
                else:
                    final[param] = m_val
            elif n_val > 0:
                final[param] = n_val
            else:
                final[param] = 0

        confidence = 0.25 + sim * 0.3
        mode = f"ML_WITH_NEIGHBOR_CHECK (sim={sim:.2f})"

    else:
        # No similar neighbor — pure ML
        if ml_pred:
            final = ml_pred.copy()
        else:
            return {"error": "No predictions available"}

        confidence = 0.20
        mode = "PURE_ML"

    # ── Step 5: Physiological sanity bounds ──
    if final.get("cmax", 0) > dose_mg / 3.0:
        final["cmax"] = dose_mg / 3.0
        corrections.append("Cmax capped at dose/Vplasma")

    if final.get("half_life", 0) > 500:
        final["half_life"] = min(final["half_life"], 200)
        corrections.append("t½ capped at 200h")

    if final.get("half_life", 0) < 0.1:
        final["half_life"] = max(final["half_life"], 0.25)
        corrections.append("t½ floored at 0.25h")

    # ── Step 6: Ensure internal consistency ──
    # AUC should approximately equal F × Dose / CL
    if final.get("clearance", 0) > 0:
        fa = max(0, min(100, 109 - 0.345 * tpsa)) / 100
        F_est = fa * 0.7
        auc_derived = F_est * dose_mg / final["clearance"]

        # If ML AUC is >5x off from derived, use average
        if final.get("auc", 0) > 0 and auc_derived > 0:
            ratio = final["auc"] / auc_derived
            if ratio > 5 or ratio < 0.2:
                final["auc"] = math.sqrt(final["auc"] * auc_derived)  # Geometric mean
                corrections.append("AUC harmonized with F×Dose/CL")

    # Round
    for p in final:
        if isinstance(final[p], float):
            final[p] = round(final[p], 6)

    return {
        "mode": mode,
        "predictions": final,
        "confidence": round(confidence, 2),
        "corrections": corrections,
        "neighbor": {
            "name": neighbor_pred["top_neighbor"] if neighbor_pred else None,
            "similarity": neighbor_pred["top_similarity"] if neighbor_pred else 0,
        } if neighbor_pred else None,
    }
