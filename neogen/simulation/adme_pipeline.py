"""
Modular ADME Prediction Pipeline
===================================
This is the CORE scientific prediction engine for NeoGen.

Architecture (v3.1 — expanded model support):
  SMILES → Molecular Features v1 (1044 dims) + v2 (2083 dims)
       → pKa Predictors (acidic + basic, IUPAC-trained, 2083 features)
       → v3 Features (2091 dims = 2083 + 8 pKa features)
       → fu Predictor v5 (2083 features, fallback: v4 @ 1044)
       → CYP Substrate Classifier (8 CYP flags)
       → Renal Classifier v1 (91.8% accuracy, 2083 features)
       → CL Predictor v7 (2083 raw features, fallback: v5 @ 1044+fu+CYP)
       → CLint Predictor v2 (2083 features, fallback: v1 @ 1044) → IVIVE
       → CL = best of (CL_direct, CL_ivive)
       → Vd Predictor v4_best (2091 features with pKa, fallback: v2 @ 1045)
       → F Predictor v2 + F decomposition (Fa × Fg × Fh)
       → BCS classification → drug-specific ka
       → MECHANISTIC DERIVATION (not ML):
           t½ = 0.693 × Vd / CL
           AUC = F × Dose / CL
           Cmax = analytical one-compartment solution

All mechanistic derivations use physics, not ML.
This is the same approach Simcyp uses.

Validated accuracy: 63.2% within 2-fold on 200 gold-standard drugs.
This module connects the trained ADME models to the production API.
"""
from __future__ import annotations

import json
import logging
import math
import numpy as np
import joblib
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "adme_pipeline"
BASELINE_DIR = Path(__file__).parent.parent.parent / "models" / "baselines"


def _compute_features(smiles: str) -> np.ndarray | None:
    """
    Compute 1044-dimensional molecular feature vector.
    MUST match EXACTLY what was used to train the ADME pipeline models.
    Training used 1024-bit Morgan FP + 20 descriptors = 1044 dims.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    n_heavy = Descriptors.HeavyAtomCount(mol)

    # EXACTLY 20 descriptors — matches training feature function
    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        Descriptors.FractionCSP3(mol),
        n_heavy,
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumHeteroatoms(mol),
        sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / max(n_heavy, 1),
        Descriptors.MolMR(mol),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8),   # n_O
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7),   # n_N
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16),  # n_S
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9),   # n_F
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 17),  # n_Cl
        Descriptors.MaxPartialCharge(mol) if n_heavy > 0 else 0,
    ]

    return np.concatenate([np.array(fp), descriptors])


def _compute_extended_features(mol) -> np.ndarray:
    """
    15 extended descriptors: ionization proxies, metabolic alerts, shape.
    MUST match compute_extended_features() from pka_vd_improvement_pipeline.py.
    """
    acidic_smarts = ["[OX2H][CX3](=O)", "[OX2H]S(=O)(=O)", "[NX3H]S(=O)(=O)",
                     "[OX2H][PX4](=O)", "c[nH]c", "[#6][F,Cl,Br,I]"]
    n_acidic = 0
    for sma in acidic_smarts:
        pat = Chem.MolFromSmarts(sma)
        if pat:
            n_acidic += len(mol.GetSubstructMatches(pat))

    basic_smarts = ["[NX3;H2,H1;!$(NC=O)]", "[nR1]1[cR1][cR1][cR1][cR1][cR1]1",
                    "[NX3](C)(C)C", "C(=N)N"]
    n_basic = 0
    for sma in basic_smarts:
        pat = Chem.MolFromSmarts(sma)
        if pat:
            n_basic += len(mol.GetSubstructMatches(pat))

    logp = Descriptors.MolLogP(mol)
    logd_proxy = logp - 0.5 * (n_acidic + n_basic)

    soft_spot_smarts = [
        "[CX4H2][CX4H2][CX4H2]", "c1ccc(O)cc1", "[NX3H1]c",
        "[SX2H]", "[CX3](=O)[OX2H]", "C=CC=C",
    ]
    n_soft_spots = 0
    for sma in soft_spot_smarts:
        pat = Chem.MolFromSmarts(sma)
        if pat:
            n_soft_spots += len(mol.GetSubstructMatches(pat))

    if n_acidic > 0 and n_basic > 0:
        ion_class = 3
    elif n_acidic > 0:
        ion_class = 1
    elif n_basic > 0:
        ion_class = 2
    else:
        ion_class = 0

    try:
        asphericity = rdMolDescriptors.CalcAsphericity(mol)
    except Exception:
        asphericity = 0
    try:
        eccentricity = rdMolDescriptors.CalcEccentricity(mol)
    except Exception:
        eccentricity = 0
    try:
        inertial_shape = rdMolDescriptors.CalcInertialShapeFactor(mol)
    except Exception:
        inertial_shape = 0
    try:
        n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    except Exception:
        n_spiro = 0
    try:
        n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    except Exception:
        n_bridgehead = 0

    min_pc = Descriptors.MinPartialCharge(mol)
    max_abs_pc = Descriptors.MaxAbsPartialCharge(mol)
    min_abs_pc = Descriptors.MinAbsPartialCharge(mol)

    return np.array([
        n_acidic, n_basic, logd_proxy,
        n_soft_spots, ion_class,
        asphericity, eccentricity, inertial_shape,
        n_spiro, n_bridgehead,
        min_pc if not np.isnan(min_pc) else 0,
        max_abs_pc if not np.isnan(max_abs_pc) else 0,
        min_abs_pc if not np.isnan(min_abs_pc) else 0,
        rdMolDescriptors.CalcLabuteASA(mol),
        rdMolDescriptors.CalcTPSA(mol, includeSandP=True),
    ], dtype=np.float32)


def _compute_features_v2(smiles: str) -> np.ndarray | None:
    """
    Compute 2083-dimensional feature vector for expanded models.
    2048-bit Morgan FP + 20 base descriptors + 15 extended = 2083 dims.
    MUST match compute_features_v2() from training scripts.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)

    n_heavy = Descriptors.HeavyAtomCount(mol)
    max_pc = Descriptors.MaxPartialCharge(mol)
    base_descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        Descriptors.FractionCSP3(mol),
        n_heavy,
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumHeteroatoms(mol),
        sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / max(n_heavy, 1),
        Descriptors.MolMR(mol),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 17),
        max_pc if not np.isnan(max_pc) else 0,
    ]

    ext = _compute_extended_features(mol)
    return np.concatenate([fp_arr, base_descriptors, ext]).astype(np.float32)


def _compute_pka_features(mol, pka_acid: float | None, pka_base: float | None) -> np.ndarray:
    """
    8 pKa-derived features for v3 (2091-dim) models.
    MUST match compute_pka_features() from pka_vd_improvement_pipeline.py.
    """
    pH = 7.4
    logp = Descriptors.MolLogP(mol)

    fi_acid = 0.0
    fi_base = 0.0
    logd_correction = 0.0

    if pka_acid is not None and pka_acid > 0:
        fi_acid = 1.0 / (1.0 + 10 ** (pka_acid - pH))
        logd_correction += math.log10(1 + 10 ** (pH - pka_acid))
    if pka_base is not None and pka_base > 0:
        fi_base = 1.0 / (1.0 + 10 ** (pH - pka_base))
        logd_correction += math.log10(1 + 10 ** (pka_base - pH))

    fraction_ionized = max(fi_acid, fi_base)
    logD = logp - logd_correction

    net_charge = 0.0
    if pka_base is not None and pka_base > 0:
        net_charge += fi_base
    if pka_acid is not None and pka_acid > 0:
        net_charge -= fi_acid

    is_zwitter = 1.0 if (pka_acid is not None and pka_acid > 0 and
                          pka_base is not None and pka_base > 0 and
                          pka_base < pka_acid) else 0.0

    delta_pka = 0.0
    if pka_acid is not None and pka_base is not None and pka_acid > 0 and pka_base > 0:
        delta_pka = pka_acid - pka_base

    return np.array([
        pka_acid if pka_acid is not None else 14.0,
        pka_base if pka_base is not None else 0.0,
        logD,
        fraction_ionized,
        net_charge,
        is_zwitter,
        delta_pka,
        logd_correction,
    ], dtype=np.float32)


def _compute_features_v3(smiles: str, pka_acid: float | None = None,
                          pka_base: float | None = None) -> np.ndarray | None:
    """
    Compute 2091-dimensional feature vector for Vd v4_best model.
    2048 Morgan + 20 base + 15 extended + 8 pKa features = 2091 dims.
    MUST match compute_features_v3() from pka_vd_improvement_pipeline.py.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)

    n_heavy = Descriptors.HeavyAtomCount(mol)
    max_pc = Descriptors.MaxPartialCharge(mol)
    base_descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        Descriptors.FractionCSP3(mol),
        n_heavy,
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumHeteroatoms(mol),
        sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / max(n_heavy, 1),
        Descriptors.MolMR(mol),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 17),
        max_pc if not np.isnan(max_pc) else 0,
    ]

    ext = _compute_extended_features(mol)
    pka_feats = _compute_pka_features(mol, pka_acid, pka_base)

    return np.concatenate([fp_arr, base_descriptors, ext, pka_feats]).astype(np.float32)


# Mordred/RDKit210 descriptor names (loaded at import time for v4 features)
_MORDRED_DESC_NAMES: list | None = None
_MORDRED_DESC_FUNCS: dict | None = None


def _load_mordred_descriptors():
    """Load the list of 202 valid RDKit descriptors used by v16/v9/v2 models."""
    global _MORDRED_DESC_NAMES, _MORDRED_DESC_FUNCS
    desc_path = MODEL_DIR / "mordred_descriptor_names.joblib"
    if desc_path.exists() and _MORDRED_DESC_NAMES is None:
        try:
            _MORDRED_DESC_NAMES = joblib.load(desc_path)
            # Build name→function map from Descriptors.descList
            all_descs = {name: func for name, func in Descriptors.descList}
            _MORDRED_DESC_FUNCS = {name: all_descs[name] for name in _MORDRED_DESC_NAMES
                                   if name in all_descs}
            logger.info(f"Loaded {len(_MORDRED_DESC_FUNCS)} RDKit210 descriptor functions")
        except Exception as e:
            logger.warning(f"Failed to load mordred descriptors: {e}")
            _MORDRED_DESC_NAMES = []
            _MORDRED_DESC_FUNCS = {}


def _compute_features_v4(smiles: str) -> np.ndarray | None:
    """
    Compute 2285-dimensional feature vector for v16/v9/v2 models.
    2083 (v2 features) + 202 (filtered RDKit210 descriptors) = 2285 dims.
    MUST match compute_full_features() from mordred_retrain_v1.py.
    """
    # Ensure descriptors are loaded
    if _MORDRED_DESC_NAMES is None:
        _load_mordred_descriptors()

    v2 = _compute_features_v2(smiles)
    if v2 is None:
        return None

    # If descriptor names not available, can't compute v4
    if not _MORDRED_DESC_NAMES or not _MORDRED_DESC_FUNCS:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    extra = []
    for name in _MORDRED_DESC_NAMES:
        func = _MORDRED_DESC_FUNCS.get(name)
        if func is None:
            extra.append(0.0)
            continue
        try:
            v = func(mol)
            if v is None or not np.isfinite(float(v)):
                extra.append(0.0)
            else:
                extra.append(float(v))
        except Exception:
            extra.append(0.0)

    extra_arr = np.array(extra, dtype=np.float32)
    return np.concatenate([v2, extra_arr])


class ModularADMEPredictor:
    """
    Production-grade modular ADME prediction pipeline.

    Prediction hierarchy:
    1. Predict fu (protein binding)
    2. Predict CYP substrate class
    3. Predict CL (with fu + CYP features) + IVIVE CL from CLint
    4. Predict Vd (with fu feature)
    5. Predict F (with decomposition Fa × Fg × Fh)
    6. Estimate BCS-specific ka
    7. DERIVE t½, AUC, Cmax mechanistically

    Each module is independently validated and documented.
    """

    def __init__(self):
        self.fu_model = None
        self.f_model = None
        self.cl_model = None
        self.vd_model = None
        self.clint_model = None
        self.cyp_model = None
        self.cyp_mlb = None
        self.hl_direct_model = None  # For ensemble t½
        self.papp_model = None  # Papp permeability predictor t½
        # Expanded v2.1 models (2083-dim features)
        self.fu_model_v5 = None       # fu with 2083 features
        self.clint_model_v2 = None    # CLint with 2083 features
        self.cl_model_v7 = None       # CL with 2083 features
        self.cl_model_v14 = None      # CL v14 mega (2083 features, 42% holdout)
        self.cl_model_v15 = None      # CL v15 cleaned (2083 features, 46% holdout)
        self.vd_model_v4 = None       # Vd with 2091 features (v3 + pKa)
        self.vd_model_v8 = None       # Vd v8 mega (2083 features, 58% holdout)
        self.hl_direct_v1 = None      # Direct half-life predictor (2083 features, 58% holdout)
        # v4 models (2285-dim: 2083 + 202 RDKit210 descriptors)
        self.cl_model_v16 = None      # CL v16 mordred (2285 features, 48% holdout)
        self.vd_model_v9 = None       # Vd v9 mordred (2285 features, 66% holdout)
        self.hl_direct_v2 = None      # HL v2 mordred (2285 features, 66% holdout)
        self.pka_acidic_model = None  # pKa acidic predictor (2083 features)
        self.pka_basic_model = None   # pKa basic predictor (2083 features)
        self.renal_classifier = None  # Renal elimination classifier (91.8% accuracy)
        self._retrieval_db = []
        self._retrieval_db_raw = []
        self.loaded = False
        self._load_models()
        self._load_retrieval_db()

    def _load_models(self):
        """Load all ADME pipeline models."""
        try:
            self.fu_model = joblib.load(MODEL_DIR / "fu_predictor_v4.joblib")
            self.f_model = joblib.load(MODEL_DIR / "F_predictor_v2.joblib")
            self.cl_model = joblib.load(MODEL_DIR / "cl_predictor_v5.joblib")
            self.vd_model = joblib.load(MODEL_DIR / "vd_predictor_v2.joblib")
            self.clint_model = joblib.load(MODEL_DIR / "clint_predictor.joblib")
            self.cyp_model = joblib.load(MODEL_DIR / "cyp_classifier.joblib")
            self.cyp_mlb = joblib.load(MODEL_DIR / "cyp_mlb.joblib")

            # Load direct t½ model for ensemble (best old model)
            for ver in ['v4', 'v3', 'v2', '']:
                suffix = f'_{ver}' if ver else ''
                path = BASELINE_DIR / f"xgb{suffix}_half_life.joblib"
                if path.exists():
                    self.hl_direct_model = joblib.load(path)
                    logger.info(f"Loaded direct t½ model: {path.name}")
                    break

            papp_path = MODEL_DIR / "papp_predictor.joblib"
            if papp_path.exists():
                self.papp_model = joblib.load(papp_path)
                logger.info("Loaded Papp model")

            # ── Load expanded v2.1 models (2083/2091-dim features) ──
            v2_models = {
                "fu_model_v5": "fu_predictor_v5_expanded.joblib",
                "clint_model_v2": "clint_predictor_v2_expanded.joblib",
                "cl_model_v7": "cl_predictor_v7_expanded.joblib",
                "cl_model_v14": "cl_predictor_v14_mega.joblib",
                "cl_model_v15": "cl_predictor_v15_cleaned.joblib",
                "vd_model_v4": "vd_predictor_v4_best.joblib",
                "vd_model_v8": "vd_predictor_v8_mega.joblib",
                "hl_direct_v1": "hl_predictor_v1_direct.joblib",
                "cl_model_v16": "cl_predictor_v16_mordred.joblib",
                "vd_model_v9": "vd_predictor_v9_mordred.joblib",
                "hl_direct_v2": "hl_predictor_v2_mordred.joblib",
                "pka_acidic_model": "pka_acidic_predictor_v1.joblib",
                "pka_basic_model": "pka_basic_predictor_v1.joblib",
                "renal_classifier": "renal_classifier_v1.joblib",
            }
            for attr, fname in v2_models.items():
                path = MODEL_DIR / fname
                if path.exists():
                    setattr(self, attr, joblib.load(path))
                    logger.info(f"Loaded expanded model: {fname}")
                else:
                    logger.warning(f"Expanded model not found: {fname}")

            # XGBoost compat shim — the joblibs in models/adme_pipeline/
            # were serialized with an older XGBoost version that exposed
            # gpu_id / device / predictor / use_label_encoder attributes
            # removed in newer releases. Without this patch, every
            # predict() call raises AttributeError. See
            # neogen/ml/_features.py::apply_xgboost_compat_shim for the
            # canonical implementation.
            _xgb_compat_defaults = {
                "gpu_id": None,
                "device": None,
                "predictor": None,
                "use_label_encoder": False,
            }
            for attr_name in list(vars(self)):
                model = getattr(self, attr_name, None)
                if model is None or not hasattr(model, "predict"):
                    continue
                for xattr, default in _xgb_compat_defaults.items():
                    if not hasattr(model, xattr):
                        try:
                            setattr(model, xattr, default)
                        except AttributeError:
                            pass

            self.loaded = True
            logger.info("ModularADMEPredictor: All models loaded successfully")
        except Exception as e:
            logger.error(f"ModularADMEPredictor: Failed to load models: {e}")
            self.loaded = False

    # ═══════════════════════════════════════════════════════
    # MODULE 1: fu PREDICTION
    # ═══════════════════════════════════════════════════════

    def predict_fu(self, features: np.ndarray,
                   features_v2: np.ndarray | None = None) -> float:
        """
        Predict fraction unbound (protein binding).
        v4: R² = 0.632 on 4,916 values (1044 features).
        v5: Expanded model with 2083 features for improved accuracy.
        """
        # Use v4/v2 original model (1044 features) — 94% holdout accuracy
        # v5 expanded (2083 features) is WORSE at 34%, so we skip it.
        if self.fu_model is None:
            return 0.5  # Default fallback

        log_fu = float(self.fu_model.predict(features.reshape(1, -1))[0])
        fu = 10 ** log_fu
        return max(0.001, min(fu, 1.0))

    # ═══════════════════════════════════════════════════════
    # MODULE 2: CYP SUBSTRATE CLASSIFICATION
    # ═══════════════════════════════════════════════════════

    def predict_cyp_flags(self, features: np.ndarray) -> np.ndarray:
        """
        Predict CYP substrate flags (8 binary).
        CYPs: 3A4, 2D6, 2C9, 2C19, 1A2, 2E1, 2C8, 2B6.
        Per-CYP accuracy: 95-100%.
        """
        if self.cyp_model is None:
            return np.zeros(8)

        try:
            return self.cyp_model.predict(features.reshape(1, -1))[0]
        except Exception:
            return np.zeros(8)

    # ═══════════════════════════════════════════════════════
    # MODULE 2b: pKa PREDICTION (NEW — IUPAC trained)
    # ═══════════════════════════════════════════════════════

    def predict_pka(self, features_v2: np.ndarray | None) -> dict:
        """
        Predict strongest acidic and basic pKa using expanded models.
        These pKa values feed into Vd v4_best (2091-dim) and renal classification.
        """
        pka_acid = None
        pka_base = None

        if features_v2 is not None:
            if self.pka_acidic_model is not None:
                try:
                    pka_acid = float(self.pka_acidic_model.predict(features_v2.reshape(1, -1))[0])
                    pka_acid = max(0.0, min(pka_acid, 14.0))
                except Exception as e:
                    logger.warning(f"pKa acidic prediction failed: {e}")
            if self.pka_basic_model is not None:
                try:
                    pka_base = float(self.pka_basic_model.predict(features_v2.reshape(1, -1))[0])
                    pka_base = max(0.0, min(pka_base, 14.0))
                except Exception as e:
                    logger.warning(f"pKa basic prediction failed: {e}")

        return {"pka_acidic": pka_acid, "pka_basic": pka_base}

    # ═══════════════════════════════════════════════════════
    # MODULE 2c: RENAL ELIMINATION CLASSIFICATION (91.8%)
    # ═══════════════════════════════════════════════════════

    def predict_renal(self, features_v2: np.ndarray | None) -> dict:
        """
        Predict whether renal elimination is a major route.
        Accuracy: 91.8% on holdout.
        """
        if features_v2 is None or self.renal_classifier is None:
            return {"renal_major": False, "renal_probability": 0.0}

        try:
            pred = int(self.renal_classifier.predict(features_v2.reshape(1, -1))[0])
            prob = float(self.renal_classifier.predict_proba(features_v2.reshape(1, -1))[0, 1])
            return {"renal_major": bool(pred), "renal_probability": round(prob, 4)}
        except Exception as e:
            logger.warning(f"Renal classification failed: {e}")
            return {"renal_major": False, "renal_probability": 0.0}

    # ═══════════════════════════════════════════════════════
    # MODULE 3: CLEARANCE PREDICTION (FIX 2: IVIVE)
    # ═══════════════════════════════════════════════════════

    def predict_clearance(self, features: np.ndarray, fu: float,
                          cyp_flags: np.ndarray, weight_kg: float = 70,
                          features_v2: np.ndarray | None = None,
                          features_v4: np.ndarray | None = None) -> dict:
        """
        Predict clearance using two methods and pick the best.

        Method A: Direct CL from ML (with fu + CYP features)
        Method B: IVIVE from CLint (CLint → well-stirred → CL)

        Returns dict with CL value and method used.
        """
        # Method A: Direct CL prediction
        # IMPORTANT: Model outputs are log10(CL in L/h) — already converted
        # during training. DO NOT convert again.
        cl_direct = None
        # Prefer CL v16 mordred (2285 features, 48% holdout — best CL model)
        if self.cl_model_v16 is not None and features_v4 is not None:
            try:
                cl_log = float(self.cl_model_v16.predict(features_v4.reshape(1, -1))[0])
                cl_direct = 10 ** cl_log
            except Exception as e:
                logger.warning(f"CL v16 prediction failed, falling back to v15: {e}")
                cl_direct = None
        # Fallback: CL v15 cleaned (2083 features, 46% holdout)
        if cl_direct is None and self.cl_model_v15 is not None and features_v2 is not None:
            try:
                cl_log = float(self.cl_model_v15.predict(features_v2.reshape(1, -1))[0])
                cl_direct = 10 ** cl_log
            except Exception as e:
                logger.warning(f"CL v15 prediction failed, falling back to v14: {e}")
                cl_direct = None
        # Fallback: CL v14 mega (2083 features, 42% holdout)
        if cl_direct is None and self.cl_model_v14 is not None and features_v2 is not None:
            try:
                cl_log = float(self.cl_model_v14.predict(features_v2.reshape(1, -1))[0])
                cl_direct = 10 ** cl_log
            except Exception as e:
                logger.warning(f"CL v14 prediction failed, falling back to v7: {e}")
                cl_direct = None
        # Fallback: CL v7 (expanded 2083 features)
        if cl_direct is None and self.cl_model_v7 is not None and features_v2 is not None:
            try:
                cl_log = float(self.cl_model_v7.predict(features_v2.reshape(1, -1))[0])
                cl_direct = 10 ** cl_log
            except Exception as e:
                logger.warning(f"CL v7 prediction failed, falling back to v5: {e}")
                cl_direct = None
        # Fallback: CL v5 (1044 + fu + 8 CYP = 1053 features)
        if cl_direct is None and self.cl_model is not None:
            cl_features = np.concatenate([features, [fu], cyp_flags]).reshape(1, -1)
            cl_log = float(self.cl_model.predict(cl_features)[0])
            cl_direct = 10 ** cl_log  # Already in L/h from training

        # Method B: IVIVE from CLint
        cl_ivive = None
        clint_model_used = self.clint_model
        clint_features = features.reshape(1, -1)
        # Prefer expanded CLint v2 model if available
        if self.clint_model_v2 is not None and features_v2 is not None:
            clint_model_used = self.clint_model_v2
            clint_features = features_v2.reshape(1, -1)
        if clint_model_used is not None:
            clint_log = float(clint_model_used.predict(clint_features)[0])
            clint_raw = 10 ** clint_log  # mL/min/kg

            # Well-stirred liver model:
            # CLh = Qh × fu × CLint_scaled / (Qh + fu × CLint_scaled)
            # where CLint_scaled = CLint × MPPGL × liver_weight
            Qh = 97.5  # Hepatic blood flow (L/h) for 70kg
            Qh_scaled = Qh * (weight_kg / 70) ** 0.75

            # CLint model was trained on ChEMBL clearance data (mL/min/kg).
            # Output is log10(mL/min/kg). Convert to L/h for well-stirred model.
            clint_lh = clint_raw * weight_kg * 60 / 1000  # mL/min/kg → L/h

            # Apply IVIVE empirical scaling factor (Hallifax & Houston 2009)
            esf = 0.43  # Average ESF for microsomal data
            clint_scaled = clint_lh * esf

            # Well-stirred model
            if Qh_scaled + fu * clint_scaled > 0:
                cl_ivive = Qh_scaled * fu * clint_scaled / (Qh_scaled + fu * clint_scaled)
            else:
                cl_ivive = 0

            # Add renal clearance estimate (20% of body weight-scaled GFR for average drug)
            gfr_lh = 7.2 * (weight_kg / 70) ** 0.75  # GFR in L/h
            cl_renal_estimate = gfr_lh * fu * 0.3  # Rough: 30% of filtered is excreted
            cl_ivive += cl_renal_estimate

        # Select best CL
        # LESSON LEARNED: geometric mean blending HURTS accuracy.
        # CL_direct (74.4% w/in 2-fold) is much better than IVIVE alone.
        # Use CL_direct as primary, IVIVE only as sanity check.
        if cl_direct is not None and cl_ivive is not None:
            # Primary: use CL_direct (trained on 25K clinical CL values)
            cl_final = cl_direct
            method = "direct_ML(primary)"

            # Sanity check: if IVIVE and direct agree within 3x, slightly adjust toward IVIVE
            if cl_ivive > 0:
                ratio = cl_direct / cl_ivive
                if 0.33 < ratio < 3.0:
                    # Mild adjustment: 85% direct + 15% IVIVE
                    cl_final = 0.85 * cl_direct + 0.15 * cl_ivive
                    method = "weighted(85%direct+15%IVIVE)"
                # If they disagree >3x, trust direct only
        elif cl_direct is not None:
            cl_final = cl_direct
            method = "direct_ML"
        elif cl_ivive is not None:
            cl_final = cl_ivive
            method = "IVIVE_well_stirred"
        else:
            cl_final = 10.0  # Emergency fallback
            method = "default_fallback"

        # Sanity bounds
        cl_final = max(0.01, min(cl_final, 5000))

        return {
            "cl": cl_final,
            "cl_direct": cl_direct,
            "cl_ivive": cl_ivive,
            "method": method,
        }

    # ═══════════════════════════════════════════════════════
    # MODULE 4: VOLUME OF DISTRIBUTION
    # ═══════════════════════════════════════════════════════

    def predict_vd(self, features: np.ndarray, fu: float, weight_kg: float = 70,
                   features_v2: np.ndarray | None = None,
                   features_v3: np.ndarray | None = None,
                   features_v4: np.ndarray | None = None) -> float:
        """
        Predict volume of distribution.
        v9_mordred: 2285 features, 66% holdout — best Vd model.
        v8_mega: 2083 features, 58% holdout — trained on 4195 drugs.
        v4_best: 2091 features (2083 + 8 pKa).
        v2: 1045 features (1044 + fu). R² = 0.483.
        """
        # Prefer v9 mordred (2285 features, 66% holdout — best Vd model)
        if self.vd_model_v9 is not None and features_v4 is not None:
            try:
                vd_log = float(self.vd_model_v9.predict(features_v4.reshape(1, -1))[0])
                vd = 10 ** vd_log
                return max(0.5, min(vd, 100000))
            except Exception as e:
                logger.warning(f"Vd v9 mordred prediction failed, falling back to v8: {e}")

        # Fallback: v8 mega model (2083 features, 58% holdout)
        if self.vd_model_v8 is not None and features_v2 is not None:
            try:
                vd_log = float(self.vd_model_v8.predict(features_v2.reshape(1, -1))[0])
                vd = 10 ** vd_log
                return max(0.5, min(vd, 100000))
            except Exception as e:
                logger.warning(f"Vd v8 mega prediction failed, falling back to v4: {e}")

        # Fallback: v4_best model with pKa features
        if self.vd_model_v4 is not None and features_v3 is not None:
            try:
                vd_log = float(self.vd_model_v4.predict(features_v3.reshape(1, -1))[0])
                vd = 10 ** vd_log
                return max(0.5, min(vd, 100000))
            except Exception as e:
                logger.warning(f"Vd v4_best prediction failed, falling back to v2: {e}")

        # Fallback to v2 (1044 + fu = 1045 features)
        if self.vd_model is None:
            return 50.0  # Default

        vd_features = np.concatenate([features, [fu]]).reshape(1, -1)
        vd_log = float(self.vd_model.predict(vd_features)[0])
        vd = 10 ** vd_log  # L

        return max(0.5, min(vd, 100000))

    # ═══════════════════════════════════════════════════════
    # MODULE 5: BIOAVAILABILITY (FIX 5: DECOMPOSITION)
    # ═══════════════════════════════════════════════════════

    def predict_F(self, features: np.ndarray, smiles: str,
                  fu: float, cl: float, weight_kg: float = 70) -> dict:
        """
        Predict bioavailability using decomposition + ML ensemble.

        F = Fa × Fg × Fh

        Fa: fraction absorbed (from TPSA/permeability)
        Fg: fraction escaping gut wall metabolism (from CYP3A4)
        Fh: fraction escaping hepatic first-pass (from CL and blood flow)
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {"F": 0.5, "Fa": 0.9, "Fg": 0.8, "Fh": 0.7, "method": "default"}

        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        mw = Descriptors.MolWt(mol)

        # ── Fa: Fraction Absorbed ──
        # TPSA-based approximation (Zhao et al.)
        # Papp model (R²=0.721) available but sigmoid conversion not calibrated yet.
        # Using TPSA which is validated at 62.3% overall accuracy.
        fa_pct = 109 - 0.345 * tpsa
        fa_pct = max(5, min(fa_pct, 100))
        if mw > 500:
            fa_pct *= max(0.3, 1.0 - (mw - 500) / 500)
        if logp < -2:
            fa_pct *= max(0.3, 1.0 + logp / 5)
        elif logp > 5:
            fa_pct *= max(0.5, 1.0 - (logp - 5) / 10)
        Fa = fa_pct / 100

        # ── Fg: Gut Wall Availability ──
        # CYP3A4 is the dominant gut wall enzyme
        # If drug is CYP3A4 substrate → more gut wall metabolism → lower Fg
        cyp_flags = self.predict_cyp_flags(features)
        is_cyp3a4 = bool(cyp_flags[0]) if len(cyp_flags) > 0 else False

        if is_cyp3a4:
            # CYP3A4 substrates lose 10-40% in gut wall
            Fg = max(0.4, 0.85 - 0.1 * logp if logp > 2 else 0.85)
        else:
            Fg = 0.95  # Non-CYP3A4 substrates: minimal gut wall extraction

        # ── Fh: Hepatic Availability ──
        # Fh = 1 - Eh (hepatic extraction ratio)
        # Eh = CL_hepatic / Qh
        Qh = 97.5 * (weight_kg / 70) ** 0.75  # Hepatic blood flow (L/h)
        Eh = min(cl / max(Qh, 1), 0.95)  # Extraction ratio capped at 0.95
        Fh = 1 - Eh

        # ── Combined F ──
        F_decomposed = Fa * Fg * Fh
        F_decomposed = max(0.01, min(F_decomposed, 0.99))

        # ── ML F prediction ──
        F_ml = 0.5
        if self.f_model is not None:
            F_ml = float(self.f_model.predict(features.reshape(1, -1))[0])
            F_ml = max(0.01, min(F_ml, 0.99))

        # ── Ensemble: 40% decomposed + 60% ML ──
        # ML F model (R²=0.428) was trained on measured data.
        # Decomposition is mechanistically grounded but crude.
        # Give ML slightly more weight to preserve calibration.
        F_final = 0.40 * F_decomposed + 0.60 * F_ml

        return {
            "F": round(F_final, 4),
            "F_decomposed": round(F_decomposed, 4),
            "F_ml": round(F_ml, 4),
            "Fa": round(Fa, 4),
            "Fg": round(Fg, 4),
            "Fh": round(Fh, 4),
            "method": "ensemble(decomposition+ML)",
        }

    # ═══════════════════════════════════════════════════════
    # MODULE 6: BCS-SPECIFIC ka (FIX 4)
    # ═══════════════════════════════════════════════════════

    def estimate_ka(self, smiles: str) -> dict:
        """
        Estimate absorption rate constant (ka) based on BCS classification.

        BCS I  (high solubility, high permeability): ka = 2.0-3.0 /h
        BCS II (low solubility, high permeability):  ka = 0.5-1.5 /h
        BCS III (high solubility, low permeability): ka = 0.5-1.0 /h
        BCS IV (low solubility, low permeability):   ka = 0.2-0.5 /h
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {"ka": 1.5, "bcs_class": "unknown", "method": "default"}

        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        mw = Descriptors.MolWt(mol)

        # Permeability proxy: logP > 0 and TPSA < 120 → high permeability
        high_perm = logp > 0 and tpsa < 120 and mw < 500

        # Solubility proxy: logP < 2 and MW < 400 → high solubility (simplified)
        high_sol = logp < 2 and mw < 400

        if high_sol and high_perm:
            bcs = 1
            ka = 2.5  # Fast absorption
        elif not high_sol and high_perm:
            bcs = 2
            ka = 1.0  # Dissolution-limited
        elif high_sol and not high_perm:
            bcs = 3
            ka = 0.8  # Permeability-limited
        else:
            bcs = 4
            ka = 0.3  # Poorly absorbed

        return {"ka": ka, "bcs_class": bcs, "method": "bcs_proxy"}

    # ═══════════════════════════════════════════════════════
    # MODULE 7: ENSEMBLE HALF-LIFE (FIX 1)
    # ═══════════════════════════════════════════════════════

    def predict_half_life(self, features: np.ndarray, cl: float, vd: float,
                          smiles: str,
                          features_v2: np.ndarray | None = None,
                          features_v4: np.ndarray | None = None) -> dict:
        """
        Ensemble half-life: blend derived + direct ML prediction.

        Derived: t½ = 0.693 × Vd / CL (mechanistic, always correct)
        Direct v1: XGBoost trained on 3335 drugs (2083 features, 58% holdout)
        Direct old: Legacy model (various feature dims)
        Ensemble: weighted average with sanity checks
        """
        # Derived t½ (mechanistic)
        if cl > 0:
            hl_derived = 0.693 * vd / cl
        else:
            hl_derived = 10.0  # Default

        # Direct ML t½ — prefer hl_direct_v2 (2285 features, 66% holdout)
        hl_direct = None
        if self.hl_direct_v2 is not None and features_v4 is not None:
            try:
                hl_log = float(self.hl_direct_v2.predict(features_v4.reshape(1, -1))[0])
                hl_direct = 10 ** hl_log
            except Exception as e:
                logger.warning(f"HL direct v2 prediction failed: {e}")

        # Fallback: hl_direct_v1 (2083 features, 58% holdout)
        if hl_direct is None and self.hl_direct_v1 is not None and features_v2 is not None:
            try:
                hl_log = float(self.hl_direct_v1.predict(features_v2.reshape(1, -1))[0])
                hl_direct = 10 ** hl_log
            except Exception as e:
                logger.warning(f"HL direct v1 prediction failed: {e}")

        # Fallback: old direct model
        if hl_direct is None and self.hl_direct_model is not None:
            try:
                for n_feat in [features.shape[0], 2068, 2058]:
                    try:
                        feat = features[:n_feat] if n_feat <= features.shape[0] else features
                        hl_log = float(self.hl_direct_model.predict(feat.reshape(1, -1))[0])
                        hl_direct = 10 ** hl_log
                        break
                    except ValueError:
                        continue
            except Exception:
                pass

        # Ensemble: direct v1 is strong (58%), so give it more weight
        if hl_direct is not None and hl_direct > 0:
            # Weight: 40% derived + 60% direct (direct is more accurate on holdout)
            hl_ensemble = 0.40 * hl_derived + 0.60 * hl_direct

            # Sanity: if derived and direct disagree by >10x, use geometric mean
            ratio = hl_derived / max(hl_direct, 0.001)
            if ratio > 10 or ratio < 0.1:
                hl_ensemble = np.sqrt(hl_derived * hl_direct)
                method = "geometric_mean(disagreement)"
            else:
                method = "ensemble(40%derived+60%direct)"
        else:
            hl_ensemble = hl_derived
            method = "derived_only"

        # Sanity bounds
        hl_ensemble = max(0.05, min(hl_ensemble, 5000))

        return {
            "half_life": hl_ensemble,
            "hl_derived": hl_derived,
            "hl_direct": hl_direct,
            "method": method,
        }

    # ═══════════════════════════════════════════════════════
    # MODULE 8: MECHANISTIC AUC AND Cmax DERIVATION
    # ═══════════════════════════════════════════════════════

    def derive_auc(self, F: float, dose_mg: float, cl: float) -> float:
        """AUC = F × Dose / CL (mechanistic, not ML)."""
        if cl <= 0:
            return 0
        auc = F * dose_mg / cl  # mg·h/L
        return max(0, auc)

    def derive_cmax(self, F: float, dose_mg: float, vd: float,
                    cl: float, ka: float) -> float:
        """
        Cmax from one-compartment analytical solution.
        Cmax = (F × Dose × ka) / (Vd × (ka - ke)) × (e^(-ke×tmax) - e^(-ka×tmax))
        """
        if vd <= 0 or cl <= 0:
            return 0

        ke = cl / vd  # Elimination rate constant (1/h)

        if ka <= ke:
            # If absorption is slower than elimination (unusual but possible)
            # Use simplified: Cmax ≈ F × Dose / Vd
            return F * dose_mg / vd

        # Time to peak
        tmax = math.log(ka / ke) / (ka - ke)

        # Peak concentration
        cmax = (F * dose_mg * ka) / (vd * (ka - ke)) * (
            math.exp(-ke * tmax) - math.exp(-ka * tmax)
        )

        return max(0, cmax)

    # ═══════════════════════════════════════════════════════
    # RETRIEVAL-AUGMENTED PREDICTION (TIER 2)
    # ═══════════════════════════════════════════════════════

    def _load_retrieval_db(self):
        """Load fingerprint database for nearest-neighbor retrieval."""
        db_path = Path(__file__).parent.parent.parent / "models" / "retrieval_fp_database.json"
        if not db_path.exists():
            return
        try:
            with open(db_path) as f:
                self._retrieval_db_raw = json.load(f)
            # Pre-compute fingerprints
            self._retrieval_db = []
            for entry in self._retrieval_db_raw:
                mol = Chem.MolFromSmiles(entry['smiles'])
                if not mol:
                    continue
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                # Support both old format (name/mw/logp/tpsa) and new format (drug_name)
                name = entry.get('name') or entry.get('drug_name', 'unknown')
                self._retrieval_db.append({
                    'name': name,
                    'fp': fp,
                    'mw': entry.get('mw', Descriptors.MolWt(mol)),
                    'logp': entry.get('logp', Descriptors.MolLogP(mol)),
                    'tpsa': entry.get('tpsa', Descriptors.TPSA(mol)),
                    'pk': entry['pk'],
                    'dose': entry.get('dose', 100),
                })
            logger.info(f"Retrieval DB loaded: {len(self._retrieval_db)} drugs")
        except Exception as e:
            logger.warning(f"Failed to load retrieval DB: {e}")
            self._retrieval_db = []

    def retrieval_predict(self, smiles: str, dose_mg: float = 100,
                          min_similarity: float = 0.25) -> dict | None:
        """
        Predict PK from nearest known drugs (retrieval-augmented).
        Validated: 56.4% within 2-fold on blind holdout (vs 39.3% ML-only).
        """
        if not hasattr(self, '_retrieval_db') or not self._retrieval_db:
            return None

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None

        query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        query_mw = Descriptors.MolWt(mol)
        query_logp = Descriptors.MolLogP(mol)

        # Find top-3 neighbors
        from rdkit import DataStructs
        similarities = []
        for db_entry in self._retrieval_db:
            sim = DataStructs.TanimotoSimilarity(query_fp, db_entry['fp'])
            if sim >= min_similarity:
                similarities.append((sim, db_entry))
        similarities.sort(key=lambda x: -x[0])
        neighbors = similarities[:3]

        if not neighbors:
            return None

        # Weighted PK prediction from neighbors
        predictions = {}
        total_weight = 0

        for sim, neighbor in neighbors:
            weight = sim ** 3
            total_weight += weight
            delta_logp = query_logp - neighbor['logp']
            delta_mw = query_mw - neighbor['mw']

            for param in ['half_life', 'clearance', 'vd', 'cmax', 'auc']:
                if param not in neighbor['pk']:
                    continue
                base_val = neighbor['pk'][param]

                if param == 'vd':
                    adj = base_val * (10 ** (0.20 * delta_logp)) * (query_mw / max(neighbor['mw'], 100)) ** 0.15
                elif param == 'clearance':
                    adj = base_val * (neighbor['mw'] / max(query_mw, 100)) ** 0.25
                    if delta_logp > 2: adj *= 0.7
                    elif delta_logp < -2: adj *= 1.3
                elif param == 'half_life':
                    vd_f = 10 ** (0.15 * delta_logp)
                    cl_f = (neighbor['mw'] / max(query_mw, 100)) ** 0.2
                    adj = base_val * vd_f / cl_f
                elif param == 'cmax':
                    adj = base_val * (dose_mg / max(neighbor['dose'], 1))
                    if abs(delta_logp) > 2: adj *= 0.8
                elif param == 'auc':
                    adj = base_val * (dose_mg / max(neighbor['dose'], 1))
                    cl_f = (neighbor['mw'] / max(query_mw, 100)) ** 0.2
                    adj /= cl_f
                else:
                    adj = base_val

                adj = max(adj, 0.001)
                predictions[param] = predictions.get(param, 0) + weight * adj

            # Also get fu and F from neighbor if available
            for extra in ['fu', 'F']:
                if extra in neighbor['pk']:
                    predictions[extra] = predictions.get(extra, 0) + weight * neighbor['pk'][extra]

        for param in predictions:
            predictions[param] /= max(total_weight, 0.001)

        return {
            'predictions': predictions,
            'best_neighbor': neighbors[0][1]['name'],
            'best_similarity': round(neighbors[0][0], 3),
            'n_neighbors': len(neighbors),
            'confidence': 'high' if neighbors[0][0] > 0.6 else 'moderate' if neighbors[0][0] > 0.4 else 'low',
        }

    # ═══════════════════════════════════════════════════════
    # MAIN PREDICTION METHOD
    # ═══════════════════════════════════════════════════════

    def predict(self, smiles: str, dose_mg: float = 100,
                route: str = "oral", weight_kg: float = 70,
                measured_fu: float = None,
                measured_F: float = None,
                measured_clint: float = None,
                measured_papp: float = None) -> dict:
        """
        Run the full modular ADME pipeline with retrieval augmentation.

        Prediction hierarchy:
        1. Retrieval from nearest known drug (56.4% blind accuracy)
        2. ML prediction as fallback (39.3% blind accuracy)
        3. Measured inputs override both (75-82% accuracy)

        Returns dict with all PK parameters + metadata.
        """
        if not self.loaded:
            return {"error": "Models not loaded", "predictions": {}}

        # Step 0: Compute molecular features (v1: 1044 dims)
        features = _compute_features(smiles)
        if features is None:
            return {"error": "Invalid SMILES", "predictions": {}}

        # Step 0a: Compute expanded features (v2: 2083 dims) for new models
        features_v2 = _compute_features_v2(smiles)

        # Step 0b: Predict pKa using expanded features (needed for Vd v4_best)
        pka_result = self.predict_pka(features_v2)
        pka_acid = pka_result["pka_acidic"]
        pka_base = pka_result["pka_basic"]

        # Step 0c: Compute v3 features (2091 dims = 2083 + 8 pKa) for Vd v4_best
        features_v3 = _compute_features_v3(smiles, pka_acid, pka_base)

        # Step 0d: Compute v4 features (2285 dims = 2083 + 202 RDKit210) for mordred models
        features_v4 = _compute_features_v4(smiles)

        # Step 0e: Predict renal elimination (91.8% accuracy)
        renal_result = self.predict_renal(features_v2)

        # Step 0.5: TRY RETRIEVAL FIRST (Tier 2 — 56.4% blind accuracy)
        # If a similar known drug exists, use its PK as the primary prediction
        retrieval_result = self.retrieval_predict(smiles, dose_mg)
        use_retrieval = retrieval_result is not None and retrieval_result['best_similarity'] > 0.30

        # Track which values are measured vs predicted
        measured_inputs = {}

        # Step 1: fu — USE MEASURED if provided, otherwise predict
        if measured_fu is not None and 0 < measured_fu <= 1:
            fu = measured_fu
            measured_inputs["fu"] = "measured"
        else:
            fu = self.predict_fu(features, features_v2)
            measured_inputs["fu"] = "predicted"

        # Step 2: Predict CYP flags
        cyp_flags = self.predict_cyp_flags(features)

        # Step 3: CL — if measured CLint provided, use IVIVE well-stirred
        if measured_clint is not None and measured_clint > 0:
            # IVIVE from measured CLint (much more accurate than ML prediction)
            Qh = 97.5 * (weight_kg / 70) ** 0.75
            # CLint in uL/min/mg → scale to L/h
            # MPPGL = 40 mg/g liver, liver = 21.4 g/kg body weight
            mppgl = 40  # mg microsomal protein per gram liver
            liver_weight_g = 21.4 * weight_kg  # g
            clint_lh = measured_clint * mppgl * liver_weight_g / 1e6 * 60  # uL/min/mg → L/h
            esf = 0.43  # Hallifax & Houston 2009
            clint_scaled = clint_lh * esf
            cl = Qh * fu * clint_scaled / (Qh + fu * clint_scaled)
            # Add renal
            gfr_lh = 7.2 * (weight_kg / 70) ** 0.75
            cl += gfr_lh * fu * 0.3
            cl = max(0.01, cl)
            cl_result = {"cl": cl, "method": "IVIVE_from_measured_CLint", "cl_direct": None, "cl_ivive": cl}
            measured_inputs["cl"] = "derived_from_measured_CLint"
        else:
            cl_result = self.predict_clearance(features, fu, cyp_flags, weight_kg, features_v2, features_v4)
            cl = cl_result["cl"]
            measured_inputs["cl"] = "predicted"

        # Step 4: Predict Vd (v9 mordred preferred, then v8 mega, then v4_best, then v2)
        vd = self.predict_vd(features, fu, weight_kg, features_v2, features_v3, features_v4)

        # Step 5: F — USE MEASURED if provided, otherwise predict
        if measured_F is not None and 0 < measured_F <= 1:
            F = measured_F
            f_result = {"F": F, "method": "measured_input", "Fa": None, "Fg": None, "Fh": None,
                        "F_decomposed": None, "F_ml": None}
            measured_inputs["F"] = "measured"
        else:
            f_result = self.predict_F(features, smiles, fu, cl, weight_kg)
            F = f_result["F"]
            measured_inputs["F"] = "predicted"

        # Step 6: BCS-specific ka (or use Papp if measured)
        if measured_papp is not None and measured_papp > 0:
            # High Papp → fast absorption, low Papp → slow
            if measured_papp > 20:  # High permeability (×10^-6 cm/s)
                ka = 2.5
            elif measured_papp > 5:
                ka = 1.5
            elif measured_papp > 1:
                ka = 0.8
            else:
                ka = 0.3
            ka_result = {"ka": ka, "bcs_class": "from_measured_papp", "method": "measured_papp"}
            measured_inputs["papp"] = "measured"
        else:
            ka_result = self.estimate_ka(smiles)
            ka = ka_result["ka"]

        # For IV route: F = 1.0, ka is infinite (instantaneous)
        if route.lower() in ("iv", "intravenous"):
            F = 1.0
            ka = 100.0

        # Step 7: Ensemble half-life
        hl_result = self.predict_half_life(features, cl, vd, smiles, features_v2, features_v4)
        half_life = hl_result["half_life"]

        # Step 8: Derive AUC and Cmax (mechanistic)
        auc = self.derive_auc(F, dose_mg, cl)
        cmax = self.derive_cmax(F, dose_mg, vd, cl, ka)

        # Step 9: BLEND WITH RETRIEVAL if available
        # Retrieval (56.4% blind) beats ML (39.3% blind) for unknown drugs
        final_preds = {
            "cmax": cmax,
            "auc": auc,
            "half_life": half_life,
            "clearance": cl,
            "vd": vd,
        }

        retrieval_info = {}
        if use_retrieval and not any(v != "predicted" for v in measured_inputs.values()):
            # No measured inputs → use retrieval as primary, ML as fallback
            ret_preds = retrieval_result['predictions']
            sim = retrieval_result['best_similarity']

            for param in ['half_life', 'clearance', 'vd', 'cmax', 'auc']:
                if param in ret_preds and ret_preds[param] > 0:
                    ml_val = final_preds[param]
                    ret_val = ret_preds[param]

                    # Weight retrieval more when similarity is high
                    if sim > 0.5:
                        final_preds[param] = ret_val  # 100% retrieval for high sim
                    elif sim > 0.35:
                        final_preds[param] = 0.7 * ret_val + 0.3 * ml_val
                    else:
                        final_preds[param] = 0.5 * ret_val + 0.5 * ml_val

            retrieval_info = {
                "used": True,
                "best_neighbor": retrieval_result['best_neighbor'],
                "similarity": retrieval_result['best_similarity'],
                "n_neighbors": retrieval_result['n_neighbors'],
            }
            source = "retrieval_augmented"
            logger.info(f"Retrieval: nearest={retrieval_result['best_neighbor']} sim={sim:.2f}")
        else:
            retrieval_info = {"used": False}
            source = "modular_adme_pipeline"

        # Determine confidence
        n_measured = sum(1 for v in measured_inputs.values() if v != "predicted")
        if n_measured >= 2:
            confidence = "high (measured ADME inputs)"
            accuracy_tier = "~75-82% within 2-fold"
        elif n_measured == 1:
            confidence = "moderate-high (1 measured input)"
            accuracy_tier = "~68-75% within 2-fold"
        elif use_retrieval and retrieval_result['best_similarity'] > 0.5:
            confidence = "moderate-high (similar known drug found)"
            accuracy_tier = "~56% within 2-fold (retrieval-augmented)"
        elif use_retrieval:
            confidence = "moderate (retrieval-augmented)"
            accuracy_tier = "~50% within 2-fold (retrieval-augmented)"
        else:
            confidence = "low (ML-only, no similar drug found)"
            accuracy_tier = "~39% within 2-fold (structure-only)"

        return {
            "predictions": {
                "cmax": round(final_preds["cmax"], 6),
                "auc": round(final_preds["auc"], 6),
                "half_life": round(final_preds["half_life"], 4),
                "clearance": round(final_preds["clearance"], 4),
                "vd": round(final_preds["vd"], 2),
            },
            "adme_details": {
                "fu": round(fu, 4),
                "F": round(F, 4),
                "F_decomposition": f_result,
                "ka": ka,
                "bcs_class": ka_result["bcs_class"],
                "cl_method": cl_result["method"],
                "cl_direct": cl_result.get("cl_direct"),
                "cl_ivive": cl_result.get("cl_ivive"),
                "hl_method": hl_result["method"],
                "hl_derived": hl_result.get("hl_derived"),
                "hl_direct": hl_result.get("hl_direct"),
                "cyp_flags": cyp_flags.tolist() if isinstance(cyp_flags, np.ndarray) else list(cyp_flags),
                "pka_acidic": pka_result.get("pka_acidic"),
                "pka_basic": pka_result.get("pka_basic"),
                "renal_major": renal_result.get("renal_major", False),
                "renal_probability": renal_result.get("renal_probability", 0.0),
            },
            "retrieval": retrieval_info,
            "measured_inputs": measured_inputs,
            "metadata": {
                "source": source,
                "pipeline_version": "v3.1",
                "confidence": confidence,
                "accuracy_tier": accuracy_tier,
                "dose_mg": dose_mg,
                "route": route,
                "weight_kg": weight_kg,
            },
        }


# ═══════════════════════════════════════════════════════
# SINGLETON ACCESSOR
# ═══════════════════════════════════════════════════════

_pipeline_instance = None

def get_adme_pipeline() -> ModularADMEPredictor:
    """Get or create the singleton ADME pipeline."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = ModularADMEPredictor()
    return _pipeline_instance
