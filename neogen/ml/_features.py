"""Shared feature computers and model-loading helpers for ``neogen.ml``.

All predictors in this package consume either the 1044-dim or the
2083-dim feature vector defined here. The formulas MUST match the
training-time code in ``neogen/simulation/adme_pipeline.py`` byte-
for-byte or the loaded joblibs will produce noise instead of
predictions. These functions are duplicated from that file
deliberately: the ``neogen.ml`` layer must not import
``neogen.simulation`` because the latter triggers loading of ~20
unrelated joblib files on import.

Feature layouts
---------------

**1044-dim** (``compute_features_1044``):
    1024-bit Morgan fingerprint (radius 2) + 20 RDKit descriptors.
    Used by: ``clint_predictor``, ``fu_predictor``.

**2083-dim** (``compute_features_2083``):
    2048-bit Morgan fingerprint (radius 2) + 20 base descriptors +
    15 extended descriptors (acidic/basic SMARTS counts, soft-spot
    counts, ionization class, shape descriptors).
    Used by: ``renal_predictor``, ``pka_predictor``.

Shared utilities
----------------
    ``apply_xgboost_compat_shim(model)``
        Sets missing ``gpu_id`` / ``device`` / ``predictor`` attrs
        on XGBoost models pickled with an older version.

    ``load_joblib_lazy(path)``
        Thread-safe one-shot load + shim. Returns ``None`` when
        the file is missing or the load fails.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from neogen.ml.feature_registry import get_schema
from neogen.ml.feature_schemas import conformer_3d_v1, legacy_2d_v1, stereo_2d_v1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureVector:
    feature_vector_id: str
    input_entity_id: str
    input_entity_type: str
    schema_version: str
    mode: str
    vector_hash: str
    feature_names_hash: str
    values: np.ndarray


def serialize_vector_float32_le(values: np.ndarray) -> bytes:
    arr = np.asarray(values, dtype="<f4")
    arr[np.isnan(arr)] = 0.0
    return arr.tobytes(order="C")


def vector_hash(values: np.ndarray) -> str:
    return hashlib.sha256(serialize_vector_float32_le(values)).hexdigest()


def feature_names_hash(feature_names: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(feature_names).encode("utf-8")).hexdigest()


def feature_vector_id_for(entity_id: str, schema_version: str, value_hash: str) -> str:
    return hashlib.sha256(
        f"FV|{entity_id}|{schema_version}|{value_hash}".encode("utf-8")
    ).hexdigest()[:32]


def compute_feature_vector(
    *,
    entity_id: str,
    entity_type: str,
    schema_version: str,
    repository: object | None = None,
    conformer_repository: object | None = None,
    smiles: str | None = None,
) -> FeatureVector:
    schema = get_schema(schema_version)
    if schema_version == "legacy_2d_v1":
        if smiles is None:
            if repository is None or not hasattr(repository, "get_stereoisomer_by_id"):
                raise ValueError("legacy_2d_v1 requires smiles or stereoisomer repository lookup")
            isomer = repository.get_stereoisomer_by_id(entity_id)
            if isomer is None:
                raise KeyError(f"unknown stereoisomer_id: {entity_id}")
            smiles = isomer.isomeric_smiles
        values = legacy_2d_v1.compute(smiles)
    elif schema_version == "stereo_2d_v1":
        if entity_type != "stereoisomer":
            raise ValueError("stereo_2d_v1 requires entity_type='stereoisomer'")
        if repository is None or not hasattr(repository, "get_stereoisomer_by_id"):
            raise ValueError("stereo_2d_v1 requires a molecule repository")
        isomer = repository.get_stereoisomer_by_id(entity_id)
        if isomer is None:
            raise KeyError(f"unknown stereoisomer_id: {entity_id}")
        values = stereo_2d_v1.compute(isomer)
    elif schema_version == "conformer_3d_v1":
        if entity_type != "conformer_set":
            raise ValueError("conformer_3d_v1 requires entity_type='conformer_set'")
        conf_repo = conformer_repository or repository
        if conf_repo is None or not hasattr(conf_repo, "get_set_by_id"):
            raise ValueError("conformer_3d_v1 requires a conformer repository")
        if repository is None or not hasattr(repository, "get_stereoisomer_by_id"):
            raise ValueError("conformer_3d_v1 requires a molecule repository")
        conformer_set = conf_repo.get_set_by_id(entity_id)
        if conformer_set is None:
            raise KeyError(f"unknown conformer_set_id: {entity_id}")
        isomer = repository.get_stereoisomer_by_id(conformer_set.stereoisomer_id)
        if isomer is None:
            raise KeyError(f"unknown stereoisomer_id: {conformer_set.stereoisomer_id}")
        values = conformer_3d_v1.compute(conformer_set, isomer)
    else:  # pragma: no cover - protected by get_schema
        raise KeyError(schema_version)

    if values is None:
        raise ValueError(f"could not compute features for {entity_id}")
    if int(values.shape[0]) != schema.dimensions:
        raise ValueError(
            f"{schema_version} produced {values.shape[0]} dims; expected {schema.dimensions}"
        )
    value_hash = vector_hash(values)
    return FeatureVector(
        feature_vector_id=feature_vector_id_for(entity_id, schema_version, value_hash),
        input_entity_id=entity_id,
        input_entity_type=entity_type,
        schema_version=schema_version,
        mode=schema.mode,
        vector_hash=value_hash,
        feature_names_hash=feature_names_hash(schema.feature_names),
        values=np.asarray(values, dtype=np.float64),
    )

# Silence the XGBoost "older pickle format" warning spam. The compat
# shim below (``apply_xgboost_compat_shim``) patches the missing
# attributes post-load, so the warning is advisory noise rather
# than a real issue. Suppress only the specific module's warnings.
try:
    import xgboost as _xgb  # noqa: F401

    warnings.filterwarnings(
        "ignore",
        message=r".*serialized model.*generated by older XGBoost.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*please use MorganGenerator.*",
    )
except ImportError:  # pragma: no cover — XGBoost missing
    pass


# ════════════════════════════════════════════════════════════════
# XGBOOST COMPATIBILITY
# ════════════════════════════════════════════════════════════════


def apply_xgboost_compat_shim(model: Any) -> None:
    """Patch attributes missing from older pickled XGBoost models.

    Some joblibs in ``models/adme_pipeline/`` were serialized with
    a version that exposed attributes which were removed or
    renamed in newer XGBoost releases. Setting them to their
    default values on the loaded object restores
    ``predict`` / ``predict_proba`` compatibility.

    Known missing attributes:
      * ``gpu_id``, ``device``, ``predictor`` — regressors
        (removed in XGBoost 2.x).
      * ``use_label_encoder`` — classifiers (deprecated in 1.6,
        removed in 2.x).

    Idempotent — safe to call on models that already have the
    attributes set.
    """
    defaults = {
        "gpu_id": None,
        "device": None,
        "predictor": None,
        "use_label_encoder": False,
    }
    for attr, default_value in defaults.items():
        if not hasattr(model, attr):
            try:
                setattr(model, attr, default_value)
            except AttributeError:  # pragma: no cover — slotted types
                pass


# ════════════════════════════════════════════════════════════════
# LAZY MODEL LOADER
# ════════════════════════════════════════════════════════════════


_loaded_cache: Dict[Path, Any] = {}
_loaded_cache_lock = threading.Lock()
_load_failed: set = set()


def load_joblib_lazy(path: Path) -> Optional[Any]:
    """One-shot, thread-safe joblib load with XGBoost compat shim.

    Returns the loaded object or ``None`` if the file is missing or
    the load fails. The failure state is cached so repeated calls
    do not re-try and spam the logs.
    """
    path = Path(path)
    if path in _load_failed:
        return None
    cached = _loaded_cache.get(path)
    if cached is not None:
        return cached

    with _loaded_cache_lock:
        # Double-check after acquiring the lock.
        if path in _load_failed:
            return None
        cached = _loaded_cache.get(path)
        if cached is not None:
            return cached

        if not path.exists():
            logger.warning("ML model not found at %s", path)
            _load_failed.add(path)
            return None

        try:
            import joblib
            model = joblib.load(path)
            apply_xgboost_compat_shim(model)
            _loaded_cache[path] = model
            logger.info("Loaded ML model %s", path.name)
            return model
        except Exception as exc:
            logger.warning(
                "Failed to load %s: %s — ML fallback disabled for this model",
                path.name, exc,
            )
            _load_failed.add(path)
            return None


# ════════════════════════════════════════════════════════════════
# 1044-DIM FEATURE VECTOR
# ════════════════════════════════════════════════════════════════
#
# 1024-bit Morgan FP (radius 2) + 20 RDKit descriptors.
# Duplicated byte-for-byte from
# ``neogen/simulation/adme_pipeline.py::_compute_features``.


def compute_features_1044(smiles: str) -> Optional[np.ndarray]:
    """Compute the 1044-dim ADME feature vector from SMILES.

    Returns ``None`` when RDKit is unavailable or SMILES fails to
    parse. dtype is float64 for joblib / sklearn compatibility.
    """
    try:
        return legacy_2d_v1.compute(smiles)
    except ImportError:
        logger.warning("RDKit not installed — ML features unavailable")
        return None


def _is_nan(v) -> bool:
    try:
        return v != v  # NaN is the only float that is not equal to itself
    except Exception:
        return False


# ════════════════════════════════════════════════════════════════
# 2083-DIM FEATURE VECTOR
# ════════════════════════════════════════════════════════════════
#
# 2048-bit Morgan FP (radius 2) + 20 base descriptors +
# 15 extended descriptors.
# Duplicated byte-for-byte from
# ``neogen/simulation/adme_pipeline.py::_compute_features_v2``.


def compute_features_2083(smiles: str) -> Optional[np.ndarray]:
    """Compute the 2083-dim ADME feature vector from SMILES.

    Used by the renal classifier and the pKa predictors. Returns
    ``None`` when RDKit is unavailable or SMILES fails to parse.
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem, Descriptors
    except ImportError:
        logger.warning("RDKit not installed — ML features unavailable")
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
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
        sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        / max(n_heavy, 1),
        Descriptors.MolMR(mol),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 17),
        max_pc if max_pc is not None and not _is_nan(max_pc) else 0,
    ]

    ext = _compute_extended_features(mol)
    return np.concatenate([fp_arr, base_descriptors, ext]).astype(np.float32)


def _compute_extended_features(mol) -> np.ndarray:
    """15 extended descriptors — duplicated from adme_pipeline."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    acidic_smarts = [
        "[OX2H][CX3](=O)",
        "[OX2H]S(=O)(=O)",
        "[NX3H]S(=O)(=O)",
        "[OX2H][PX4](=O)",
        "c[nH]c",
        "[#6][F,Cl,Br,I]",
    ]
    n_acidic = 0
    for sma in acidic_smarts:
        pat = Chem.MolFromSmarts(sma)
        if pat:
            n_acidic += len(mol.GetSubstructMatches(pat))

    basic_smarts = [
        "[NX3;H2,H1;!$(NC=O)]",
        "[nR1]1[cR1][cR1][cR1][cR1][cR1]1",
        "[NX3](C)(C)C",
        "C(=N)N",
    ]
    n_basic = 0
    for sma in basic_smarts:
        pat = Chem.MolFromSmarts(sma)
        if pat:
            n_basic += len(mol.GetSubstructMatches(pat))

    logp = Descriptors.MolLogP(mol)
    logd_proxy = logp - 0.5 * (n_acidic + n_basic)

    soft_spot_smarts = [
        "[CX4H2][CX4H2][CX4H2]",
        "c1ccc(O)cc1",
        "[NX3H1]c",
        "[SX2H]",
        "[CX3](=O)[OX2H]",
        "C=CC=C",
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

    return np.array(
        [
            n_acidic,
            n_basic,
            logd_proxy,
            n_soft_spots,
            ion_class,
            asphericity,
            eccentricity,
            inertial_shape,
            n_spiro,
            n_bridgehead,
            min_pc if min_pc is not None and not _is_nan(min_pc) else 0,
            max_abs_pc if max_abs_pc is not None and not _is_nan(max_abs_pc) else 0,
            min_abs_pc if min_abs_pc is not None and not _is_nan(min_abs_pc) else 0,
            rdMolDescriptors.CalcLabuteASA(mol),
            rdMolDescriptors.CalcTPSA(mol, includeSandP=True),
        ],
        dtype=np.float32,
    )
