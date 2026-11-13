"""ADME property lookup from clinical gold-standard data.

Provides measured CL, Vd, t1/2, fu, and F for drugs that match
by canonical SMILES.  This is the highest-confidence data source
-- measured clinical values always trump ML predictions.

Data sources (in priority order):
  1. data/clinical/gold_standard_pk.json   (200 drugs, full PK)
  2. data/drugbank/parsed/expanded_known_drugs.json (2 049 drugs)
  3. data/drugbank/parsed/fu_training.json  (1 415 drugs, fu only)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Literature drug names — drugs defined in literature_params.py always take
# priority over ADME lookup values. If a drug name matches an entry here,
# the lookup functions return None so the engine uses the curated literature
# values instead.
# ---------------------------------------------------------------------------

_literature_names: Optional[Set[str]] = None

# Lookup block-list: canonical SMILES the user wants forced out of every
# lookup (gold/expanded/fu_training/literature). Used to measure TRUE ML
# generalization on benchmark sets where the holdout drugs would otherwise
# be returned verbatim from a curated table.
#
# Activation:
#   NEOGEN_BLOCK_HOLDOUT_LOOKUP=1   # block the 50 blind-holdout drugs
#   NEOGEN_BLOCK_LOOKUP_FILE=path/to/blocklist.json  # custom list
#
# blocklist.json schema: a JSON list of canonical SMILES strings, OR
# a list of {"smiles": "..."} objects.
_blocked_smiles: Optional[Set[str]] = None


def _load_blocked_smiles() -> Set[str]:
    """Load the set of canonical SMILES blocked from all lookups."""
    global _blocked_smiles
    if _blocked_smiles is not None:
        return _blocked_smiles

    blocked: Set[str] = set()

    if os.environ.get("NEOGEN_BLOCK_HOLDOUT_LOOKUP", "").strip() in ("1", "true", "yes"):
        path = _repo_root() / "data" / "validation" / "blind_holdout_50.json"
        if path.exists():
            try:
                with open(path) as f:
                    entries = json.load(f)
                for entry in entries:
                    cs = _canonical(entry.get("smiles", ""))
                    if cs:
                        blocked.add(cs)
            except Exception as exc:
                logger.warning("[ADME-lookup] block-holdout load failed: %s", exc)

    extra = os.environ.get("NEOGEN_BLOCK_LOOKUP_FILE", "").strip()
    if extra:
        try:
            with open(extra) as f:
                items = json.load(f)
            for item in items:
                smi = item if isinstance(item, str) else item.get("smiles", "")
                cs = _canonical(smi)
                if cs:
                    blocked.add(cs)
        except Exception as exc:
            logger.warning("[ADME-lookup] custom blocklist load failed: %s", exc)

    _blocked_smiles = blocked
    if blocked:
        logger.info("[ADME-lookup] %d SMILES blocked from lookups", len(blocked))
    return _blocked_smiles


def _is_blocked(smiles: str) -> bool:
    """Return True if this SMILES is on the lookup block-list."""
    cs = _canonical(smiles)
    return cs is not None and cs in _load_blocked_smiles()


def _is_blocked_name(name: Optional[str]) -> bool:
    """Return True if this drug name (case-insensitive) maps to a blocked SMILES."""
    if name is None:
        return False
    blocked = _load_blocked_smiles()
    if not blocked:
        return False
    # Resolve name->SMILES via the holdout file (used as canonical name source).
    path = _repo_root() / "data" / "validation" / "blind_holdout_50.json"
    if not path.exists():
        return False
    try:
        with open(path) as f:
            entries = json.load(f)
        for entry in entries:
            if entry.get("name", "").lower() == name.lower():
                cs = _canonical(entry.get("smiles", ""))
                return cs is not None and cs in blocked
    except Exception:
        return False
    return False


def _load_literature_names() -> Set[str]:
    """Lazily load the set of drug names with literature parameters."""
    global _literature_names
    if _literature_names is not None:
        return _literature_names
    try:
        from neogen.core.constants.literature_params import LITERATURE_DRUGS
        _literature_names = set(LITERATURE_DRUGS.keys())
    except Exception:
        _literature_names = set()
    return _literature_names


def _is_literature_drug(name: Optional[str]) -> bool:
    """Return True if the drug name matches a literature_params.py entry."""
    if name is None:
        return False
    key = name.lower().replace("-", "_").replace(" ", "_")
    return key in _load_literature_names()

# ---------------------------------------------------------------------------
# Canonical SMILES helper
# ---------------------------------------------------------------------------

def _canonical(smiles: str) -> Optional[str]:
    """Return RDKit canonical SMILES, or None on parse failure."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Lazy-loaded caches (module-level singletons)
# ---------------------------------------------------------------------------

_gold_cache: Optional[Dict[str, dict]] = None
_expanded_cache: Optional[Dict[str, dict]] = None
_fu_cache: Optional[Dict[str, float]] = None


def _repo_root() -> Path:
    """Find the repository root (directory containing 'data/clinical/')."""
    # Walk up from this file's location, looking for the specific
    # data/clinical subdir to avoid matching neogen/data.
    p = Path(__file__).resolve().parent
    for _ in range(10):
        if (p / "data" / "clinical").is_dir():
            return p
        p = p.parent
    # Fallback: cwd
    return Path.cwd()


def _load_gold() -> Dict[str, dict]:
    """Load gold_standard_pk.json, keyed by canonical SMILES."""
    global _gold_cache
    if _gold_cache is not None:
        return _gold_cache

    path = _repo_root() / "data" / "clinical" / "gold_standard_pk.json"
    _gold_cache = {}
    if not path.exists():
        logger.warning("Gold-standard PK file not found: %s", path)
        return _gold_cache

    with open(path) as f:
        entries = json.load(f)

    for entry in entries:
        smiles = entry.get("smiles", "")
        cs = _canonical(smiles)
        if cs:
            _gold_cache[cs] = entry

    logger.info(
        "[ADME-lookup] Loaded %d gold-standard entries from %s",
        len(_gold_cache), path,
    )
    return _gold_cache


def _load_expanded() -> Dict[str, dict]:
    """Load expanded_known_drugs.json, keyed by canonical SMILES."""
    global _expanded_cache
    if _expanded_cache is not None:
        return _expanded_cache

    path = _repo_root() / "data" / "drugbank" / "parsed" / "expanded_known_drugs.json"
    _expanded_cache = {}
    if not path.exists():
        logger.warning("Expanded known-drugs file not found: %s", path)
        return _expanded_cache

    with open(path) as f:
        entries = json.load(f)

    for entry in entries:
        smiles = entry.get("smiles", "")
        cs = _canonical(smiles)
        if cs:
            _expanded_cache[cs] = entry

    logger.info(
        "[ADME-lookup] Loaded %d expanded-drug entries from %s",
        len(_expanded_cache), path,
    )
    return _expanded_cache


def _load_fu() -> Dict[str, float]:
    """Load fu_training.json, keyed by canonical SMILES -> fu value."""
    global _fu_cache
    if _fu_cache is not None:
        return _fu_cache

    path = _repo_root() / "data" / "drugbank" / "parsed" / "fu_training.json"
    _fu_cache = {}
    if not path.exists():
        logger.warning("fu_training file not found: %s", path)
        return _fu_cache

    with open(path) as f:
        entries = json.load(f)

    for entry in entries:
        smiles = entry.get("smiles", "")
        fu = entry.get("fu")
        cs = _canonical(smiles)
        if cs and fu is not None:
            _fu_cache[cs] = float(fu)

    logger.info(
        "[ADME-lookup] Loaded %d fu entries from %s",
        len(_fu_cache), path,
    )
    return _fu_cache


# ---------------------------------------------------------------------------
# Unified lookup helper (gold > expanded > fu_training)
# ---------------------------------------------------------------------------

def _lookup(smiles: str) -> tuple[Optional[dict], Optional[dict], Optional[float]]:
    """Return (gold_entry, expanded_entry, fu_value) for a SMILES.

    Any or all may be None if no match is found.
    """
    cs = _canonical(smiles)
    if cs is None:
        return None, None, None

    gold = _load_gold().get(cs)
    expanded = _load_expanded().get(cs)
    fu_val = _load_fu().get(cs)
    return gold, expanded, fu_val


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup_fu(smiles: str, drug_name: Optional[str] = None) -> Optional[float]:
    """Return measured fraction unbound in plasma, or None.

    Priority: gold_standard > expanded_known_drugs > fu_training.
    Returns None if the drug has a literature_params.py entry (literature wins).
    """
    if _is_literature_drug(drug_name):
        return None
    if _is_blocked(smiles) or _is_blocked_name(drug_name):
        return None

    gold, expanded, fu_val = _lookup(smiles)

    if gold and gold.get("fu") is not None:
        return float(gold["fu"])
    if expanded and expanded.get("fu") is not None:
        return float(expanded["fu"])
    if fu_val is not None:
        return fu_val
    return None


def lookup_clearance_L_per_h(
    smiles: str, drug_name: Optional[str] = None
) -> Optional[float]:
    """Return measured systemic clearance in L/h, or None.

    Priority: gold_standard > expanded_known_drugs.
    Returns None if:
      - the drug has a literature_params.py entry (literature wins), OR
      - the CL value < 0.5 L/h for a small molecule (MW < 600), which
        indicates the value is likely CL/F (oral clearance) not true
        systemic CL.
    """
    if _is_literature_drug(drug_name):
        return None
    if _is_blocked(smiles) or _is_blocked_name(drug_name):
        return None

    gold, expanded, _ = _lookup(smiles)

    cl = None
    if gold and gold.get("clearance_l_h") is not None:
        cl = float(gold["clearance_l_h"])
    elif expanded and expanded.get("clearance") is not None:
        cl = float(expanded["clearance"])

    if cl is None:
        return None

    # CL sanity filter: CL < 0.5 L/h for a small molecule (MW < 600)
    # is almost certainly CL/F (oral clearance) not true systemic CL.
    # DrugBank expanded_known_drugs reports CL/F for many oral drugs.
    # Return None so the engine computes CL from CLint instead.
    mw = _get_mw_from_lookup(gold, expanded)
    if cl < 0.5 and mw is not None and mw < 600.0:
        logger.info(
            "[ADME-lookup] CL=%.4f L/h for MW=%.1f — likely CL/F, "
            "suppressing lookup (drug_name=%s)",
            cl, mw, drug_name,
        )
        return None

    return cl


def _get_mw_from_lookup(
    gold: Optional[dict], expanded: Optional[dict]
) -> Optional[float]:
    """Extract molecular weight from a gold or expanded entry."""
    if gold and gold.get("molecular_weight") is not None:
        return float(gold["molecular_weight"])
    if expanded and expanded.get("molecular_weight") is not None:
        return float(expanded["molecular_weight"])
    # Try to compute from SMILES if neither has it
    if gold and gold.get("smiles"):
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            mol = Chem.MolFromSmiles(gold["smiles"])
            if mol:
                return Descriptors.MolWt(mol)
        except Exception:
            pass
    if expanded and expanded.get("smiles"):
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            mol = Chem.MolFromSmiles(expanded["smiles"])
            if mol:
                return Descriptors.MolWt(mol)
        except Exception:
            pass
    return None


def lookup_vd_L(smiles: str, drug_name: Optional[str] = None) -> Optional[float]:
    """Return measured volume of distribution in L, or None.

    Priority: gold_standard > expanded_known_drugs.
    Returns None if the drug has a literature_params.py entry (literature wins).
    """
    if _is_literature_drug(drug_name):
        return None
    if _is_blocked(smiles) or _is_blocked_name(drug_name):
        return None

    gold, expanded, _ = _lookup(smiles)

    if gold and gold.get("vd_l") is not None:
        return float(gold["vd_l"])
    if expanded and expanded.get("vd") is not None:
        return float(expanded["vd"])
    return None


def lookup_half_life_h(smiles: str, drug_name: Optional[str] = None) -> Optional[float]:
    """Return measured terminal half-life in hours, or None.

    Priority: gold_standard > expanded_known_drugs.
    Returns None if the drug has a literature_params.py entry (literature wins).
    """
    if _is_literature_drug(drug_name):
        return None
    if _is_blocked(smiles) or _is_blocked_name(drug_name):
        return None

    gold, expanded, _ = _lookup(smiles)

    if gold and gold.get("half_life_h") is not None:
        return float(gold["half_life_h"])
    if expanded and expanded.get("half_life") is not None:
        return float(expanded["half_life"])
    return None


def lookup_bioavailability(smiles: str, drug_name: Optional[str] = None) -> Optional[float]:
    """Return measured oral bioavailability (0-1), or None.

    Priority: gold_standard > expanded_known_drugs.
    Returns None if the drug has a literature_params.py entry (literature wins).
    """
    if _is_literature_drug(drug_name):
        return None
    if _is_blocked(smiles) or _is_blocked_name(drug_name):
        return None

    gold, expanded, _ = _lookup(smiles)

    if gold and gold.get("F") is not None:
        return float(gold["F"])
    if expanded and expanded.get("F") is not None:
        return float(expanded["F"])
    return None
