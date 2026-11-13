"""
DDInter 2.0 Drug-Drug Interaction Lookup
=========================================
Loads 222K+ DDI records from DDInter 2.0 and provides fast
name-based lookup for drug pairs.

Data format: [{DDInterID_A, Drug_A, DDInterID_B, Drug_B, Level, source}, ...]
Level values: "Major", "Moderate", "Minor"
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

# Singleton: loaded once on first call
_DDI_INDEX: dict | None = None


def _normalize(name: str) -> str:
    """Normalize drug name for matching: lowercase, strip whitespace."""
    return name.strip().lower()


def _load_ddinter() -> dict:
    """
    Load DDInter unified JSON and build a bidirectional lookup index.
    Returns dict: {normalized_drug_name -> {normalized_partner -> [records]}}
    """
    global _DDI_INDEX
    if _DDI_INDEX is not None:
        return _DDI_INDEX

    # Try multiple paths
    candidates = [
        Path(__file__).parent.parent.parent / "data" / "ddi_engine" / "ddinter_unified.json",
        Path("data/ddi_engine/ddinter_unified.json"),
    ]

    data = None
    for p in candidates:
        if p.exists():
            try:
                with open(p) as f:
                    data = json.load(f)
                logger.info(f"Loaded DDInter data from {p}: {len(data)} records")
                break
            except Exception as e:
                logger.warning(f"Failed to load {p}: {e}")

    if data is None:
        logger.warning("DDInter data not found. DDI lookup will return empty results.")
        _DDI_INDEX = {}
        return _DDI_INDEX

    # Build bidirectional index
    index = defaultdict(lambda: defaultdict(list))
    for record in data:
        a = _normalize(record.get("Drug_A", ""))
        b = _normalize(record.get("Drug_B", ""))
        if a and b:
            entry = {
                "level": record.get("Level", "Unknown"),
                "drug_a": record.get("Drug_A", ""),
                "drug_b": record.get("Drug_B", ""),
                "source": record.get("source", "DDInter"),
            }
            index[a][b].append(entry)
            index[b][a].append(entry)

    _DDI_INDEX = dict(index)
    unique_drugs = len(_DDI_INDEX)
    logger.info(f"DDInter index built: {unique_drugs} unique drugs")
    return _DDI_INDEX


def lookup_ddi(drug_a_name: str, drug_b_name: str) -> list[dict]:
    """
    Look up DDI records between two drugs by name.

    Returns list of matching records, each with:
    - level: "Major" / "Moderate" / "Minor"
    - drug_a, drug_b: original drug names from DDInter
    - source: data source identifier
    """
    index = _load_ddinter()
    if not index:
        return []

    a = _normalize(drug_a_name)
    b = _normalize(drug_b_name)

    partners = index.get(a, {})
    matches = partners.get(b, [])
    return matches


def get_all_interactions(drug_name: str) -> list[dict]:
    """Get all known DDI partners for a given drug."""
    index = _load_ddinter()
    if not index:
        return []

    key = _normalize(drug_name)
    partners = index.get(key, {})

    results = []
    for partner_name, records in partners.items():
        # Take the highest severity if multiple records
        severity_order = {"major": 3, "moderate": 2, "minor": 1, "unknown": 0}
        best = max(records, key=lambda r: severity_order.get(r["level"].lower(), 0))
        results.append({
            "partner": partner_name,
            "level": best["level"],
            "drug_a": best["drug_a"],
            "drug_b": best["drug_b"],
        })

    # Sort by severity (major first)
    severity_order = {"Major": 3, "Moderate": 2, "Minor": 1}
    results.sort(key=lambda r: severity_order.get(r["level"], 0), reverse=True)
    return results
