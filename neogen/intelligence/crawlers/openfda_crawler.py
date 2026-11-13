"""
Stage 1: Drug Discovery via openFDA API
==========================================
Discovers FDA-approved drugs and builds a canonical drug registry.
Uses the openFDA drug label API (free, no key required).

API docs: https://open.fda.gov/apis/drug/label/
Rate limit: 240 requests/minute without API key
"""
from __future__ import annotations

import requests
import json
import time
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
OPENFDA_DRUG_URL = "https://api.fda.gov/drug/drugsfda.json"


def discover_drugs(
    max_drugs: int = 500,
    output_path: str = "data/fda_pipeline/drug_registry.json",
    therapeutic_classes: list[str] = None,
) -> list[dict]:
    """
    Discover FDA-approved drugs via openFDA API.

    Returns list of drug records with:
    - generic_name, brand_name, nda_number, route, dosage_form
    - active_ingredients, therapeutic_class
    - openfda identifiers (rxnorm, unii, atc)
    """
    logger.info("Discovering FDA-approved drugs via openFDA API...")

    drugs = []
    seen_names = set()
    skip = 0
    limit = 100  # Max per request

    while len(drugs) < max_drugs:
        params = {
            "search": 'openfda.product_type:"HUMAN PRESCRIPTION DRUG"',
            "limit": limit,
            "skip": skip,
        }

        try:
            resp = requests.get(OPENFDA_LABEL_URL, params=params, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"openFDA returned {resp.status_code} at skip={skip}")
                break

            data = resp.json()
            results = data.get("results", [])
            if not results:
                break

            for result in results:
                openfda = result.get("openfda", {})
                generic = openfda.get("generic_name", [""])[0].lower() if openfda.get("generic_name") else ""
                brand = openfda.get("brand_name", [""])[0] if openfda.get("brand_name") else ""

                if not generic or generic in seen_names:
                    continue
                seen_names.add(generic)

                drug_record = {
                    "generic_name": generic,
                    "brand_name": brand,
                    "active_ingredients": openfda.get("substance_name", []),
                    "route": openfda.get("route", [""])[0] if openfda.get("route") else "",
                    "dosage_form": openfda.get("dosage_form", [""])[0] if openfda.get("dosage_form") else "",
                    "nda_number": openfda.get("application_number", [""])[0] if openfda.get("application_number") else "",
                    "manufacturer": openfda.get("manufacturer_name", [""])[0] if openfda.get("manufacturer_name") else "",
                    "unii": openfda.get("unii", []),
                    "rxnorm": openfda.get("rxcui", []),
                    "spl_id": openfda.get("spl_id", [""])[0] if openfda.get("spl_id") else "",
                    "product_type": openfda.get("product_type", [""])[0] if openfda.get("product_type") else "",
                    # Extract available label sections
                    "has_clinical_pharmacology": bool(result.get("clinical_pharmacology")),
                    "has_drug_interactions": bool(result.get("drug_interactions")),
                    "has_warnings": bool(result.get("warnings")),
                    "has_adverse_reactions": bool(result.get("adverse_reactions")),
                    "has_dosage": bool(result.get("dosage_and_administration")),
                    "has_contraindications": bool(result.get("contraindications")),
                    # Store raw clinical pharmacology text for extraction
                    "clinical_pharmacology_text": (
                        result.get("clinical_pharmacology", [""])[0][:5000]
                        if result.get("clinical_pharmacology") else ""
                    ),
                    "drug_interactions_text": (
                        result.get("drug_interactions", [""])[0][:3000]
                        if result.get("drug_interactions") else ""
                    ),
                    "mechanism_text": (
                        result.get("mechanism_of_action", [""])[0][:2000]
                        if result.get("mechanism_of_action") else ""
                    ),
                    "warnings_text": (
                        result.get("warnings", [""])[0][:2000]
                        if result.get("warnings") else ""
                    ),
                    "contraindications_text": (
                        result.get("contraindications", [""])[0][:1000]
                        if result.get("contraindications") else ""
                    ),
                    "adverse_reactions_text": (
                        result.get("adverse_reactions", [""])[0][:3000]
                        if result.get("adverse_reactions") else ""
                    ),
                }

                drugs.append(drug_record)

                if len(drugs) >= max_drugs:
                    break

            skip += limit
            logger.info(f"  Discovered {len(drugs)} drugs so far (skip={skip})")
            time.sleep(0.3)  # Rate limiting

        except Exception as e:
            logger.error(f"Error at skip={skip}: {e}")
            time.sleep(1)
            skip += limit

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(drugs, f, indent=2)

    logger.info(f"Discovered {len(drugs)} unique drugs. Saved to {output_path}")

    # Stats
    with_cp = sum(1 for d in drugs if d["has_clinical_pharmacology"])
    with_ddi = sum(1 for d in drugs if d["has_drug_interactions"])
    logger.info(f"  With Clinical Pharmacology section: {with_cp}")
    logger.info(f"  With Drug Interactions section: {with_ddi}")

    return drugs


def get_drug_label(generic_name: str) -> Optional[dict]:
    """Get full drug label for a specific drug."""
    params = {
        "search": f'openfda.generic_name:"{generic_name}"',
        "limit": 1,
    }
    try:
        resp = requests.get(OPENFDA_LABEL_URL, params=params, timeout=15)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                return results[0]
    except Exception as e:
        logger.error(f"Error fetching label for {generic_name}: {e}")
    return None


def get_adverse_events(generic_name: str, limit: int = 100) -> list[dict]:
    """Get adverse event reports from FAERS for a drug."""
    url = "https://api.fda.gov/drug/event.json"
    params = {
        "search": f'patient.drug.openfda.generic_name:"{generic_name}"',
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": limit,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("results", [])
    except Exception as e:
        logger.error(f"Error fetching FAERS for {generic_name}: {e}")
    return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    drugs = discover_drugs(max_drugs=200)
    print(f"\nDiscovered {len(drugs)} drugs")
