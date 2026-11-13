"""Validation cohort definitions and loaders.

Defines the frozen case types for all four validation tiers and
loaders that read them from on-disk sources:

  Tier 1: blind_holdout_50.json (single-drug PK ground truth)
  Tier 2: ddi_reference_pairs.json (clinical DDI AUCR references)
  Tier 3: tier3_bioavailability.json (oral F with decomposition when known)
  Tier 4: invariant checks (defined in code, not data)

The cohort module has zero dependency on the engine — it only
produces case objects. Harnesses in tier*.py are responsible for
running the engine and comparing predictions.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple

logger = logging.getLogger(__name__)


# Anchor — repo root relative to this file.
# neogen/validation/cohort.py → repo root is 3 parents up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_VALIDATION = _REPO_ROOT / "data" / "validation"


# ════════════════════════════════════════════════════════════════
# TIER 1 — Single-drug PK
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Tier1Case:
    """A single-drug PK validation case.

    At least one of (observed_half_life_h, observed_cl_L_per_h,
    observed_vd_L) must be present for the case to be scorable.
    """

    name: str
    smiles: str
    observed_half_life_h: Optional[float] = None
    observed_cl_L_per_h: Optional[float] = None
    observed_vd_L: Optional[float] = None
    observed_fu: Optional[float] = None
    observed_f_oral: Optional[float] = None
    literature_available: bool = False
    source: str = ""

    @property
    def has_scorable_metric(self) -> bool:
        return any(
            v is not None
            for v in (
                self.observed_half_life_h,
                self.observed_cl_L_per_h,
                self.observed_vd_L,
            )
        )


def load_tier1_blind_holdout(
    path: Optional[Path] = None,
    *,
    literature_names: Optional[set] = None,
) -> Tuple[Tier1Case, ...]:
    """Load the blind holdout 50-drug cohort from data/validation/.

    Args:
        path: Override path (default: data/validation/blind_holdout_50.json).
        literature_names: Set of drug names (lowercase) known to the
            literature DB. Used to flag which cases can be built via
            DrugFactory.create(name) vs. create_from_smiles(smiles).

    Returns:
        Tuple of Tier1Case objects. Cases missing both name and smiles
        are skipped with a warning.
    """
    path = path or (_DATA_VALIDATION / "blind_holdout_50.json")
    if not path.exists():
        logger.warning("Tier 1 cohort not found at %s", path)
        return ()

    with path.open() as fh:
        raw = json.load(fh)

    if not isinstance(raw, list):
        logger.error("Tier 1 cohort at %s is not a JSON list", path)
        return ()

    lit = {n.lower() for n in (literature_names or set())}
    cases: list[Tier1Case] = []
    for entry in raw:
        name = entry.get("name", "").strip()
        smiles = entry.get("smiles", "").strip()
        if not name or not smiles:
            logger.debug("Skipping tier1 entry missing name/smiles: %s", entry)
            continue
        case = Tier1Case(
            name=name,
            smiles=smiles,
            observed_half_life_h=entry.get("half_life_h"),
            observed_cl_L_per_h=entry.get("clearance_lh"),
            observed_vd_L=entry.get("vd_l"),
            observed_fu=entry.get("fu"),
            observed_f_oral=entry.get("bioavailability"),
            literature_available=name.lower() in lit,
            source=str(path.name),
        )
        cases.append(case)

    logger.info("Loaded %d Tier 1 cases from %s", len(cases), path.name)
    return tuple(cases)


# ════════════════════════════════════════════════════════════════
# TIER 2 — DDI pair AUCR
# ════════════════════════════════════════════════════════════════


DDIMode = Literal["reversible", "induction", "mbi"]


@dataclass(frozen=True)
class Tier2Case:
    """A single DDI victim/perpetrator pair reference.

    Mirrors the DDICase used by tests/test_v1_p3_acceptance_gate.py
    so the two harnesses can share a cohort file. All pairs are
    hepatic-CYP only in V1; gut-wall MBI and transporter DDIs are
    V1.1 scope.
    """

    label: str
    victim_name: str
    victim_dose_mg: float
    victim_route: str
    perpetrator_name: str
    perpetrator_iu_um: float
    mode: DDIMode
    clinical_aucr: float
    citation: str


def load_tier2_pairs(
    path: Optional[Path] = None,
) -> Tuple[Tier2Case, ...]:
    """Load DDI reference pairs from data/validation/ddi_reference_pairs.json."""
    path = path or (_DATA_VALIDATION / "ddi_reference_pairs.json")
    if not path.exists():
        logger.warning("Tier 2 cohort not found at %s", path)
        return ()

    with path.open() as fh:
        raw = json.load(fh)

    pairs: list[Tier2Case] = []
    for entry in raw.get("pairs", []):
        try:
            pairs.append(
                Tier2Case(
                    label=entry["label"],
                    victim_name=entry["victim_name"],
                    victim_dose_mg=float(entry["victim_dose_mg"]),
                    victim_route=entry.get("victim_route", "iv_bolus"),
                    perpetrator_name=entry["perpetrator_name"],
                    perpetrator_iu_um=float(entry["perpetrator_iu_um"]),
                    mode=entry["mode"],
                    clinical_aucr=float(entry["clinical_aucr"]),
                    citation=entry.get("citation", ""),
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed tier2 entry %s: %s", entry, exc)
            continue

    logger.info("Loaded %d Tier 2 pairs from %s", len(pairs), path.name)
    return tuple(pairs)


# ════════════════════════════════════════════════════════════════
# TIER 3 — Bioavailability decomposition
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Tier3Case:
    """A bioavailability validation case.

    The engine can report empirical F (from AUC_oral/AUC_iv) and
    a decomposed F = fa × fg × fh from the absorption + clearance
    engines. Observed fa/fg/fh are rarely published, so they are
    all optional — at minimum, observed_f_oral is required.
    """

    drug_name: str
    oral_dose_mg: float
    iv_dose_mg: float
    observed_f_oral: float
    observed_fa: Optional[float] = None
    observed_fg: Optional[float] = None
    observed_fh: Optional[float] = None
    citation: str = ""


def load_tier3_cohort(
    path: Optional[Path] = None,
) -> Tuple[Tier3Case, ...]:
    """Load bioavailability cases from data/validation/tier3_bioavailability.json."""
    path = path or (_DATA_VALIDATION / "tier3_bioavailability.json")
    if not path.exists():
        logger.warning("Tier 3 cohort not found at %s", path)
        return ()

    with path.open() as fh:
        raw = json.load(fh)

    cases: list[Tier3Case] = []
    for entry in raw.get("cases", []):
        try:
            cases.append(
                Tier3Case(
                    drug_name=entry["drug_name"],
                    oral_dose_mg=float(entry.get("oral_dose_mg", 10.0)),
                    iv_dose_mg=float(entry.get("iv_dose_mg", 10.0)),
                    observed_f_oral=float(entry["observed_f_oral"]),
                    observed_fa=entry.get("observed_fa"),
                    observed_fg=entry.get("observed_fg"),
                    observed_fh=entry.get("observed_fh"),
                    citation=entry.get("citation", ""),
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed tier3 entry %s: %s", entry, exc)
            continue

    logger.info("Loaded %d Tier 3 cases from %s", len(cases), path.name)
    return tuple(cases)


# ════════════════════════════════════════════════════════════════
# DATA PATH HELPER
# ════════════════════════════════════════════════════════════════


def data_validation_dir() -> Path:
    """Return the absolute path to data/validation/. Useful for tests."""
    return _DATA_VALIDATION
