"""Phase II metabolism substrate predictor — SMARTS-based heuristic.

Predicts UGT/SULT substrate likelihood from molecular structure using
RDKit SMARTS pattern matching.  This fills the gap where the
:class:`PhaseIIMetabolismProfile` entity exists but SMILES-only drugs
default to all-zero fm values.

Strategy
--------
For drugs where no literature Phase II data exists, detect functional
groups known to undergo glucuronidation, sulfation, or aldehyde oxidase
metabolism and assign conservative default fm fractions:

  * **Phenols** (ArOH) -> UGT1A1/1A9 substrates (paracetamol, propofol)
  * **Primary/secondary amines** -> UGT1A4 substrates (lamotrigine)
  * **Carboxylic acids** -> UGT2B7 substrates (acyl glucuronidation)
  * **Aliphatic alcohols** -> UGT2B7 substrates (morphine, zidovudine)
  * **Sulfonamides** -> SULT1A1 substrates (less common)

The assigned fm values are intentionally conservative (low).  They
represent a "non-zero prior" that enables downstream DDI and clearance
models to account for Phase II pathways rather than forcing 100% CYP.

Limitations
-----------
* No ML model -- purely rule-based from SMARTS patterns.
* Cannot distinguish UGT1A1 from UGT1A9 (both glucuronidate phenols).
* Does not account for steric effects or competing pathways.
* Total Phase II fm is capped at 0.60 to avoid overcorrection.
"""
from __future__ import annotations

import logging
from typing import Optional

from neogen.core.entities.drug import PhaseIIMetabolismProfile

logger = logging.getLogger(__name__)


def predict_phase2_profile(smiles: str) -> Optional[PhaseIIMetabolismProfile]:
    """Predict Phase II metabolism profile from SMILES via SMARTS matching.

    Parameters
    ----------
    smiles
        Valid SMILES string.

    Returns
    -------
    PhaseIIMetabolismProfile or None
        Profile with non-zero fm values for detected functional groups.
        Returns ``None`` if RDKit is unavailable or SMILES is invalid.
    """
    try:
        from rdkit import Chem
    except ImportError:
        logger.debug("RDKit not available; skipping Phase II prediction")
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.debug("Invalid SMILES for Phase II prediction: %s", smiles)
        return None

    # SMARTS patterns for Phase II-susceptible functional groups
    phenol_pat = Chem.MolFromSmarts("[OH]c")         # aromatic OH
    alc_oh_pat = Chem.MolFromSmarts("[OH][CX4]")     # aliphatic OH
    cooh_pat = Chem.MolFromSmarts("C(=O)[OH]")       # carboxylic acid
    pri_amine_pat = Chem.MolFromSmarts("[NH2][CX4]")  # primary amine
    sec_amine_pat = Chem.MolFromSmarts("[NH]([CX4])[CX4]")  # secondary amine
    aromatic_amine_pat = Chem.MolFromSmarts("[NH2]c")  # aromatic primary amine

    fm_ugt2b7 = 0.0
    fm_ugt1a1 = 0.0
    fm_ugt1a4 = 0.0
    fm_sult1a1 = 0.0

    # Phenol -> UGT1A1/1A9 glucuronidation
    if phenol_pat is not None and mol.HasSubstructMatch(phenol_pat):
        fm_ugt1a1 += 0.15
        logger.debug("Phase II: phenol detected -> fm_ugt1a1 += 0.15")

    # Carboxylic acid -> UGT2B7 acyl glucuronidation
    if cooh_pat is not None and mol.HasSubstructMatch(cooh_pat):
        fm_ugt2b7 += 0.15
        logger.debug("Phase II: carboxylic acid detected -> fm_ugt2b7 += 0.15")

    # Aliphatic alcohol -> UGT2B7 glucuronidation
    if alc_oh_pat is not None and mol.HasSubstructMatch(alc_oh_pat):
        fm_ugt2b7 += 0.10
        logger.debug("Phase II: aliphatic OH detected -> fm_ugt2b7 += 0.10")

    # Primary / secondary amines -> UGT1A4 N-glucuronidation
    has_amine = False
    if pri_amine_pat is not None and mol.HasSubstructMatch(pri_amine_pat):
        has_amine = True
    if sec_amine_pat is not None and mol.HasSubstructMatch(sec_amine_pat):
        has_amine = True
    if aromatic_amine_pat is not None and mol.HasSubstructMatch(aromatic_amine_pat):
        has_amine = True
    if has_amine:
        fm_ugt1a4 += 0.10
        logger.debug("Phase II: amine detected -> fm_ugt1a4 += 0.10")

    # Phenol -> also a minor SULT1A1 substrate
    if phenol_pat is not None and mol.HasSubstructMatch(phenol_pat):
        fm_sult1a1 += 0.05
        logger.debug("Phase II: phenol detected -> fm_sult1a1 += 0.05")

    # Cap total Phase II fm at 0.60
    total = fm_ugt1a1 + fm_ugt2b7 + fm_ugt1a4 + fm_sult1a1
    if total > 0.60:
        scale = 0.60 / total
        fm_ugt1a1 *= scale
        fm_ugt2b7 *= scale
        fm_ugt1a4 *= scale
        fm_sult1a1 *= scale

    # Only return a non-default profile if we detected something
    if total == 0.0:
        return None

    profile = PhaseIIMetabolismProfile(
        fm_ugt1a1=round(fm_ugt1a1, 4),
        fm_ugt2b7=round(fm_ugt2b7, 4),
        fm_ugt1a4=round(fm_ugt1a4, 4),
        fm_sult1a1=round(fm_sult1a1, 4),
    )
    logger.info(
        "Phase II prediction: UGT1A1=%.3f, UGT2B7=%.3f, UGT1A4=%.3f, "
        "SULT1A1=%.3f (total=%.3f)",
        profile.fm_ugt1a1, profile.fm_ugt2b7, profile.fm_ugt1a4,
        profile.fm_sult1a1, profile.total_phase2_fm,
    )
    return profile
