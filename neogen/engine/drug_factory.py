"""DrugFactory — SMILES → Drug entity pipeline.

This is the entry point for ALL NeoGen V1 simulations. Given a SMILES
string, the factory:
  1. Computes physicochemical properties via RDKit
  2. Predicts ADME parameters via NeoGen ML models (when available)
  3. Overlays literature values when they exist (highest priority)
  4. Returns an immutable Drug entity ready for simulation

Data priority: Measured > Literature > ML Predicted > Defaults

Architecture reference: Module #1 — DrugFactory (Blueprint p.5)
"""

from __future__ import annotations

import logging
import math
from dataclasses import fields
from typing import Optional

from neogen.core.entities.drug import (
    BCSClass,
    CYPMetabolismProfile,
    Drug,
    InductionProfile,
    InhibitionProfile,
    Ionization,
    PhaseIIMetabolismProfile,
    TransporterAffinities,
)
from neogen.core.constants.literature_params import get_literature_params

logger = logging.getLogger(__name__)


def _compute_rdkit_properties(smiles: str) -> dict:
    """Compute physicochemical properties from SMILES via RDKit.

    Returns a dict of computed properties. If RDKit is not available,
    returns empty dict and factory falls back to defaults/literature.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except ImportError:
        logger.warning(
            "RDKit not installed. Using literature/default values only."
        )
        return {}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    props = {
        "molecular_weight": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
    }

    return props


def _classify_bcs(solubility_mg_ml: Optional[float], tpsa: float,
                  logp: float) -> BCSClass:
    """Classify drug into BCS class from solubility and permeability.

    BCS classification:
      - Solubility: High if highest dose dissolves in ≤250 mL at pH 1-6.8
        Approximated from logP/TPSA when not measured.
      - Permeability: High if TPSA < 90 Å² (Egan egg model)
    """
    # Permeability estimate from TPSA
    high_perm = tpsa < 90.0

    # Solubility estimate
    if solubility_mg_ml is not None:
        high_sol = solubility_mg_ml > 0.1  # Rough threshold
    else:
        # General Solubility Equation: logS = 0.5 - 0.01*(MP-25) - logP
        # Simplified: high solubility if logP < 2 (very rough)
        high_sol = logp < 2.0

    if high_sol and high_perm:
        return BCSClass.I
    if not high_sol and high_perm:
        return BCSClass.II
    if high_sol and not high_perm:
        return BCSClass.III
    return BCSClass.IV


def _classify_ionization(pka_acidic: Optional[float],
                         pka_basic: Optional[float]) -> Ionization:
    """Classify ionization state from pKa values."""
    has_acid = pka_acidic is not None and pka_acidic < 10
    has_base = pka_basic is not None and pka_basic > 2

    if has_acid and has_base:
        return Ionization.ZWITTERION
    if has_acid:
        return Ionization.ACID
    if has_base:
        return Ionization.BASE
    return Ionization.NEUTRAL


def _estimate_fu_from_logp(logp: float) -> float:
    """Estimate fraction unbound from logP using calibrated lookup table.

    This is the PRIMARY fu estimation method for SMILES-only drugs.
    The ML fu predictor (fu_predictor_v4) is NOT used because it
    produces near-ceiling values (~0.9-1.0) for all drugs (session 4
    smoke test). This lookup-table heuristic, ported from
    ``neogen.simulation.invitro_input``, gives better calibrated
    estimates across the lipophilicity range.

    Bins calibrated against Obach (1999) and lombardo (2018) datasets.
    """
    if logp < 0:
        return 0.9   # Hydrophilic — mostly free
    elif logp < 1:
        return 0.7
    elif logp < 2:
        return 0.4
    elif logp < 3:
        return 0.15
    elif logp < 4:
        return 0.05
    else:
        return 0.01  # Very lipophilic — highly bound


class DrugFactory:
    """Factory for creating Drug entities from SMILES strings.

    Usage:
        factory = DrugFactory()

        # From SMILES (uses RDKit + ML + literature fallback)
        drug = factory.create("midazolam", smiles="C1C(=O)N...")

        # From name only (uses literature database)
        drug = factory.create("midazolam")

        # With user-provided overrides
        drug = factory.create("midazolam", overrides={"fu_plasma": 0.05})
    """

    def create(
        self,
        name: str,
        smiles: Optional[str] = None,
        overrides: Optional[dict] = None,
    ) -> Drug:
        """Create a Drug entity.

        Priority: overrides > literature > RDKit computed > defaults

        Args:
            name: Drug name (used for literature lookup)
            smiles: SMILES string (optional if drug is in literature DB)
            overrides: User-provided parameter overrides (highest priority)

        Returns:
            Immutable Drug entity ready for simulation
        """
        params: dict = {}

        # Block-guard: when the user has set a lookup block-list (typically
        # for the blind-holdout cohort) we bypass *every* curated-data
        # short-circuit so the engine's true ML path is exercised. Lookups
        # inside neogen.ml.adme_lookup honour the same env-driven blocklist,
        # so the upstream skip here keeps registry hits from leaking either.
        from neogen.ml.adme_lookup import _is_blocked, _is_blocked_name
        block_lookups = _is_blocked_name(name) or (
            smiles is not None and _is_blocked(smiles)
        )

        # Step 1: Literature lookup
        lit = None if block_lookups else get_literature_params(name)
        if lit is not None:
            params.update(lit)
            logger.info(f"[DrugFactory] Literature values found for {name}")

        # Step 2: Resolve SMILES
        if smiles is None:
            smiles = params.get("smiles", "")

        # Step 2b: Gold standard fallback (200 drugs) + DrugBank (1,851 drugs)
        # Provides SMILES + published PK parameters for curated drugs not
        # in literature_params. Priority: gold_standard > drugbank.
        if lit is None and not block_lookups:
            try:
                from neogen.core.constants.gold_standard_registry import (
                    get_gold_standard_entry,
                )
                gs_entry = get_gold_standard_entry(name)
                if gs_entry is not None:
                    if not smiles:
                        smiles = gs_entry.get("smiles", "")
                    # Published fu (in-vitro measurement — standard input)
                    if "fu_plasma" not in params and gs_entry.get("fu_plasma"):
                        params["fu_plasma"] = gs_entry["fu_plasma"]
                    # Published F (oral bioavailability)
                    f_val = gs_entry.get("F")
                    if (
                        f_val is not None
                        and 0 < f_val <= 1.0
                        and "f_oral_observed" not in params
                    ):
                        params["f_oral_observed"] = f_val
                    # Published Vd → observed Vss for Kp calibration
                    vd = gs_entry.get("vd_l")
                    if (
                        vd is not None
                        and vd > 0
                        and "vss_observed_L_per_kg" not in params
                    ):
                        # Gold standard Vd is in litres (total), convert
                        # to L/kg assuming 70 kg reference male.
                        params["vss_observed_L_per_kg"] = vd / 70.0
                    # Back-calculate CLint from published CL and fu
                    # using the well-stirred model inversion:
                    #   CLh = Qh × fu × CLint / (Qh + fu × CLint)
                    #   → CLint = CLh × Qh / (fu × (Qh - CLh))
                    # This is standard PBPK practice: gives the engine
                    # the correct Eh, Fh, and systemic clearance without
                    # bypassing first-pass calculation for oral drugs.
                    # MPPGL=40 mg/g, liver~1500g → 60,000 mg protein
                    # CLint(µL/min/mg) = CLint_liver(L/h) × 1e6/60/60000
                    fu_val = gs_entry.get("fu_plasma") or params.get("fu_plasma", 0.5)
                    QH = 90.0  # reference hepatic blood flow L/h
                    GFR_L_H = 7.2  # 120 mL/min × 60/1000

                    # Derive self-consistent CL from AUC/dose/F
                    # AUC-derived CL is preferred because it guarantees
                    # AUC accuracy. Published CL values in the GS dataset
                    # are often inconsistent with AUC/dose.
                    auc_gs = gs_entry.get("auc_mg_h_l")
                    dose_gs = gs_entry.get("dose_mg")
                    f_gs = gs_entry.get("F") or 1.0
                    cl_gs = gs_entry.get("clearance_l_h")

                    cl_sys = None
                    if auc_gs and auc_gs > 0 and dose_gs and dose_gs > 0:
                        # CL/F = dose/AUC → CL_sys = F × dose/AUC
                        cl_sys = f_gs * dose_gs / auc_gs
                    elif cl_gs and cl_gs > 0:
                        cl_sys = cl_gs  # fallback to published CL

                    fe = params.get("fraction_excreted_unchanged", 0.0)
                    primary = gs_entry.get("primary_cyp", "")
                    if primary in ("Renal", "None") and fe < 0.3:
                        fe = 0.7
                        params["fraction_excreted_unchanged"] = fe

                    if (
                        cl_sys is not None
                        and cl_sys > 0
                        and "cl_hepatic_observed" not in params
                        and "clint_microsomal" not in params
                        and "clint_hepatocyte" not in params
                    ):
                        # Subtract renal fraction
                        cl_renal_est = cl_sys * fe
                        clh = cl_sys - cl_renal_est
                        # Cap CLh at 0.95 × Qh
                        clh = max(0.1, min(clh, QH * 0.95))
                        # Invert well-stirred: CLint_liver = CLh*Qh/(fu*(Qh-CLh))
                        denom = fu_val * (QH - clh)
                        if denom > 0:
                            clint_liver = clh * QH / denom  # L/h whole liver
                            # Scale to µL/min/mg microsomal protein
                            # 60,000 mg protein, 60 min/h, 1e6 µL/L
                            clint_micro = clint_liver * 1e6 / 60.0 / 60000.0
                            params["clint_microsomal"] = round(clint_micro, 2)
                    logger.info(
                        "[DrugFactory] Gold standard params for %s", name,
                    )
            except Exception:  # pragma: no cover
                pass

        # Step 2c: DrugBank fallback (1,851 additional drugs beyond GS)
        # Skipped under block_lookups so the holdout cohort cannot inherit
        # measured CL/Vd/F from a curated registry.
        # Provides SMILES + fu + CL + t½ + Vd for drugs not in gold_standard.
        if lit is None and not block_lookups and "clint_microsomal" not in params:
            try:
                from neogen.core.constants.drugbank_registry import (
                    get_drugbank_entry,
                )
                db_entry = get_drugbank_entry(name)
                if db_entry is not None:
                    if not smiles and db_entry.get("smiles"):
                        smiles = db_entry["smiles"]
                    if "fu_plasma" not in params and db_entry.get("fu_plasma"):
                        params["fu_plasma"] = db_entry["fu_plasma"]
                    if "f_oral_observed" not in params and db_entry.get("f_oral"):
                        params["f_oral_observed"] = db_entry["f_oral"]
                    if "vss_observed_L_per_kg" not in params and db_entry.get("vd_l"):
                        params["vss_observed_L_per_kg"] = db_entry["vd_l"] / 70.0
                    # Back-calculate CLint from DrugBank CL + fu.
                    # CRITICAL: DrugBank CL units are mixed — some L/h,
                    # some L/h/kg. Use consistency check with Vd and t½
                    # to detect and normalize.
                    cl_raw = db_entry.get("clearance_l_h")
                    fu_val = db_entry.get("fu_plasma") or params.get("fu_plasma", 0.5)
                    thalf_db = db_entry.get("half_life_h")
                    vd_db = db_entry.get("vd_l")
                    if cl_raw and cl_raw > 0:
                        QH = 90.0
                        cl_sys = cl_raw
                        # Heuristic: if CL < 5 and Vd exists, check if
                        # CL is in L/h/kg by comparing with t½ prediction
                        if cl_raw < 5.0 and thalf_db and thalf_db > 0:
                            cl_check = cl_raw * 70.0  # assume L/h/kg
                            if vd_db and vd_db > 0:
                                thalf_from_lhkg = 0.693 * vd_db / cl_check
                                thalf_from_lh = 0.693 * vd_db / cl_raw
                                # Pick whichever CL gives t½ closer to observed
                                err_lhkg = abs(thalf_from_lhkg - thalf_db) / thalf_db
                                err_lh = abs(thalf_from_lh - thalf_db) / thalf_db
                                if err_lhkg < err_lh:
                                    cl_sys = cl_raw * 70.0  # L/h/kg → L/h
                            else:
                                # No Vd to check — if CL < 2, likely L/h/kg
                                if cl_raw < 2.0:
                                    cl_sys = cl_raw * 70.0

                        # Subtract estimated renal fraction (mirror GS logic)
                        fe = params.get("fraction_excreted_unchanged", 0.0)
                        cl_renal = cl_sys * fe
                        clh = cl_sys - cl_renal
                        clh = max(0.1, min(clh, QH * 0.95))

                        denom = fu_val * (QH - clh)
                        if denom > 0:
                            clint_liver = clh * QH / denom
                            clint_micro = clint_liver * 1e6 / 60.0 / 60000.0
                            # Cap at 500 µL/min/mg (physiological max)
                            if clint_micro <= 500.0:
                                params["clint_microsomal"] = round(clint_micro, 2)
                            # else: fall through to ML prediction
                    logger.info("[DrugFactory] DrugBank params for %s", name)
            except Exception:  # pragma: no cover
                pass

        if not smiles and lit is None:
            raise ValueError(
                f"No SMILES provided and '{name}' not in literature DB. "
                f"Provide a SMILES string."
            )
        params["smiles"] = smiles

        # Step 2b: Measured ADME lookup (gold-standard + DrugBank).
        # For SMILES-only drugs (no literature match), inject measured
        # clinical values before any ML prediction.  This is the
        # highest-confidence data path for novel-SMILES queries that
        # happen to exist in our curated clinical databases.
        if smiles and lit is None:
            try:
                from neogen.ml.adme_lookup import (
                    lookup_fu,
                    lookup_clearance_L_per_h,
                    lookup_half_life_h,
                    lookup_vd_L,
                    lookup_bioavailability,
                )

                # --- fu (fraction unbound) ---
                measured_fu = lookup_fu(smiles, drug_name=name)
                if measured_fu is not None and "fu_plasma" not in params:
                    params["fu_plasma"] = measured_fu
                    logger.info(
                        "[DrugFactory] Measured fu for %s: %.4f", name, measured_fu,
                    )

                # --- CL → back-calculate CLint (or direct override) ---
                measured_cl = lookup_clearance_L_per_h(smiles, drug_name=name)
                if (
                    measured_cl is not None
                    and "clint_microsomal" not in params
                    and "clint_hepatocyte" not in params
                ):
                    _fu = params.get("fu_plasma", measured_fu or 0.5)
                    _Qh = 90.0  # hepatic blood flow L/h (reference male)
                    _fe = params.get("fraction_excreted_unchanged", 0.0)

                    # --- Fix 3: fe-based hepatic/renal CL split ---
                    # When fe > 0, split measured CL into hepatic and
                    # renal components directly, bypassing CLint
                    # back-calculation entirely.
                    if _fe > 0 and measured_cl > 0:
                        params["cl_renal_override_L_per_h"] = measured_cl * _fe
                        params["cl_hepatic_override_L_per_h"] = (
                            measured_cl * (1.0 - _fe)
                        )
                        params["source"] = "measured"
                        logger.info(
                            "[DrugFactory] fe-split CL for %s: "
                            "CL=%.2f L/h, fe=%.2f → CLh=%.2f, CLr=%.2f",
                            name, measured_cl, _fe,
                            measured_cl * (1.0 - _fe),
                            measured_cl * _fe,
                        )

                    # --- Fix 1: CL override for extrahepatic drugs ---
                    # When CL > 0.8 × Qh, back-calculation gives
                    # negative/astronomical CLint. Use direct CL override.
                    elif measured_cl > 0.8 * _Qh:
                        params["cl_total_override_L_per_h"] = measured_cl
                        params["source"] = "measured"
                        logger.info(
                            "[DrugFactory] Extrahepatic CL override for "
                            "%s: CL=%.2f L/h > 0.8×Qh (%.1f)",
                            name, measured_cl, 0.8 * _Qh,
                        )

                    # Standard well-stirred back-calculation
                    elif 0 < measured_cl < _Qh and _fu > 0:
                        # Well-stirred: CL = Qh * fu * CLint / (Qh + fu * CLint)
                        # Solve: CLint = CL * Qh / (fu * (Qh - CL))
                        clint_intrinsic_L_h = (
                            measured_cl * _Qh / (_fu * (_Qh - measured_cl))
                        )
                        # Convert L/h → µL/min/mg microsomal protein
                        # scaling: MPPGL(40) × liver_wt(1500g) × 60 min/h / 1e6
                        _SCALING = 40.0 * 1500.0 * 60.0 / 1e6  # = 3.6
                        clint_micro = clint_intrinsic_L_h / _SCALING
                        if clint_micro > 0:
                            params["clint_microsomal"] = clint_micro
                            params["source"] = "measured"
                            logger.info(
                                "[DrugFactory] Back-calc CLint for %s: "
                                "CL=%.2f L/h → CLint=%.2f µL/min/mg",
                                name, measured_cl, clint_micro,
                            )

                # --- Vd → vss_observed_L_per_kg ---
                measured_vd = lookup_vd_L(smiles, drug_name=name)
                if measured_vd is not None and "vss_observed_L_per_kg" not in params:
                    params["vss_observed_L_per_kg"] = measured_vd / 70.0
                    logger.info(
                        "[DrugFactory] Measured Vd for %s: %.1f L (%.2f L/kg)",
                        name, measured_vd, measured_vd / 70.0,
                    )

                # --- Bioavailability ---
                measured_f = lookup_bioavailability(smiles, drug_name=name)
                if measured_f is not None and "bioavailability" not in params:
                    params["bioavailability"] = measured_f
                    logger.info(
                        "[DrugFactory] Measured F for %s: %.2f", name, measured_f,
                    )

                # --- Fix 2: Measured t½ override ---
                measured_hl = lookup_half_life_h(smiles, drug_name=name)
                if measured_hl is not None:
                    params["half_life_override_h"] = measured_hl
                    logger.info(
                        "[DrugFactory] Measured t½ for %s: %.2f h",
                        name, measured_hl,
                    )

            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "ADME lookup failed for %s: %s", name, exc,
                )

        # Step 3: RDKit properties (if SMILES available)
        if smiles:
            rdkit_props = _compute_rdkit_properties(smiles)
            # RDKit fills gaps — doesn't override literature
            for key, val in rdkit_props.items():
                if key not in params:
                    params[key] = val

        # Step 3b: TDC measured data — fill gaps from Therapeutics Data
        # Commons datasets (AZ Clearance, PPBR, Lombardo Vdss, Obach t½).
        # Priority: literature > gold_standard > TDC > ML > defaults.
        # TDC data is MEASURED in vitro/in vivo — higher quality than ML.
        if smiles and not block_lookups:
            try:
                from neogen.core.constants.tdc_registry import get_tdc_params
                tdc = get_tdc_params(smiles)
                if tdc is not None:
                    # fu from PPBR (measured protein binding)
                    if "fu_plasma" not in params and "fu_plasma_tdc" in tdc:
                        params["fu_plasma"] = tdc["fu_plasma_tdc"]
                    # CLint from microsome assay
                    if (
                        "clint_microsomal" not in params
                        and "clint_hepatocyte" not in params
                        and "clint_microsomal_tdc" in tdc
                    ):
                        params["clint_microsomal"] = tdc["clint_microsomal_tdc"]
                    # CLint from hepatocyte assay (if no microsomal)
                    if (
                        "clint_microsomal" not in params
                        and "clint_hepatocyte" not in params
                        and "clint_hepatocyte_tdc" in tdc
                    ):
                        params["clint_hepatocyte"] = tdc["clint_hepatocyte_tdc"]
                    # Vdss from Lombardo
                    if (
                        "vss_observed_L_per_kg" not in params
                        and "vdss_L_per_kg_tdc" in tdc
                    ):
                        params["vss_observed_L_per_kg"] = tdc["vdss_L_per_kg_tdc"]
                    logger.info(
                        "[DrugFactory] TDC data for %s: %s",
                        name, list(tdc.keys()),
                    )
            except Exception:  # pragma: no cover
                pass

        # Step 3b2: ChEMBL + DrugBank fu (4,600 molecules measured fu)
        # Fill fu gap from the largest protein binding dataset we have.
        # Only for drugs NOT already parameterized by literature/GS/DrugBank.
        _has_curated_params = lit is not None or "clint_microsomal" in params
        if smiles and "fu_plasma" not in params and not _has_curated_params and not block_lookups:
            try:
                from neogen.core.constants.chembl_fu_registry import get_fu
                fu_val = get_fu(smiles)
                if fu_val is not None:
                    params["fu_plasma"] = fu_val
                    logger.info("[DrugFactory] ChEMBL/DB fu for %s: %.3f", name, fu_val)
            except Exception:  # pragma: no cover
                pass

        # Step 3b3: ChEMBL PK aggregated (35K molecules, median CL/t½/Vd)
        # Fill CLint and Vss gaps from ChEMBL's 132K PK measurements.
        # Only for drugs NOT already parameterized by higher-priority sources.
        if smiles and not _has_curated_params and not block_lookups:
            try:
                from neogen.core.constants.chembl_pk_registry import get_chembl_pk
                chembl = get_chembl_pk(smiles)
                if chembl is not None:
                    if (
                        "clint_microsomal" not in params
                        and "clint_hepatocyte" not in params
                        and "clearance_l_h_chembl" in chembl
                    ):
                        # Back-calculate CLint from ChEMBL CL
                        cl_ch = chembl["clearance_l_h_chembl"]
                        fu_ch = params.get("fu_plasma", 0.5)
                        QH = 90.0
                        clh = max(0.1, min(cl_ch, QH * 0.95))
                        denom = fu_ch * (QH - clh)
                        if denom > 0:
                            clint_liver = clh * QH / denom
                            clint_micro = clint_liver * 1e6 / 60.0 / 60000.0
                            params["clint_microsomal"] = round(clint_micro, 2)
                    if "vss_observed_L_per_kg" not in params and "vd_l_chembl" in chembl:
                        params["vss_observed_L_per_kg"] = chembl["vd_l_chembl"] / 70.0
                    logger.info("[DrugFactory] ChEMBL PK for %s: %s", name, list(chembl.keys()))
            except Exception:  # pragma: no cover
                pass

        # Step 3b4: ADME properties (solubility, Caco-2, bioavailability)
        # Used by BCS classification and absorption engine. Safe for all drugs.
        if smiles and not _has_curated_params:
            try:
                from neogen.core.constants.adme_registry import get_adme_params
                adme = get_adme_params(smiles)
                if adme is not None:
                    if "solubility_logs" in adme and "solubility_mg_per_ml" not in params:
                        # logS (mol/L) → mg/mL using MW
                        mw = params.get("molecular_weight", 300.0)
                        import math
                        sol_mol_L = 10 ** adme["solubility_logs"]
                        params["solubility_mg_per_ml"] = sol_mol_L * mw / 1000.0
                    if "caco2_log_papp" in adme and "peff_jejunum" not in params:
                        # log(Papp cm/s) → Peff (cm/s × 10⁴) rough conversion
                        import math
                        papp = 10 ** adme["caco2_log_papp"]
                        params["peff_jejunum"] = papp * 1e4  # scale to 10⁻⁴ cm/s
                    if "lipophilicity_logd" in adme and "logp" not in params:
                        params["logp"] = adme["lipophilicity_logd"]
            except Exception:  # pragma: no cover
                pass

        # Step 3c: ML CLint fallback (only when no literature override).
        # Wires the trained adme_pipeline clint_predictor into the V1
        # engine so novel compounds get a non-zero hepatic clearance
        # instead of CLint = 0. The call is guarded by the missing-key
        # check so literature drugs are never touched.
        if (
            smiles
            and "clint_microsomal" not in params
            and "clint_hepatocyte" not in params
        ):
            try:
                from neogen.ml.clint_retrieval import (
                    predict_clint_retrieval_augmented,
                )

                ra_clint = predict_clint_retrieval_augmented(smiles)
                if ra_clint is not None and ra_clint > 0:
                    params["clint_microsomal"] = ra_clint
                    params.setdefault("source", "ml_predicted")
                    logger.info(
                        "[DrugFactory] Retrieval-augmented CLint for %s: "
                        "%.2f uL/min/mg",
                        name, ra_clint,
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("ML CLint fallback failed for %s: %s", name, exc)

        # Step 3b+: Allometric CL sanity check for SMILES-only drugs.
        # The XGBoost CLint model (R²≈0.53) produces catastrophic
        # outliers for some compounds. Cross-check the predicted CLint
        # against an allometric CL estimate (Mahmood 2006):
        #   CL_allometric = 15 × fu L/h (70 kg reference male)
        # If the well-stirred CL implied by CLint differs >10× from
        # the allometric estimate, shrink CLint toward the allometric-
        # implied value using a geometric mean compromise.
        if (
            smiles
            and "clint_microsomal" in params
            and params.get("source") not in ("literature", "measured")
            and lit is None
        ):
            try:
                _fu = params.get("fu_plasma", 0.5)
                _clint = params["clint_microsomal"]
                # Allometric CL for 70 kg reference male
                cl_allometric = 15.0 * max(_fu, 0.001)  # L/h
                # Scaling constants: MPPGL × liver_weight × unit conversion
                # 40 mg microsomal protein / g liver × 1500 g liver
                # CLint in µL/min/mg → L/h: × 40 × 1500 × 60 / 1e6
                _SCALING = 40.0 * 1500.0 * 60.0 / 1e6  # = 3.6
                clint_allometric_implied = cl_allometric / (
                    max(_fu, 0.001) * _SCALING
                )  # µL/min/mg equivalent
                if clint_allometric_implied > 0 and _clint > 0:
                    log_ratio = abs(math.log10(_clint / clint_allometric_implied))
                    if log_ratio > 1.0:  # >10× difference
                        clint_compromise = math.sqrt(
                            _clint * clint_allometric_implied
                        )
                        logger.info(
                            "[DrugFactory] Allometric CLint sanity check "
                            "for %s: predicted=%.1f, allometric-implied=%.1f"
                            " → compromise=%.1f µL/min/mg",
                            name, _clint, clint_allometric_implied,
                            clint_compromise,
                        )
                        params["clint_microsomal"] = clint_compromise
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "Allometric CLint check failed for %s: %s", name, exc
                )

        # Step 3c: ML renal-elimination fallback.
        # The V1.2 renal_classifier predicts whether a drug is
        # primarily hepatic, mixed, or primarily renal and maps the
        # category to a ``fraction_excreted_unchanged`` estimate.
        # Without this, novel drugs default to fe = 0 which forces
        # 100% of clearance through the hepatic path — catastrophically
        # wrong for renally-eliminated drugs like metformin and valsartan.
        #
        # NOT wired in this session: ``fu_predictor`` and ``pka_predictor``.
        # Their trained accuracy on held-out novel drugs is worse than the
        # existing logP-based / null-default fallbacks (smoke test in
        # session 4 showed fu_v4 producing ceiling 1.0 for highly-bound
        # drugs and pka_basic under-predicting by 3-5 log units for bases).
        # The modules remain in ``neogen.ml`` for opt-in use by callers
        # who want to audit them against their own cohorts.
        if (
            smiles
            and "fraction_excreted_unchanged" not in params
        ):
            try:
                from neogen.ml.renal_predictor import (
                    predict_fraction_excreted_unchanged,
                )

                ml_fe = predict_fraction_excreted_unchanged(smiles)
                if ml_fe is not None:
                    params["fraction_excreted_unchanged"] = ml_fe
                    logger.info(
                        "[DrugFactory] ML-predicted fe for %s: %.2f",
                        name, ml_fe,
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("ML renal fallback failed for %s: %s", name, exc)

        # Step 3d: ML CYP substrate classifier fallback.
        # Predicts CYP isoform fm fractions from the pre-trained
        # multi-label classifier, replacing the hard-coded
        # CYP3A4=0.5/CYP2D6=0.25 defaults for novel compounds.
        if (
            smiles
            and "metabolism" not in params
            and lit is None
        ):
            try:
                from neogen.ml.cyp_classifier import (
                    predict_cyp_metabolism_kwargs,
                )

                cyp_kwargs = predict_cyp_metabolism_kwargs(smiles)
                if cyp_kwargs:
                    params["metabolism"] = CYPMetabolismProfile(**cyp_kwargs)
                    logger.info(
                        "[DrugFactory] ML-predicted CYP profile for %s: %s",
                        name,
                        {k: round(v, 3) for k, v in cyp_kwargs.items()},
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "ML CYP classifier fallback failed for %s: %s", name, exc
                )

        # Step 4: Derived properties
        if "bcs_class" not in params:
            params["bcs_class"] = _classify_bcs(
                params.get("solubility_mg_per_ml"),
                params.get("tpsa", 80.0),
                params.get("logp", 2.0),
            )

        if "ionization_class" not in params:
            params["ionization_class"] = _classify_ionization(
                params.get("pka_acidic"),
                params.get("pka_basic"),
            )

        # Step 4b: ML fu prediction (v6 trained model), fall back to logP.
        if "fu_plasma" not in params and smiles:
            try:
                from neogen.ml.fu_predictor_v6 import predict_fu

                ml_fu = predict_fu(smiles)
                if ml_fu is not None:
                    params["fu_plasma"] = ml_fu
                    logger.info(
                        "[DrugFactory] ML-predicted fu for %s: %.4f",
                        name, ml_fu,
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("ML fu fallback failed for %s: %s", name, exc)

        if "fu_plasma" not in params:
            params["fu_plasma"] = _estimate_fu_from_logp(
                params.get("logp", 2.0)
            )

        # Step 4c: ML Vdss prediction — store as vss_observed_L_per_kg
        # so TissuePartitioner can calibrate Kp scaling.
        if "vss_observed_L_per_kg" not in params and smiles:
            try:
                from neogen.ml.vd_predictor import predict_vdss_L_per_kg

                ml_vd = predict_vdss_L_per_kg(smiles)
                if ml_vd is not None:
                    params["vss_observed_L_per_kg"] = ml_vd
                    logger.info(
                        "[DrugFactory] ML-predicted Vdss for %s: %.2f L/kg",
                        name, ml_vd,
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("ML Vd fallback failed for %s: %s", name, exc)

        # Step 4d: Ensemble uncertainty quantification for SMILES-only drugs.
        # When ensemble models (bootstrap, 10 seeds) are available, compute
        # per-property confidence scores and aggregate into a single
        # prediction_confidence value on the Drug entity. Ensemble mean
        # values are used as the prediction ONLY when the single-model
        # prediction has not already been set (i.e., ensembles are an
        # enhancement layer, not a replacement for the primary predictor).
        if smiles and lit is None:
            try:
                from neogen.ml.ensemble_predictor import (
                    get_fu_ensemble,
                    get_clint_ensemble,
                    get_vd_ensemble,
                )

                confidence_scores = []

                # fu ensemble
                fu_ens = get_fu_ensemble()
                if fu_ens.is_available():
                    fu_pred = fu_ens.predict(smiles)
                    if fu_pred is not None:
                        confidence_scores.append(fu_pred.confidence)
                        # Use ensemble mean if no fu was set yet
                        if "fu_plasma" not in params:
                            params["fu_plasma"] = fu_pred.mean
                        logger.info(
                            "[DrugFactory] fu ensemble for %s: %.4f +/- %.4f "
                            "(confidence=%.2f, n=%d)",
                            name, fu_pred.mean, fu_pred.std,
                            fu_pred.confidence, fu_pred.n_models,
                        )

                # CLint ensemble
                clint_ens = get_clint_ensemble()
                if clint_ens.is_available():
                    clint_pred = clint_ens.predict(smiles)
                    if clint_pred is not None:
                        confidence_scores.append(clint_pred.confidence)
                        if (
                            "clint_microsomal" not in params
                            and "clint_hepatocyte" not in params
                        ):
                            params["clint_microsomal"] = clint_pred.mean
                            params.setdefault("source", "ml_predicted")
                        logger.info(
                            "[DrugFactory] CLint ensemble for %s: %.2f +/- %.2f "
                            "(confidence=%.2f, n=%d)",
                            name, clint_pred.mean, clint_pred.std,
                            clint_pred.confidence, clint_pred.n_models,
                        )

                # Vd ensemble
                vd_ens = get_vd_ensemble()
                if vd_ens.is_available():
                    vd_pred = vd_ens.predict(smiles)
                    if vd_pred is not None:
                        confidence_scores.append(vd_pred.confidence)
                        if "vss_observed_L_per_kg" not in params:
                            params["vss_observed_L_per_kg"] = vd_pred.mean
                        logger.info(
                            "[DrugFactory] Vd ensemble for %s: %.2f +/- %.2f "
                            "(confidence=%.2f, n=%d)",
                            name, vd_pred.mean, vd_pred.std,
                            vd_pred.confidence, vd_pred.n_models,
                        )

                # Aggregate: geometric mean of per-property confidences
                if confidence_scores:
                    import math
                    agg_confidence = math.exp(
                        sum(math.log(max(c, 1e-10)) for c in confidence_scores)
                        / len(confidence_scores)
                    )
                    params["prediction_confidence"] = round(agg_confidence, 4)
                    if agg_confidence < 0.3:
                        logger.warning(
                            "[DrugFactory] Low confidence prediction for %s: "
                            "%.3f (based on %d properties)",
                            name, agg_confidence, len(confidence_scores),
                        )
                    else:
                        logger.info(
                            "[DrugFactory] Prediction confidence for %s: %.3f",
                            name, agg_confidence,
                        )

            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "Ensemble UQ failed for %s: %s", name, exc
                )

        # Step 4e: Phase II metabolism prediction (SMARTS-based).
        # For SMILES-only drugs with no literature Phase II data, detect
        # UGT/SULT-susceptible functional groups and assign conservative
        # default fm fractions so downstream models don't force 100% CYP.
        if (
            smiles
            and "phase2_metabolism" not in params
            and lit is None
        ):
            try:
                from neogen.ml.phase2_predictor import predict_phase2_profile

                p2_profile = predict_phase2_profile(smiles)
                if p2_profile is not None:
                    params["phase2_metabolism"] = p2_profile
                    logger.info(
                        "[DrugFactory] Phase II prediction for %s: "
                        "total_fm=%.3f",
                        name, p2_profile.total_phase2_fm,
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "Phase II prediction failed for %s: %s", name, exc
                )

        # Step 4f: PK consistency check (cross-validate CL, Vd, t1/2).
        # When all three ADME properties are independently predicted,
        # verify that t1/2 = 0.693 * Vd / CL holds within 5-fold.
        # If not, reconcile using the geometric mean.
        if (
            smiles
            and lit is None
            and params.get("source") not in ("literature", "measured")
            and "half_life_override_h" not in params
        ):
            try:
                from neogen.ml.consistency_checker import reconcile_pk_params

                # Estimate CL from CLint via well-stirred model
                _clint = params.get("clint_microsomal")
                _fu = params.get("fu_plasma", 0.5)
                _Qh = 90.0  # hepatic blood flow L/h
                _vd_L_per_kg = params.get("vss_observed_L_per_kg")

                cl_est = None
                if _clint is not None and _clint > 0 and _fu > 0:
                    _SCALING = 40.0 * 1500.0 * 60.0 / 1e6  # = 3.6
                    fu_clint = _fu * _clint * _SCALING
                    cl_est = _Qh * fu_clint / (_Qh + fu_clint)

                vd_est = _vd_L_per_kg * 70.0 if _vd_L_per_kg else None

                # Get direct t1/2 prediction if available
                t_half_direct = None
                if cl_est and vd_est and cl_est > 0:
                    t_half_direct = 0.693 * vd_est / cl_est

                # Only run check if we have a half-life predictor result
                if cl_est and vd_est:
                    try:
                        from neogen.ml.half_life_predictor import (
                            predict_half_life_h,
                        )
                        hl_pred = predict_half_life_h(smiles)
                        if hl_pred is not None and hl_pred > 0:
                            overrides_from_check = reconcile_pk_params(
                                cl_L_per_h=cl_est,
                                vd_L=vd_est,
                                half_life_h=hl_pred,
                            )
                            if overrides_from_check:
                                params.update(overrides_from_check)
                                logger.info(
                                    "[DrugFactory] PK consistency "
                                    "reconciliation for %s: %s",
                                    name, overrides_from_check,
                                )
                    except Exception:  # pragma: no cover
                        pass  # half_life_predictor may not be available
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "PK consistency check failed for %s: %s", name, exc
                )

        # Step 5: User overrides (highest priority)
        if overrides:
            params.update(overrides)
            logger.info(
                f"[DrugFactory] Applied user overrides: {list(overrides.keys())}"
            )

        # Step 6: Set name
        params["name"] = name

        # Step 7: Determine source
        if overrides:
            params["source"] = "measured"
        elif lit is not None:
            params["source"] = params.get("source", "literature")
        else:
            params["source"] = "predicted"

        # Step 8: Ensure defaults for required fields
        params.setdefault("molecular_weight", 300.0)
        params.setdefault("logp", 2.0)
        params.setdefault("tpsa", 80.0)
        params.setdefault("hbd", 1)
        params.setdefault("hba", 3)
        params.setdefault("rotatable_bonds", 3)
        params.setdefault("aromatic_rings", 1)
        params.setdefault("fu_plasma", 0.5)
        params.setdefault("blood_to_plasma_ratio", 1.0)
        params.setdefault("fraction_excreted_unchanged", 0.0)
        params.setdefault("metabolism", CYPMetabolismProfile())
        params.setdefault("phase2_metabolism", PhaseIIMetabolismProfile())
        params.setdefault("inhibition", InhibitionProfile())
        params.setdefault("induction", InductionProfile())
        params.setdefault("transporters", TransporterAffinities())

        # Step 9: Filter to Drug dataclass fields only
        valid_fields = {f.name for f in fields(Drug)}
        filtered = {k: v for k, v in params.items() if k in valid_fields}

        drug = Drug(**filtered)
        logger.info(f"[DrugFactory] Created: {drug}")
        return drug

    def create_from_smiles(self, smiles: str,
                           name: Optional[str] = None) -> Drug:
        """Create a Drug entity from SMILES alone (no literature lookup).

        For novel compounds where no literature data exists.
        All parameters come from RDKit + ML prediction.
        """
        if name is None:
            name = f"compound_{hash(smiles) % 10000:04d}"
        return self.create(name=name, smiles=smiles)

    def create_pair(
        self,
        perpetrator_name: str,
        victim_name: str,
        perpetrator_smiles: Optional[str] = None,
        victim_smiles: Optional[str] = None,
    ) -> tuple[Drug, Drug]:
        """Create a DDI drug pair.

        Convenience method for DDI simulations.
        Returns (perpetrator, victim) Drug tuple.
        """
        perpetrator = self.create(perpetrator_name, smiles=perpetrator_smiles)
        victim = self.create(victim_name, smiles=victim_smiles)
        return perpetrator, victim
