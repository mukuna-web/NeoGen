"""
Hepatic Clearance — V1 Engine
==============================
Well-stirred liver model for hepatic extraction and clearance.

The liver is the primary site of drug metabolism. This module computes:
  1. Hepatic extraction ratio (Eh)
  2. Hepatic clearance (CLh)
  3. First-pass hepatic availability (fh = 1 − Eh)

Well-Stirred Model (most widely used in PBPK):
  Eh = fu × CLint / (Qh + fu × CLint)

Where:
  fu     = fraction unbound in plasma
  CLint  = intrinsic clearance (mL/min/mg or scaled to L/h)
  Qh     = hepatic blood flow (L/h)

Scaling from in vitro CLint:
  CLint_scaled (L/h) = CLint_microsomal (µL/min/mg) × MPPGL × liver_weight(g) × 60/1e6

References:
  Rowland M et al. J Pharmacokinet Biopharm 1973; 1:123-136
  Pang KS, Rowland M. J Pharmacokinet Biopharm 1977; 5:625-653
  Ito K, Houston JB. Pharm Res 2005; 22:103-112
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from neogen.core.entities.drug import Drug
from neogen.core.entities.physiology import PhysiologyModel
from neogen.engine.ddi.transporter_ddi import (
    DEFAULT_OATP1B1_UPTAKE_FACTOR,
    OATP1B1InhibitorExposure,
    compute_oatp1b1_uptake_factor,
)

logger = logging.getLogger(__name__)

# Reference scaling constants
MPPGL_REFERENCE = 40.0   # mg microsomal protein per gram liver
LIVER_DENSITY = 1.08     # g/mL


@dataclass(frozen=True)
class HepaticClearanceResult:
    """Immutable result of hepatic clearance calculation."""

    # Scaled intrinsic clearance
    clint_scaled_L_per_h: float

    # Hepatic blood flow used
    qh_L_per_h: float

    # Extraction ratio (0–1)
    extraction_ratio: float

    # Systemic hepatic clearance (L/h)
    cl_hepatic_L_per_h: float

    # First-pass hepatic availability (fh = 1 − Eh)
    fh: float

    # Fraction unbound used
    fu_plasma: float

    # Model used
    model: str = "well_stirred"

    # V1.2 Track B: active OATP1B1 uptake amplification actually applied.
    # 1.0 for non-substrates. Equals the drug's baseline uptake factor
    # (default 15.0) when no OATP1B1 inhibitors are supplied — matches
    # V1.1 byte-for-byte. Falls toward 1.0 as perpetrator [I]/IC50 rises.
    oatp1b1_uptake_factor: float = 1.0

    @property
    def half_life_hepatic_h(self) -> float:
        """Hepatic contribution to elimination half-life.
        Approximate: t½ ≈ 0.693 × Vd / CL (needs Vd from outside)."""
        return 0.0  # Requires Vd — set by orchestrator


class HepaticClearanceEngine:
    """
    Computes hepatic clearance using the well-stirred liver model.

    Consumes V1 frozen entities directly.

    Usage:
        engine = HepaticClearanceEngine()
        result = engine.compute(drug, physiology)
        print(result.extraction_ratio, result.cl_hepatic_L_per_h, result.fh)
    """

    def compute(
        self,
        drug: Drug,
        physiology: PhysiologyModel,
        oatp1b1_inhibitors: Optional[Sequence[OATP1B1InhibitorExposure]] = None,
    ) -> HepaticClearanceResult:
        """
        Compute hepatic clearance from drug CLint and physiology.

        Steps:
          1. Scale CLint from in vitro → whole liver (L/h)
          2. Apply well-stirred model: Eh = fu×CLint / (Qh + fu×CLint)
          3. CLh = Qh × Eh
          4. fh = 1 − Eh

        Args:
            drug: V1 Drug with clint_microsomal or clint_hepatocyte, fu_plasma
            physiology: V1 PhysiologyModel with liver compartment
            oatp1b1_inhibitors: Optional V1.2 Track B OATP1B1 perpetrator
                exposures. When None or empty, the result is byte-identical
                to V1.1 (flat baseline uptake factor applied). When
                populated, uptake is suppressed per
                :func:`compute_oatp1b1_uptake_factor`.

        Returns:
            Frozen HepaticClearanceResult
        """
        fu = drug.fu_plasma_safe  # ≥ 0.001

        # ── Step 0: Observed CLh override (non-CYP drugs) ──
        # For paracetamol (UGT/SULT), aspirin (esterase), digoxin (P-gp),
        # microsomal CLint dramatically under-represents actual hepatic CL.
        # Use the clinical CLh directly when available.
        if drug.cl_hepatic_observed is not None and drug.cl_hepatic_observed > 0:
            qh = self._hepatic_blood_flow(physiology)
            cl_h = min(drug.cl_hepatic_observed, qh * 0.99)
            eh = cl_h / qh if qh > 0 else 0.0
            fh = 1.0 - eh
            logger.info(
                "Hepatic CL: using observed CLh=%.2f L/h (Eh=%.3f, fh=%.3f)",
                cl_h, eh, fh,
            )
            return HepaticClearanceResult(
                clint_scaled_L_per_h=0.0,
                qh_L_per_h=round(qh, 2),
                extraction_ratio=round(eh, 4),
                cl_hepatic_L_per_h=round(cl_h, 4),
                fh=round(fh, 4),
                fu_plasma=fu,
            )

        # ── Step 1: Scale CLint to whole-liver L/h ──
        clint_scaled = self._scale_clint(drug, physiology)

        # ── Step 1b: OATP1B1 hepatic uptake enhancement ──
        # For OATP1B1 substrates (statins, valsartan, repaglinide), active
        # sinusoidal uptake raises the effective CLint by 10-30× because
        # hepatocyte drug concentrations exceed plasma concentrations at
        # steady state. This is a well-established SCCM (extended clearance)
        # correction — without it, single-drug PK for statins is dramatically
        # overpredicted (F and AUC both ~10× too high).
        #
        # V1.1: flat 15× amplification for every OATP1B1 substrate.
        # V1.2 Track B: per-drug baseline factor from
        #   ``TransporterAffinities.oatp1b1_uptake_factor`` (falls back to
        #   15.0), modulated by co-administered OATP1B1 inhibitor IC50s.
        #   When no inhibitors are supplied the baseline factor is
        #   returned unchanged — V1.1 parity.
        #
        # Reference: Shitara et al. 2006, Kusuhara 2013, Varma 2012.
        uptake_factor = compute_oatp1b1_uptake_factor(
            drug, oatp1b1_inhibitors or ()
        )
        if uptake_factor > 1.0:
            clint_scaled = clint_scaled * uptake_factor
            if oatp1b1_inhibitors:
                logger.info(
                    "Hepatic CL: OATP1B1 substrate — CLint × %.2f "
                    "(modulated from base by %d inhibitor(s))",
                    uptake_factor,
                    len(oatp1b1_inhibitors),
                )
            else:
                logger.info(
                    "Hepatic CL: OATP1B1 substrate — CLint × %.1f",
                    uptake_factor,
                )

        # ── Step 1c: Phase II metabolism contribution ──
        # Phase II enzymes (UGT, SULT, aldehyde oxidase) metabolize drug
        # in the same hepatocytes as CYP enzymes. When a drug has both
        # CYP and Phase II metabolism, the total hepatic CLint should
        # include both contributions.
        #
        # Approximation: Phase II enzymes share the same in-vivo scaling
        # as CYP enzymes (MPPGL-based). We scale the CYP-derived CLint
        # by the ratio of total enzymatic fm to CYP-only fm:
        #   CLint_total = CLint_CYP × (fm_CYP + fm_Phase2) / fm_CYP
        #
        # This is a first-order approximation. More accurate would be
        # separate UGT microsomal scaling (MPPGL × UGT abundance), but
        # that requires UGT-specific in-vitro CLint data we don't have
        # for most drugs.
        phase2_fm = drug.phase2_metabolism.total_phase2_fm
        if phase2_fm > 0 and clint_scaled > 0:
            cyp_fm = drug.metabolism.total_cyp_fm
            total_fm = cyp_fm + phase2_fm
            if total_fm > 0 and cyp_fm > 0:
                phase2_scaling = total_fm / cyp_fm
                clint_scaled *= phase2_scaling
                logger.info(
                    "Hepatic CL: Phase II scaling for %s — "
                    "fm_CYP=%.2f, fm_Phase2=%.2f → CLint × %.2f",
                    drug.name, cyp_fm, phase2_fm, phase2_scaling,
                )
            elif cyp_fm == 0:
                # Drug has Phase II metabolism but no CYP metabolism.
                # CLint from in-vitro assays may already include Phase II
                # (e.g., hepatocyte CLint captures all pathways). No
                # scaling needed — the existing CLint is the total.
                logger.info(
                    "Hepatic CL: Phase II-only metabolism for %s "
                    "(fm_Phase2=%.2f, no CYP) — CLint used as-is",
                    drug.name, phase2_fm,
                )

        # ── Step 2: Hepatic blood flow ──
        # Qh = hepatic artery + portal vein (gut + spleen + pancreas)
        qh = self._hepatic_blood_flow(physiology)

        # ── Step 3: Well-stirred model ──
        # Eh = fu × CLint / (Qh + fu × CLint)
        fu_clint = fu * clint_scaled
        extraction_ratio = fu_clint / (qh + fu_clint) if (qh + fu_clint) > 0 else 0.0

        # Physiological bound: extraction can't exceed 1.0
        extraction_ratio = min(extraction_ratio, 0.99)

        # ── Step 4: Hepatic clearance and availability ──
        cl_hepatic = qh * extraction_ratio
        fh = 1.0 - extraction_ratio

        logger.info(
            "Hepatic CL: CLint_scaled=%.2f L/h, Qh=%.1f L/h, "
            "Eh=%.3f, CLh=%.2f L/h, fh=%.3f",
            clint_scaled, qh, extraction_ratio, cl_hepatic, fh,
        )

        return HepaticClearanceResult(
            clint_scaled_L_per_h=round(clint_scaled, 4),
            qh_L_per_h=round(qh, 2),
            extraction_ratio=round(extraction_ratio, 4),
            cl_hepatic_L_per_h=round(cl_hepatic, 4),
            fh=round(fh, 4),
            fu_plasma=fu,
            oatp1b1_uptake_factor=round(uptake_factor, 4),
        )

    def compute_first_pass_gut(
        self,
        drug: Drug,
        physiology: PhysiologyModel,
    ) -> float:
        """
        Estimate gut wall first-pass availability (fg) using the Qgut model.

        Qgut model (Yang et al. 2007):
          fg = Qgut / (Qgut + fu_gut × CLint_gut)

        Where:
          Qgut = Qvilli × fu_gut / (Qvilli + fu_gut × CLperm)
          Simplified: Qgut ≈ gut blood flow for well-absorbed drugs

        For drugs without gut CLint data, fg is estimated from CYP3A4
        gut abundance relative to liver.
        """
        fu_gut = drug.fu_gut_effective

        # Gut blood flow (villous flow)
        q_gut = physiology.gut.blood_flow_L_per_h

        # Estimate gut wall CLint from liver CLint × (gut CYP3A4 / liver CYP3A4)
        # CYP3A4 gut:liver abundance ratio ≈ 1.46/137 ≈ 0.0107
        # But scaled by gut weight (~0.4 kg) vs liver weight (~1.8 kg)
        gut_liver_ratio = (1.46 / 137.0) * (400.0 / 1800.0)  # ≈ 0.0024

        clint_liver = self._scale_clint(drug, physiology)
        fm_3a4 = drug.metabolism.cyp3a4

        # Only CYP3A4 is meaningfully present in gut wall
        clint_gut = clint_liver * fm_3a4 * gut_liver_ratio

        # Qgut model
        fu_gut_clint = fu_gut * clint_gut
        fg = q_gut / (q_gut + fu_gut_clint) if (q_gut + fu_gut_clint) > 0 else 1.0

        # Bound: fg between 0.01 and 1.0
        fg = max(0.01, min(fg, 1.0))

        logger.info("Gut first-pass: fg=%.3f (CLint_gut=%.3f, fm_3a4=%.2f)", fg, clint_gut, fm_3a4)
        return round(fg, 4)

    # ── Internal helpers ──

    @staticmethod
    def _scale_clint(drug: Drug, physiology: PhysiologyModel) -> float:
        """
        Scale in-vitro CLint to whole-liver intrinsic clearance (L/h).

        Priority:
          0. Michaelis-Menten (Km/Vmax) → CLint at [S]=Km (saturable)
          1. CLint from hepatocytes (µL/min/10^6 cells) → scale by HPGL
          2. CLint from microsomes (µL/min/mg protein) → scale by MPPGL
          3. Default to 0 (no hepatic metabolism)

        Scaling:
          CLint_liver (µL/min) = CLint × scaling_factor × liver_weight(g)
          CLint_liver (L/h) = CLint_liver (µL/min) × 60 / 1e6
        """
        liver = physiology.liver
        liver_weight_g = liver.volume_L * 1000.0 * LIVER_DENSITY

        # ── Michaelis-Menten saturable metabolism ──
        # When Km and Vmax are set, use CLint = Vmax/(2*Km) which
        # represents the effective CLint at therapeutic concentrations
        # near Km (a ~50% reduction from the linear Vmax/Km value).
        # This captures non-linear kinetics for drugs like omeprazole
        # (CYP2C19 Km≈4 µM) and theophylline (CYP1A2 Km≈40 µM).
        if drug.km_um is not None and drug.vmax_pmol_per_min_per_mg is not None:
            mppgl = liver.mppgl if liver.mppgl is not None else MPPGL_REFERENCE
            # CLint at [S]=Km: Vmax/(Km+Km) = Vmax/(2*Km)
            # Units: pmol/min/mg → µL/min/mg (divide by 1000 for volume)
            # Actually Vmax/(2*Km) has units of (pmol/min/mg)/(µM)
            # = (pmol/min/mg)/(nmol/mL) = (pmol·mL)/(min·mg·nmol)
            # = (1e-12 mol · 1e-3 L)/(min · mg · 1e-9 mol)
            # = 1e-6 L/min/mg = µL/min/mg ✓
            clint_mm = drug.vmax_pmol_per_min_per_mg / (2.0 * drug.km_um)
            clint_total = clint_mm * mppgl * liver_weight_g
            clint_scaled = clint_total * 60.0 / 1e6  # → L/h
            logger.info(
                "Michaelis-Menten CLint: Vmax=%.1f, Km=%.1f µM, "
                "CLint_eff=%.2f µL/min/mg → %.2f L/h",
                drug.vmax_pmol_per_min_per_mg, drug.km_um,
                clint_mm, clint_scaled,
            )
            return clint_scaled

        if drug.clint_hepatocyte is not None and drug.clint_hepatocyte > 0:
            # Hepatocyte CLint: µL/min/10^6 cells
            hpgl = liver.hpgl if liver.hpgl is not None else 120.0
            clint_total = drug.clint_hepatocyte * hpgl * liver_weight_g
            return clint_total * 60.0 / 1e6  # → L/h

        if drug.clint_microsomal is not None and drug.clint_microsomal > 0:
            # Microsomal CLint: µL/min/mg protein
            mppgl = liver.mppgl if liver.mppgl is not None else MPPGL_REFERENCE
            clint_total = drug.clint_microsomal * mppgl * liver_weight_g
            return clint_total * 60.0 / 1e6  # → L/h

        logger.warning("No CLint data for %s — hepatic clearance will be 0", drug.name)
        return 0.0

    @staticmethod
    def _hepatic_blood_flow(physiology: PhysiologyModel) -> float:
        """
        Total hepatic blood flow = hepatic artery + portal vein.
        Portal vein = gut + spleen + pancreas flows.

        In PhysiologyModel, liver.blood_flow_L_per_h already includes
        the total hepatic flow (artery + portal). Use it directly.
        """
        return physiology.liver.blood_flow_L_per_h
