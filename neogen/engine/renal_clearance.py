"""
Renal Clearance — V1 Engine
=============================
Mechanistic renal clearance using three physiological components:

  CLrenal = (CLfiltration + CLsecretion) × (1 − freabsorption)

1. Glomerular Filtration (passive)
   CLfilt = GFR × fu × sieving_coefficient
   Only free drug passes the glomerular membrane.

2. Tubular Secretion (active transport)
   Michaelis-Menten kinetics via OAT1/OAT3 (anions) or OCT2/MATE1 (cations).
   CLsec = Vmax / (Km + C)

3. Tubular Reabsorption (passive, pH-dependent)
   Henderson-Hasselbalch: only unionized drug is reabsorbed from urine.
   Acidic urine traps basic drugs; alkaline urine traps acidic drugs.

References:
  Masereeuw R, Russel FGM. Drug Metab Rev 2001; 33:299-351
  Morrissey KM et al. Annu Rev Pharmacol Toxicol 2013; 53:503-529
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional

from neogen.core.entities.drug import Drug, Ionization
from neogen.core.entities.physiology import PhysiologyModel

logger = logging.getLogger(__name__)

# Reference GFR (healthy adult)
REFERENCE_GFR_ML_PER_MIN = 120.0
REFERENCE_GFR_L_PER_H = REFERENCE_GFR_ML_PER_MIN * 60.0 / 1000.0  # 7.2 L/h

# Urine reference values
REFERENCE_URINE_PH = 6.0
REFERENCE_URINE_FLOW_ML_PER_MIN = 1.0


@dataclass(frozen=True)
class RenalClearanceResult:
    """Immutable result of mechanistic renal clearance calculation."""

    # Components (L/h)
    cl_filtration_L_per_h: float
    cl_secretion_L_per_h: float
    cl_reabsorbed_L_per_h: float

    # Net clearance (L/h)
    cl_renal_net_L_per_h: float

    # Fractions
    reabsorption_fraction: float
    fraction_filtered: float
    fraction_secreted: float

    # Parameters used
    gfr_ml_per_min: float
    urine_ph: float
    fu_plasma: float

    # Transport mechanism(s)
    transport: str  # "passive_only", "OAT1_OAT3", "OCT2_MATE1", "PGP", etc.

    # Whether cl_renal_override_L_per_h was used (bypasses model)
    is_override: bool = False


class RenalClearanceEngine:
    """
    Computes mechanistic renal clearance from V1 Drug entity.

    Usage:
        engine = RenalClearanceEngine()
        result = engine.compute(drug, physiology)
        print(result.cl_renal_net_L_per_h)
    """

    def compute(
        self,
        drug: Drug,
        physiology: PhysiologyModel,
        gfr_ml_per_min: float = REFERENCE_GFR_ML_PER_MIN,
        urine_ph: float = REFERENCE_URINE_PH,
        urine_flow_ml_per_min: float = REFERENCE_URINE_FLOW_ML_PER_MIN,
        plasma_conc_mg_per_L: float = 1.0,
    ) -> RenalClearanceResult:
        """
        Compute net renal clearance for a drug.

        For drugs with known fe (fraction excreted unchanged), we can
        also cross-validate: CLrenal_expected ≈ fe × CLtotal.

        Args:
            drug: V1 Drug entity
            physiology: V1 PhysiologyModel (for renal blood flow)
            gfr_ml_per_min: Glomerular filtration rate (default 120)
            urine_ph: Urine pH (default 6.0)
            urine_flow_ml_per_min: Urine flow rate (default 1.0)
            plasma_conc_mg_per_L: Plasma concentration for M-M secretion

        Returns:
            Frozen RenalClearanceResult
        """
        fu = drug.fu_plasma_safe
        mw = drug.molecular_weight

        # ── 0. Direct CLr override ──
        # When cl_renal_override_L_per_h is set on the Drug entity,
        # return it directly without running the mechanistic model.
        if drug.cl_renal_override_L_per_h is not None:
            cl_override = drug.cl_renal_override_L_per_h
            logger.info(
                "Renal CL: using override CLr=%.2f L/h", cl_override,
            )
            return RenalClearanceResult(
                cl_filtration_L_per_h=0.0,
                cl_secretion_L_per_h=0.0,
                cl_reabsorbed_L_per_h=0.0,
                cl_renal_net_L_per_h=round(cl_override, 4),
                reabsorption_fraction=0.0,
                fraction_filtered=0.0,
                fraction_secreted=0.0,
                gfr_ml_per_min=gfr_ml_per_min,
                urine_ph=urine_ph,
                fu_plasma=fu,
                transport="override",
                is_override=True,
            )

        # ── 1. Glomerular Filtration ──
        cl_filt = self._filtration_clearance(gfr_ml_per_min, fu, mw)

        # ── 2. Tubular Secretion ──
        # Intact-nephron hypothesis: in CKD, tubular secretion capacity
        # declines proportionally to functional nephron mass (tracked by
        # GFR). At reference GFR (120) the factor is 1.0, so legacy
        # behavior is preserved. In advanced CKD (GFR=30), secretion is
        # scaled to 0.25× reference — matching observed metformin AUC
        # increases and the approach used by Simcyp/PK-Sim.
        cl_sec_ref, transport = self._secretion_clearance(
            drug, plasma_conc_mg_per_L
        )
        nephron_fraction = max(gfr_ml_per_min / REFERENCE_GFR_ML_PER_MIN, 0.0)
        cl_sec = cl_sec_ref * nephron_fraction

        # ── 3. Tubular Reabsorption ──
        reab_frac = self._reabsorption_fraction(drug, urine_ph, urine_flow_ml_per_min)

        # ── Net Clearance ──
        cl_gross = cl_filt + cl_sec
        cl_reabsorbed = cl_gross * reab_frac
        cl_net = max(cl_gross - cl_reabsorbed, 0.0)

        # Bound: renal clearance can't exceed renal blood flow
        q_kidney = physiology.kidney.blood_flow_L_per_h
        cl_net = min(cl_net, q_kidney * fu)

        frac_filt = cl_filt / cl_gross if cl_gross > 0 else 1.0
        frac_sec = cl_sec / cl_gross if cl_gross > 0 else 0.0

        logger.info(
            "Renal CL: filt=%.2f, sec=%.2f, reab_frac=%.2f, net=%.2f L/h (%s)",
            cl_filt, cl_sec, reab_frac, cl_net, transport,
        )

        return RenalClearanceResult(
            cl_filtration_L_per_h=round(cl_filt, 4),
            cl_secretion_L_per_h=round(cl_sec, 4),
            cl_reabsorbed_L_per_h=round(cl_reabsorbed, 4),
            cl_renal_net_L_per_h=round(cl_net, 4),
            reabsorption_fraction=round(reab_frac, 4),
            fraction_filtered=round(frac_filt, 4),
            fraction_secreted=round(frac_sec, 4),
            gfr_ml_per_min=gfr_ml_per_min,
            urine_ph=urine_ph,
            fu_plasma=fu,
            transport=transport,
        )

    # ── Component calculations ──

    @staticmethod
    def _filtration_clearance(gfr_ml_per_min: float, fu: float, mw: float) -> float:
        """
        CLfilt = GFR × fu × sieving_coefficient

        Sieving: MW < 10 kDa → 1.0, MW 10-60 kDa → linear decline, MW > 60 kDa → 0.
        """
        gfr_L_per_h = gfr_ml_per_min * 60.0 / 1000.0

        if mw < 10_000:
            sieving = 1.0
        elif mw < 60_000:
            sieving = max(0.0, 1.0 - (mw - 10_000) / 50_000)
        else:
            sieving = 0.0

        return gfr_L_per_h * fu * sieving

    @staticmethod
    def _secretion_clearance(
        drug: Drug,
        plasma_conc: float,
    ) -> tuple[float, str]:
        """
        Active tubular secretion via OAT or OCT transporters.

        Priority order:
          1. Explicit transporter substrate flags on ``drug.transporters`` —
             high-capacity secretion (e.g. metformin, cimetidine).
          2. Fallback heuristic: ionization class + fe_unchanged — catches
             drugs where the substrate flags are not yet populated.

        Michaelis-Menten saturation is applied at high plasma concentrations
        using a per-transporter Km.
        """
        transport = "passive_only"
        cl_sec = 0.0
        km_mg_per_L = 100.0

        transporters = drug.transporters

        # ── Explicit substrate flags (preferred) ──
        if getattr(transporters, "is_oct2_substrate", False):
            # OCT2 + MATE1 basolateral/apical pair. Typical high-capacity
            # values for metformin, cimetidine, pindolol: 20-35 L/h.
            cl_sec = 25.0
            transport = "OCT2_MATE1"
            # OCT2 Km for metformin ≈ 1-2 mM ≈ 170 mg/L — plasma rarely saturates.
            km_mg_per_L = 200.0

        elif getattr(transporters, "is_oat1_substrate", False) or getattr(
            transporters, "is_oat3_substrate", False
        ):
            # OAT1/OAT3 anionic secretion. Representative drugs: para-
            # aminohippurate (CL ≈ 500 mL/min ≈ 30 L/h), furosemide,
            # methotrexate. Use a moderate default.
            cl_sec = 20.0
            transport = "OAT1_OAT3"
            km_mg_per_L = 100.0

        # ── P-gp mediated apical efflux in renal proximal tubules ──
        # P-gp (ABCB1/MDR1) is expressed on the apical (luminal) membrane
        # of renal proximal tubule cells and actively secretes substrates
        # into urine. This is the dominant secretion pathway for digoxin
        # (Drescher 2003, Tanigawara 1992, Fromm 1999).
        #
        # Priority: explicit per-drug override > default P-gp CLsec.
        elif getattr(drug, "renal_secretion_cl_L_per_h", None) is not None:
            cl_sec = drug.renal_secretion_cl_L_per_h
            transport = "PGP_explicit"
            km_mg_per_L = 500.0  # P-gp: high capacity, rarely saturated

        elif getattr(transporters, "is_pgp_substrate", False):
            # Default P-gp renal secretion clearance for substrates
            # without a drug-specific override: 5.0 L/h (Drescher 2003
            # digoxin value — reasonable default for P-gp substrates).
            cl_sec = 5.0
            transport = "PGP"
            km_mg_per_L = 500.0

        # ── Heuristic fallback (ionization + fe_unchanged) ──
        elif drug.ionization_class == Ionization.ACID:
            if drug.fraction_excreted_unchanged > 0.3:
                cl_sec = 3.0
                transport = "OAT1_OAT3_heuristic"

        elif drug.ionization_class == Ionization.BASE:
            if drug.fraction_excreted_unchanged > 0.3:
                cl_sec = 4.0
                transport = "OCT2_MATE1_heuristic"

        # Michaelis-Menten saturation at high concentrations
        if cl_sec > 0 and plasma_conc > 0:
            cl_sec = cl_sec * km_mg_per_L / (km_mg_per_L + plasma_conc)

        return cl_sec, transport

    @staticmethod
    def _reabsorption_fraction(
        drug: Drug,
        urine_ph: float,
        urine_flow_ml_per_min: float,
    ) -> float:
        """
        Passive tubular reabsorption — pH-dependent.

        Henderson-Hasselbalch determines fraction unionized in urine.
        Only unionized + lipophilic drug passively diffuses back.
        High urine flow → less contact time → less reabsorption.
        """
        ion_class = drug.ionization_class
        logp = drug.logp

        # Fraction unionized at urine pH
        if ion_class == Ionization.ACID and drug.pka_acidic is not None:
            f_unionized = 1.0 / (1.0 + 10.0 ** (urine_ph - drug.pka_acidic))
        elif ion_class == Ionization.BASE and drug.pka_basic is not None:
            f_unionized = 1.0 / (1.0 + 10.0 ** (drug.pka_basic - urine_ph))
        elif ion_class == Ionization.NEUTRAL:
            f_unionized = 0.5  # Moderate — depends on lipophilicity
        else:
            f_unionized = 0.3  # Zwitterion — moderate reabsorption

        # Lipophilicity factor: logP > 1 → highly reabsorbed
        logp_factor = min(1.0, max(0.0, (logp + 1.0) / 4.0))

        # Urine flow factor: high flow → less time → less reabsorption
        flow_factor = min(2.0, 1.0 / max(urine_flow_ml_per_min, 0.1))

        # Base reabsorption
        base_reab = 0.5  # Moderate default

        reab = base_reab * f_unionized * logp_factor * flow_factor

        # Cap: can't reabsorb everything
        return min(reab, 0.95)

    # ── fe-based calibration ──

    @staticmethod
    def calibrate_with_fe(
        renal_result: RenalClearanceResult,
        drug: Drug,
        cl_hepatic_L_per_h: float,
    ) -> RenalClearanceResult:
        """Constrain mechanistic CLr using the known fe when available.

        When a drug has a literature ``fraction_excreted_unchanged`` (fe),
        the relationship ``fe = CLr / (CLr + CLh)`` provides a strong
        constraint.  The mechanistic reabsorption model often under-predicts
        tubular reabsorption for small, high-permeability drugs (e.g.
        caffeine, theophylline), which inflates CLr by 5-20×.

        This post-hoc calibration caps CLr to the value implied by fe:

            CLr_target = fe × CLh / (1 − fe)

        Only applied when fe < 0.5 (metabolism-dominated drugs where the
        constraint is informative) and when the mechanistic CLr exceeds
        the target by > 50%.

        Returns:
            A new (possibly unchanged) frozen ``RenalClearanceResult``.
        """
        from dataclasses import replace as dc_replace

        fe = drug.fraction_excreted_unchanged
        if fe is None or fe <= 0.0 or fe >= 0.5:
            return renal_result  # No constraint or renally-dominated drug

        if cl_hepatic_L_per_h <= 0.0:
            return renal_result

        clr_target = fe * cl_hepatic_L_per_h / (1.0 - fe)
        clr_mech = renal_result.cl_renal_net_L_per_h

        # Only calibrate if mechanistic CLr overshoots by > 50 %
        if clr_mech <= clr_target * 1.5:
            return renal_result

        logger.info(
            "Renal fe-calibration: CLr_mech=%.2f → CLr_fe=%.2f L/h "
            "(fe=%.2f, CLh=%.2f)",
            clr_mech, clr_target, fe, cl_hepatic_L_per_h,
        )
        return dc_replace(renal_result, cl_renal_net_L_per_h=round(clr_target, 4))
