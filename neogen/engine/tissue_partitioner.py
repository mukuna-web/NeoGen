"""
Tissue Partition Coefficient (Kp) Prediction — V1 Engine
=========================================================
Clean implementation of Rodgers & Rowland (2005/2006) method using
V1 frozen entity system (Drug + PhysiologyModel).

Two methods:
  Method 1 (R&R 2005): Strong bases — includes acidic phospholipid binding
  Method 2 (R&R 2006): Acids, neutrals, zwitterions, weak bases

References:
  Rodgers T, Rowland M. J Pharm Sci 2005; 94:1259-1276
  Rodgers T, Rowland M. Pharm Res 2006; 23:56-68
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Dict, Optional

from neogen.core.entities.drug import Drug, Ionization
from neogen.core.entities.physiology import PhysiologyModel, OrganCompartment

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# TISSUE COMPOSITION DATA (Rodgers & Rowland / ICRP 89)
# ════════════════════════════════════════════════════════════════
# These are the fractional tissue volumes needed for Kp prediction.
# Separate from the PhysiologyModel TissueComposition (which stores
# organ-level fractions for the compartment model).

@dataclass(frozen=True)
class RRTissueData:
    """Rodgers-Rowland tissue composition fractions."""
    f_water: float           # Intracellular water
    f_ew: float              # Extracellular water
    f_neutral_lipid: float   # Neutral lipids
    f_phospholipid: float    # Phospholipids
    f_acidic_phospholipid: float  # Acidic phospholipids (AP)
    ph_iw: float = 7.0      # Intracellular pH


# Rodgers & Rowland 2005/2006 tissue composition table
RR_TISSUES: Dict[str, RRTissueData] = {
    "lung":     RRTissueData(0.446, 0.336, 0.003, 0.009,  0.0068, 7.0),
    "liver":    RRTissueData(0.573, 0.161, 0.014, 0.024,  0.0488, 7.0),
    "gut":      RRTissueData(0.282, 0.265, 0.038, 0.0125, 0.0111, 7.0),
    "kidney":   RRTissueData(0.483, 0.273, 0.012, 0.0124, 0.0113, 7.0),
    "brain":    RRTissueData(0.620, 0.162, 0.039, 0.0532, 0.0406, 7.0),
    "heart":    RRTissueData(0.456, 0.320, 0.014, 0.0111, 0.0127, 7.0),
    "muscle":   RRTissueData(0.630, 0.118, 0.010, 0.0073, 0.0064, 7.0),
    "fat":      RRTissueData(0.017, 0.135, 0.853, 0.0023, 0.0004, 7.0),
    "skin":     RRTissueData(0.382, 0.291, 0.060, 0.0044, 0.0033, 7.0),
    "bone":     RRTissueData(0.346, 0.100, 0.017, 0.0017, 0.0009, 7.0),
    "spleen":   RRTissueData(0.579, 0.207, 0.0077, 0.0113, 0.0091, 7.0),
    "pancreas": RRTissueData(0.463, 0.120, 0.041, 0.009,  0.0046, 7.0),
}

# Plasma composition (reference)
PLASMA_PH = 7.4
PLASMA_F_WATER = 0.945
PLASMA_F_NEUTRAL_LIPID = 0.0023
PLASMA_F_PHOSPHOLIPID = 0.0009

# Reference body weight used to convert `drug.vss_observed_L_per_kg`
# to an absolute target volume (L). The V1 reference physiology is
# keyed to a 70 kg adult male (ICRP 89).
REFERENCE_BODY_WEIGHT_KG = 70.0

# Clamp bounds on the observed-Vss calibration scale factor. These
# stop a bad literature value from pushing Kp into non-physiological
# territory (e.g. < 0.01 or > 500). The upper bound is set at 200×
# to accommodate extreme cases like itraconazole (logP 5.66,
# fu 0.002), where R&R under-predicts tissue distribution by ~100×.
_VSS_CALIBRATION_SCALE_MIN = 0.05
_VSS_CALIBRATION_SCALE_MAX = 200.0


# ════════════════════════════════════════════════════════════════
# PARTITION RESULT
# ════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PartitionCoefficients:
    """Immutable container for all tissue:plasma partition coefficients."""
    lung: float
    liver: float
    gut: float
    kidney: float
    brain: float
    heart: float
    muscle: float
    fat: float
    skin: float
    bone: float
    spleen: float
    pancreas: float
    vss_predicted_L: float
    method: str

    def for_organ(self, organ_name: str) -> float:
        """Look up Kp by organ name (case-insensitive)."""
        key = organ_name.lower().replace(" ", "_")
        # Map common aliases
        aliases = {"adipose": "fat", "gut_wall": "gut", "rest": "muscle"}
        key = aliases.get(key, key)
        val = getattr(self, key, None)
        if val is None:
            logger.warning("No Kp for organ '%s', defaulting to 1.0", organ_name)
            return 1.0
        return val

    def as_dict(self) -> Dict[str, float]:
        """Return all Kp values as a flat dictionary."""
        return {
            "lung": self.lung,
            "liver": self.liver,
            "gut": self.gut,
            "kidney": self.kidney,
            "brain": self.brain,
            "heart": self.heart,
            "muscle": self.muscle,
            "fat": self.fat,
            "skin": self.skin,
            "bone": self.bone,
            "spleen": self.spleen,
            "pancreas": self.pancreas,
        }


# ════════════════════════════════════════════════════════════════
# TISSUE PARTITIONER
# ════════════════════════════════════════════════════════════════

class TissuePartitioner:
    """
    Predicts tissue:plasma partition coefficients (Kp) for all organs
    using the Rodgers & Rowland mechanistic method.

    Consumes V1 frozen entities directly — no mutable state, no ML.

    Usage:
        partitioner = TissuePartitioner()
        kp = partitioner.predict(drug, physiology)
        print(kp.liver, kp.brain, kp.vss_predicted_L)
    """

    def predict(
        self,
        drug: Drug,
        physiology: PhysiologyModel,
    ) -> PartitionCoefficients:
        """
        Compute Kp for all tissues from drug physicochemical properties.

        Uses Method 1 (R&R 2005) for strong bases, Method 2 (R&R 2006)
        for acids, neutrals, and zwitterions.

        Args:
            drug: V1 Drug entity with logp, pka, fu_plasma, ionization_class
            physiology: V1 PhysiologyModel (used for Vss calculation)

        Returns:
            Frozen PartitionCoefficients with Kp per organ + predicted Vss.
        """
        logp = drug.logp
        fu = drug.fu_plasma_safe  # Clamped ≥ 0.001
        ion_class = drug.ionization_class
        bp_ratio = drug.blood_to_plasma_ratio

        # Determine effective pKa
        pka = self._effective_pka(drug)

        # Determine R&R method
        # Method 1 (R&R 2005): ALL bases (strong and weak)
        # Method 2 (R&R 2006): acids, neutrals, zwitterions
        use_method_1 = ion_class == Ionization.BASE

        # Octanol:water and vegetable-oil:water partitioning
        p_ow = 10.0 ** logp
        log_p_vow = logp - 0.7  # vegetable oil:water offset
        p_vow = 10.0 ** log_p_vow

        # Ionization ratio at plasma pH
        ratio_plasma = self._ionization_ratio(pka, PLASMA_PH, ion_class)

        # Compute Kp per tissue
        kp_values: Dict[str, float] = {}
        for tissue_name, tc in RR_TISSUES.items():
            ratio_tissue = self._ionization_ratio(pka, tc.ph_iw, ion_class)
            kp = self._compute_single_kp(
                tc, ratio_plasma, ratio_tissue,
                p_ow, p_vow, fu, use_method_1,
            )
            kp_values[tissue_name] = kp

        # V1.2 Fix 3: ionization-class Kp correction.
        # R&R systematically over-predicts Kp for lipophilic bases
        # (acidic phospholipid term in Method 1 is too aggressive)
        # and under-predicts for acids. Apply empirical scalar per
        # Poulin & Theil 2002, Rodgers 2005 validation set.
        _ION_KP_CORRECTION = {
            Ionization.BASE: 0.65,
            Ionization.ACID: 1.4,
            Ionization.NEUTRAL: 1.0,
            Ionization.ZWITTERION: 0.9,
        }
        kp_corr = _ION_KP_CORRECTION.get(ion_class, 1.0)
        if kp_corr != 1.0:
            kp_values = {k: v * kp_corr for k, v in kp_values.items()}

        # Lysosomal trapping correction for weak/moderate bases.
        # Lysosomes occupy ~1% of cell volume at pH ~5.0. Weak bases
        # (pKa 6-10) accumulate 10-1000x in lysosomes via ion trapping:
        #   ratio = (1 + 10^(pKa - pH_lyso)) / (1 + 10^(pKa - pH_cyto))
        # This primarily affects Vd for amines (CNS drugs, antipsychotics,
        # beta-blockers, antimalarials). De Duve 1974, MacIntyre 1988.
        if ion_class == Ionization.BASE and pka is not None:
            pH_lyso = 5.0
            pH_cyto = 7.2
            lyso_ratio = (1.0 + 10.0 ** (pka - pH_lyso)) / (
                1.0 + 10.0 ** (pka - pH_cyto)
            )
            # Lysosomes = ~1% of tissue volume, so net effect
            # on Kp is: Kp_corrected = Kp * (1 + 0.01 * lyso_ratio)
            lyso_kp_factor = 1.0 + 0.01 * min(lyso_ratio, 500.0)
            kp_values = {k: v * lyso_kp_factor for k, v in kp_values.items()}

        # Predict Vss from Kp values and organ volumes
        vss = self._predict_vss(kp_values, physiology)

        method = "Rodgers_Rowland_2005" if use_method_1 else "Rodgers_Rowland_2006"

        # Optional: calibrate Kp values so the predicted Vss matches a
        # literature-reported observed Vss. R&R has known structural
        # limits (high-logP bases, zwitterions, transporter-mediated
        # tissue binding) where the mechanistic prediction is off by
        # 3-100×. When `drug.vss_observed_L_per_kg` is provided, we
        # apply a single scalar to all tissue Kp values so the final
        # predicted Vss matches the target. Tissue *ratios* (the
        # liver:muscle:fat shape from R&R) are preserved, so local
        # pharmacokinetics (extraction ratios, tissue residence times)
        # still follow the mechanistic geometry.
        if drug.vss_observed_L_per_kg is not None:
            kp_values, vss = self._calibrate_to_observed_vss(
                kp_values, vss, drug.vss_observed_L_per_kg, physiology
            )
            method = f"{method}+Vss_calibrated"

        return PartitionCoefficients(
            lung=kp_values["lung"],
            liver=kp_values["liver"],
            gut=kp_values["gut"],
            kidney=kp_values["kidney"],
            brain=kp_values["brain"],
            heart=kp_values["heart"],
            muscle=kp_values["muscle"],
            fat=kp_values["fat"],
            skin=kp_values["skin"],
            bone=kp_values["bone"],
            spleen=kp_values["spleen"],
            pancreas=kp_values["pancreas"],
            vss_predicted_L=round(vss, 2),
            method=method,
        )

    # ── Internal helpers ──

    @staticmethod
    def _effective_pka(drug: Drug) -> float:
        """Extract the relevant pKa for ionization calculations."""
        if drug.ionization_class == Ionization.BASE:
            return drug.pka_basic if drug.pka_basic is not None else 7.0
        if drug.ionization_class == Ionization.ACID:
            return drug.pka_acidic if drug.pka_acidic is not None else 7.0
        if drug.ionization_class == Ionization.ZWITTERION:
            # Use basic pKa for tissue distribution (conservative)
            return drug.pka_basic if drug.pka_basic is not None else 7.0
        # Neutral
        return 7.0

    @staticmethod
    def _ionization_ratio(pka: float, ph: float, ion_class: Ionization) -> float:
        """
        Calculate ionization ratio at given pH.

        For bases: ratio = 10^(pKa - pH)  (high when pH < pKa → ionized)
        For acids: ratio = 10^(pH - pKa)  (high when pH > pKa → ionized)
        """
        if ion_class == Ionization.BASE:
            return 10.0 ** (pka - ph)
        if ion_class == Ionization.ACID:
            return 10.0 ** (ph - pka)
        if ion_class == Ionization.ZWITTERION:
            # Approximate: use base-like behavior for tissue binding
            return 10.0 ** (pka - ph)
        # Neutral — fully unionized
        return 0.0

    @staticmethod
    def _compute_single_kp(
        tc: RRTissueData,
        ratio_plasma: float,
        ratio_tissue: float,
        p_ow: float,
        p_vow: float,
        fu_plasma: float,
        use_method_1: bool,
    ) -> float:
        """
        Compute Kp for a single tissue using the Rodgers & Rowland method.

        The canonical R&R (2005, 2006) equation, in the notation used by
        Simcyp/PK-Sim, is:

          Kp = (Kpu_tissue × fu_plasma) / Kpu_plasma

        where ``Kpu_plasma`` is the unbound partitioning fraction of plasma
        (water + neutral lipid + phospholipid contributions — NOT 1/fu_p).
        The ``fu_plasma`` multiplier converts from an unbound partition
        ratio into a total-drug tissue:plasma ratio and is essential for
        highly-bound drugs (acids with fu_p < 0.05). Earlier V1 revisions
        dropped this factor in the name of a "Berezhkovskiy correction",
        which caused Kp_fat values of 200+ for NSAIDs and statins.

        Method 1 (bases): Includes acidic phospholipid (AP) ion-pair binding.
          Ka_AP = 0.5 × P^0.65
          Kpu_t = f_ew + [(1+Rt)/(1+Rp)] × [f_iw + Ka_AP × f_AP]
                  + P_vow × f_nl + (0.3P + 0.7) × f_pl

        Method 2 (acids/neutrals): No AP binding.
          Kpu_t = f_ew + [(1+Rt)/(1+Rp)] × f_iw
                  + P_vow × f_nl + (0.3P + 0.7) × f_pl

        Reference:
          Berezhkovskiy LM. J Pharm Sci 2004; 93:1628-1640
        """
        denom_ion = max(1.0 + ratio_plasma, 0.01)
        ion_factor = (1.0 + ratio_tissue) / denom_ion

        # Phospholipid partitioning term
        pl_term = (0.3 * p_ow + 0.7) * tc.f_phospholipid

        if use_method_1:
            # Method 1: strong bases bind to acidic phospholipids
            ka_ap = 0.5 * (p_ow ** 0.65)
            kp_u_tissue = (
                tc.f_ew
                + ion_factor * (tc.f_water + ka_ap * tc.f_acidic_phospholipid)
                + p_vow * tc.f_neutral_lipid
                + pl_term
            )
        else:
            # Method 2: acids, neutrals, zwitterions
            kp_u_tissue = (
                tc.f_ew
                + ion_factor * tc.f_water
                + p_vow * tc.f_neutral_lipid
                + pl_term
            )

        # Plasma reference (unbound partitioning in plasma)
        kp_u_plasma = (
            PLASMA_F_WATER
            + p_vow * PLASMA_F_NEUTRAL_LIPID
            + (0.3 * p_ow + 0.7) * PLASMA_F_PHOSPHOLIPID
        )

        if kp_u_plasma <= 0:
            return 1.0

        # Rodgers & Rowland (2005): Kp = Kpu_tissue × fu_plasma / Kpu_plasma
        # The fu_plasma multiplier is the canonical correction for highly
        # protein-bound drugs. Guarding fu_plasma > 0 defensively — a
        # zero-fu drug would be a modelling error upstream.
        fu_eff = max(fu_plasma, 1.0e-4)
        kp = (kp_u_tissue * fu_eff) / kp_u_plasma

        # Physiological bounds: Kp can't be < 0.01 or > 500
        return max(0.01, min(kp, 500.0))

    @staticmethod
    def _calibrate_to_observed_vss(
        kp_values: Dict[str, float],
        vss_predicted_L: float,
        vss_observed_L_per_kg: float,
        physiology: PhysiologyModel,
    ) -> "tuple[Dict[str, float], float]":
        """
        Rescale tissue Kp values so the predicted Vss matches a
        literature-reported observed Vss.

        Vss = V_plasma + Σ (V_tissue × Kp_tissue)

        Holding V_plasma fixed, the tissue contribution is
            V_tissue_total = Vss - V_plasma

        To hit `target_tissue_vss = V_obs - V_plasma`, we multiply
        every Kp by
            scale = target_tissue_vss / current_tissue_vss

        This preserves tissue *ratios* (liver:muscle:fat) so local
        partitioning geometry remains mechanistic from R&R; only the
        magnitude is calibrated. The scale factor is clamped to
        [_VSS_CALIBRATION_SCALE_MIN, _VSS_CALIBRATION_SCALE_MAX] so a
        bad literature value cannot blow up the Kp bounds.
        """
        v_plasma = physiology.plasma.volume_L
        target_vss = vss_observed_L_per_kg * REFERENCE_BODY_WEIGHT_KG

        tissue_vss_current = vss_predicted_L - v_plasma
        tissue_vss_target = target_vss - v_plasma

        if tissue_vss_current <= 0.0 or tissue_vss_target <= 0.0:
            return kp_values, vss_predicted_L

        scale = tissue_vss_target / tissue_vss_current
        scale = max(_VSS_CALIBRATION_SCALE_MIN,
                    min(_VSS_CALIBRATION_SCALE_MAX, scale))

        # ── Pass 1: uniform scale ──
        calibrated = {
            name: max(0.01, min(kp * scale, 500.0))
            for name, kp in kp_values.items()
        }

        # ── Pass 2: tissue-specific Kp caps ──────────────────────
        # Uniform scaling can inflate Kp for low-flow/high-volume
        # tissues (fat, skin, bone) to unrealistic levels, creating
        # artificially slow terminal compartments that over-predict
        # t½.  Cap each tissue at a physiological maximum, then
        # redistribute any Vss deficit to well-perfused organs.
        _KP_MAX: Dict[str, float] = {
            "fat": 50.0,       # even amiodarone ≈ 30-50
            "skin": 30.0,
            "bone": 15.0,
            "brain": 15.0,     # BBB limits accumulation
            "muscle": 30.0,
            "heart": 50.0,
            "spleen": 50.0,
            "pancreas": 50.0,
            "lung": 80.0,
            "liver": 150.0,    # active transporters can genuinely
            "kidney": 100.0,   #   concentrate drugs
            "gut": 100.0,
        }

        # Well-perfused tissues that can absorb the Vss deficit
        # without materially affecting terminal t½ (high Q/V ratio).
        _REDISTRIB_TISSUES = {"liver", "kidney", "gut", "muscle"}

        organ_map = {
            "lung": physiology.lung, "liver": physiology.liver,
            "gut": physiology.gut, "kidney": physiology.kidney,
            "brain": physiology.brain, "heart": physiology.heart,
            "muscle": physiology.muscle, "fat": physiology.fat,
            "skin": physiology.skin, "bone": physiology.bone,
            "spleen": physiology.spleen, "pancreas": physiology.pancreas,
        }

        deficit_L = 0.0
        for name, kp in list(calibrated.items()):
            cap = _KP_MAX.get(name, 500.0)
            if kp > cap:
                organ = organ_map.get(name)
                vol = organ.volume_L if organ else 0.0
                deficit_L += vol * (kp - cap)
                calibrated[name] = cap

        # Redistribute deficit to well-perfused tissues.
        if deficit_L > 0.0:
            redist_vol = sum(
                organ_map[t].volume_L for t in _REDISTRIB_TISSUES
                if t in calibrated and t in organ_map
            )
            if redist_vol > 0.0:
                extra_kp = deficit_L / redist_vol
                for t in _REDISTRIB_TISSUES:
                    if t in calibrated:
                        cap = _KP_MAX.get(t, 500.0)
                        calibrated[t] = min(calibrated[t] + extra_kp, cap)

        # Recompute Vss after the (possibly clamped) rescale so the
        # returned object is self-consistent.
        vss_new = TissuePartitioner._predict_vss(calibrated, physiology)
        return calibrated, vss_new

    @staticmethod
    def _predict_vss(
        kp_values: Dict[str, float],
        physiology: PhysiologyModel,
    ) -> float:
        """
        Predict steady-state volume of distribution.
        Vss = Vplasma + Σ(Vtissue × Kp_tissue)
        """
        # Map organ names to PhysiologyModel compartments
        organ_map = {
            "lung": physiology.lung,
            "liver": physiology.liver,
            "gut": physiology.gut,
            "kidney": physiology.kidney,
            "brain": physiology.brain,
            "heart": physiology.heart,
            "muscle": physiology.muscle,
            "fat": physiology.fat,
            "skin": physiology.skin,
            "bone": physiology.bone,
            "spleen": physiology.spleen,
            "pancreas": physiology.pancreas,
        }

        vss = physiology.plasma.volume_L
        for tissue_name, kp in kp_values.items():
            organ = organ_map.get(tissue_name)
            if organ is not None:
                vss += organ.volume_L * kp

        return vss
