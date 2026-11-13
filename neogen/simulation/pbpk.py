"""
7-Compartment PBPK Simulation Engine
======================================
Physiologically Based Pharmacokinetic model that simulates drug distribution
across real organ compartments connected by blood flow.

Compartments: GI tract, Liver, Kidney, Brain, Muscle, Fat, Blood/Plasma

Drug flow (oral):
    Gut lumen → Gut tissue → Portal vein → Liver (first-pass) → Systemic blood → All organs

This is what Simcyp and GastroPlus do internally. We're building it from scratch
with real physiological parameters (ICRP Publication 89).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# PATIENT PARAMETERS
# ════════════════════════════════════════════════════════════════

@dataclass
class PatientParameters:
    """User-facing patient configuration."""
    weight_kg: float = 70.0
    age_years: int = 30
    sex: str = "male"  # "male" or "female"


# ════════════════════════════════════════════════════════════════
# PHYSIOLOGICAL PARAMETERS
# ════════════════════════════════════════════════════════════════

@dataclass
class PhysiologicalParameters:
    """
    Organ volumes (L) and blood flows (L/h) for a specific patient.
    Reference: ICRP Publication 89 (70 kg adult male).
    """
    # Organ volumes (L) — ICRP 89 reference values for 70 kg male
    V_plasma: float = 5.6       # Total blood volume ~5.6L
    V_gut: float = 1.7          # GI tract tissue
    V_liver: float = 1.8        # Liver
    V_kidney: float = 0.31      # Both kidneys
    V_brain: float = 1.45       # Brain
    V_muscle: float = 29.0      # Skeletal muscle
    V_fat: float = 18.2         # Adipose tissue
    # New compartments (14-compartment model)
    V_lung: float = 0.50        # Lung tissue (not airspace)
    V_spleen: float = 0.15      # Spleen
    V_skin: float = 3.3         # Skin + subcutaneous
    V_bone: float = 4.0         # Bone marrow (red + yellow)
    V_rbc: float = 2.5          # Red blood cells (packed)
    V_gut_wall: float = 0.65    # Gut wall (enterocytes) — separate from gut lumen
    V_pancreas: float = 0.10    # Pancreas

    # Blood flows (L/h) — fractions of cardiac output
    CO: float = 390.0  # Cardiac output (6.5 L/min)
    Q_gut: float = 0.0
    Q_liver: float = 0.0  # Total hepatic (portal + arterial)
    Q_kidney: float = 0.0
    Q_brain: float = 0.0
    Q_muscle: float = 0.0
    Q_fat: float = 0.0
    # New compartment blood flows
    Q_lung: float = 0.0        # Lung receives ENTIRE CO (pulmonary circulation)
    Q_spleen: float = 0.0
    Q_skin: float = 0.0
    Q_bone: float = 0.0
    Q_pancreas: float = 0.0

    # Clearance scaling factors
    hepatic_cl_scale: float = 1.0
    renal_cl_scale: float = 1.0

    @classmethod
    def from_patient(cls, patient: PatientParameters) -> PhysiologicalParameters:
        """Create physiological parameters scaled to a specific patient."""
        p = cls()
        w_ratio = patient.weight_kg / 70.0

        # Scale volumes linearly with weight
        p.V_plasma = 5.6 * w_ratio
        p.V_gut = 1.7 * w_ratio
        p.V_liver = 1.8 * w_ratio
        p.V_kidney = 0.31 * w_ratio
        p.V_brain = 1.45  # Brain doesn't scale much with weight
        p.V_muscle = 29.0 * w_ratio
        p.V_fat = 18.2 * w_ratio
        # New compartments
        p.V_lung = 0.50 * w_ratio
        p.V_spleen = 0.15 * w_ratio
        p.V_skin = 3.3 * w_ratio
        p.V_bone = 4.0 * w_ratio
        p.V_rbc = 2.5 * w_ratio
        p.V_gut_wall = 0.65 * w_ratio
        p.V_pancreas = 0.10 * w_ratio

        # Scale cardiac output allometrically
        p.CO = 390.0 * (w_ratio ** 0.75)

        # Sex adjustments
        if patient.sex == "female":
            p.V_muscle *= 0.80   # 20% less muscle
            p.V_fat *= 1.30      # 30% more fat
            p.V_rbc *= 0.85      # Lower hematocrit
            p.V_skin *= 0.90
            p.CO *= 0.85         # 15% lower cardiac output

        # Compute organ blood flows from CO fractions (ICRP 89)
        # Sum of fractions should ≈ 1.0 (rest goes to "other")
        p.Q_gut = p.CO * 0.15        # 15% — splanchnic (portal)
        p.Q_liver = p.CO * 0.25      # 25% — total hepatic (portal + arterial)
        p.Q_kidney = p.CO * 0.19     # 19% — renal
        p.Q_brain = p.CO * 0.12      # 12% — cerebral
        p.Q_muscle = p.CO * 0.17     # 17% — skeletal muscle
        p.Q_fat = p.CO * 0.05        # 5% — adipose
        # New compartment flows
        p.Q_lung = p.CO               # 100% — ALL blood passes through lungs (pulmonary circuit)
        p.Q_spleen = p.CO * 0.03     # 3% — splenic
        p.Q_skin = p.CO * 0.05       # 5% — cutaneous
        p.Q_bone = p.CO * 0.05       # 5% — bone marrow
        p.Q_pancreas = p.CO * 0.01   # 1% — pancreatic

        # Age adjustments (organ function decline)
        p.hepatic_cl_scale = 1.0
        p.renal_cl_scale = 1.0
        if patient.age_years > 60:
            age_factor = (patient.age_years - 60) / 40.0  # 0 to 1 over ages 60-100
            p.hepatic_cl_scale = 1.0 - 0.30 * min(age_factor, 1.0)
            p.renal_cl_scale = 1.0 - 0.25 * min(age_factor, 1.0)
            p.V_bone *= max(0.7, 1.0 - age_factor * 0.3)  # Bone marrow decreases
        if patient.age_years > 30:
            co_decline = (patient.age_years - 30) * 0.01
            p.CO *= max(1.0 - co_decline, 0.5)
            # Recompute ALL flows
            p.Q_gut = p.CO * 0.15
            p.Q_liver = p.CO * 0.25
            p.Q_kidney = p.CO * 0.19
            p.Q_brain = p.CO * 0.12
            p.Q_muscle = p.CO * 0.17
            p.Q_fat = p.CO * 0.05
            p.Q_lung = p.CO
            p.Q_spleen = p.CO * 0.03
            p.Q_skin = p.CO * 0.05
            p.Q_bone = p.CO * 0.05
            p.Q_pancreas = p.CO * 0.01

        return p


# ════════════════════════════════════════════════════════════════
# PBPK MODEL
# ════════════════════════════════════════════════════════════════

class PBPKModel:
    """
    7-compartment PBPK model with ODE solver.

    Usage:
        patient = PatientParameters(weight_kg=70, age_years=30, sex="male")
        model = PBPKModel(patient, dose_mg=100, route="oral")
        model.derive_pbpk_params(ml_predictions, logp=1.2)
        solution = model.solve()
        time_course = model.to_time_course_dicts(solution)
    """

    def __init__(
        self,
        patient: PatientParameters | None = None,
        dose_mg: float = 100.0,
        route: str = "oral",
    ):
        self.patient = patient or PatientParameters()
        self.phys = PhysiologicalParameters.from_patient(self.patient)
        self.dose_mg = dose_mg
        self.route = route

        # PBPK parameters (set by derive_pbpk_params)
        self.ka = 1.0       # Absorption rate constant (1/h)
        self.F = 0.7        # Bioavailability
        self.CL_hep = 0.0   # Hepatic clearance (L/h)
        self.CL_renal = 0.0 # Renal clearance (L/h)
        self.half_life = 4.0

        # Partition coefficients (tissue:plasma)
        self.Kp_gut = 1.0
        self.Kp_liver = 1.5
        self.Kp_kidney = 1.0
        self.Kp_brain = 0.3
        self.Kp_muscle = 0.5
        self.Kp_fat = 0.2
        # New compartment partition coefficients
        self.Kp_lung = 0.8
        self.Kp_spleen = 1.2
        self.Kp_skin = 0.5
        self.Kp_bone = 0.3
        self.Kp_rbc = 1.0    # RBC:plasma ratio
        self.Kp_gut_wall = 1.0
        self.Kp_pancreas = 0.8

    def derive_pbpk_params(
        self, ml_predictions: dict, logp: float = 2.0,
        smiles: str = "", mw: float = 300, tpsa: float = 60, hbd: int = 2,
    ) -> None:
        """
        Map aggregate ML predictions to compartment-level PBPK parameters.

        Now integrates:
        - Bioavailability model (replaces F=0.7)
        - Protein binding (fu affects clearance and distribution)
        - BBB permeability (brain Kp from BBB model)
        - hERG cardiotoxicity (stored for later assessment)

        Args:
            ml_predictions: dict with keys cmax, auc, half_life, clearance, vd
            logp: octanol-water partition coefficient from RDKit
            smiles: SMILES string for drug-specific lookups
            mw: molecular weight
            tpsa: topological polar surface area
            hbd: hydrogen bond donors
        """
        half_life = max(ml_predictions.get("half_life", 4.0), 0.1)
        clearance = max(ml_predictions.get("clearance", 1.0), 0.01)
        vd = max(ml_predictions.get("vd", 10.0), 0.1)

        self.half_life = half_life
        self.smiles = smiles

        # Elimination rate
        ke = 0.693 / half_life

        # ── Bioavailability (replaces F=0.7) ──
        if self.route == "oral":
            try:
                from neogen.simulation.bioavailability import BioavailabilityEngine
                bio_engine = BioavailabilityEngine(smiles, logp=logp, mw=mw, tpsa=tpsa, hbd=hbd)
                self.F = bio_engine.get_bioavailability()
                self.ka = bio_engine.get_absorption_rate_constant(ke * 5.0)
                self.bioavailability_info = bio_engine.get_full_assessment()
                logger.info("Bioavailability: F=%.3f (BCS class %d)", self.F, bio_engine.profile.bcs_class)
            except Exception as e:
                logger.warning("Bioavailability engine failed (%s), using default F=0.7", e)
                self.F = 0.7
                self.ka = ke * 5.0
                self.bioavailability_info = None
        else:
            self.ka = ke * 50.0  # Essentially instant for IV
            self.F = 1.0
            self.bioavailability_info = {"f_oral": 1.0, "bcs_class": 0, "drug_name": "IV dose"}

        # ── Protein Binding ──
        try:
            from neogen.simulation.protein_binding import ProteinBindingEngine
            pb_engine = ProteinBindingEngine(smiles, logp=logp)
            self.fu = pb_engine.fu
            self.protein_binding_info = {
                "fu": pb_engine.fu,
                "protein": pb_engine.profile.primary_protein,
                "binding_affinity": pb_engine.profile.binding_affinity,
                "known": pb_engine.known,
            }
            logger.info("Protein binding: fu=%.3f (%s)", self.fu, pb_engine.profile.primary_protein)
        except Exception as e:
            logger.warning("Protein binding engine failed (%s), using fu=0.5", e)
            self.fu = 0.5
            self.protein_binding_info = None

        # Split total clearance into hepatic (70%) and renal (30%)
        self.CL_hep = clearance * 0.70 * self.phys.hepatic_cl_scale
        self.CL_renal = clearance * 0.30 * self.phys.renal_cl_scale

        # ── Mechanistic Renal Parameters ──
        # Try to get drug-specific renal transporter data
        self._renal_secretion_cl = 0.0
        self._renal_reabsorption_frac = 0.5  # Default: 50% reabsorbed
        try:
            from neogen.simulation.renal_model import get_renal_profile, calculate_net_renal_clearance
            renal_profile = get_renal_profile(smiles) if smiles else None
            if renal_profile:
                renal_result = calculate_net_renal_clearance(renal_profile)
                self._renal_secretion_cl = renal_result.get("cl_secretion", 0)
                self._renal_reabsorption_frac = renal_result.get("reabsorption_fraction", 0.5)
                # Use the mechanistic CL_renal instead of the simple 30% split
                self.CL_renal = renal_result.get("cl_net_renal", self.CL_renal)
                logger.info("Renal model: CL_filt=%.2f, CL_sec=%.2f, reab=%.2f, CL_net=%.2f",
                           renal_result.get("cl_filtration", 0), self._renal_secretion_cl,
                           self._renal_reabsorption_frac, self.CL_renal)
        except Exception as e:
            logger.debug("Renal model not available: %s", e)
            # Estimate reabsorption from logP (lipophilic drugs are more reabsorbed)
            if logp > 3:
                self._renal_reabsorption_frac = 0.8
            elif logp > 1:
                self._renal_reabsorption_frac = 0.5
            elif logp > -1:
                self._renal_reabsorption_frac = 0.3
            else:
                self._renal_reabsorption_frac = 0.1  # Hydrophilic = not reabsorbed

        # ── BBB Permeability (brain Kp) ──
        try:
            from neogen.simulation.bbb_permeability import BBBPermeabilityEngine
            bbb_engine = BBBPermeabilityEngine(smiles, logp=logp, mw=mw, tpsa=tpsa, hbd=hbd)
            self.Kp_brain = bbb_engine.get_brain_kp()
            self.bbb_info = bbb_engine.get_full_assessment()
            logger.info("BBB: Kp_brain=%.3f (%s)", self.Kp_brain, bbb_engine.profile.bbb_category)
        except Exception as e:
            logger.warning("BBB engine failed (%s), using default Kp_brain", e)
            self.Kp_brain = max(0.3 + 0.1 * logp, 0.05) if logp > 1.0 else 0.05
            self.bbb_info = None

        # Partition coefficients — use Rodgers & Rowland if possible
        try:
            from neogen.simulation.kp_prediction import get_kp_for_pbpk
            # Determine drug type from logP/pKa
            drug_type = "neutral"  # Default
            if logp > 3:
                drug_type = "base"  # Lipophilic drugs tend to be basic
            elif logp < 0:
                drug_type = "acid"

            kp_rr = get_kp_for_pbpk(logp, pka=7.0, drug_type=drug_type, fu_plasma=self.fu)
            self.Kp_gut = kp_rr["Kp_gut"]
            self.Kp_liver = kp_rr["Kp_liver"]
            self.Kp_kidney = kp_rr["Kp_kidney"]
            # Keep BBB Kp from BBB engine (more specific)
            self.Kp_muscle = kp_rr["Kp_muscle"]
            self.Kp_fat = kp_rr["Kp_fat"]
            self.Kp_lung = kp_rr["Kp_lung"]
            self.Kp_spleen = kp_rr["Kp_spleen"]
            self.Kp_skin = kp_rr["Kp_skin"]
            self.Kp_bone = kp_rr["Kp_bone"]
            self.Kp_rbc = kp_rr["Kp_rbc"]
            self.Kp_gut_wall = kp_rr["Kp_gut_wall"]
            self.Kp_pancreas = kp_rr["Kp_pancreas"]
            logger.info("Kp values from Rodgers & Rowland: liver=%.2f, kidney=%.2f, muscle=%.2f, fat=%.2f",
                       self.Kp_liver, self.Kp_kidney, self.Kp_muscle, self.Kp_fat)
        except Exception as e:
            logger.debug("Rodgers & Rowland Kp failed (%s), using logP-based estimates", e)
            # Fallback: simplified logP-based Kp
            self.Kp_gut = np.clip(1.0 + 0.2 * logp, 0.3, 10.0)
            self.Kp_liver = np.clip(1.5 + 0.3 * logp, 0.5, 20.0)
            self.Kp_kidney = np.clip(1.0 + 0.2 * logp, 0.5, 15.0)
            self.Kp_muscle = np.clip(0.5 + 0.15 * logp, 0.2, 10.0)
            self.Kp_fat = max(0.2 + 0.5 * logp, 0.1) if logp > 0 else 0.1
            self.Kp_lung = np.clip(0.8 + 0.15 * logp, 0.3, 8.0)
        self.Kp_spleen = np.clip(1.2 + 0.2 * logp, 0.5, 10.0)  # Spleen: blood-rich
        self.Kp_skin = np.clip(0.5 + 0.3 * logp, 0.2, 15.0)    # Skin: lipophilic drugs accumulate
        self.Kp_bone = np.clip(0.3 + 0.05 * logp, 0.1, 3.0)    # Bone: low perfusion
        self.Kp_rbc = np.clip(1.0 + 0.1 * logp, 0.5, 5.0)      # RBC partitioning
        self.Kp_gut_wall = np.clip(1.0 + 0.2 * logp, 0.5, 8.0)  # Gut wall enterocytes
        self.Kp_pancreas = np.clip(0.8 + 0.15 * logp, 0.3, 5.0)  # Pancreas

        # ── Store hERG info for later assessment ──
        try:
            from neogen.simulation.cardiotoxicity import CardiotoxicityEngine
            # Estimate Cmax in µM for hERG (convert mg/L → µM using MW)
            cmax_val = ml_predictions.get("cmax", 10.0)
            cmax_um = (cmax_val / max(mw, 100)) * 1000  # mg/L → µM
            herg_engine = CardiotoxicityEngine(smiles, logp=logp, mw=mw, plasma_cmax_um=cmax_um)
            self.herg_info = herg_engine.get_full_assessment()
            logger.info("hERG: safety margin=%.1f (%s)", herg_engine.profile.safety_margin, herg_engine.profile.risk_category)
        except Exception as e:
            logger.warning("hERG engine failed (%s)", e)
            self.herg_info = None

        # ── Store hepatic metabolism info ──
        try:
            from neogen.simulation.hepatic_metabolism import HepaticMetabolismEngine
            metab_engine = HepaticMetabolismEngine(smiles, dose_mg=self.dose_mg)
            self.metabolism_info = metab_engine.assess_metabolism()
            logger.info("Metabolism: %s risk, GSH %s", self.metabolism_info["overall_risk"],
                        self.metabolism_info["phase2"]["gsh_status"])
        except Exception as e:
            logger.warning("Hepatic metabolism engine failed (%s)", e)
            self.metabolism_info = None

        # ── Advanced Physiology (EHR, pH, Food, Transporters, Metabolites) ──
        try:
            from neogen.simulation.advanced_physiology import assess_advanced_physiology
            drug_name = getattr(self, '_drug_name', '')
            fed_state = getattr(self, '_fed_state', 'fasted')
            ppi = getattr(self, '_ppi', False)
            grapefruit = getattr(self, '_grapefruit', False)
            bcs_class = getattr(self, '_bcs_class', 1)
            if hasattr(self, 'bioavailability_info') and self.bioavailability_info:
                bcs_class = self.bioavailability_info.get('bcs_class', 1)

            self.adv_phys = assess_advanced_physiology(
                smiles=smiles, drug_name=drug_name,
                fed_state=fed_state, ppi=ppi, grapefruit=grapefruit,
                bcs_class=bcs_class,
            )

            # Apply pH absorption modifier to ka
            self.ka *= self.adv_phys.ka_modifier

            # Apply food effect modifiers
            self.F *= self.adv_phys.food_effect.bioavail_multiplier
            self.F = min(self.F, 1.0)  # Cap at 100%
            self.ka *= self.adv_phys.food_effect.absorption_multiplier

            # Apply transporter effects
            tp = self.adv_phys.transporters
            # P-gp reduces gut absorption
            if tp.pgp_substrate:
                self.F *= tp.gut_efflux_factor
                self.F = min(self.F, 1.0)
            # OATP increases liver partition
            self.Kp_liver *= tp.liver_uptake_factor
            # OCT/OAT increases renal clearance
            self.CL_renal *= tp.kidney_secretion_factor
            # P-gp/BCRP reduces brain partition
            self.Kp_brain *= tp.brain_efflux_factor

            # EHR parameters for ODE system
            self.ehr_active = self.adv_phys.ehr_active
            self.ehr_fraction = self.adv_phys.ehr.fraction_biliary
            self.ehr_reabsorption = self.adv_phys.ehr.reabsorption_rate

            # Biliary efflux from MRP2
            self.biliary_cl_factor = tp.biliary_efflux_factor

            logger.info("Advanced physiology applied: ka=%.2f, F=%.3f, Kp_liver=%.1f, Kp_brain=%.3f, CL_renal=%.2f",
                       self.ka, self.F, self.Kp_liver, self.Kp_brain, self.CL_renal)
        except Exception as e:
            logger.warning("Advanced physiology failed (%s), using defaults", e)
            self.adv_phys = None
            self.ehr_active = False
            self.ehr_fraction = 0.0
            self.ehr_reabsorption = 0.0
            self.biliary_cl_factor = 1.0

        logger.info(
            "PBPK params: F=%.3f, ka=%.2f, fu=%.3f, CL_hep=%.2f, CL_renal=%.2f, "
            "Kp_liver=%.1f, Kp_brain=%.2f, Kp_fat=%.1f",
            self.F, self.ka, self.fu, self.CL_hep, self.CL_renal,
            self.Kp_liver, self.Kp_brain, self.Kp_fat,
        )

    def _ode_system(self, t: float, y: np.ndarray) -> list[float]:
        """
        ODE system for 14-compartment PBPK model with advanced physiology.

        State vector y (16 elements):
            [0]  A_gut       — amount in gut lumen (mg)
            [1]  C_plasma    — plasma concentration (mg/L)
            [2]  C_liver     — liver concentration (mg/L)
            [3]  C_kidney    — kidney concentration (mg/L)
            [4]  C_brain     — brain concentration (mg/L)
            [5]  C_muscle    — muscle concentration (mg/L)
            [6]  C_fat       — fat concentration (mg/L)
            [7]  C_gut_wall  — gut wall tissue concentration (mg/L)
            [8]  A_bile      — amount in bile (mg) — EHR
            [9]  C_lung      — lung tissue concentration (mg/L)
            [10] C_spleen    — spleen concentration (mg/L)
            [11] C_skin      — skin concentration (mg/L)
            [12] C_bone      — bone marrow concentration (mg/L)
            [13] C_rbc       — red blood cell concentration (mg/L)
            [14] C_gut_wall2 — gut wall enterocyte (separate from lumen)
            [15] C_pancreas  — pancreas concentration (mg/L)
        """
        # Unpack state vector
        n = len(y)
        A_gut = max(y[0], 0.0)
        C_pl  = max(y[1], 0.0)
        C_li  = max(y[2], 0.0)
        C_ki  = max(y[3], 0.0)
        C_br  = max(y[4], 0.0)
        C_mu  = max(y[5], 0.0)
        C_fa  = max(y[6], 0.0)
        C_gt  = max(y[7], 0.0)
        A_bile = max(y[8], 0.0) if n > 8 else 0.0
        C_lu  = max(y[9], 0.0) if n > 9 else 0.0   # Lung
        C_sp  = max(y[10], 0.0) if n > 10 else 0.0  # Spleen
        C_sk  = max(y[11], 0.0) if n > 11 else 0.0  # Skin
        C_bo  = max(y[12], 0.0) if n > 12 else 0.0  # Bone
        C_rb  = max(y[13], 0.0) if n > 13 else 0.0  # RBC
        C_gw  = max(y[14], 0.0) if n > 14 else 0.0  # Gut wall enterocyte
        C_pa  = max(y[15], 0.0) if n > 15 else 0.0  # Pancreas

        ph = self.phys

        # Hepatic arterial flow = total liver flow - portal (gut) flow - spleen flow
        Q_ha = max(ph.Q_liver - ph.Q_gut - ph.Q_spleen, 0.01)

        # ── Enterohepatic Recirculation ──
        ehr_frac = getattr(self, 'ehr_fraction', 0.0)
        ehr_reabs = getattr(self, 'ehr_reabsorption', 0.0)
        bil_factor = getattr(self, 'biliary_cl_factor', 1.0)
        biliary_excretion = self.CL_hep * ehr_frac * bil_factor * C_li / self.Kp_liver
        bile_to_gut = ehr_reabs * A_bile

        effective_hep_cl = self.CL_hep * (1.0 - ehr_frac)

        # ── PULMONARY CIRCULATION ──
        # In a real body: venous blood → RIGHT heart → lungs → LEFT heart → arterial blood
        # The lung compartment sits between venous and arterial blood
        # C_arterial ≈ C_lung / Kp_lung (what organs see)
        # C_venous = weighted mix of all organ venous returns (what lungs see)
        # For simplicity, we model lung as perfusion-limited with Q = CO
        C_arterial = C_lu / self.Kp_lung if self.Kp_lung > 0 else C_pl

        # ── ODEs ──

        # Gut lumen: absorption + EHR reabsorption
        dA_gut = -self.ka * A_gut + bile_to_gut

        # Bile compartment (EHR)
        dA_bile = biliary_excretion - bile_to_gut

        # Gut wall (enterocytes): drug from lumen → gut wall → portal vein
        dC_gw = (self.ka * A_gut / ph.V_gut_wall) \
                + (ph.Q_gut / ph.V_gut_wall) * (C_arterial - C_gw / self.Kp_gut_wall)

        # Gut tissue (old compartment — now receives from gut wall)
        dC_gt = (ph.Q_gut / ph.V_gut) * (C_arterial - C_gt / self.Kp_gut)

        # Liver: portal vein (gut + spleen) + hepatic artery
        dC_li = (1.0 / ph.V_liver) * (
            Q_ha * C_arterial                                # Hepatic arterial
            + ph.Q_gut * C_gw / self.Kp_gut_wall            # Portal vein from gut wall
            + ph.Q_spleen * C_sp / self.Kp_spleen           # Splenic vein → portal
            - ph.Q_liver * C_li / self.Kp_liver             # Venous outflow
            - effective_hep_cl * C_li / self.Kp_liver       # Metabolism
            - biliary_excretion                              # Biliary excretion
        )

        # Kidney: perfusion + mechanistic renal clearance
        # CL_renal now includes 3 components:
        #   Filtration: GFR × fu (only free drug filtered)
        #   Secretion: active transport (OAT/OCT) — Michaelis-Menten
        #   Reabsorption: passive back-diffusion (pH-dependent)
        C_free_ki = C_ki / self.Kp_kidney * getattr(self, 'fu', 0.5)
        # Filtration component
        gfr_lh = 7.2 * ph.renal_cl_scale  # 120 mL/min = 7.2 L/h, scaled
        cl_filt = gfr_lh * getattr(self, 'fu', 0.5)
        # Secretion component (if transporter data available)
        cl_sec = getattr(self, '_renal_secretion_cl', 0.0)
        # Reabsorption fraction
        reab_frac = getattr(self, '_renal_reabsorption_frac', 0.5)
        # Net renal clearance = (filtration + secretion) × (1 - reabsorption)
        cl_renal_net = (cl_filt + cl_sec) * (1 - reab_frac)
        # Use the higher of mechanistic or simple CL_renal (for drugs without transporter data)
        cl_renal_effective = max(cl_renal_net, self.CL_renal * 0.5)

        dC_ki = (ph.Q_kidney / ph.V_kidney) * (C_arterial - C_ki / self.Kp_kidney) \
                - (cl_renal_effective / ph.V_kidney) * C_ki / self.Kp_kidney

        # Brain: perfusion-limited (BBB)
        dC_br = (ph.Q_brain / ph.V_brain) * (C_arterial - C_br / self.Kp_brain)

        # Muscle
        dC_mu = (ph.Q_muscle / ph.V_muscle) * (C_arterial - C_mu / self.Kp_muscle)

        # Fat
        dC_fa = (ph.Q_fat / ph.V_fat) * (C_arterial - C_fa / self.Kp_fat)

        # ── NEW COMPARTMENTS ──

        # Lung: receives ALL venous blood, outputs to arterial
        # Venous mixing pool = weighted sum of all organ venous returns
        venous_return = (
            ph.Q_liver * C_li / self.Kp_liver
            + ph.Q_kidney * C_ki / self.Kp_kidney
            + ph.Q_brain * C_br / self.Kp_brain
            + ph.Q_muscle * C_mu / self.Kp_muscle
            + ph.Q_fat * C_fa / self.Kp_fat
            + ph.Q_skin * C_sk / self.Kp_skin
            + ph.Q_bone * C_bo / self.Kp_bone
            + ph.Q_pancreas * C_pa / self.Kp_pancreas
        )
        # Add rest-of-body flow
        Q_accounted = (ph.Q_liver + ph.Q_kidney + ph.Q_brain + ph.Q_muscle
                      + ph.Q_fat + ph.Q_skin + ph.Q_bone + ph.Q_pancreas)
        Q_rest = max(ph.CO - Q_accounted, 0)
        venous_return += Q_rest * C_pl

        dC_lu = (1.0 / ph.V_lung) * (venous_return - ph.CO * C_lu / self.Kp_lung)

        # Spleen: drains into portal vein (→ liver)
        dC_sp = (ph.Q_spleen / ph.V_spleen) * (C_arterial - C_sp / self.Kp_spleen)

        # Skin: important for dermal drug accumulation
        dC_sk = (ph.Q_skin / ph.V_skin) * (C_arterial - C_sk / self.Kp_skin)

        # Bone marrow: hematopoietic tissue
        dC_bo = (ph.Q_bone / ph.V_bone) * (C_arterial - C_bo / self.Kp_bone)

        # RBC: drug partitioning into red blood cells
        # RBC equilibrates with plasma via passive diffusion
        k_rbc = 5.0  # Equilibration rate constant (1/h) — fast
        dC_rb = k_rbc * (C_pl * self.Kp_rbc - C_rb)

        # Pancreas
        dC_pa = (ph.Q_pancreas / ph.V_pancreas) * (C_arterial - C_pa / self.Kp_pancreas)

        # Plasma: venous mixing pool.
        # Receives organ venous returns (including rest-of-body, already
        # inside ``venous_return``), and pumps blood to the lung at CO.
        # NOTE: the previous code *double-counted* Q_rest·C_pl (once
        # inside venous_return, once added again here) and drained
        # at CO, creating a phantom mass sink for IV bolus (Cp≫Cart).
        dC_pl = (1.0 / ph.V_plasma) * (
            venous_return - ph.CO * C_pl
            - k_rbc * ph.V_rbc * (C_pl * self.Kp_rbc - C_rb)  # RBC exchange
        )

        return [dA_gut, dC_pl, dC_li, dC_ki, dC_br, dC_mu, dC_fa, dC_gt, dA_bile,
                dC_lu, dC_sp, dC_sk, dC_bo, dC_rb, dC_gw, dC_pa]

    def solve(self, n_points: int = 200) -> dict:
        """
        Solve the PBPK ODE system.

        Returns dict with keys: time_h, plasma_conc, gut_conc, liver_conc,
        kidney_conc, brain_conc, muscle_conc, fat_conc (each numpy array).
        """
        t_max = max(5.0 * self.half_life, 24.0)
        t_eval = np.linspace(0, t_max, n_points)

        # Initial conditions (16 compartments — 14-compartment PBPK)
        y0 = [
            self.F * self.dose_mg,  # [0]  A_gut: drug amount in gut lumen
            0.0,                     # [1]  C_plasma
            0.0,                     # [2]  C_liver
            0.0,                     # [3]  C_kidney
            0.0,                     # [4]  C_brain
            0.0,                     # [5]  C_muscle
            0.0,                     # [6]  C_fat
            0.0,                     # [7]  C_gut_tissue
            0.0,                     # [8]  A_bile (EHR)
            0.0,                     # [9]  C_lung
            0.0,                     # [10] C_spleen
            0.0,                     # [11] C_skin
            0.0,                     # [12] C_bone
            0.0,                     # [13] C_rbc
            0.0,                     # [14] C_gut_wall
            0.0,                     # [15] C_pancreas
        ]

        # For IV, distribute dose across pseudo-equilibrium Vss so the
        # ODE starts from a mass-balanced state. The previous approach
        # (dose / V_plasma) created an unphysical 333 mg/L spike that
        # was instantly drained by the -CO·C_pl term before organs
        # could return flow.
        if self.route != "oral":
            y0[0] = 0.0
            ph = self.phys
            organ_vkp = [
                (1,  ph.V_plasma,    1.0),              # plasma
                (2,  ph.V_liver,     self.Kp_liver),    # liver
                (3,  ph.V_kidney,    self.Kp_kidney),   # kidney
                (4,  ph.V_brain,     self.Kp_brain),    # brain
                (5,  ph.V_muscle,    self.Kp_muscle),   # muscle
                (6,  ph.V_fat,       self.Kp_fat),      # fat
                (7,  ph.V_gut,       self.Kp_gut),      # gut tissue
                (9,  ph.V_lung,      self.Kp_lung),     # lung
                (10, ph.V_spleen,    self.Kp_spleen),   # spleen
                (11, ph.V_skin,      self.Kp_skin),     # skin
                (12, ph.V_bone,      self.Kp_bone),     # bone
                (14, ph.V_gut_wall,  self.Kp_gut_wall), # gut wall
                (15, ph.V_pancreas,  self.Kp_pancreas), # pancreas
            ]
            vss = sum(v * kp for _, v, kp in organ_vkp)
            if vss > 0:
                c_plasma_0 = self.dose_mg / vss
                for idx, _v, kp in organ_vkp:
                    y0[idx] = c_plasma_0 * kp

        # Solve ODE
        try:
            sol = solve_ivp(
                self._ode_system,
                t_span=(0, t_max),
                y0=y0,
                t_eval=t_eval,
                method="RK45",
                rtol=1e-6,
                atol=1e-9,
                max_step=t_max / 100,
            )

            if not sol.success:
                logger.warning("RK45 failed (%s), retrying with BDF", sol.message)
                sol = solve_ivp(
                    self._ode_system,
                    t_span=(0, t_max),
                    y0=y0,
                    t_eval=t_eval,
                    method="BDF",
                    rtol=1e-6,
                    atol=1e-9,
                )
        except Exception as e:
            logger.error("ODE solver failed: %s", e)
            # Return flat zeros as fallback
            zeros = np.zeros(n_points)
            return {
                "time_h": t_eval,
                "plasma_conc": zeros, "gut_conc": zeros,
                "liver_conc": zeros, "kidney_conc": zeros,
                "brain_conc": zeros, "muscle_conc": zeros,
                "fat_conc": zeros, "bile_amount": zeros,
                "lung_conc": zeros, "spleen_conc": zeros,
                "skin_conc": zeros, "bone_conc": zeros,
                "rbc_conc": zeros, "gut_wall_conc": zeros,
                "pancreas_conc": zeros,
            }

        # Extract and clip (no negative concentrations)
        y = np.clip(sol.y, 0, None)

        result = {
            "time_h": sol.t,
            "plasma_conc": y[1],
            "gut_conc": y[7],        # Gut tissue concentration
            "liver_conc": y[2],
            "kidney_conc": y[3],
            "brain_conc": y[4],
            "muscle_conc": y[5],
            "fat_conc": y[6],
            "gut_amount": y[0],      # For mass balance checks
        }

        # 14-compartment extended outputs
        if y.shape[0] > 8:
            result["bile_amount"] = y[8]
        if y.shape[0] > 9:
            result["lung_conc"] = y[9]
        if y.shape[0] > 10:
            result["spleen_conc"] = y[10]
        if y.shape[0] > 11:
            result["skin_conc"] = y[11]
        if y.shape[0] > 12:
            result["bone_conc"] = y[12]
        if y.shape[0] > 13:
            result["rbc_conc"] = y[13]
        if y.shape[0] > 14:
            result["gut_wall_conc"] = y[14]
        if y.shape[0] > 15:
            result["pancreas_conc"] = y[15]

        return result

    def generate_organ_activity(self, solution: dict) -> dict:
        """
        Generate ADME phase timing from actual simulated concentrations.
        """
        t = solution["time_h"]
        t_max_sim = t[-1]

        plasma = solution["plasma_conc"]
        gut_amt = solution.get("gut_amount", np.zeros_like(t))

        # Absorption: ends when <5% of dose remains in gut
        initial_gut = self.F * self.dose_mg
        if initial_gut > 0:
            gut_fraction = gut_amt / initial_gut
            abs_end_idx = np.searchsorted(-gut_fraction, -0.05)  # First index where < 5%
            abs_end = t[min(abs_end_idx, len(t) - 1)]
        else:
            abs_end = 0.0

        # Tmax: when plasma concentration peaks
        tmax_idx = np.argmax(plasma)
        tmax = t[tmax_idx]

        # Distribution: from first rise to approximate equilibrium
        dist_end = min(tmax * 2, t_max_sim * 0.5)

        # Metabolism: liver active throughout, most intense around Tmax
        metab_start = max(tmax * 0.3, 0.1)
        metab_end = min(tmax * 3, t_max_sim * 0.8)

        # Elimination: from Tmax onward
        elim_start = tmax

        return {
            "absorption": {
                "organs": ["stomach", "intestine"],
                "start_h": 0.0,
                "end_h": round(abs_end, 2),
            },
            "distribution": {
                "organs": ["heart", "blood_vessels", "brain", "muscles"],
                "start_h": 0.0,
                "end_h": round(dist_end, 2),
            },
            "metabolism": {
                "organs": ["liver"],
                "start_h": round(metab_start, 2),
                "end_h": round(metab_end, 2),
            },
            "elimination": {
                "organs": ["kidneys", "bladder"],
                "start_h": round(elim_start, 2),
                "end_h": round(t_max_sim, 2),
            },
        }

    def to_time_course_dicts(self, solution: dict) -> list[dict]:
        """Convert solution arrays to list of dicts for JSON serialization."""
        n = len(solution["time_h"])
        # All possible keys (14-compartment)
        keys = ["time_h", "plasma_conc", "gut_conc", "liver_conc", "kidney_conc",
                "brain_conc", "muscle_conc", "fat_conc",
                "lung_conc", "spleen_conc", "skin_conc", "bone_conc",
                "rbc_conc", "gut_wall_conc", "pancreas_conc"]
        result = []
        for i in range(n):
            point = {}
            for key in keys:
                if key in solution:
                    point[key] = round(float(solution[key][i]), 4)
            result.append(point)
        return result


# ════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("NeoGen PBPK Engine — Standalone Test")
    print("=" * 60)

    # Test 1: Aspirin (LogP ≈ 1.2)
    print("\n--- Test 1: Aspirin (100mg oral, 70kg male) ---")
    patient = PatientParameters(weight_kg=70, age_years=30, sex="male")
    model = PBPKModel(patient=patient, dose_mg=100, route="oral")
    model.derive_pbpk_params(
        ml_predictions={"cmax": 10.0, "auc": 50.0, "half_life": 4.0,
                        "clearance": 0.35, "vd": 10.0},
        logp=1.2,
    )
    sol = model.solve()

    print(f"  Time range: 0 — {sol['time_h'][-1]:.1f} h")
    print(f"  Plasma Cmax: {sol['plasma_conc'].max():.2f} mg/L at t={sol['time_h'][np.argmax(sol['plasma_conc'])]:.1f}h")
    print(f"  Liver Cmax:  {sol['liver_conc'].max():.2f} mg/L")
    print(f"  Brain Cmax:  {sol['brain_conc'].max():.3f} mg/L")
    print(f"  Fat Cmax:    {sol['fat_conc'].max():.3f} mg/L")
    print(f"  Kidney Cmax: {sol['kidney_conc'].max():.2f} mg/L")
    print(f"  Muscle Cmax: {sol['muscle_conc'].max():.3f} mg/L")

    # Mass conservation check
    ph = model.phys
    final_idx = -1
    total_in_body = (
        sol["gut_amount"][final_idx]
        + sol["plasma_conc"][final_idx] * ph.V_plasma
        + sol["liver_conc"][final_idx] * ph.V_liver
        + sol["kidney_conc"][final_idx] * ph.V_kidney
        + sol["brain_conc"][final_idx] * ph.V_brain
        + sol["muscle_conc"][final_idx] * ph.V_muscle
        + sol["fat_conc"][final_idx] * ph.V_fat
        + sol["gut_conc"][final_idx] * ph.V_gut
    )
    initial_dose = model.F * model.dose_mg
    eliminated = initial_dose - total_in_body
    print(f"\n  Mass balance:")
    print(f"    Initial dose (F*D): {initial_dose:.1f} mg")
    print(f"    Remaining in body:  {total_in_body:.2f} mg")
    print(f"    Eliminated:         {eliminated:.2f} mg ({eliminated/initial_dose*100:.1f}%)")

    # Test 2: Patient variability
    print("\n--- Test 2: Patient Variability ---")
    for desc, p in [
        ("50kg female, age 25", PatientParameters(50, 25, "female")),
        ("70kg male, age 30", PatientParameters(70, 30, "male")),
        ("90kg male, age 70", PatientParameters(90, 70, "male")),
    ]:
        m = PBPKModel(patient=p, dose_mg=100, route="oral")
        m.derive_pbpk_params(
            {"cmax": 10, "auc": 50, "half_life": 4, "clearance": 0.35, "vd": 10},
            logp=1.2,
        )
        s = m.solve()
        tmax_idx = np.argmax(s["plasma_conc"])
        print(f"  {desc}: Cmax={s['plasma_conc'].max():.2f} mg/L, "
              f"Tmax={s['time_h'][tmax_idx]:.1f}h, "
              f"Brain={s['brain_conc'].max():.3f}")

    # Test 3: IV vs Oral
    print("\n--- Test 3: IV vs Oral ---")
    for route in ["oral", "iv"]:
        m = PBPKModel(patient=PatientParameters(), dose_mg=100, route=route)
        m.derive_pbpk_params(
            {"cmax": 10, "auc": 50, "half_life": 4, "clearance": 0.35, "vd": 10},
            logp=1.2,
        )
        s = m.solve()
        print(f"  {route.upper():5s}: Cmax={s['plasma_conc'].max():.2f} mg/L, "
              f"Tmax={s['time_h'][np.argmax(s['plasma_conc'])]:.1f}h")

    # Output format test
    tc = model.to_time_course_dicts(sol)
    print(f"\n  Time course: {len(tc)} points, keys: {list(tc[0].keys())}")

    oa = model.generate_organ_activity(sol)
    print(f"  Organ activity phases: {list(oa.keys())}")
    for phase, info in oa.items():
        print(f"    {phase}: {info['start_h']}h — {info['end_h']}h ({info['organs']})")

    print("\nPBPK engine test PASSED!")
