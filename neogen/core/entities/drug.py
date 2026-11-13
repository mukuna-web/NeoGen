"""Drug entity — the root object of the NeoGen V1 simulation engine.

Every simulation begins with a Drug. The DrugFactory produces Drug instances
from SMILES strings by computing physicochemical properties (RDKit), predicting
ADME parameters (ML models), and overlaying literature values when available.

The Drug entity is IMMUTABLE after construction. If you need to modify
parameters (e.g., for sensitivity analysis), create a new Drug via
drug.with_overrides(fu=0.05).

Architecture reference: Layer 1 — Drug Definition & Input (Blueprint p.3)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from neogen.core.entities.pd import PDProfile


class BCSClass(Enum):
    """Biopharmaceutics Classification System.

    Determines dissolution and absorption behavior:
      I:   High solubility + High permeability → fast, complete absorption
      II:  Low solubility + High permeability → dissolution-limited
      III: High solubility + Low permeability → permeation-limited
      IV:  Low solubility + Low permeability → poor candidate
    """

    I = "BCS_I"
    II = "BCS_II"
    III = "BCS_III"
    IV = "BCS_IV"


class Ionization(Enum):
    """Ionization class — determines pH-dependent solubility and reabsorption."""

    ACID = "acid"
    BASE = "base"
    NEUTRAL = "neutral"
    ZWITTERION = "zwitterion"
    AMPHOLYTE = "ampholyte"


@dataclass(frozen=True)
class CYPMetabolismProfile:
    """Fraction metabolized (fm) by each CYP isoform for this drug.

    fm values must sum to ≤ 1.0. The remainder (1 - sum) is non-CYP
    clearance (renal, UGT, other).

    This is the MOST IMPORTANT parameter for DDI prediction:
      fm=0.9 via CYP3A4 → 10-fold AUC change with strong inhibitor
      fm=0.3 via CYP3A4 → 1.4-fold AUC change with same inhibitor
    """

    cyp3a4: float = 0.0
    cyp2d6: float = 0.0
    cyp2c9: float = 0.0
    cyp2c19: float = 0.0
    cyp1a2: float = 0.0
    cyp2c8: float = 0.0
    cyp2b6: float = 0.0
    cyp2e1: float = 0.0

    @property
    def total_cyp_fm(self) -> float:
        """Total fraction metabolized by CYP enzymes."""
        return (
            self.cyp3a4
            + self.cyp2d6
            + self.cyp2c9
            + self.cyp2c19
            + self.cyp1a2
            + self.cyp2c8
            + self.cyp2b6
            + self.cyp2e1
        )

    @property
    def non_cyp_fraction(self) -> float:
        """Fraction cleared by non-CYP pathways (renal, UGT, etc.)."""
        return 1.0 - min(self.total_cyp_fm, 1.0)

    @property
    def dominant_enzyme(self) -> Optional[str]:
        """The CYP isoform responsible for the largest fraction."""
        enzymes = {
            "CYP3A4": self.cyp3a4,
            "CYP2D6": self.cyp2d6,
            "CYP2C9": self.cyp2c9,
            "CYP2C19": self.cyp2c19,
            "CYP1A2": self.cyp1a2,
            "CYP2C8": self.cyp2c8,
            "CYP2B6": self.cyp2b6,
            "CYP2E1": self.cyp2e1,
        }
        max_enzyme = max(enzymes, key=enzymes.get)
        return max_enzyme if enzymes[max_enzyme] > 0.0 else None

    def fm_for(self, enzyme_name: str) -> float:
        """Get fm for a specific enzyme by name (e.g., 'CYP3A4')."""
        mapping = {
            "CYP3A4": self.cyp3a4,
            "CYP2D6": self.cyp2d6,
            "CYP2C9": self.cyp2c9,
            "CYP2C19": self.cyp2c19,
            "CYP1A2": self.cyp1a2,
            "CYP2C8": self.cyp2c8,
            "CYP2B6": self.cyp2b6,
            "CYP2E1": self.cyp2e1,
        }
        return mapping.get(enzyme_name.upper(), 0.0)


@dataclass(frozen=True)
class InhibitionProfile:
    """CYP inhibition constants (Ki, µM) for this drug acting as an inhibitor.

    A drug with ki_cyp3a4 = 0.01 µM is a POTENT CYP3A4 inhibitor (ketoconazole).
    A drug with ki_cyp3a4 = 100 µM is essentially non-inhibitory.

    CRITICAL: Use UNBOUND [I], not total. Highly protein-bound inhibitors are
    overpredicted by 10-100x if using total concentration.
    """

    ki_cyp3a4: Optional[float] = None
    ki_cyp2d6: Optional[float] = None
    ki_cyp2c9: Optional[float] = None
    ki_cyp2c19: Optional[float] = None
    ki_cyp1a2: Optional[float] = None
    ki_cyp2c8: Optional[float] = None
    ki_cyp2b6: Optional[float] = None
    ki_cyp2e1: Optional[float] = None

    # Mechanism-based inhibition (MBI) parameters
    # kinact = maximal inactivation rate (min⁻¹)
    # k_i_mbi = concentration for half-maximal inactivation (µM)
    kinact_cyp3a4: Optional[float] = None
    k_i_mbi_cyp3a4: Optional[float] = None

    def ki_for(self, enzyme_name: str) -> Optional[float]:
        """Get Ki for a specific enzyme by name."""
        mapping = {
            "CYP3A4": self.ki_cyp3a4,
            "CYP2D6": self.ki_cyp2d6,
            "CYP2C9": self.ki_cyp2c9,
            "CYP2C19": self.ki_cyp2c19,
            "CYP1A2": self.ki_cyp1a2,
            "CYP2C8": self.ki_cyp2c8,
            "CYP2B6": self.ki_cyp2b6,
            "CYP2E1": self.ki_cyp2e1,
        }
        return mapping.get(enzyme_name.upper())

    @property
    def is_inhibitor(self) -> bool:
        """Whether this drug inhibits any CYP enzyme."""
        return any(
            v is not None
            for v in [
                self.ki_cyp3a4,
                self.ki_cyp2d6,
                self.ki_cyp2c9,
                self.ki_cyp2c19,
                self.ki_cyp1a2,
                self.ki_cyp2c8,
                self.ki_cyp2b6,
                self.ki_cyp2e1,
            ]
        )

    @property
    def is_mbi(self) -> bool:
        """Whether this drug is a mechanism-based inhibitor."""
        return (
            self.kinact_cyp3a4 is not None and self.k_i_mbi_cyp3a4 is not None
        )


@dataclass(frozen=True)
class InductionProfile:
    """CYP induction parameters for this drug acting as an inducer.

    Induction modeled as: fold_increase = 1 + Indmax × [I] / (IndC50 + [I])
    where Indmax = maximal fold-induction, IndC50 = concentration for half-max.
    """

    indmax_cyp3a4: Optional[float] = None
    indc50_cyp3a4: Optional[float] = None
    indmax_cyp2b6: Optional[float] = None
    indc50_cyp2b6: Optional[float] = None
    indmax_cyp1a2: Optional[float] = None
    indc50_cyp1a2: Optional[float] = None
    indmax_cyp2c9: Optional[float] = None
    indc50_cyp2c9: Optional[float] = None

    @property
    def is_inducer(self) -> bool:
        """Whether this drug induces any CYP enzyme."""
        return any(
            v is not None
            for v in [
                self.indmax_cyp3a4,
                self.indmax_cyp2b6,
                self.indmax_cyp1a2,
                self.indmax_cyp2c9,
            ]
        )

    def params_for(
        self, enzyme_name: str
    ) -> tuple[Optional[float], Optional[float]]:
        """Get (Indmax, IndC50) for a specific CYP by name.

        Returns ``(None, None)`` if the CYP has no induction params or
        is not an inducible CYP in V1 (CYP2D6 is famously non-inducible).
        """
        key = enzyme_name.upper()
        mapping = {
            "CYP3A4": (self.indmax_cyp3a4, self.indc50_cyp3a4),
            "CYP2B6": (self.indmax_cyp2b6, self.indc50_cyp2b6),
            "CYP1A2": (self.indmax_cyp1a2, self.indc50_cyp1a2),
            "CYP2C9": (self.indmax_cyp2c9, self.indc50_cyp2c9),
        }
        return mapping.get(key, (None, None))


@dataclass(frozen=True)
class PhaseIIMetabolismProfile:
    """Non-CYP Phase II conjugation enzyme fractions.

    Phase II enzymes (UGT, SULT, aldehyde oxidase) metabolize drug in
    the same hepatocytes as CYP enzymes via conjugation reactions.
    Approximately 30% of marketed drugs have significant non-CYP
    clearance through these pathways.

    fm values represent the fraction of total hepatic clearance
    attributable to each Phase II enzyme. Combined with CYP fm values,
    they should sum to <= 1.0. The remainder is non-enzymatic clearance
    (renal, biliary, hydrolysis, etc.).
    """

    fm_ugt1a1: float = 0.0    # bilirubin, irinotecan
    fm_ugt2b7: float = 0.0    # morphine, zidovudine
    fm_ugt1a4: float = 0.0    # lamotrigine, olanzapine
    fm_sult1a1: float = 0.0   # paracetamol, minoxidil
    fm_aldehyde_oxidase: float = 0.0  # zaleplon, famciclovir

    @property
    def total_phase2_fm(self) -> float:
        """Total fraction metabolized by Phase II enzymes."""
        return (
            self.fm_ugt1a1
            + self.fm_ugt2b7
            + self.fm_ugt1a4
            + self.fm_sult1a1
            + self.fm_aldehyde_oxidase
        )


@dataclass(frozen=True)
class TransporterAffinities:
    """Drug interactions with membrane transporters.

    P-gp (gut): efflux pump. Inhibition → increased absorption (higher fa).
    OATP1B1 (liver): hepatic uptake. Inhibition → decreased liver uptake → higher systemic.
    OCT2 (kidney): renal secretion. Inhibition → decreased renal clearance.
    """

    is_pgp_substrate: bool = False
    is_pgp_inhibitor: bool = False
    pgp_ic50: Optional[float] = None  # µM

    is_oatp1b1_substrate: bool = False
    is_oatp1b1_inhibitor: bool = False
    oatp1b1_ic50: Optional[float] = None  # µM
    # V1.2 Track B: active uptake enhancement factor for OATP1B1
    # substrates. Active sinusoidal uptake concentrates drug in
    # hepatocytes, producing effective CLint that is 5-30x the
    # microsomal-scaled CLint (Shitara 2006, Kusuhara 2013). V1.1
    # used a flat 15x for every OATP1B1 substrate; V1.2 Track B
    # makes it per-drug. Drugs without an override fall back to
    # the V1.1 default of 15x. Under OATP1B1 inhibition, this
    # factor is modulated by the IC50 formula in
    # :mod:`neogen.engine.ddi.transporter_ddi`. Set to 1.0 to
    # disable uptake enhancement (e.g. for weak OATP substrates
    # where active uptake contribution is negligible).
    oatp1b1_uptake_factor: Optional[float] = None

    # V1.2 Fix 5: P-gp efflux absorption penalty factor [0, 1].
    # For P-gp substrates, effective Peff is reduced by this factor
    # in the absorption engine. Digoxin: 0.70 (Fabs ≈ 0.70).
    # None means no penalty (default for non-substrates).
    pgp_absorption_factor: Optional[float] = None

    is_oct2_substrate: bool = False
    is_oct2_inhibitor: bool = False
    oct2_ic50: Optional[float] = None  # µM

    is_bcrp_substrate: bool = False
    is_bcrp_inhibitor: bool = False
    bcrp_ic50: Optional[float] = None  # µM



@dataclass(frozen=True)
class Drug:
    """Root entity for NeoGen V1 simulation.

    Represents a single drug molecule with all properties needed for
    mechanistic PBPK simulation and DDI prediction.

    Construction:
        Use DrugFactory.build(smiles) to create Drug instances.
        DrugFactory handles RDKit computation, ML prediction, and
        literature override logic.

    Immutability:
        Drug is frozen. Use drug.with_overrides() to create variants
        for sensitivity analysis.
    """

    # ── Identity ────────────────────────────────────────────────
    name: str
    smiles: str

    # ── Physicochemical (RDKit-computed) ────────────────────────
    molecular_weight: float  # Da
    logp: float  # Partition coefficient (octanol/water)
    pka_acidic: Optional[float] = None  # Strongest acidic pKa
    pka_basic: Optional[float] = None  # Strongest basic pKa
    tpsa: float = 0.0  # Topological polar surface area (Å²)
    hbd: int = 0  # Hydrogen bond donors
    hba: int = 0  # Hydrogen bond acceptors
    rotatable_bonds: int = 0
    aromatic_rings: int = 0
    ionization_class: Ionization = Ionization.NEUTRAL

    # ── Biopharmaceutics ────────────────────────────────────────
    bcs_class: BCSClass = BCSClass.II
    solubility_mg_per_ml: Optional[float] = None  # Aqueous solubility at pH 6.8
    peff_jejunum: Optional[float] = None  # Effective permeability (×10⁻⁴ cm/s)

    # Pre-absorptive degradation: fraction of dose lost to gastric acid
    # degradation before reaching the small intestine. Default 0.0 (no
    # degradation). Set to 0.40 for omeprazole (acid-labile PPI).
    pre_absorptive_degradation_fraction: float = 0.0

    # ── Gut-Wall Metabolism (V1.2) ──────────────────────────────
    # Per-drug gut extraction coefficient that scales enterocyte
    # CYP3A4-mediated loss during absorption. The V1 model used a
    # fixed 0.3 constant for every drug, which systematically
    # under-predicted gut-wall first-pass for high-CLint drugs
    # like simvastatin (clinical Fg ~ 0.35 vs V1-predicted ~0.87).
    #
    # When set, PermeabilityEngine uses this override; otherwise it
    # falls back to the global default of 0.3. The proper Qgut
    # model derives this from the drug-specific enterocyte CLint,
    # which V1.2 does not yet track — we use a calibrated scalar
    # as a practical proxy until the full model lands in V1.3.
    #
    # Reasonable range:
    #   0.20 — moderate-fm CYP3A4 substrates (diazepam, nifedipine)
    #   0.30 — V1 default (midazolam clinical Fg ~0.57 matches)
    #   0.50 — high-fm substrates (atorvastatin)
    #   0.80 — extreme-fm + high-CLint (simvastatin, saquinavir)
    gut_extraction_alpha: Optional[float] = None

    # ── Gut-Wall CYP2D6 Extraction (V1.2) ──────────────────────
    # Per-drug CYP2D6 gut-wall extraction coefficient. CYP2D6 is
    # present in the gut wall at ~10% of CYP3A4 abundance. For drugs
    # with significant CYP2D6 metabolism (fm_2D6 >= 0.30), this
    # parameter enables an additional fg reduction beyond the CYP3A4
    # gut-wall model. None = no CYP2D6 gut-wall extraction (default).
    gut_cyp2d6_extraction_alpha: Optional[float] = None

    # ── Protein Binding ─────────────────────────────────────────
    fu_plasma: float = 0.5  # Fraction unbound in plasma (0-1)
    fu_gut: Optional[float] = None  # Fraction unbound in gut (enterocyte)
    blood_to_plasma_ratio: float = 1.0  # B:P ratio

    # ── Distribution Calibration (optional literature override) ──
    # When populated, the TissuePartitioner scales its R&R-derived Kp
    # values so the predicted Vss matches this observed value. Tissue
    # *ratios* (liver:muscle:fat) remain mechanistic from R&R; only the
    # magnitude is calibrated. This is the standard PBPK practice for
    # drugs where R&R is known to systematically mis-predict
    # distribution (strong lipophilic bases, zwitterions, P-gp
    # substrates, tissue-binding drugs like digoxin).
    vss_observed_L_per_kg: Optional[float] = None  # clinical Vss in L/kg

    # ── Intrinsic Clearance ─────────────────────────────────────
    clint_microsomal: Optional[float] = None  # µL/min/mg microsomal protein
    clint_hepatocyte: Optional[float] = None  # µL/min/million hepatocytes
    # V1.2 Fix 4: pre-scaled CLint override (L/h) derived from
    # published Eh, bypassing MPPGL scaling. When set, takes
    # absolute priority over clint_microsomal and clint_hepatocyte.
    clint_scaled_override_L_per_h: Optional[float] = None

    # ── Michaelis-Menten Kinetics (saturable metabolism) ────────
    # For drugs whose therapeutic concentrations approach Km, linear
    # CLint (= Vmax/Km) overestimates clearance. When both are set,
    # HepaticClearanceEngine uses CLint = Vmax/(2*Km) instead of the
    # standard microsomal/hepatocyte scaling, capturing the ~50%
    # reduction in effective CLint at concentrations near Km.
    km_um: Optional[float] = None  # Michaelis constant (µM)
    vmax_pmol_per_min_per_mg: Optional[float] = None  # Vmax (pmol/min/mg protein)

    # ── Direct CL Override (V1.2 structural fix) ────────────────
    # When measured CL > 0.8 × Qh, the well-stirred back-calculation
    # produces negative or astronomical CLint. For these extrahepatic-
    # clearance drugs, bypass the mechanistic hepatic/renal split and
    # inject measured total CL directly into the simulation.
    cl_total_override_L_per_h: Optional[float] = None

    # Measured clinical t½ bypass. The 14-compartment ODE's terminal
    # phase includes redistribution artifacts that extend the simulated
    # t½ beyond the clinical monoexponential value. When set, this
    # value replaces the ODE-derived half-life in PK metrics.
    half_life_override_h: Optional[float] = None

    # Hepatic/renal CL split overrides (V1.2 structural fix).
    # When measured total CL and fe are both available, we can split:
    #   CLr = CL × fe,  CLh = CL × (1 - fe)
    # These bypass HepaticClearanceEngine and RenalClearanceEngine.
    cl_hepatic_override_L_per_h: Optional[float] = None
    cl_renal_override_L_per_h: Optional[float] = None

    # ── Observed Hepatic Clearance Override ─────────────────────
    # For drugs cleared predominantly by non-CYP pathways (UGT, SULT,
    # esterases, P-gp efflux) where the microsomal CLint dramatically
    # under-represents total hepatic clearance.  When set, the
    # HepaticClearanceEngine returns this value directly instead of
    # computing from CLint via well-stirred model.  Units: L/h.
    cl_hepatic_observed: Optional[float] = None

    # ── Observed Oral Bioavailability ──────────────────────────
    # Published absolute oral bioavailability (F). When set, the
    # orchestrator uses this to correct the mechanistic Fa×Fg×Fh
    # estimate for oral doses. This is standard PBPK practice
    # for drugs where absorption is known from IV/oral crossover
    # studies. Range [0, 1]. ``None`` means "use mechanistic estimate".
    f_oral_observed: Optional[float] = None

    # ── Renal ───────────────────────────────────────────────────
    fraction_excreted_unchanged: float = 0.0  # fe (0-1)
    # Per-drug active tubular secretion clearance override (L/h).
    # When set, the RenalClearanceEngine adds this as an additional
    # secretion component (e.g. P-gp mediated secretion in renal
    # proximal tubules). Digoxin: 5.0 L/h (Drescher 2003).
    renal_secretion_cl_L_per_h: Optional[float] = None

    # ── CYP Metabolism ──────────────────────────────────────────
    metabolism: CYPMetabolismProfile = field(
        default_factory=CYPMetabolismProfile
    )
    # ── Phase II Metabolism (UGT/SULT/AO) ──────────────────────
    phase2_metabolism: PhaseIIMetabolismProfile = field(
        default_factory=PhaseIIMetabolismProfile
    )
    inhibition: InhibitionProfile = field(default_factory=InhibitionProfile)
    induction: InductionProfile = field(default_factory=InductionProfile)

    # ── Transporters ────────────────────────────────────────────
    transporters: TransporterAffinities = field(
        default_factory=TransporterAffinities
    )

    # ── Pharmacodynamics (V1 P4) ────────────────────────────────
    # Optional: drugs without PD data are still valid; the P4 engine
    # skips any missing sub-models. See neogen/core/entities/pd.py.
    pd_profile: Optional["PDProfile"] = None

    # ── Statin Prodrug Correction ──────────────────────────────────
    # Fraction of total circulating drug that exists as the
    # pharmacologically active hydroxy-acid form. Published clinical
    # PK data (Cmax, AUC) report the acid form only. When set, PK
    # metrics (Cmax, AUC) are corrected by multiplying by this
    # fraction after simulation.
    #   Atorvastatin: 0.30 (Lennernas 2003)
    #   Simvastatin:  0.15 (Vickers 1990)
    active_fraction: Optional[float] = None

    # ── Prediction Confidence (V1.2 UQ) ──────────────────────────
    # Aggregate confidence score from bootstrap ensemble models.
    # Range [0, 1]: higher = more reliable ML predictions.
    # Computed as the geometric mean of per-property confidence
    # scores (1 / (1 + CV) where CV = ensemble_std / |mean|).
    # None for literature drugs or when ensemble models are
    # unavailable. When < 0.3, DrugFactory logs a warning.
    prediction_confidence: Optional[float] = None

    # ── Data Source Tracking ────────────────────────────────────
    source: str = "predicted"  # "predicted", "literature", "hybrid"

    def with_overrides(self, **kwargs) -> Drug:
        """Create a new Drug with specific parameter overrides.

        Useful for sensitivity analysis:
            drug_low_fu = drug.with_overrides(fu_plasma=0.01)
            drug_high_fu = drug.with_overrides(fu_plasma=0.10)
        """
        return replace(self, **kwargs)

    @property
    def fu_plasma_safe(self) -> float:
        """fu clamped to physiological minimum (never exactly zero)."""
        return max(self.fu_plasma, 0.001)

    @property
    def fu_gut_effective(self) -> float:
        """fu in gut — if not measured, assume equal to fu_plasma."""
        if self.fu_gut is not None:
            return max(self.fu_gut, 0.001)
        return self.fu_plasma_safe

    @property
    def is_cyp3a4_substrate(self) -> bool:
        """Whether CYP3A4 contributes meaningfully to metabolism."""
        return self.metabolism.cyp3a4 > 0.1

    @property
    def is_highly_bound(self) -> bool:
        """fu < 0.1 — highly protein-bound. Displacement interactions possible."""
        return self.fu_plasma < 0.1

    @property
    def is_narrow_therapeutic_index(self) -> bool:
        """Placeholder — should be set from literature for drugs like warfarin."""
        return False

    @property
    def has_pd_profile(self) -> bool:
        """Whether any PD data (effect comp, Emax, receptor, toxicity) is attached."""
        return self.pd_profile is not None and self.pd_profile.has_any_pd_data

    def __repr__(self) -> str:
        return (
            f"Drug(name={self.name!r}, MW={self.molecular_weight:.1f}, "
            f"logP={self.logp:.2f}, fu={self.fu_plasma:.3f}, "
            f"BCS={self.bcs_class.value}, source={self.source!r})"
        )
