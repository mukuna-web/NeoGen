"""Curated CYP Enzyme Database — kinetic parameters for DDI prediction.

This is the SCIENTIFIC BACKBONE of NeoGen V1's DDI engine.
Every parameter here is sourced from peer-reviewed literature,
FDA DDI guidance (2020), or validated PBPK models.

Sources:
  - FDA Guidance: In Vitro DDI Studies (2020), Table 6
  - Simcyp compound library (cross-validation)
  - Galetin et al. (2005): fm values
  - Rowland Yeo et al. (2011): CYP abundance
  - Gertz et al. (2010): Gut wall CYP3A4

CRITICAL NOTES:
  1. Ki values must be UNBOUND. Using total [I] overpredicts by 10-100x.
  2. fm determines DDI magnitude: fm=0.9 → 10x AUC change; fm=0.3 → 1.4x.
  3. CYP2D6 is NOT inducible. This is unique among CYPs.
  4. Gut wall CYP3A4 abundance is ~1% of liver but clinically important.
"""

from neogen.core.entities.enzyme import Enzyme, EnzymeFamily, EnzymeLocation

# ═══════════════════════════════════════════════════════════════════
# CYP ENZYME DEFINITIONS
# Abundance: Rowland Yeo et al. (2011), Simcyp V21
# kdeg: Obach et al. (2007), Yang et al. (2008)
# ═══════════════════════════════════════════════════════════════════

CYP3A4 = Enzyme(
    name="CYP3A4",
    family=EnzymeFamily.CYP,
    abundance_liver_pmol_per_mg=137.0,  # Most abundant CYP in liver
    abundance_gut_pmol_per_mg=1.46,  # ~1% of liver, but clinically critical
    kdeg_liver=0.0193,  # t½ ≈ 36h
    kdeg_gut=0.029,  # t½ ≈ 24h (faster gut turnover)
    is_inducible=True,
    induction_receptor="PXR",
    is_polymorphic=False,
    locations=(EnzymeLocation.LIVER, EnzymeLocation.GUT_WALL),
)

CYP2D6 = Enzyme(
    name="CYP2D6",
    family=EnzymeFamily.CYP,
    abundance_liver_pmol_per_mg=8.0,  # Low abundance but ~25% of drugs
    abundance_gut_pmol_per_mg=0.0,  # Negligible in gut
    kdeg_liver=0.0193,
    kdeg_gut=0.0,
    is_inducible=False,  # ← UNIQUE: CYP2D6 cannot be induced
    induction_receptor=None,
    is_polymorphic=True,
    poor_metabolizer_frequency=0.07,  # ~7% Caucasian
    locations=(EnzymeLocation.LIVER,),
)

CYP2C9 = Enzyme(
    name="CYP2C9",
    family=EnzymeFamily.CYP,
    abundance_liver_pmol_per_mg=73.0,
    abundance_gut_pmol_per_mg=0.0,
    kdeg_liver=0.0193,
    kdeg_gut=0.0,
    is_inducible=True,
    induction_receptor="PXR",
    is_polymorphic=True,
    poor_metabolizer_frequency=0.03,  # ~3% Caucasian
    locations=(EnzymeLocation.LIVER,),
)

CYP2C19 = Enzyme(
    name="CYP2C19",
    family=EnzymeFamily.CYP,
    abundance_liver_pmol_per_mg=14.0,
    abundance_gut_pmol_per_mg=0.0,
    kdeg_liver=0.0193,
    kdeg_gut=0.0,
    is_inducible=True,
    induction_receptor="PXR",
    is_polymorphic=True,
    poor_metabolizer_frequency=0.03,  # ~3% Caucasian, ~15% Asian
    locations=(EnzymeLocation.LIVER,),
)

CYP1A2 = Enzyme(
    name="CYP1A2",
    family=EnzymeFamily.CYP,
    abundance_liver_pmol_per_mg=52.0,
    abundance_gut_pmol_per_mg=0.0,
    kdeg_liver=0.0193,
    kdeg_gut=0.0,
    is_inducible=True,
    induction_receptor="AhR",  # Aryl hydrocarbon receptor (smoking!)
    is_polymorphic=False,
    locations=(EnzymeLocation.LIVER,),
)

CYP2C8 = Enzyme(
    name="CYP2C8",
    family=EnzymeFamily.CYP,
    abundance_liver_pmol_per_mg=24.0,
    abundance_gut_pmol_per_mg=0.0,
    kdeg_liver=0.0193,
    kdeg_gut=0.0,
    is_inducible=True,
    induction_receptor="PXR",
    is_polymorphic=True,
    poor_metabolizer_frequency=0.02,
    locations=(EnzymeLocation.LIVER,),
)

CYP2B6 = Enzyme(
    name="CYP2B6",
    family=EnzymeFamily.CYP,
    abundance_liver_pmol_per_mg=17.0,
    abundance_gut_pmol_per_mg=0.0,
    kdeg_liver=0.0193,
    kdeg_gut=0.0,
    is_inducible=True,
    induction_receptor="CAR",
    is_polymorphic=True,
    poor_metabolizer_frequency=0.04,
    locations=(EnzymeLocation.LIVER,),
)

CYP2E1 = Enzyme(
    name="CYP2E1",
    family=EnzymeFamily.CYP,
    abundance_liver_pmol_per_mg=49.0,
    abundance_gut_pmol_per_mg=0.0,
    kdeg_liver=0.0193,
    kdeg_gut=0.0,
    is_inducible=True,
    induction_receptor="ethanol",  # Substrate-mediated stabilization
    is_polymorphic=False,
    locations=(EnzymeLocation.LIVER,),
)


# ═══════════════════════════════════════════════════════════════════
# ENZYME REGISTRY — lookup by name
# ═══════════════════════════════════════════════════════════════════

ENZYME_REGISTRY: dict[str, Enzyme] = {
    "CYP3A4": CYP3A4,
    "CYP2D6": CYP2D6,
    "CYP2C9": CYP2C9,
    "CYP2C19": CYP2C19,
    "CYP1A2": CYP1A2,
    "CYP2C8": CYP2C8,
    "CYP2B6": CYP2B6,
    "CYP2E1": CYP2E1,
}


def get_enzyme(name: str) -> Enzyme:
    """Get enzyme by name (case-insensitive)."""
    key = name.upper().replace("-", "").replace(" ", "")
    if key not in ENZYME_REGISTRY:
        raise ValueError(
            f"Unknown enzyme: {name}. "
            f"Available: {list(ENZYME_REGISTRY.keys())}"
        )
    return ENZYME_REGISTRY[key]


# ═══════════════════════════════════════════════════════════════════
# FDA INDEX INHIBITORS — Ki values (µM, UNBOUND)
# Source: FDA DDI Guidance (2020), Table 6 + published studies
# These are the gold-standard inhibitors used in clinical DDI studies.
# ═══════════════════════════════════════════════════════════════════

FDA_INDEX_INHIBITORS: dict[str, dict[str, float]] = {
    # Strong CYP3A4 inhibitors (expected AUC ratio ≥5)
    "ketoconazole": {"CYP3A4": 0.015},  # Strongest. fu~0.01
    "itraconazole": {"CYP3A4": 0.0027},
    "clarithromycin": {"CYP3A4": 3.3},  # Also MBI
    "ritonavir": {"CYP3A4": 0.017},  # Multi-CYP inhibitor

    # Moderate CYP3A4 inhibitors (AUC ratio 2-5)
    "erythromycin": {"CYP3A4": 5.0},  # Also MBI
    "diltiazem": {"CYP3A4": 4.0},  # Also MBI
    "verapamil": {"CYP3A4": 1.2},
    "fluconazole": {"CYP3A4": 7.0, "CYP2C9": 7.0, "CYP2C19": 3.0},

    # CYP2D6 inhibitors
    "fluoxetine": {"CYP2D6": 0.07},  # Potent, long-acting
    "paroxetine": {"CYP2D6": 0.065},  # Potent
    "quinidine": {"CYP2D6": 0.04},  # Index inhibitor for CYP2D6
    "bupropion": {"CYP2D6": 6.5},  # Moderate

    # CYP2C9 inhibitors
    "fluconazole_2c9": {"CYP2C9": 7.0},
    "amiodarone": {"CYP2C9": 2.0},

    # CYP2C19 inhibitors
    "omeprazole": {"CYP2C19": 2.0},
    "fluvoxamine": {"CYP2C19": 0.2, "CYP1A2": 0.2},

    # CYP1A2 inhibitors
    "fluvoxamine_1a2": {"CYP1A2": 0.2},
    "ciprofloxacin": {"CYP1A2": 5.0},
}


# ═══════════════════════════════════════════════════════════════════
# FDA INDEX SUBSTRATES — fm values (fraction metabolized)
# Source: FDA DDI Guidance (2020), Galetin et al. (2005)
#
# fm is the MOST IMPORTANT parameter for DDI quantification.
# AUC_ratio = 1 / (1 - fm + fm/(1 + [I]/Ki))
# ═══════════════════════════════════════════════════════════════════

FDA_INDEX_SUBSTRATES: dict[str, dict[str, float]] = {
    # CYP3A4 substrates
    "midazolam": {"CYP3A4": 0.96},  # Index substrate — almost pure 3A4
    "simvastatin": {"CYP3A4": 0.95},
    "lovastatin": {"CYP3A4": 0.90},
    "atorvastatin": {"CYP3A4": 0.70, "CYP2C8": 0.10},
    "nifedipine": {"CYP3A4": 0.90},
    "cyclosporine": {"CYP3A4": 0.85},
    "tacrolimus": {"CYP3A4": 0.95},
    "buspirone": {"CYP3A4": 0.95},
    "felodipine": {"CYP3A4": 0.90},
    "triazolam": {"CYP3A4": 0.95},
    "alprazolam": {"CYP3A4": 0.80},
    "sildenafil": {"CYP3A4": 0.75, "CYP2C9": 0.20},

    # CYP2D6 substrates
    "dextromethorphan": {"CYP2D6": 0.90},  # Index substrate
    "codeine": {"CYP2D6": 0.80, "CYP3A4": 0.10},  # Prodrug!
    "metoprolol": {"CYP2D6": 0.80},
    "tramadol": {"CYP2D6": 0.60, "CYP3A4": 0.30},
    "tamoxifen": {"CYP2D6": 0.70, "CYP3A4": 0.20},

    # CYP2C9 substrates
    "s_warfarin": {"CYP2C9": 0.85},  # Index substrate — narrow TI
    "celecoxib": {"CYP2C9": 0.75, "CYP3A4": 0.15},
    "phenytoin": {"CYP2C9": 0.80, "CYP2C19": 0.10},
    "losartan": {"CYP2C9": 0.60, "CYP3A4": 0.30},

    # CYP2C19 substrates
    "omeprazole_sub": {"CYP2C19": 0.80, "CYP3A4": 0.15},
    "clopidogrel": {"CYP2C19": 0.45, "CYP3A4": 0.35, "CYP2B6": 0.10},
    "diazepam": {"CYP2C19": 0.50, "CYP3A4": 0.40},

    # CYP1A2 substrates
    "caffeine": {"CYP1A2": 0.95},  # Index substrate
    "theophylline": {"CYP1A2": 0.90},
    "tizanidine": {"CYP1A2": 0.95},
}


# ═══════════════════════════════════════════════════════════════════
# MECHANISM-BASED INHIBITORS — kinact and KI
# Source: FDA DDI Guidance, Grimm et al. (2009)
# These inhibitors DESTROY the enzyme irreversibly.
# Recovery requires new enzyme synthesis (~3-5 days).
# ═══════════════════════════════════════════════════════════════════

MBI_PARAMETERS: dict[str, dict] = {
    "clarithromycin": {
        "enzyme": "CYP3A4",
        "kinact_per_min": 0.054,  # Maximal inactivation rate
        "KI_um": 5.49,  # Half-maximal inactivation conc
    },
    "erythromycin": {
        "enzyme": "CYP3A4",
        "kinact_per_min": 0.040,
        "KI_um": 8.0,
    },
    "diltiazem": {
        "enzyme": "CYP3A4",
        "kinact_per_min": 0.029,
        "KI_um": 1.5,
    },
    "verapamil": {
        "enzyme": "CYP3A4",
        "kinact_per_min": 0.026,
        "KI_um": 2.0,
    },
    "mibefradil": {
        "enzyme": "CYP3A4",
        "kinact_per_min": 0.089,
        "KI_um": 0.22,  # Withdrawn from market (too strong)
    },
}


# ═══════════════════════════════════════════════════════════════════
# CYP INDUCERS — Indmax and IndC50
# Source: FDA DDI Guidance, Fahmi et al. (2008)
# Induction increases enzyme synthesis via nuclear receptors.
# ═══════════════════════════════════════════════════════════════════

INDUCTION_PARAMETERS: dict[str, dict] = {
    "rifampicin": {
        "CYP3A4": {"Indmax": 12.0, "IndC50_um": 0.31},  # Strongest inducer
        "CYP2C9": {"Indmax": 3.0, "IndC50_um": 0.5},
        "CYP2C19": {"Indmax": 3.0, "IndC50_um": 0.5},
        "CYP2B6": {"Indmax": 8.0, "IndC50_um": 0.3},
    },
    "carbamazepine": {
        "CYP3A4": {"Indmax": 5.0, "IndC50_um": 10.0},
        "CYP2B6": {"Indmax": 3.0, "IndC50_um": 15.0},
    },
    "phenytoin": {
        "CYP3A4": {"Indmax": 4.0, "IndC50_um": 5.0},
        "CYP2C9": {"Indmax": 2.0, "IndC50_um": 8.0},
    },
    "phenobarbital": {
        "CYP3A4": {"Indmax": 4.0, "IndC50_um": 20.0},
        "CYP2B6": {"Indmax": 5.0, "IndC50_um": 15.0},
    },
    "efavirenz": {
        "CYP3A4": {"Indmax": 3.0, "IndC50_um": 2.0},
        "CYP2B6": {"Indmax": 4.0, "IndC50_um": 3.0},
    },
    "st_johns_wort": {
        "CYP3A4": {"Indmax": 5.0, "IndC50_um": 0.1},  # Hyperforin
    },
    # CYP1A2 inducers (via AhR)
    "smoking": {
        "CYP1A2": {"Indmax": 4.0, "IndC50_um": 0.01},  # Polycyclic aromatics
    },
    "omeprazole_inducer": {
        "CYP1A2": {"Indmax": 1.5, "IndC50_um": 5.0},
    },
}


# ═══════════════════════════════════════════════════════════════════
# TRANSPORTER PARAMETERS — IC50 values (µM)
# Source: ISWG recommendations, FDA DDI Guidance
# ═══════════════════════════════════════════════════════════════════

TRANSPORTER_INHIBITORS: dict[str, dict] = {
    # P-gp inhibitors (gut → ↑ absorption; kidney → ↓ secretion)
    "cyclosporine": {"P-gp": 1.0, "OATP1B1": 0.1},
    "quinidine": {"P-gp": 2.0},
    "verapamil_pgp": {"P-gp": 5.0},
    "ritonavir_pgp": {"P-gp": 3.0, "OATP1B1": 0.3},
    "ketoconazole_pgp": {"P-gp": 3.0},

    # OATP1B1 inhibitors (liver → ↓ hepatic uptake → ↑ systemic)
    "rifampicin_single": {"OATP1B1": 0.48},  # Single dose inhibits!
    "gemfibrozil": {"OATP1B1": 4.0, "CYP2C8": 4.0},
    "cyclosporine_oatp": {"OATP1B1": 0.1},

    # OCT2 inhibitors (kidney → ↓ secretion)
    "cimetidine": {"OCT2": 100.0},
    "trimethoprim": {"OCT2": 50.0},
}
