"""
Full CYP Enzyme Kinetics Model
================================
Michaelis-Menten kinetics for 15 CYP isoforms with:
- Vmax (pmol/min/pmol CYP) — maximum metabolic rate
- Km (µM) — substrate concentration at 50% Vmax
- Enzyme abundance (pmol/mg microsomal protein)
- MPPGL (mg protein/g liver) — scales in vitro → in vivo
- Liver weight scaling

This replaces our simple "CYP3A4 = yes/no" with real enzyme kinetics.

Reference: Rowland Yeo et al., Drug Metab Dispos 2004
           Zanger & Schwab, Pharmacol Ther 2013
           Simcyp CYP abundance data
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CYPIsoform:
    """Single CYP enzyme isoform with kinetic parameters."""
    name: str
    abundance_pmol_mg: float      # pmol CYP / mg microsomal protein (adult liver mean)
    abundance_cv: float = 0.3     # Coefficient of variation (population variability)
    substrates: list[str] = field(default_factory=list)
    inhibitors: list[str] = field(default_factory=list)
    inducers: list[str] = field(default_factory=list)
    percent_of_total: float = 0   # % of total hepatic CYP content
    primary_reactions: str = ""    # What this enzyme does


# ════════════════════════════════════════════════════════════════
# CYP ISOFORM DATABASE (15 isoforms)
# ════════════════════════════════════════════════════════════════

CYP_ISOFORMS = {
    "CYP3A4": CYPIsoform(
        name="CYP3A4", abundance_pmol_mg=108.0, abundance_cv=0.50,
        percent_of_total=30,
        primary_reactions="Oxidation of ~50% of all marketed drugs",
        substrates=["midazolam", "nifedipine", "cyclosporine", "simvastatin", "atorvastatin",
                    "tacrolimus", "diazepam", "erythromycin", "fentanyl", "carbamazepine"],
        inhibitors=["ketoconazole", "itraconazole", "ritonavir", "clarithromycin",
                    "grapefruit_juice", "verapamil", "diltiazem"],
        inducers=["rifampicin", "carbamazepine", "phenytoin", "St_Johns_Wort", "efavirenz"],
    ),
    "CYP3A5": CYPIsoform(
        name="CYP3A5", abundance_pmol_mg=2.0, abundance_cv=1.0,
        percent_of_total=1,
        primary_reactions="Overlaps with CYP3A4; polymorphic (only 20% Caucasians express)",
        substrates=["tacrolimus", "midazolam", "nifedipine"],
        inhibitors=["ketoconazole"],
        inducers=["rifampicin"],
    ),
    "CYP2D6": CYPIsoform(
        name="CYP2D6", abundance_pmol_mg=10.0, abundance_cv=0.70,
        percent_of_total=2,
        primary_reactions="Metabolizes ~25% of drugs despite low abundance. Highly polymorphic.",
        substrates=["codeine", "tramadol", "dextromethorphan", "metoprolol", "tamoxifen",
                    "fluoxetine", "paroxetine", "risperidone", "atomoxetine", "venlafaxine"],
        inhibitors=["fluoxetine", "paroxetine", "quinidine", "bupropion", "terbinafine"],
        inducers=[],  # CYP2D6 is NOT inducible — unique among CYPs
    ),
    "CYP2C9": CYPIsoform(
        name="CYP2C9", abundance_pmol_mg=73.0, abundance_cv=0.40,
        percent_of_total=20,
        primary_reactions="NSAIDs, warfarin (S-enantiomer), phenytoin, losartan",
        substrates=["warfarin", "phenytoin", "losartan", "ibuprofen", "diclofenac",
                    "celecoxib", "glipizide", "tolbutamide"],
        inhibitors=["fluconazole", "amiodarone", "sulfaphenazole", "miconazole"],
        inducers=["rifampicin"],
    ),
    "CYP2C19": CYPIsoform(
        name="CYP2C19", abundance_pmol_mg=14.0, abundance_cv=0.60,
        percent_of_total=4,
        primary_reactions="PPIs, clopidogrel activation, diazepam, some antidepressants",
        substrates=["omeprazole", "clopidogrel", "diazepam", "citalopram",
                    "voriconazole", "proguanil", "phenytoin"],
        inhibitors=["omeprazole", "fluoxetine", "fluvoxamine", "ticlopidine"],
        inducers=["rifampicin", "carbamazepine"],
    ),
    "CYP1A2": CYPIsoform(
        name="CYP1A2", abundance_pmol_mg=45.0, abundance_cv=0.50,
        percent_of_total=13,
        primary_reactions="Caffeine, theophylline, clozapine, olanzapine, PAH activation",
        substrates=["caffeine", "theophylline", "clozapine", "olanzapine",
                    "tizanidine", "melatonin", "ropivacaine"],
        inhibitors=["fluvoxamine", "ciprofloxacin", "cimetidine"],
        inducers=["smoking", "charbroiled_meat", "cruciferous_vegetables", "omeprazole"],
    ),
    "CYP2E1": CYPIsoform(
        name="CYP2E1", abundance_pmol_mg=49.0, abundance_cv=0.40,
        percent_of_total=7,
        primary_reactions="Ethanol, paracetamol→NAPQI, volatile anesthetics, carcinogens",
        substrates=["paracetamol", "ethanol", "isoflurane", "sevoflurane",
                    "chlorzoxazone", "dapsone"],
        inhibitors=["disulfiram", "diethyldithiocarbamate"],
        inducers=["ethanol_chronic", "isoniazid", "obesity", "diabetes", "fasting"],
    ),
    "CYP2B6": CYPIsoform(
        name="CYP2B6", abundance_pmol_mg=17.0, abundance_cv=0.80,
        percent_of_total=5,
        primary_reactions="Efavirenz, methadone, bupropion, cyclophosphamide",
        substrates=["efavirenz", "methadone", "bupropion", "cyclophosphamide",
                    "propofol", "ketamine", "artemisinin"],
        inhibitors=["ticlopidine", "clopidogrel"],
        inducers=["rifampicin", "phenobarbital", "carbamazepine"],
    ),
    "CYP2C8": CYPIsoform(
        name="CYP2C8", abundance_pmol_mg=24.0, abundance_cv=0.50,
        percent_of_total=7,
        primary_reactions="Paclitaxel, repaglinide, amodiaquine, rosiglitazone",
        substrates=["paclitaxel", "repaglinide", "amodiaquine", "rosiglitazone",
                    "cerivastatin", "ibuprofen"],
        inhibitors=["gemfibrozil_glucuronide", "trimethoprim", "montelukast"],
        inducers=["rifampicin"],
    ),
    "CYP2A6": CYPIsoform(
        name="CYP2A6", abundance_pmol_mg=36.0, abundance_cv=0.60,
        percent_of_total=4,
        primary_reactions="Nicotine, coumarin, tegafur, letrozole",
        substrates=["nicotine", "coumarin", "tegafur", "letrozole", "valproic_acid"],
        inhibitors=["methoxsalen", "tranylcypromine", "pilocarpine"],
        inducers=["phenobarbital", "dexamethasone"],
    ),
    "CYP2J2": CYPIsoform(
        name="CYP2J2", abundance_pmol_mg=1.5, abundance_cv=0.50,
        percent_of_total=1,
        primary_reactions="Arachidonic acid epoxidation, astemizole, terfenadine",
        substrates=["astemizole", "terfenadine", "ebastine"],
        inhibitors=["telmisartan", "flunarizine"],
        inducers=[],
    ),
    "CYP4A11": CYPIsoform(
        name="CYP4A11", abundance_pmol_mg=5.0, abundance_cv=0.40,
        percent_of_total=1,
        primary_reactions="Fatty acid omega-hydroxylation, 20-HETE synthesis",
        substrates=["lauric_acid", "palmitic_acid"],
        inhibitors=["HET0016"],
        inducers=["fibrates", "fasting"],
    ),
    "CYP1A1": CYPIsoform(
        name="CYP1A1", abundance_pmol_mg=1.0, abundance_cv=0.80,
        percent_of_total=0.3,
        primary_reactions="PAH activation (benzo[a]pyrene), estradiol hydroxylation. Mainly extrahepatic.",
        substrates=["benzo_a_pyrene", "estradiol"],
        inhibitors=["alpha_naphthoflavone"],
        inducers=["smoking", "charbroiled_meat", "dioxins"],
    ),
    "CYP1B1": CYPIsoform(
        name="CYP1B1", abundance_pmol_mg=0.5, abundance_cv=0.90,
        percent_of_total=0.1,
        primary_reactions="Estrogen metabolism, procarcinogen activation. Mainly extrahepatic (breast, prostate).",
        substrates=["estradiol", "benzo_a_pyrene"],
        inhibitors=["resveratrol"],
        inducers=["dioxins", "PAHs"],
    ),
    "CYP3A7": CYPIsoform(
        name="CYP3A7", abundance_pmol_mg=0.0, abundance_cv=1.0,
        percent_of_total=0,
        primary_reactions="Fetal CYP — dominant in fetal liver, replaced by CYP3A4 after birth",
        substrates=["dehydroepiandrosterone_sulfate", "retinoic_acid"],
        inhibitors=["ketoconazole"],
        inducers=[],
    ),
}


# ════════════════════════════════════════════════════════════════
# DRUG-SPECIFIC CYP KINETICS
# ════════════════════════════════════════════════════════════════

@dataclass
class DrugCYPKinetics:
    """Michaelis-Menten kinetics for a specific drug-CYP pair."""
    drug: str
    cyp: str
    vmax: float          # pmol/min/pmol CYP
    km: float            # µM (Michaelis constant)
    fm: float = 0.0      # Fraction metabolized by this CYP (0-1)
    clint: float = 0.0   # In vitro intrinsic clearance (µL/min/pmol CYP) = Vmax/Km
    metabolite: str = ""  # Primary metabolite formed


DRUG_CYP_KINETICS = [
    # Caffeine — primarily CYP1A2
    DrugCYPKinetics("caffeine", "CYP1A2", vmax=3.8, km=320, fm=0.95, metabolite="paraxanthine"),
    DrugCYPKinetics("caffeine", "CYP2E1", vmax=0.2, km=1000, fm=0.03, metabolite="theobromine"),
    DrugCYPKinetics("caffeine", "CYP3A4", vmax=0.1, km=800, fm=0.02, metabolite="trimethyluric_acid"),

    # Paracetamol — CYP2E1 produces toxic NAPQI
    DrugCYPKinetics("paracetamol", "CYP2E1", vmax=12.0, km=1150, fm=0.10, metabolite="NAPQI_toxic"),
    DrugCYPKinetics("paracetamol", "CYP1A2", vmax=3.0, km=2800, fm=0.05, metabolite="NAPQI_toxic"),
    DrugCYPKinetics("paracetamol", "CYP3A4", vmax=2.0, km=3200, fm=0.03, metabolite="NAPQI_toxic"),

    # Warfarin — CYP2C9 for S-warfarin (active)
    DrugCYPKinetics("warfarin", "CYP2C9", vmax=5.6, km=4.0, fm=0.85, metabolite="7-hydroxywarfarin"),
    DrugCYPKinetics("warfarin", "CYP3A4", vmax=1.2, km=10.0, fm=0.10, metabolite="10-hydroxywarfarin"),
    DrugCYPKinetics("warfarin", "CYP1A2", vmax=0.5, km=15.0, fm=0.05, metabolite="6-hydroxywarfarin"),

    # Midazolam — CYP3A4 probe substrate
    DrugCYPKinetics("midazolam", "CYP3A4", vmax=18.0, km=3.3, fm=0.94, metabolite="1-hydroxymidazolam"),
    DrugCYPKinetics("midazolam", "CYP3A5", vmax=4.0, km=8.0, fm=0.06, metabolite="1-hydroxymidazolam"),

    # Omeprazole — CYP2C19 (polymorphic)
    DrugCYPKinetics("omeprazole", "CYP2C19", vmax=8.0, km=6.0, fm=0.80, metabolite="5-hydroxyomeprazole"),
    DrugCYPKinetics("omeprazole", "CYP3A4", vmax=2.0, km=25.0, fm=0.20, metabolite="omeprazole_sulfone"),

    # Codeine — CYP2D6 activates it to morphine
    DrugCYPKinetics("codeine", "CYP2D6", vmax=1.5, km=150, fm=0.10, metabolite="morphine_active"),
    DrugCYPKinetics("codeine", "CYP3A4", vmax=8.0, km=250, fm=0.80, metabolite="norcodeine_inactive"),
    DrugCYPKinetics("codeine", "UGT2B7", vmax=3.0, km=500, fm=0.10, metabolite="codeine-6-glucuronide"),

    # Diazepam
    DrugCYPKinetics("diazepam", "CYP3A4", vmax=4.0, km=50, fm=0.50, metabolite="temazepam"),
    DrugCYPKinetics("diazepam", "CYP2C19", vmax=3.5, km=30, fm=0.45, metabolite="nordiazepam"),
    DrugCYPKinetics("diazepam", "CYP2B6", vmax=0.3, km=100, fm=0.05, metabolite="nordiazepam"),

    # Metoprolol — CYP2D6 (polymorphic)
    DrugCYPKinetics("metoprolol", "CYP2D6", vmax=15.0, km=25, fm=0.80, metabolite="alpha-hydroxymetoprolol"),
    DrugCYPKinetics("metoprolol", "CYP3A4", vmax=2.0, km=100, fm=0.15, metabolite="O-demethylmetoprolol"),

    # Simvastatin — CYP3A4
    DrugCYPKinetics("simvastatin", "CYP3A4", vmax=25.0, km=2.0, fm=0.95, metabolite="simvastatin_acid"),

    # Clopidogrel — CYP2C19 activates the prodrug
    DrugCYPKinetics("clopidogrel", "CYP2C19", vmax=5.0, km=20, fm=0.45, metabolite="active_thiol_metabolite"),
    DrugCYPKinetics("clopidogrel", "CYP3A4", vmax=3.0, km=35, fm=0.30, metabolite="2-oxo-clopidogrel"),
    DrugCYPKinetics("clopidogrel", "CYP2B6", vmax=1.5, km=40, fm=0.15, metabolite="2-oxo-clopidogrel"),
    DrugCYPKinetics("clopidogrel", "CYP1A2", vmax=0.5, km=50, fm=0.10, metabolite="2-oxo-clopidogrel"),
]


# ════════════════════════════════════════════════════════════════
# KINETIC CALCULATIONS
# ════════════════════════════════════════════════════════════════

# Scaling factors for IVIVE
MPPGL = 40.0         # mg microsomal protein per gram liver (adult)
LIVER_WEIGHT_G = 1500  # Average liver weight (grams)


def calculate_clint_per_cyp(
    vmax: float, km: float, substrate_conc_um: float = 1.0,
) -> float:
    """
    Calculate in vitro intrinsic clearance for a single CYP.
    CLint = Vmax / (Km + [S])

    At low [S] (most therapeutic concentrations), CLint ≈ Vmax/Km
    """
    return vmax / (km + substrate_conc_um)


def calculate_total_hepatic_clearance(
    drug_name: str,
    substrate_conc_um: float = 1.0,
    cyp_modifications: dict | None = None,
    liver_weight_g: float = LIVER_WEIGHT_G,
) -> dict:
    """
    Calculate total hepatic clearance from individual CYP contributions.

    CLhep = Σ (CLint_i × abundance_i × MPPGL × liver_weight) for each CYP

    Args:
        drug_name: Drug name
        substrate_conc_um: Substrate concentration (µM)
        cyp_modifications: Dict of {CYP_name: activity_fraction} for DDI/polymorphism
            e.g., {"CYP2D6": 0.0} for PM, {"CYP3A4": 0.1} for ketoconazole inhibition
        liver_weight_g: Patient liver weight

    Returns:
        Detailed clearance breakdown by CYP
    """
    if cyp_modifications is None:
        cyp_modifications = {}

    key = drug_name.lower().replace(" ", "_").replace("-", "_")
    kinetics = [k for k in DRUG_CYP_KINETICS if k.drug == key]

    if not kinetics:
        return {
            "drug": drug_name,
            "error": "No CYP kinetics data available",
            "available_drugs": list(set(k.drug for k in DRUG_CYP_KINETICS)),
        }

    contributions = []
    total_clint_scaled = 0

    for kin in kinetics:
        # Get CYP abundance
        cyp_info = CYP_ISOFORMS.get(kin.cyp)
        abundance = cyp_info.abundance_pmol_mg if cyp_info else 10.0

        # Apply modification (DDI, polymorphism)
        activity_factor = cyp_modifications.get(kin.cyp, 1.0)

        # Calculate CLint for this CYP
        clint = calculate_clint_per_cyp(kin.vmax, kin.km, substrate_conc_um)

        # Scale: CLint × abundance × MPPGL × liver_weight
        # Units: (µL/min/pmol) × (pmol/mg) × (mg/g) × g = µL/min → convert to L/h
        clint_scaled = clint * abundance * activity_factor * MPPGL * liver_weight_g / 1e6 * 60  # µL/min → L/h

        total_clint_scaled += clint_scaled

        contributions.append({
            "cyp": kin.cyp,
            "fm": round(kin.fm, 3),
            "vmax": kin.vmax,
            "km": kin.km,
            "clint_per_pmol": round(clint, 4),
            "abundance": abundance,
            "activity_factor": activity_factor,
            "cl_contribution_lh": round(clint_scaled, 4),
            "metabolite": kin.metabolite,
            "is_toxic": "toxic" in kin.metabolite.lower(),
            "is_active": "active" in kin.metabolite.lower(),
        })

    # Sort by contribution
    contributions.sort(key=lambda x: x["cl_contribution_lh"], reverse=True)

    # Calculate fractions
    for c in contributions:
        c["fraction_of_total"] = round(c["cl_contribution_lh"] / total_clint_scaled, 3) if total_clint_scaled > 0 else 0

    # Clinical implications
    implications = _generate_cyp_implications(drug_name, contributions, cyp_modifications)

    return {
        "drug": drug_name,
        "total_clint_scaled_lh": round(total_clint_scaled, 3),
        "contributions": contributions,
        "primary_cyp": contributions[0]["cyp"] if contributions else "unknown",
        "clinical_implications": implications,
        "polymorphism_impact": _assess_polymorphism_impact(contributions),
    }


def _generate_cyp_implications(drug_name: str, contributions: list, modifications: dict) -> list[str]:
    """Generate clinical implications from CYP kinetics."""
    implications = []

    if contributions:
        primary = contributions[0]
        if primary["fraction_of_total"] > 0.8:
            implications.append(
                f"Predominantly metabolized by {primary['cyp']} ({primary['fraction_of_total']*100:.0f}%). "
                f"Strong {primary['cyp']} inhibitors will dramatically increase drug levels."
            )

        # Check for active metabolites
        for c in contributions:
            if c["is_active"]:
                implications.append(
                    f"{c['cyp']} produces the ACTIVE metabolite ({c['metabolite']}). "
                    f"Inhibiting {c['cyp']} will REDUCE drug efficacy."
                )
            if c["is_toxic"]:
                implications.append(
                    f"{c['cyp']} produces a TOXIC metabolite ({c['metabolite']}). "
                    f"Inducing {c['cyp']} will increase toxicity risk."
                )

    # DDI warnings
    for cyp, factor in modifications.items():
        if factor < 0.3:
            implications.append(f"{cyp} is strongly inhibited ({factor*100:.0f}% activity) — expect increased drug levels")
        elif factor > 2.0:
            implications.append(f"{cyp} is induced ({factor*100:.0f}% activity) — expect decreased drug levels")

    return implications


def _assess_polymorphism_impact(contributions: list) -> dict:
    """Assess impact of known genetic polymorphisms."""
    polymorphic_cyps = {"CYP2D6", "CYP2C19", "CYP2C9", "CYP2B6", "CYP3A5"}

    affected = []
    for c in contributions:
        if c["cyp"] in polymorphic_cyps and c["fraction_of_total"] > 0.1:
            affected.append({
                "cyp": c["cyp"],
                "fraction": c["fraction_of_total"],
                "impact": (
                    "HIGH — Poor metabolizers will have dramatically different drug levels"
                    if c["fraction_of_total"] > 0.5
                    else "MODERATE — Polymorphism may affect drug levels"
                    if c["fraction_of_total"] > 0.2
                    else "LOW — Minor impact expected"
                ),
            })

    return {
        "has_polymorphic_pathway": len(affected) > 0,
        "affected_cyps": affected,
        "recommendation": (
            "Pharmacogenomic testing recommended before prescribing"
            if any(a["fraction"] > 0.5 for a in affected)
            else "Consider pharmacogenomic testing for dose optimization"
            if affected
            else "No significant polymorphism concerns"
        ),
    }


def get_cyp_isoform(name: str) -> CYPIsoform | None:
    """Look up a CYP isoform."""
    return CYP_ISOFORMS.get(name.upper())


def get_drug_kinetics(drug_name: str) -> list[DrugCYPKinetics]:
    """Get all CYP kinetics entries for a drug."""
    key = drug_name.lower().replace(" ", "_")
    return [k for k in DRUG_CYP_KINETICS if k.drug == key]
