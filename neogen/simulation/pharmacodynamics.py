"""
Pharmacodynamics Engine
========================
Models what drugs DO to the body — not just where they go, but their actual
biological effects at the organ and cellular level.

Three levels of simulation:
1. ORGAN EFFECTS — how each organ responds to the drug
2. DISEASE INTERACTION — how drugs fight disease entities (HIV, cancer, bacteria)
3. CELLULAR MECHANISMS — receptor binding, signal cascades, ion channels
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 1. ORGAN-LEVEL PHARMACODYNAMIC EFFECTS
# ════════════════════════════════════════════════════════════════

@dataclass
class OrganEffect:
    """How a drug affects a specific organ."""
    organ: str                        # liver, heart, brain, kidney, lung, stomach, intestine, muscle
    effect_type: str                  # "therapeutic", "side_effect", "toxic"
    mechanism: str                    # What the drug does (e.g., "COX-1/2 inhibition")
    visual_effect: str                # What to show on the 3D body
    description: str                  # Human-readable explanation
    concentration_threshold: float = 0.0  # Min concentration to trigger (mg/L)
    intensity_scale: str = "linear"   # "linear", "sigmoidal", "threshold"
    color_shift: str = ""             # Hex color to shift organ to
    particle_type: str = ""           # "healing", "inflammation", "damage", "electrical", "blocking"
    particle_color: str = ""          # Hex color for effect particles


@dataclass
class CellularMechanism:
    """Cell-level drug mechanism for zoom view."""
    cell_type: str                    # "hepatocyte", "cardiomyocyte", "neuron", "t_cell", etc.
    receptor: str                     # Target receptor/enzyme name
    receptor_type: str                # "GPCR", "ion_channel", "enzyme", "nuclear", "transporter"
    action: str                       # "agonist", "antagonist", "inhibitor", "blocker", "activator"
    binding_affinity: str             # "high", "moderate", "low"
    downstream_effects: list = field(default_factory=list)  # Signal cascade
    visual_description: str = ""      # What to animate at cell level


@dataclass
class DiseaseEntity:
    """A disease entity that can be visualized and targeted by drugs."""
    entity_type: str                  # "virus", "bacteria", "cancer_cell", "autoimmune", "plaque"
    name: str                         # "HIV virion", "MRSA", "tumor cell"
    target_organ: str                 # Where these entities live
    visual_shape: str                 # "sphere_spiky", "rod", "irregular", "cluster"
    color: str                        # Entity color
    size: float = 0.1                 # Relative size
    count: int = 20                   # How many to show
    behavior: str = "float"           # "float", "attach", "multiply", "invade"
    vulnerability: list = field(default_factory=list)  # Which drug classes kill it


@dataclass
class DrugVsDisease:
    """How a drug interacts with disease entities."""
    drug_class: str                   # "NRTI", "protease_inhibitor", "antibiotic", etc.
    target_entity: str                # "HIV", "bacteria", "cancer_cell"
    mechanism: str                    # How the drug fights the disease
    visual_animation: str             # What to show
    effectiveness: float = 0.8        # 0-1 how effective
    time_to_effect_h: float = 1.0     # Hours until visible effect


# ════════════════════════════════════════════════════════════════
# DRUG PHARMACODYNAMIC DATABASE
# ════════════════════════════════════════════════════════════════

DRUG_PD_DATABASE = {
    # ── ASPIRIN ──────────────────────────────────────
    "aspirin": {
        "drug_class": "NSAID",
        "primary_target": "COX-1/COX-2",
        "organ_effects": [
            OrganEffect("stomach", "side_effect", "COX-1 inhibition reduces protective mucus",
                        "Stomach lining thins, red inflammation spots appear",
                        "Aspirin blocks COX-1 in stomach → less protective prostaglandins → risk of GI bleeding",
                        concentration_threshold=5.0, color_shift="#ff4444",
                        particle_type="inflammation", particle_color="#ff3333"),
            OrganEffect("heart", "therapeutic", "Irreversible platelet COX-1 inhibition",
                        "Blood flow improves, platelets stop clumping",
                        "Aspirin permanently disables COX-1 in platelets → prevents clotting → reduces heart attack risk",
                        concentration_threshold=1.0, color_shift="#44ff88",
                        particle_type="healing", particle_color="#00ff88"),
            OrganEffect("brain", "therapeutic", "COX-2 inhibition reduces inflammation",
                        "Pain signals diminish, inflammation markers decrease",
                        "Blocks prostaglandin synthesis → reduces pain signaling and inflammation",
                        concentration_threshold=2.0, color_shift="#44ddff",
                        particle_type="blocking", particle_color="#00ccff"),
            OrganEffect("kidney", "side_effect", "Reduced renal prostaglandins",
                        "Kidney blood flow slightly decreases",
                        "Prostaglandins maintain renal blood flow → aspirin reduces this → risk in dehydration",
                        concentration_threshold=10.0, color_shift="#ffaa44",
                        particle_type="damage", particle_color="#ffaa00"),
        ],
        "cellular": [
            CellularMechanism("platelet", "COX-1", "enzyme", "inhibitor", "high",
                             ["Arachidonic acid blocked", "No TXA2 production", "Platelet aggregation stops"],
                             "Drug molecule enters platelet, binds to COX-1 enzyme (acetylation of Ser530), enzyme shape changes, TXA2 pathway blocked"),
            CellularMechanism("gastric_epithelial", "COX-1", "enzyme", "inhibitor", "high",
                             ["Prostaglandin E2 reduced", "Mucus secretion decreases", "Acid damage increases"],
                             "Drug binds COX-1 in stomach cells, protective mucus layer thins, H+ ions penetrate"),
        ],
    },

    # ── CAFFEINE ─────────────────────────────────────
    "caffeine": {
        "drug_class": "Methylxanthine",
        "primary_target": "Adenosine A1/A2A receptors",
        "organ_effects": [
            OrganEffect("brain", "therapeutic", "Adenosine A1/A2A receptor antagonism",
                        "Neural activity increases, alertness markers rise",
                        "Caffeine blocks adenosine receptors → prevents drowsiness signal → increased alertness and focus",
                        concentration_threshold=1.0, color_shift="#ffff44",
                        particle_type="electrical", particle_color="#ffff00"),
            OrganEffect("heart", "side_effect", "Positive chronotropic effect",
                        "Heart beats faster, rhythm may become irregular",
                        "Blocks adenosine → removes brake on heart rate → tachycardia at high doses",
                        concentration_threshold=5.0, color_shift="#ff6644",
                        particle_type="electrical", particle_color="#ff4444"),
            OrganEffect("kidney", "therapeutic", "Increased renal blood flow + diuresis",
                        "Kidney filtration increases, urine output rises",
                        "Caffeine increases GFR and inhibits sodium reabsorption → diuretic effect",
                        concentration_threshold=2.0, color_shift="#44ccff",
                        particle_type="healing", particle_color="#00bbff"),
            OrganEffect("stomach", "side_effect", "Increased gastric acid secretion",
                        "Acid production increases in stomach",
                        "Stimulates parietal cells → more HCl → heartburn/reflux risk",
                        concentration_threshold=3.0, color_shift="#ffaa44",
                        particle_type="inflammation", particle_color="#ffaa00"),
        ],
        "cellular": [
            CellularMechanism("neuron", "Adenosine A2A", "GPCR", "antagonist", "high",
                             ["Adenosine can't bind", "cAMP levels stay high", "Dopamine signaling enhanced", "Wakefulness maintained"],
                             "Caffeine molecule fits into adenosine receptor pocket, blocks adenosine from binding, neuron stays active instead of slowing down"),
        ],
    },

    # ── PARACETAMOL (Acetaminophen) ──────────────────
    "paracetamol": {
        "drug_class": "Analgesic/Antipyretic",
        "primary_target": "COX-2 (central) + TRPV1",
        "organ_effects": [
            OrganEffect("brain", "therapeutic", "Central COX-2 inhibition + endocannabinoid pathway",
                        "Pain perception decreases, temperature normalizes",
                        "Acts primarily in CNS → reduces prostaglandin E2 in hypothalamus → lowers fever and pain",
                        concentration_threshold=2.0, color_shift="#44ff88",
                        particle_type="healing", particle_color="#00ff88"),
            OrganEffect("liver", "toxic", "CYP2E1 produces NAPQI (toxic metabolite)",
                        "Liver cells show stress, GSH depletes, damage zones appear",
                        "5% metabolized to NAPQI via CYP2E1 → normally detoxified by glutathione → at overdose GSH runs out → liver cell death",
                        concentration_threshold=20.0, color_shift="#ff2222",
                        particle_type="damage", particle_color="#ff0000"),
            OrganEffect("kidney", "side_effect", "Direct tubular toxicity at high doses",
                        "Kidney function slightly stressed",
                        "High doses can cause renal papillary necrosis, especially with chronic use",
                        concentration_threshold=30.0, color_shift="#ffaa44",
                        particle_type="damage", particle_color="#ff8800"),
        ],
        "cellular": [
            CellularMechanism("neuron", "COX-2 (central)", "enzyme", "inhibitor", "moderate",
                             ["PGE2 in hypothalamus decreases", "Thermoregulatory set point normalizes", "Pain signal transmission reduced"],
                             "Drug crosses BBB, enters CNS neurons, inhibits COX-2 in brain, fever signal stops"),
            CellularMechanism("hepatocyte", "CYP2E1", "enzyme", "substrate", "high",
                             ["NAPQI formed", "GSH conjugation (normal)", "If GSH depleted: protein adducts", "Mitochondrial damage", "Cell death"],
                             "Drug enters hepatocyte, CYP2E1 converts small fraction to NAPQI (glowing red toxic molecule), GSH (green shield) neutralizes it, but at overdose GSH runs out and NAPQI attacks cell proteins"),
        ],
    },

    # ── IBUPROFEN ────────────────────────────────────
    "ibuprofen": {
        "drug_class": "NSAID",
        "primary_target": "COX-1/COX-2 (reversible)",
        "organ_effects": [
            OrganEffect("brain", "therapeutic", "COX-2 inhibition in pain pathways",
                        "Pain and inflammation signals decrease",
                        "Reversibly inhibits COX-2 → reduces prostaglandin-mediated inflammation and pain",
                        concentration_threshold=2.0, color_shift="#44ddff",
                        particle_type="healing", particle_color="#00ccff"),
            OrganEffect("stomach", "side_effect", "COX-1 inhibition in GI mucosa",
                        "Stomach lining protection decreases",
                        "Blocks protective prostaglandins → GI erosion risk (less than aspirin since reversible)",
                        concentration_threshold=8.0, color_shift="#ffaa44",
                        particle_type="inflammation", particle_color="#ffaa00"),
            OrganEffect("kidney", "side_effect", "Renal prostaglandin reduction",
                        "Kidney blood flow decreases",
                        "Prostaglandins maintain renal perfusion → ibuprofen reduces → risky in dehydration/CKD",
                        concentration_threshold=15.0, color_shift="#ff8844",
                        particle_type="damage", particle_color="#ff6600"),
            OrganEffect("heart", "side_effect", "Increased thrombotic risk",
                        "Slight increase in clotting tendency",
                        "COX-2 inhibition reduces prostacyclin → tips balance toward clotting → CV risk with chronic use",
                        concentration_threshold=20.0, color_shift="#ffaa44",
                        particle_type="inflammation", particle_color="#ff8800"),
        ],
        "cellular": [
            CellularMechanism("macrophage", "COX-2", "enzyme", "inhibitor", "high",
                             ["Arachidonic acid pathway blocked", "PGE2/PGI2 reduced", "Inflammation cascade stops", "Swelling decreases"],
                             "Drug enters inflammatory cell, sits in COX-2 active site (reversible binding), prostaglandin factory shuts down, inflammation resolves"),
        ],
    },

    # ── METFORMIN ────────────────────────────────────
    "metformin": {
        "drug_class": "Biguanide (Antidiabetic)",
        "primary_target": "AMPK pathway / mitochondrial complex I",
        "organ_effects": [
            OrganEffect("liver", "therapeutic", "Reduces hepatic glucose production",
                        "Liver glucose output decreases, sugar production slows",
                        "Activates AMPK → suppresses gluconeogenesis → liver stops overproducing glucose",
                        concentration_threshold=0.5, color_shift="#44ff88",
                        particle_type="healing", particle_color="#00ff88"),
            OrganEffect("stomach", "side_effect", "GI intolerance",
                        "Stomach shows irritation, motility increases",
                        "Accumulates in GI mucosa → increased serotonin release → nausea/diarrhea in 30% of patients",
                        concentration_threshold=2.0, color_shift="#ffaa44",
                        particle_type="inflammation", particle_color="#ffaa00"),
            OrganEffect("muscle", "therapeutic", "Increases glucose uptake",
                        "Muscle cells absorb more glucose from blood",
                        "AMPK activation → GLUT4 transporter translocation → muscles take up more glucose → blood sugar drops",
                        concentration_threshold=1.0, color_shift="#44ddff",
                        particle_type="healing", particle_color="#00ccff"),
        ],
        "cellular": [
            CellularMechanism("hepatocyte", "AMPK / Complex I", "enzyme", "activator", "moderate",
                             ["Mitochondrial Complex I partially inhibited", "AMP:ATP ratio rises", "AMPK activated",
                              "Gluconeogenic genes suppressed", "Glucose production stops"],
                             "Drug enters hepatocyte mitochondria, partially blocks Complex I, energy sensor AMPK activates, liver stops making excess glucose"),
        ],
    },

    # ── ATORVASTATIN ─────────────────────────────────
    "atorvastatin": {
        "drug_class": "Statin (HMG-CoA reductase inhibitor)",
        "primary_target": "HMG-CoA reductase",
        "organ_effects": [
            OrganEffect("liver", "therapeutic", "HMG-CoA reductase inhibition",
                        "Cholesterol synthesis drops, LDL receptors increase",
                        "Blocks cholesterol synthesis → liver upregulates LDL receptors → pulls LDL from blood → LDL drops 30-50%",
                        concentration_threshold=0.01, color_shift="#44ff88",
                        particle_type="healing", particle_color="#00ff88"),
            OrganEffect("muscle", "side_effect", "Mitochondrial dysfunction in myocytes",
                        "Muscle fibers show stress, occasional damage",
                        "CoQ10 synthesis reduced → mitochondrial dysfunction → myalgia in 5-10% → rare rhabdomyolysis",
                        concentration_threshold=0.1, color_shift="#ffaa44",
                        particle_type="damage", particle_color="#ff8800"),
        ],
        "cellular": [
            CellularMechanism("hepatocyte", "HMG-CoA reductase", "enzyme", "inhibitor", "high",
                             ["Mevalonate pathway blocked", "Cholesterol synthesis stops", "SREBP activated",
                              "LDL receptor gene upregulated", "More LDL receptors on cell surface", "Blood LDL captured"],
                             "Drug binds HMG-CoA reductase in liver cell ER, cholesterol factory stops, cell senses low cholesterol, makes more LDL receptors, pulls LDL from bloodstream"),
        ],
    },
}

# ════════════════════════════════════════════════════════════════
# DISEASE ENTITY DATABASE
# ════════════════════════════════════════════════════════════════

DISEASE_ENTITIES = {
    "hiv": {
        "entities": [
            DiseaseEntity("virus", "HIV Virion", "blood", "sphere_spiky",
                         "#44cc44", 0.08, 30, "float",
                         ["NRTI", "NNRTI", "protease_inhibitor", "integrase_inhibitor", "entry_inhibitor"]),
            DiseaseEntity("virus", "HIV-infected CD4+ T-cell", "blood", "sphere_damaged",
                         "#cc4444", 0.15, 10, "attach",
                         ["NRTI", "NNRTI", "protease_inhibitor"]),
        ],
        "drug_interactions": {
            "NRTI": DrugVsDisease("NRTI", "HIV",
                "Nucleoside analog incorporates into viral DNA → chain termination → virus can't replicate",
                "Drug particles enter HIV-infected cells, viral DNA strand stops growing, new virions stop forming",
                0.85, 2.0),
            "NNRTI": DrugVsDisease("NNRTI", "HIV",
                "Binds reverse transcriptase directly → enzyme can't function → no viral DNA made",
                "Drug binds to reverse transcriptase enzyme (shape change), viral replication machinery jams",
                0.80, 1.0),
            "protease_inhibitor": DrugVsDisease("protease_inhibitor", "HIV",
                "Blocks HIV protease → immature non-infectious virions produced",
                "New HIV particles form but are misshapen and non-functional, virus count drops",
                0.90, 4.0),
            "integrase_inhibitor": DrugVsDisease("integrase_inhibitor", "HIV",
                "Blocks integrase → viral DNA can't insert into host genome",
                "Drug blocks integration step, viral DNA stays unintegrated, cell cleared",
                0.95, 1.5),
        },
    },

    "bacterial_infection": {
        "entities": [
            DiseaseEntity("bacteria", "Bacteria", "blood", "rod",
                         "#66aa44", 0.06, 40, "multiply",
                         ["penicillin", "cephalosporin", "fluoroquinolone", "macrolide", "aminoglycoside"]),
        ],
        "drug_interactions": {
            "penicillin": DrugVsDisease("penicillin", "bacteria",
                "Inhibits cell wall synthesis → bacteria lyse (burst)",
                "Drug molecules attach to bacterial surface, cell wall weakens, bacteria swell and burst (lysis animation)",
                0.90, 1.0),
            "fluoroquinolone": DrugVsDisease("fluoroquinolone", "bacteria",
                "Inhibits DNA gyrase → bacteria can't replicate DNA → death",
                "Drug enters bacteria, DNA untangling stops, bacteria can't divide, cell death",
                0.85, 2.0),
            "macrolide": DrugVsDisease("macrolide", "bacteria",
                "Binds 50S ribosomal subunit → protein synthesis stops",
                "Drug enters bacteria, attaches to ribosome, protein factory jams, bacteria starve",
                0.75, 3.0),
        },
    },

    "cancer": {
        "entities": [
            DiseaseEntity("cancer_cell", "Tumor Cell", "liver", "irregular",
                         "#884488", 0.12, 15, "multiply",
                         ["alkylating", "antimetabolite", "topoisomerase_inhibitor", "kinase_inhibitor"]),
        ],
        "drug_interactions": {
            "alkylating": DrugVsDisease("alkylating", "cancer_cell",
                "Cross-links DNA strands → cell can't replicate → apoptosis",
                "Drug molecules enter nucleus, DNA strands get tangled/cross-linked, cell triggers self-destruct (apoptosis)",
                0.70, 6.0),
            "antimetabolite": DrugVsDisease("antimetabolite", "cancer_cell",
                "Mimics normal metabolites → incorporated into DNA/RNA → errors → death",
                "Fake building blocks enter cell, get incorporated into DNA, errors accumulate, cell dies",
                0.65, 12.0),
            "kinase_inhibitor": DrugVsDisease("kinase_inhibitor", "cancer_cell",
                "Blocks overactive growth signaling (EGFR/VEGF/BCR-ABL) → growth stops",
                "Drug binds to kinase receptor on cell surface, growth signal blocked, cell stops dividing",
                0.80, 4.0),
        },
    },

    "hypertension": {
        "entities": [
            DiseaseEntity("plaque", "Arterial Plaque", "heart", "irregular",
                         "#ccaa44", 0.08, 8, "attach",
                         ["ace_inhibitor", "arb", "ccb", "beta_blocker"]),
        ],
        "drug_interactions": {
            "ace_inhibitor": DrugVsDisease("ace_inhibitor", "plaque",
                "Blocks ACE → less angiotensin II → vessels relax → BP drops",
                "Drug blocks ACE enzyme in blood vessels, vessel walls relax and widen, blood flow improves",
                0.85, 1.0),
            "beta_blocker": DrugVsDisease("beta_blocker", "plaque",
                "Blocks beta-1 receptors on heart → heart rate and contractility decrease → BP drops",
                "Drug binds heart beta receptors, heart slows down, pumps less forcefully, pressure drops",
                0.80, 0.5),
        },
    },

    "inflammation": {
        "entities": [
            DiseaseEntity("autoimmune", "Inflammatory Mediator", "stomach", "cluster",
                         "#ff6644", 0.05, 25, "float",
                         ["NSAID", "corticosteroid", "anti_TNF", "anti_IL6"]),
        ],
        "drug_interactions": {
            "NSAID": DrugVsDisease("NSAID", "inflammation",
                "Blocks COX enzymes → reduces prostaglandin-mediated inflammation",
                "Drug molecules neutralize inflammatory particles, redness and swelling decrease",
                0.75, 0.5),
            "corticosteroid": DrugVsDisease("corticosteroid", "inflammation",
                "Enters nucleus → suppresses inflammatory gene transcription → broad anti-inflammatory",
                "Drug enters cell nucleus, shuts down inflammatory gene expression, immune response calms",
                0.95, 2.0),
        },
    },

    "diabetes": {
        "entities": [
            DiseaseEntity("metabolic", "Excess Glucose", "blood", "sphere",
                         "#ffaa00", 0.04, 50, "float",
                         ["insulin", "metformin", "sulfonylurea", "sglt2_inhibitor", "glp1_agonist"]),
        ],
        "drug_interactions": {
            "metformin": DrugVsDisease("metformin", "glucose",
                "Reduces hepatic glucose production + increases muscle glucose uptake",
                "Liver stops releasing glucose, muscle cells absorb more, blood glucose particles decrease",
                0.70, 2.0),
            "sglt2_inhibitor": DrugVsDisease("sglt2_inhibitor", "glucose",
                "Blocks glucose reabsorption in kidney → glucose excreted in urine",
                "Kidney filters glucose normally but drug prevents reabsorption, glucose leaves via urine",
                0.80, 1.0),
        },
    },
}

# ════════════════════════════════════════════════════════════════
# SMILES → DRUG NAME MAPPING (for lookup)
# ════════════════════════════════════════════════════════════════

SMILES_TO_DRUG = {
    "CC(=O)OC1=CC=CC=C1C(=O)O": "aspirin",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": "caffeine",
    "CC(=O)NC1=CC=C(O)C=C1": "paracetamol",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O": "ibuprofen",
    "CN(C)CCOC(C1=CC=CC=C1)C1=CC=CC=C1": "diphenhydramine",
    "CC(C)NCC(O)C1=CC=C(O)C(O)=C1": "metformin",  # approximate
    "CCO": "ethanol",
}


# ════════════════════════════════════════════════════════════════
# MAIN ASSESSMENT FUNCTION
# ════════════════════════════════════════════════════════════════

def assess_pharmacodynamics(
    smiles: str = "",
    drug_name: str = "",
    organ_concentrations: dict = None,
    diseases: list = None,
    dose_mg: float = 100.0,
) -> dict:
    """
    Assess pharmacodynamic effects for a drug at given organ concentrations.

    Returns a dict with:
    - organ_effects: list of active effects per organ
    - cellular_mechanisms: list of cell-level mechanisms
    - disease_interactions: how the drug fights active diseases
    """
    # Resolve drug name from SMILES if not provided
    if not drug_name or len(drug_name) < 3:
        drug_name = SMILES_TO_DRUG.get(smiles, "")

    name_lower = drug_name.lower().strip()

    # Find drug in PD database
    drug_pd = None
    matched_name = ""
    for known_name, pd_data in DRUG_PD_DATABASE.items():
        if name_lower and len(name_lower) >= 3 and (known_name in name_lower or name_lower in known_name):
            drug_pd = pd_data
            matched_name = known_name
            break

    result = {
        "drug_name": matched_name or drug_name or "Unknown",
        "drug_class": "",
        "primary_target": "",
        "organ_effects": [],
        "cellular_mechanisms": [],
        "disease_interactions": [],
    }

    if not drug_pd:
        return result

    result["drug_class"] = drug_pd["drug_class"]
    result["primary_target"] = drug_pd["primary_target"]

    # Evaluate organ effects based on concentration
    if organ_concentrations:
        for effect in drug_pd.get("organ_effects", []):
            organ_conc = organ_concentrations.get(f"{effect.organ}_conc", 0)

            # Check if concentration exceeds threshold
            if organ_conc >= effect.concentration_threshold:
                # Calculate intensity (0-1) based on how far above threshold
                if effect.intensity_scale == "threshold":
                    intensity = 1.0 if organ_conc >= effect.concentration_threshold else 0.0
                elif effect.intensity_scale == "sigmoidal":
                    ec50 = effect.concentration_threshold * 2
                    intensity = organ_conc / (organ_conc + ec50)
                else:  # linear
                    intensity = min(1.0, organ_conc / (effect.concentration_threshold * 3))

                result["organ_effects"].append({
                    "organ": effect.organ,
                    "effect_type": effect.effect_type,
                    "mechanism": effect.mechanism,
                    "visual_effect": effect.visual_effect,
                    "description": effect.description,
                    "intensity": round(intensity, 2),
                    "concentration": round(organ_conc, 3),
                    "threshold": effect.concentration_threshold,
                    "color_shift": effect.color_shift,
                    "particle_type": effect.particle_type,
                    "particle_color": effect.particle_color,
                })
    else:
        # No concentrations — return all effects with default intensity
        for effect in drug_pd.get("organ_effects", []):
            result["organ_effects"].append({
                "organ": effect.organ,
                "effect_type": effect.effect_type,
                "mechanism": effect.mechanism,
                "visual_effect": effect.visual_effect,
                "description": effect.description,
                "intensity": 0.5,
                "concentration": 0,
                "threshold": effect.concentration_threshold,
                "color_shift": effect.color_shift,
                "particle_type": effect.particle_type,
                "particle_color": effect.particle_color,
            })

    # Cellular mechanisms
    for mech in drug_pd.get("cellular", []):
        result["cellular_mechanisms"].append({
            "cell_type": mech.cell_type,
            "receptor": mech.receptor,
            "receptor_type": mech.receptor_type,
            "action": mech.action,
            "binding_affinity": mech.binding_affinity,
            "downstream_effects": mech.downstream_effects,
            "visual_description": mech.visual_description,
        })

    # Disease interactions
    if diseases:
        drug_class = drug_pd["drug_class"]
        for disease_id in diseases:
            disease_key = disease_id.lower().replace("_", "").replace("-", "")
            for db_key, disease_data in DISEASE_ENTITIES.items():
                if db_key in disease_key or disease_key in db_key:
                    for interact_class, interaction in disease_data.get("drug_interactions", {}).items():
                        if interact_class.lower() in drug_class.lower() or drug_class.lower() in interact_class.lower():
                            result["disease_interactions"].append({
                                "drug_class": interact_class,
                                "target": interaction.target_entity,
                                "mechanism": interaction.mechanism,
                                "visual_animation": interaction.visual_animation,
                                "effectiveness": interaction.effectiveness,
                                "time_to_effect_h": interaction.time_to_effect_h,
                            })
                    # Also include the entities for visualization
                    for entity in disease_data.get("entities", []):
                        result.setdefault("disease_entities", []).append({
                            "type": entity.entity_type,
                            "name": entity.name,
                            "target_organ": entity.target_organ,
                            "visual_shape": entity.visual_shape,
                            "color": entity.color,
                            "size": entity.size,
                            "count": entity.count,
                            "behavior": entity.behavior,
                        })

    return result
