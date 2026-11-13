"""
Immune System Model (Innate + Adaptive)
========================================
Models the body's immune response and how drugs/diseases affect it.

Innate Immunity (fast, non-specific):
  - Neutrophils: first responders, phagocytose bacteria (hours)
  - Macrophages: phagocytose, present antigens, release cytokines (hours-days)
  - NK cells: kill virus-infected cells and tumor cells
  - Complement: cascade of 30+ proteins that lyse pathogens
  - Dendritic cells: bridge to adaptive immunity (antigen presentation)

Adaptive Immunity (slow, specific, has memory):
  - CD4+ T cells (helper): coordinate immune response, release cytokines
  - CD8+ T cells (cytotoxic): kill virus-infected cells directly
  - B cells → Plasma cells: produce antibodies (IgG, IgM, IgA, IgE)
  - Memory cells: remember pathogens for faster response on re-exposure

Key Cytokines:
  - IL-1: fever, inflammation, acute phase response
  - IL-6: major inflammatory mediator, suppresses CYP3A4/1A2
  - TNF-α: inflammation, cachexia, septic shock
  - IFN-γ: antiviral, macrophage activation
  - IL-10: anti-inflammatory (negative feedback)
  - IL-4: drives Th2 response, IgE production (allergies)

Drug Effects on Immunity:
  - Corticosteroids: suppress almost everything (broad immunosuppression)
  - NSAIDs: reduce prostaglandins → anti-inflammatory but don't suppress adaptive
  - Biologics: targeted (anti-TNF, anti-IL-6, checkpoint inhibitors)
  - Chemotherapy: kills rapidly dividing cells (including immune cells)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ImmuneState:
    """Complete immune system state."""
    # Innate cells (cells/µL blood)
    neutrophils: float = 4500     # Normal: 2500-7500
    monocytes: float = 500        # Normal: 200-800
    nk_cells: float = 200         # Normal: 100-400
    eosinophils: float = 200      # Normal: 100-500
    basophils: float = 50         # Normal: 20-100

    # Adaptive cells
    cd4_t_cells: float = 800      # Normal: 500-1500
    cd8_t_cells: float = 400      # Normal: 200-800
    b_cells: float = 200          # Normal: 100-500
    cd4_cd8_ratio: float = 2.0    # Normal: 1.0-3.0

    # Cytokines (pg/mL — picograms per mL)
    il_1: float = 5.0             # Normal: <10
    il_6: float = 3.0             # Normal: <7
    tnf_alpha: float = 8.0        # Normal: <15
    ifn_gamma: float = 10.0       # Normal: <20
    il_10: float = 5.0            # Normal: <20 (anti-inflammatory)
    il_4: float = 2.0             # Normal: <10

    # Immunoglobulins (mg/dL)
    igg: float = 1000             # Normal: 700-1600
    igm: float = 100              # Normal: 40-230
    iga: float = 200              # Normal: 70-400
    ige: float = 50               # Normal: <100 IU/mL

    # Complement
    c3: float = 120               # Normal: 90-180 mg/dL
    c4: float = 25                # Normal: 10-40 mg/dL

    # Functional scores
    innate_function: float = 100  # % of normal
    adaptive_function: float = 100
    overall_immunocompetence: float = 100


@dataclass
class ImmuneDrugEffect:
    """How a drug affects the immune system."""
    name: str
    drug_class: str
    mechanism: str

    # Cell count effects (% change)
    neutrophil_change: float = 0
    lymphocyte_change: float = 0  # All lymphocytes
    cd4_change: float = 0
    nk_change: float = 0

    # Cytokine effects (% change)
    il6_change: float = 0
    tnf_change: float = 0
    il1_change: float = 0
    il10_change: float = 0

    # Functional effects
    innate_suppression: float = 0    # % reduction in innate function
    adaptive_suppression: float = 0  # % reduction in adaptive function
    infection_risk_increase: float = 0  # % increase in infection susceptibility

    # Special effects
    causes_neutropenia: bool = False
    causes_lymphopenia: bool = False
    reactivation_risk: list[str] = field(default_factory=list)  # Latent infections that may reactivate


IMMUNE_DRUG_EFFECTS = {
    "prednisolone": ImmuneDrugEffect(
        name="Prednisolone (Corticosteroid)", drug_class="Glucocorticoid",
        mechanism="Binds glucocorticoid receptor → suppresses NF-κB → reduces cytokine production, inhibits T cell activation, causes lymphopenia + neutrophilia",
        neutrophil_change=30, lymphocyte_change=-50, cd4_change=-60, nk_change=-30,
        il6_change=-70, tnf_change=-60, il1_change=-50, il10_change=20,
        innate_suppression=30, adaptive_suppression=60, infection_risk_increase=40,
        causes_lymphopenia=True,
        reactivation_risk=["TB", "Hepatitis B", "Strongyloides", "Herpes zoster"],
    ),
    "methotrexate": ImmuneDrugEffect(
        name="Methotrexate", drug_class="DMARD / Antifolate",
        mechanism="Inhibits dihydrofolate reductase → blocks purine synthesis → suppresses rapidly dividing lymphocytes",
        neutrophil_change=-20, lymphocyte_change=-30, cd4_change=-25,
        il6_change=-30, tnf_change=-20,
        innate_suppression=15, adaptive_suppression=35, infection_risk_increase=25,
        causes_neutropenia=True,
        reactivation_risk=["Hepatitis B", "Herpes zoster"],
    ),
    "infliximab": ImmuneDrugEffect(
        name="Infliximab (Remicade)", drug_class="Anti-TNF biologic",
        mechanism="Chimeric monoclonal antibody that binds and neutralizes TNF-α → potent anti-inflammatory but increases infection risk",
        tnf_change=-90, il6_change=-40, il1_change=-30,
        innate_suppression=20, adaptive_suppression=15, infection_risk_increase=30,
        reactivation_risk=["TB (MUST screen before starting)", "Hepatitis B", "Histoplasmosis"],
    ),
    "tocilizumab": ImmuneDrugEffect(
        name="Tocilizumab (Actemra)", drug_class="Anti-IL-6 receptor biologic",
        mechanism="Blocks IL-6 receptor → reduces inflammation. CRITICAL: restores CYP3A4 activity (IL-6 suppresses CYP3A4)",
        il6_change=-95, tnf_change=-10, il1_change=-20,
        innate_suppression=15, adaptive_suppression=10, infection_risk_increase=20,
        reactivation_risk=["TB", "Hepatitis B"],
    ),
    "rituximab": ImmuneDrugEffect(
        name="Rituximab", drug_class="Anti-CD20 biologic",
        mechanism="Depletes CD20+ B cells → reduces antibody production. B cells recover over 6-12 months.",
        lymphocyte_change=-40, il6_change=-20,
        innate_suppression=5, adaptive_suppression=50, infection_risk_increase=35,
        causes_lymphopenia=True,
        reactivation_risk=["Hepatitis B (can be fatal)", "JC virus (PML)"],
    ),
    "cyclosporine": ImmuneDrugEffect(
        name="Cyclosporine", drug_class="Calcineurin inhibitor",
        mechanism="Blocks calcineurin in T cells → inhibits IL-2 production → suppresses T cell activation and proliferation",
        cd4_change=-40, il6_change=-30, tnf_change=-20,
        innate_suppression=10, adaptive_suppression=50, infection_risk_increase=30,
        reactivation_risk=["CMV", "EBV (lymphoma risk)", "BK virus"],
    ),
    "nivolumab": ImmuneDrugEffect(
        name="Nivolumab (Opdivo)", drug_class="Checkpoint inhibitor (anti-PD-1)",
        mechanism="Blocks PD-1 receptor → UNLEASHES T cells against tumor. Can cause autoimmune side effects (immune-related adverse events).",
        cd4_change=30, nk_change=20,
        il6_change=40, tnf_change=30, ifn_gamma_change=50,
        innate_suppression=-20, adaptive_suppression=-40,  # ENHANCES immunity
        infection_risk_increase=-10,  # Actually reduces infection risk
    ),
    "ibuprofen": ImmuneDrugEffect(
        name="Ibuprofen", drug_class="NSAID",
        mechanism="Inhibits COX-1/2 → reduces prostaglandin synthesis → anti-inflammatory but does NOT suppress adaptive immunity",
        il6_change=-15, tnf_change=-10, il1_change=-10,
        innate_suppression=5, adaptive_suppression=0, infection_risk_increase=0,
    ),
    "aspirin": ImmuneDrugEffect(
        name="Aspirin", drug_class="NSAID / COX inhibitor",
        mechanism="Irreversibly inhibits COX → reduces prostaglandins and thromboxane. Minimal immune suppression.",
        il6_change=-10, tnf_change=-5,
        innate_suppression=3, adaptive_suppression=0, infection_risk_increase=0,
    ),
    "caffeine": ImmuneDrugEffect(
        name="Caffeine", drug_class="Adenosine antagonist",
        mechanism="Blocks adenosine receptors → mildly modulates immune function. Adenosine normally suppresses inflammation.",
        il6_change=5, tnf_change=5,
        innate_suppression=-5, adaptive_suppression=0, infection_risk_increase=0,
    ),
}


def simulate_immune_effects(
    drug_name: str,
    dose_fraction: float = 1.0,
    baseline_state: ImmuneState | None = None,
    disease_context: str = "none",
) -> dict:
    """Simulate drug effects on the immune system."""
    state = baseline_state or ImmuneState()
    key = drug_name.lower().replace(" ", "_").replace("-", "_")
    drug = IMMUNE_DRUG_EFFECTS.get(key)

    if not drug:
        return {
            "drug": drug_name,
            "has_immune_effect": False,
            "message": "No immune profile available",
        }

    effect = dose_fraction

    # Apply cell count changes
    new_neutrophils = max(0, state.neutrophils * (1 + drug.neutrophil_change * effect / 100))
    new_lymphocytes = max(0, state.cd4_t_cells * (1 + drug.lymphocyte_change * effect / 100))
    new_cd4 = max(0, state.cd4_t_cells * (1 + drug.cd4_change * effect / 100))

    # Cytokine changes
    new_il6 = max(0, state.il_6 * (1 + drug.il6_change * effect / 100))
    new_tnf = max(0, state.tnf_alpha * (1 + drug.tnf_change * effect / 100))
    new_il1 = max(0, state.il_1 * (1 + drug.il1_change * effect / 100))

    # Functional scores
    innate = max(0, 100 - drug.innate_suppression * effect)
    adaptive = max(0, 100 - drug.adaptive_suppression * effect)
    overall = (innate * 0.4 + adaptive * 0.6)

    # Infection risk
    infection_risk = drug.infection_risk_increase * effect

    # Warnings
    warnings = []
    if new_neutrophils < 1500:
        warnings.append(f"NEUTROPENIA ({new_neutrophils:.0f}/µL) — increased bacterial/fungal infection risk")
    if new_neutrophils < 500:
        warnings.append("SEVERE NEUTROPENIA — febrile neutropenia protocol required")
    if new_cd4 < 200:
        warnings.append(f"CD4 < 200 — AIDS-defining level. PJP prophylaxis required.")
    if drug.reactivation_risk:
        warnings.append(f"Screen for latent infections before starting: {', '.join(drug.reactivation_risk)}")
    if new_il6 < 1.0:
        warnings.append("IL-6 profoundly suppressed — CYP3A4 may be RESTORED. Adjust CYP3A4 substrate doses.")

    # CYP interaction (IL-6 suppresses CYP3A4)
    cyp3a4_change = 0
    if drug.il6_change < -50:
        cyp3a4_change = abs(drug.il6_change) * 0.5  # CYP3A4 activity INCREASES when IL-6 drops
        warnings.append(
            f"Anti-IL-6 effect will INCREASE CYP3A4 activity by ~{cyp3a4_change:.0f}%. "
            f"Reduce doses of CYP3A4 substrates (simvastatin, cyclosporine, tacrolimus)."
        )

    return {
        "drug": drug.name,
        "drug_class": drug.drug_class,
        "mechanism": drug.mechanism,
        "has_immune_effect": True,
        "cell_counts": {
            "neutrophils": {"before": state.neutrophils, "after": round(new_neutrophils), "unit": "/µL"},
            "cd4_t_cells": {"before": state.cd4_t_cells, "after": round(new_cd4), "unit": "/µL"},
            "total_lymphocytes": {"before": state.cd4_t_cells + state.cd8_t_cells, "after": round(new_lymphocytes), "unit": "/µL"},
        },
        "cytokines": {
            "IL-6": {"before": state.il_6, "after": round(new_il6, 1), "change_pct": round(drug.il6_change * effect), "unit": "pg/mL"},
            "TNF-α": {"before": state.tnf_alpha, "after": round(new_tnf, 1), "change_pct": round(drug.tnf_change * effect), "unit": "pg/mL"},
            "IL-1": {"before": state.il_1, "after": round(new_il1, 1), "change_pct": round(drug.il1_change * effect), "unit": "pg/mL"},
        },
        "functional_scores": {
            "innate_immunity": round(innate, 1),
            "adaptive_immunity": round(adaptive, 1),
            "overall_immunocompetence": round(overall, 1),
        },
        "infection_risk": {
            "increase_percent": round(infection_risk, 1),
            "reactivation_risk": drug.reactivation_risk,
            "neutropenia": drug.causes_neutropenia,
            "lymphopenia": drug.causes_lymphopenia,
        },
        "cyp_interaction": {
            "cyp3a4_activity_change_pct": round(cyp3a4_change, 1),
            "mechanism": "IL-6 suppresses CYP3A4. Anti-IL-6 drugs restore CYP3A4." if cyp3a4_change > 0 else "No CYP interaction",
        },
        "warnings": warnings,
    }
