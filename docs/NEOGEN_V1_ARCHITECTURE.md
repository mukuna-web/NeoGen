# NeoGen V1 Architecture Blueprint

## Mechanistic Drug-Drug Interaction & Bioavailability Simulation Engine

**Classification:** Internal Technical Planning Document
**Author:** NeoGen Engineering
**Date:** April 2026
**Status:** Architecture Specification — Pre-Implementation

---

# A. EXECUTIVE TECHNICAL FRAMING

## What NeoGen Must Become

NeoGen is not a lookup table. It is not a black-box ML predictor that says "these two drugs interact." It must become a **mechanistic causal simulation engine** that models what physically happens when two or more drug molecules enter a human body simultaneously — from the moment of ingestion through absorption, first-pass metabolism, systemic distribution, enzyme competition, transporter interference, receptor occupancy, and elimination.

## The Core Technical Question

Given Drug A (SMILES, dose, route, formulation) and Drug B (same), and a Patient Profile (weight, age, sex, organ function, genotype), simulate the time-concentration curves of both drugs and all relevant metabolites across all organs over 0-96 hours, and quantify how the presence of each drug alters the other's:

- Bioavailability (F)
- Systemic exposure (AUC, Cmax, Tmax)
- Organ-level concentrations
- Metabolic pathway flux
- Receptor occupancy
- Toxicity potential

## Why Mechanistic, Not Black-Box

| Property                  | Black-Box ML              | Mechanistic Simulation           |
| ------------------------- | ------------------------- | -------------------------------- |
| Novel drug pairs          | Cannot extrapolate        | Predicts from first principles   |
| Explains WHY              | No                        | Yes — traceable causal chain     |
| FDA submission compatible | No                        | Yes — PBPK accepted in NDAs      |
| Fails gracefully          | Silently wrong            | Reports uncertainty per module   |
| Scientific moat           | Replicable with more data | Requires deep domain integration |
| Bioavailability modeling  | Scalar lookup             | Dynamic emergent property        |

NeoGen's current system (v2.1) uses ML to predict individual PK parameters + a 14-compartment PBPK shell. The V1 mechanistic engine replaces the black-box predictions with causal sub-models while retaining ML where mechanistic models are infeasible.

## Architecture Philosophy

**Hybrid mechanistic-empirical.** Not everything can be or should be mechanistic:

| Component                | Approach                                          | Rationale                                  |
| ------------------------ | ------------------------------------------------- | ------------------------------------------ |
| GI dissolution           | Semi-mechanistic (Noyes-Whitney)                  | Physics-based, well-understood             |
| Intestinal permeability  | ML-predicted + mechanistic correction             | Papp hard to compute from structure        |
| CYP metabolism kinetics  | Mechanistic (Michaelis-Menten)                    | Km/Vmax from literature, essential for DDI |
| Protein binding (fu)     | ML-predicted                                      | Binding thermodynamics too complex for V1  |
| Tissue partitioning (Kp) | Semi-mechanistic (Rodgers-Rowland)                | Physicochemistry-based equations exist     |
| Transporter kinetics     | Mechanistic where data exists, empirical fallback | P-gp/OATP well-characterized               |
| CYP induction dynamics   | Semi-mechanistic (Emax model + enzyme turnover)   | Nuclear receptor pathway known             |
| Cardiac ion channels     | Mechanistic (CiPA Hodgkin-Huxley)                 | Already implemented in NeoGen              |
| Receptor PD              | Emax model (empirical)                            | Full receptor kinetics deferred            |
| Systems biology pathways | Deferred to V2                                    | Requires pathway databases + ODE networks  |

---

# B. FULL LAYERED ARCHITECTURE

## Layer 1 — Drug Definition & Input

### Biology

A drug is a molecule with defined physicochemical properties that determine every downstream behavior: how it dissolves, crosses membranes, binds proteins, gets metabolized, and interacts with targets.

### Computation

The Drug entity is the root object. Everything downstream queries it.

### Variables Required

```
Drug {
  smiles: string                    # Canonical SMILES
  name: string
  molecular_weight: float           # g/mol
  logP: float                       # Octanol-water partition coefficient
  logD_7_4: float                   # Distribution coefficient at pH 7.4
  pKa: float[]                      # Ionization constants (acid and base)
  polar_surface_area: float         # TPSA in Angstroms^2
  hydrogen_bond_donors: int
  hydrogen_bond_acceptors: int
  rotatable_bonds: int
  aromatic_rings: int
  molecular_volume: float           # Angstroms^3
  is_acid: bool
  is_base: bool
  is_zwitterion: bool
  is_neutral: bool
  bcs_class: enum{I, II, III, IV}   # Biopharmaceutics Classification
  lipinski_violations: int

  # ML-predicted or literature
  fu_plasma: float                  # Fraction unbound in plasma
  fu_gut: float                     # Fraction unbound in gut
  blood_plasma_ratio: float         # B:P ratio

  # Metabolism profile
  cyp_substrate_profile: {          # fm per enzyme
    CYP3A4: float,
    CYP2D6: float,
    CYP2C9: float,
    CYP2C19: float,
    CYP1A2: float,
    CYP2E1: float,
    CYP2B6: float,
    UGT: float,
    OTHER: float,
  }
  cyp_inhibition_profile: {        # Ki values (uM) per enzyme
    CYP3A4: {Ki: float, mechanism: enum{reversible, irreversible, TDI}},
    CYP2D6: {Ki: float, mechanism: ...},
    ...
  }
  cyp_induction_profile: {         # Emax, EC50 per nuclear receptor
    PXR: {Emax: float, EC50: float},
    CAR: {Emax: float, EC50: float},
    AhR: {Emax: float, EC50: float},
  }

  # Transporter profile
  transporter_substrate: {P_gp: bool, BCRP: bool, OATP1B1: bool, ...}
  transporter_inhibition: {P_gp: {IC50: float}, OATP1B1: {IC50: float}, ...}

  # Clearance
  renal_fraction: float            # fe — fraction excreted unchanged in urine

  # PD (optional)
  target_receptor: string
  target_Ki: float
  target_Emax: float
  target_EC50: float
}
```

### Data Sources

- **Structure-derived:** RDKit computes MW, logP, TPSA, pKa, HBD, HBA, rotatable bonds, BCS classification from SMILES
- **ML-predicted:** fu, CYP profiles, transporter profiles from NeoGen's 20+ XGBoost models
- **Literature:** DrugBank, ChEMBL, DDInter 2.0 for known Ki, Km, Vmax values
- **Hybrid:** When literature value exists, use it. When not, fall back to ML prediction with uncertainty flag.

### MVP Simplification

- Accept ML-predicted values for fu, CYP profiles, transporter substrate status
- Require SMILES only as input
- Defer: pKa-dependent solubility profiles, polymorphic form effects

---

## Layer 2 — Formulation & Administration

### Biology

How a drug enters the body determines the initial conditions for the entire simulation. Oral tablets must dissolve, cross the GI wall, and survive first-pass metabolism. IV bypasses all of this.

### Computation

A Dose event initializes the simulation with a drug amount at a specific body location.

### Variables

```
DoseEvent {
  drug: Drug
  route: enum{oral, iv_bolus, iv_infusion, sublingual, im, sc, topical, inhaled}
  dose_mg: float
  formulation: enum{solution, immediate_release, enteric_coated, sustained_release}
  timing_hours: float              # Time of administration (for multi-dose)
  fed_state: enum{fasted, fed}     # Affects gastric emptying and dissolution

  # For IV infusion
  infusion_rate_mg_per_hr: float
  infusion_duration_hr: float
}
```

### Route-Specific Initialization

| Route       | Initial Compartment           | Bioavailability             | First-Pass           |
| ----------- | ----------------------------- | --------------------------- | -------------------- |
| IV bolus    | Central plasma                | F = 1.0 by definition       | None                 |
| IV infusion | Central plasma (rate-limited) | F = 1.0                     | None                 |
| Oral        | Stomach → GI lumen            | F = fa × fg × fh (EMERGENT) | Gut wall + Liver     |
| Sublingual  | Buccal mucosa → systemic      | F ~ 0.3-0.7                 | Partial liver bypass |
| IM/SC       | Depot → systemic              | F ~ 0.7-1.0                 | Liver only           |

### MVP Simplification

- Support oral and IV only
- Immediate-release formulation only
- Fasted state default
- Defer: sustained-release dissolution profiles, food effect modeling, IM/SC depot kinetics

---

## Layer 3 — Absorption & Bioavailability

### Biology

For oral drugs, absorption is the most complex layer. The drug must:

1. Dissolve in GI fluid (dissolution)
2. Survive gastric pH and enzymes (stability)
3. Cross the intestinal epithelium (permeability — passive + active)
4. Survive gut wall metabolism by CYP3A4 and UGTs in enterocytes (fg)
5. Survive efflux back into the lumen by P-gp and BCRP
6. Travel via portal vein to the liver
7. Survive hepatic first-pass metabolism (fh)

**Bioavailability is NOT a single number. It is the emergent product of all these sub-processes:**

```
F = fa × fg × fh
```

Where:

- **fa** = fraction absorbed across intestinal wall (dissolution + permeability + stability)
- **fg** = fraction surviving gut wall metabolism (enterocyte CYP3A4, UGTs)
- **fh** = fraction surviving hepatic first-pass (liver CYPs, UGTs)

### Computational Sub-Modules

#### 3a. Dissolution Module

```
Input:  Drug (solubility, particle_size, formulation), GI_conditions (pH, volume, motility)
Output: dissolved_drug_concentration(t) in GI lumen

Equation: Noyes-Whitney
  dM/dt = (D × A / h) × (Cs - C)

  D  = diffusion coefficient (Stokes-Einstein from MW)
  A  = surface area of particles (from particle size distribution)
  h  = diffusion layer thickness (~30 um)
  Cs = saturation solubility at local pH (Henderson-Hasselbalch from pKa)
  C  = current dissolved concentration

pH profile along GI:
  Stomach: pH 1.5-3.5 (fasted) / 4.5-6.0 (fed)
  Duodenum: pH 5.5-6.5
  Jejunum: pH 6.0-7.0
  Ileum: pH 7.0-7.5
  Colon: pH 5.5-7.0
```

**MVP:** For BCS Class I (high solubility, high permeability), assume complete dissolution. Only model dissolution explicitly for BCS Class II (low solubility).

#### 3b. Permeability Module

```
Input:  Drug (logP, PSA, MW, HBD, pKa), GI_segment
Output: effective_permeability (Peff, cm/s)

Approach: ML-predicted Papp (Caco-2) → scaled to human Peff
  Peff_human = alpha × Papp_predicted + beta  (calibration from ~200 drugs)

Passive transcellular: dominates for lipophilic drugs (logP > 1)
Passive paracellular: relevant for small, hydrophilic drugs (MW < 200)
Active transport: P-gp efflux reduces net absorption, OATP/PEPT1 uptake increases it

Absorption rate from GI segment:
  dA_absorbed/dt = Peff × SA_segment × C_dissolved × (1 - fraction_effluxed)

  SA by segment (cm^2):
    Duodenum: ~1,800
    Jejunum: ~18,400
    Ileum: ~6,000
    Colon: ~900
```

**MVP:** Use ML-predicted Peff. Add P-gp efflux correction for known P-gp substrates. Defer paracellular transport.

#### 3c. Gut Wall Metabolism Module (fg)

```
Input:  Drug (CYP3A4 substrate? fm_CYP3A4), enterocyte CYP3A4 abundance
Output: fg — fraction escaping gut wall metabolism

Model: Qgut model (Yang et al.)
  fg = Qgut / (Qgut + fu_gut × CLint_gut)

  Qgut = effective gut blood flow corrected for permeability
       = Peff × SA / (1 + Peff × SA / Q_enterocyte)
  CLint_gut = CYP3A4_gut_abundance × Vmax_CYP3A4 / (Km_CYP3A4 + C_gut)

Key: Intestinal CYP3A4 abundance ≈ 1% of hepatic, but oral drugs see it FIRST.
     Grapefruit effect: irreversible CYP3A4 inhibition in gut wall.
     P-gp and CYP3A4 co-localize — drugs effluxed back get re-exposed to CYP3A4.
```

**MVP:** Calculate fg from ML-predicted fm_CYP3A4 using the Qgut model with population-average CYP3A4 gut abundance. Defer P-gp/CYP3A4 interplay loop.

#### 3d. Hepatic First-Pass Module (fh)

```
Input:  Drug (CLint_hepatic, fu_plasma), hepatic blood flow (Qh)
Output: fh — fraction escaping hepatic first-pass

Model: Well-stirred liver model
  E_h = fu × CLint / (Qh + fu × CLint)
  fh = 1 - E_h

  Qh = hepatic blood flow ≈ 1500 mL/min (70 kg adult)
  CLint = sum of all hepatic metabolic pathways
        = CLint_CYP3A4 + CLint_CYP2D6 + CLint_CYP2C9 + ... + CLint_UGT + CLint_other

  Each pathway: CLint_CYPx = (Vmax_x / Km_x) × liver_abundance_x
```

**MVP:** Use well-stirred model. CLint from ML-predicted total intrinsic clearance, partitioned by fm values. Defer parallel-tube and dispersion models.

### How Bioavailability Changes in DDI Context

**THIS IS THE CRITICAL INSIGHT:** When Drug B inhibits CYP3A4, it changes Drug A's bioavailability in two ways:

1. **Increases fg:** Less gut wall metabolism → more drug survives to portal vein
2. **Increases fh:** Less hepatic first-pass → more drug reaches systemic circulation

The result: Drug A's effective dose increases without changing the actual dose. This is why ketoconazole + simvastatin causes 10-fold AUC increase — it's not just inhibiting systemic clearance, it's also boosting bioavailability.

```
F_baseline = fa × fg_baseline × fh_baseline
F_with_inhibitor = fa × fg_inhibited × fh_inhibited

AUC_ratio = F_with_inhibitor / F_baseline × CL_baseline / CL_inhibited
```

### Bioavailability Architecture Decision for NeoGen

**Bioavailability must be an EMERGENT PROPERTY, not a scalar input.**

It is computed as the product of three independently modeled sub-modules (fa, fg, fh), each of which can be perturbed by co-administered drugs. Storing F as a single number makes DDI modeling impossible because you can't distinguish between a drug with low F due to poor absorption (fa) versus high first-pass metabolism (fh).

```
Bioavailability is NOT stored as a field.
It is COMPUTED by the simulation at runtime:
  F_effective(t) = fa(t) × fg(t, enzyme_state) × fh(t, enzyme_state)
```

---

## Layer 4 — Distribution (Plasma & Tissue)

### Biology

After reaching systemic circulation, a drug distributes into tissues based on blood flow, protein binding, tissue affinity, and transporter activity. Only the **unbound fraction** is pharmacologically active and available for metabolism.

### Computation: PBPK Compartmental Model

NeoGen's existing 14-compartment model provides the structural foundation. Each compartment represents an organ/tissue with:

- Blood flow (Q_organ)
- Tissue volume (V_organ)
- Tissue:plasma partition coefficient (Kp)
- Transporter expression levels

```
Compartments:
  1. Arterial blood pool
  2. Venous blood pool
  3. Lung
  4. Heart
  5. Brain (with BBB — P-gp mediated)
  6. Liver (metabolically active — receives portal + arterial)
  7. Gut (metabolically active — CYP3A4 + P-gp)
  8. Kidney (filtration + secretion + reabsorption)
  9. Muscle
  10. Adipose
  11. Skin
  12. Bone
  13. Spleen
  14. Rest-of-body (aggregated)
```

### Tissue Partitioning (Kp Prediction)

```
Method: Rodgers-Rowland (2005/2006) equations

For neutral and weak bases:
  Kp = (fu_p / fu_t) × (1 + Rat × Ka × [AP]t)

For acids:
  Kp = fu_p × (Vwt + 0.3×Vlt + ... + Ka×[AP]t) / fu_t

Inputs needed per tissue:
  - Fractional tissue water (Vwt)
  - Fractional tissue lipid (Vlt, Vnlt)
  - Acidic phospholipid concentration [AP]t
  - Tissue protein concentration

Inputs needed per drug:
  - fu_plasma (ML-predicted or literature)
  - logP, pKa, blood_plasma_ratio
  - Drug type (acid, base, neutral, zwitterion)
```

### Mass Balance ODE per Compartment

```
Non-eliminating organ:
  V_t × dC_t/dt = Q_t × (C_art - C_t/Kp_t)

Liver (eliminating, receives portal vein):
  V_liver × dC_liver/dt = Q_ha × C_art + Q_pv × C_gut/Kp_gut
                          - (Q_ha + Q_pv) × C_liver/Kp_liver
                          - CLint_hepatic × fu_p × C_liver/Kp_liver

Kidney (eliminating):
  V_kidney × dC_kidney/dt = Q_kidney × (C_art - C_kidney/Kp_kidney)
                           - CL_renal × fu_p × C_kidney/Kp_kidney

Venous mixing:
  Q_total × C_ven = sum(Q_t × C_t/Kp_t) for all tissues
```

### Protein Binding DDI

From the PDF: "If Drug A displaces Drug B from albumin, Drug B's free fraction temporarily increases, potentially causing toxicity. However, the body usually compensates by increasing clearance of the now-unbound drug. Clinically significant protein binding displacement DDIs are rarer than commonly taught."

**MVP:** Use static fu_plasma per drug. Do not model displacement dynamics. Flag pairs where both drugs are >95% protein bound as requiring manual review. Defer dynamic fu modeling to V2.

---

## Layer 5 — Metabolism

### Biology

Drug metabolism converts lipophilic molecules into hydrophilic metabolites for excretion. The liver is primary. Phase I (CYP450 oxidation/reduction/hydrolysis) and Phase II (UGT conjugation, sulfation, etc.) operate in sequence or parallel.

### Computation: Enzyme Kinetics Engine

This is the **most DDI-critical layer.** Every metabolic DDI operates through this module.

```
MetabolismEngine {

  For each Drug in simulation:
    For each Enzyme (CYP3A4, CYP2D6, CYP2C9, CYP2C19, CYP1A2, CYP2E1, CYP2B6, UGT2B7, ...):

      # Base metabolic rate: Michaelis-Menten
      rate_base = (Vmax × C_unbound) / (Km + C_unbound)

      # Apply DDI modifiers
      rate_actual = rate_base × inhibition_factor × induction_factor

      # Inhibition (competitive reversible):
      inhibition_factor = 1 / (1 + [I_unbound] / Ki)

      # Inhibition (mechanism-based / irreversible):
      dE_inactive/dt = k_inact × [I] × E_active / (KI + [I])
      dE_active/dt = k_synthesis - k_degradation × E_active - k_inact × [I] × E_active / (KI + [I])

      # Induction (via nuclear receptor):
      E_induced = E_baseline × (1 + Emax × [I] / (EC50 + [I]))
      dE_total/dt = k_synthesis × fold_induction - k_degradation × E_total

      # Net enzyme activity at time t:
      E_effective(t) = E_active(t) × (1 - fraction_reversibly_inhibited)

      # Metabolic clearance contribution:
      CLint_enzyme = (Vmax / Km) × E_effective(t) / E_baseline
}
```

### CYP Enzyme Parameters Required

From the PDF — the Big Five handle ~90% of drug metabolism:

| Enzyme  | % Drugs | Key for DDI                    | Inducible?      |
| ------- | ------- | ------------------------------ | --------------- |
| CYP3A4  | ~50%    | Most DDIs occur here           | Yes (PXR)       |
| CYP2D6  | ~25%    | Polymorphic, high fm drugs     | **NO** (unique) |
| CYP2C9  | ~10%    | Warfarin, narrow TI drugs      | Yes (PXR)       |
| CYP2C19 | ~5%     | Clopidogrel prodrug activation | Yes (PXR)       |
| CYP1A2  | ~5%     | Smoking-inducible              | Yes (AhR)       |

### Metabolite Tracking

Critical for: prodrug activation (codeine → morphine via CYP2D6), active metabolites (diazepam → nordiazepam), toxic metabolites (acetaminophen → NAPQI via CYP2E1).

```
Metabolite {
  parent: Drug
  formation_enzyme: Enzyme
  formation_rate: Michaelis-Menten params
  is_active: bool
  own_metabolism: Drug  # Metabolite can itself be metabolized
  own_pk_params: {...}
}
```

**MVP:** Track primary metabolites for high-fm drugs. Flag prodrugs (codeine, clopidogrel, enalapril). Do NOT model secondary metabolites or the full metabolite tree. Defer NAPQI-type reactive intermediate modeling.

---

## Layer 6 — Elimination & Clearance

### Biology

Final removal of drug and metabolites. Two primary routes: renal (urine) and biliary (feces). Minor: pulmonary, sweat.

### Computation

```
Total Clearance:
  CL_total = CL_hepatic + CL_renal + CL_biliary + CL_other

Hepatic Clearance (well-stirred model):
  CL_hepatic = Qh × fu × CLint / (Qh + fu × CLint)

Renal Clearance:
  CL_renal = (GFR × fu) + CL_secretion - CL_reabsorption

  GFR ≈ 120 mL/min (healthy adult)
  Secretion: OAT1/3, OCT2 mediated (active transport, saturable)
  Reabsorption: passive, pH-dependent for weak acids/bases

Biliary Clearance:
  CL_biliary: MRP2-mediated excretion of glucuronide conjugates
  Enterohepatic recirculation: glucuronide hydrolyzed by gut bacteria → parent reabsorbed
```

### DDI at Elimination Level

| Mechanism               | Example (from PDF)        | Effect                                  |
| ----------------------- | ------------------------- | --------------------------------------- |
| OAT inhibition          | Probenecid + penicillin   | Reduced renal secretion → higher levels |
| OCT2 inhibition         | Cimetidine + metformin    | Reduced renal clearance                 |
| P-gp inhibition (renal) | Quinidine + digoxin       | 2-fold digoxin increase                 |
| MRP2 inhibition         | Probenecid + methotrexate | Reduced biliary excretion               |

**MVP:** Model renal clearance as GFR × fu × (1 - reabsorption_fraction) + secretion. Use fe (fraction excreted unchanged) from literature/ML. Defer enterohepatic recirculation and transporter-mediated secretion kinetics.

---

## Layer 7 — Drug-Drug Interaction Resolution

### Biology

DDIs are not a separate phenomenon — they are emergent behaviors from Layers 3-6 operating simultaneously on multiple drugs. But the DDI Resolution layer is the **orchestrator** that manages multi-drug competition for shared resources.

### Mechanistic DDI Classification (from PDF)

#### 7a. Absorption-Phase DDIs (Layer 3)

| DDI Type                | Mechanism                                      | Variables Affected               | Model                             |
| ----------------------- | ---------------------------------------------- | -------------------------------- | --------------------------------- |
| Gastric pH alteration   | PPI raises pH → changes dissolution            | Solubility(pH), dissolution rate | Henderson-Hasselbalch             |
| Chelation               | Metal ions bind drug → insoluble complex       | fa                               | Binary flag + fa reduction factor |
| GI motility change      | Opioids slow, metoclopramide speeds            | Transit time, absorption window  | Transit time parameter            |
| P-gp inhibition (gut)   | Inhibitor blocks efflux → increased absorption | fa, net Peff                     | IC50-based Peff correction        |
| CYP3A4 inhibition (gut) | Inhibitor blocks gut wall metabolism           | fg                               | Ki-based CLint_gut reduction      |

**MVP:** Model CYP3A4 gut inhibition (highest clinical impact — grapefruit effect). Flag pH interactions for BCS Class II drugs. Defer chelation, motility.

#### 7b. Metabolic DDIs (Layer 5) — HIGHEST PRIORITY

| DDI Type                     | Mechanism                                            | Equation                                           | Time Course                       |
| ---------------------------- | ---------------------------------------------------- | -------------------------------------------------- | --------------------------------- |
| Competitive inhibition       | Drug occupies enzyme active site                     | AUC_ratio = 1/(1 + [I]/Ki)                         | Fast onset (inhibitor Css)        |
| Mechanism-based inhibition   | Reactive intermediate inactivates enzyme permanently | dE/dt = k_synth - k_deg×E - k_inact×[I]×E/(KI+[I]) | Onset 1-2 days, offset 1-3 days   |
| Induction                    | Nuclear receptor → increased transcription           | E_ss = E_base × (1 + Emax×[I]/(EC50+[I]))          | Onset 1-2 weeks, offset 1-2 weeks |
| Mixed inhibition + induction | Same drug does both (ritonavir)                      | Net effect = time-dependent balance                | Complex, requires ODE tracking    |

**The fm Rule (from PDF):** `AUC_ratio = 1 / (1 - fm + fm/(1 + [I]/Ki))`

This is the **single most important equation for DDI prediction.** It connects:

- How much the victim drug depends on the inhibited enzyme (fm)
- How potent the inhibitor is ([I]/Ki)
- The resulting exposure change (AUC ratio)

#### 7c. Transporter DDIs (Layer 4/6)

| DDI Type           | Where                   | Clinical Example (PDF)   | Effect                                 |
| ------------------ | ----------------------- | ------------------------ | -------------------------------------- |
| P-gp inhibition    | Gut, liver, kidney, BBB | Quinidine + digoxin      | ↑ absorption, ↓ excretion              |
| OATP1B1 inhibition | Liver sinusoidal        | Cyclosporine + statins   | ↑ systemic exposure (↓ hepatic uptake) |
| OAT/OCT inhibition | Kidney proximal tubule  | Probenecid + penicillin  | ↓ renal clearance                      |
| BCRP inhibition    | Gut, liver              | Elacridar + rosuvastatin | ↑ statin exposure                      |

**MVP:** Implement P-gp and OATP1B1 interactions with IC50-based inhibition. These account for the majority of clinically significant transporter DDIs.

#### 7d. Pharmacodynamic DDIs (Layer 8)

| Type         | Example (PDF)                                    | Model                              |
| ------------ | ------------------------------------------------ | ---------------------------------- |
| Additive     | Opioid + benzodiazepine → respiratory depression | Effect_total = Effect_A + Effect_B |
| Synergistic  | TMP + SMX → folate pathway                       | Effect_total > Effect_A + Effect_B |
| Antagonistic | Warfarin + Vitamin K                             | Effect_total < Effect_A + Effect_B |

**MVP:** Classify PD DDIs by literature lookup (additive/synergistic/antagonistic flag). Defer quantitative PD interaction modeling.

### Multi-Drug Interaction Resolver

When 3+ drugs compete for the same enzyme:

```
For CYP3A4 with N inhibitors:
  Total_inhibition = product of (1/(1 + [Ii]/Ki_i)) for i = 1..N

For victim drug metabolized by CYP3A4 (fm_3A4):
  AUC_ratio = 1 / (1 - fm_3A4 + fm_3A4 × Total_inhibition)
```

This handles polypharmacy — the common clinical scenario (74% of Americans take 2+ medications).

---

## Layer 8 — Pharmacodynamic / Target Effect

### Biology

PD describes the relationship between drug concentration at the effect site and the biological response. This is where "what the drug does to the body" gets modeled.

### Computation

```
Emax Model (standard):
  E = Emax × C^n / (EC50^n + C^n)

  Emax = maximum effect
  EC50 = concentration producing 50% of max effect
  n = Hill coefficient (steepness)
  C = unbound concentration at effect site

Receptor Occupancy:
  Occupancy = C / (Kd + C)

  For competitive antagonism with 2 drugs:
  Occupancy_A = [A]/(Kd_A × (1 + [B]/Kd_B) + [A])
```

**MVP:** Implement Emax model for drugs with known PD parameters. Receptor-level competition modeling for same-target drugs only. Defer downstream signaling.

---

## Layer 9 — Systems Biology / Pathway Response

### Biology

Drug effects propagate through biological pathways. CYP inhibition changes metabolite profiles. Pathway perturbation can cause off-target toxicity.

### Computation

This is the most ambitious layer and is **deferred to V2/V3.**

```
V2 targets:
  - Acetaminophen toxicity pathway (CYP2E1 → NAPQI → glutathione depletion)
  - Serotonin accumulation pathway (MAO inhibition + serotonergic drug)
  - QT prolongation pathway (multi-channel ion current integration — already in NeoGen CiPA)

V3 targets:
  - Full pathway modeling via KEGG/Reactome integration
  - Graph-based perturbation propagation
  - Organ-level response functions (hepatotoxicity scores, nephrotoxicity scores)
```

**MVP:** Use the existing 7-channel CiPA cardiac model as the only systems-level pathway. Everything else is risk-flagging, not mechanistic modeling.

---

## Layer 10 — Simulation / Time Evolution Engine

### Biology

Drug concentrations change continuously over time. Multiple drugs evolve simultaneously, their concentrations affecting each other's metabolism in real-time.

### Computation: ODE System

The entire simulation is a system of coupled ordinary differential equations solved over time.

```
State Vector (for N drugs, M compartments, E enzymes):
  y = [C_drug1_comp1, C_drug1_comp2, ..., C_drug1_compM,
       C_drug2_comp1, ..., C_drug2_compM,
       ...,
       C_drugN_compM,
       E_active_CYP3A4, E_active_CYP2D6, ...,  # Enzyme pool states
       metabolite_1_comp1, ...]

System size: ~(N_drugs × M_compartments) + E_enzymes + metabolites
  For 2 drugs, 14 compartments, 7 enzymes: ~40-50 ODEs
  For 5 drugs: ~100-120 ODEs

Solver: scipy.integrate.solve_ivp with BDF method (stiff ODE solver)
  - Variable time step (auto-adjusted for stiffness)
  - Relative tolerance: 1e-6
  - Absolute tolerance: 1e-9
  - Event detection for dose events (multiple doses over 24-96 hours)
```

### Simulation Timeline

```
t = 0:          First dose administered
t = 0-2h:       Absorption phase (GI dissolution, permeability, gut wall metabolism)
t = 0.5-4h:     Distribution phase (tissue partitioning equilibration)
t = 2-8h:       Pseudo-distribution equilibrium
t = 0-96h:      Continuous: metabolism, elimination, enzyme state evolution
t = dosing_times: Multiple dose events (e.g., q12h, q8h, q24h dosing)
```

### Output at Each Timepoint

```
SimulationState(t) {
  for each drug:
    C_plasma_total(t)
    C_plasma_unbound(t)
    C_tissue[organ](t) for all 14 compartments
    amount_absorbed(t)
    amount_metabolized(t)
    amount_eliminated_renal(t)
    amount_eliminated_biliary(t)
    bioavailability_effective(t)

  for each enzyme:
    E_active(t) / E_baseline — fold-change from baseline

  derived PK metrics:
    AUC(0-t), AUC(0-inf)
    Cmax, Tmax
    half_life_terminal
    clearance_apparent
}
```

### Simulation Strategy Recommendation

| Approach                 | Best For                                                      | NeoGen Use                     |
| ------------------------ | ------------------------------------------------------------- | ------------------------------ |
| Compartmental ODE (PBPK) | Core PK + DDI simulation                                      | **V1 PRIMARY**                 |
| Hybrid ODE + rule-based  | When pathway triggers (e.g., glutathione depletion threshold) | V2 for toxicity pathways       |
| Stochastic simulation    | Population variability, Monte Carlo                           | V2 for population PK           |
| Agent-based physiology   | Individual cell-level modeling                                | **Not recommended** — overkill |
| Graph-based pathway      | Systems biology integration                                   | V3                             |

**V1: Deterministic PBPK ODE system.** Single patient, multiple drugs, coupled enzyme dynamics. This is what Simcyp, GastroPlus, and PK-Sim all use. It's the FDA-accepted standard.

---

## Layer 11 — Validation & Benchmarking

### Validation Strategy

#### Tier 1: Single-Drug PK Validation

```
For 50+ drugs with known clinical PK data:
  Compare predicted vs observed:
    - AUC (within 2-fold = pass, per FDA guidance)
    - Cmax (within 2-fold)
    - Tmax (within 30 min for oral drugs)
    - Half-life (within 2-fold)
    - Bioavailability F (within 20% absolute)

  Data sources: Goodman & Gilman, Rowland & Tozer, FDA drug labels

  Acceptance criterion: ≥70% of drugs within 2-fold for AUC and Cmax
  Current NeoGen performance: 65% (within 2-fold on 50-drug holdout)
```

#### Tier 2: DDI Pair Validation

```
For known DDI pairs from FDA guidance and clinical studies:
  Compare predicted vs observed AUC_ratio:
    - Ketoconazole + midazolam: observed AUC_ratio ~15-fold
    - Rifampin + midazolam: observed AUC_ratio ~0.04 (96% decrease)
    - Fluconazole + warfarin: observed AUC_ratio ~2-fold
    - Ritonavir + nirmatrelvir: boosting ratio

  Top 10 DDI pairs from PDF Section 13.2 as mandatory benchmark set

  Acceptance: ≥60% within 2-fold of observed AUC_ratio for V1
  Stretch: ≥80% for V2
```

#### Tier 3: Bioavailability Validation

```
For drugs with known fa, fg, fh decomposition:
  - Midazolam: F = 0.3-0.4, fg ≈ 0.57, fh ≈ 0.57
  - Simvastatin: F = 0.05, high first-pass
  - Propranolol: F = 0.25, high extraction

  Verify that F_predicted = fa × fg × fh matches known F
  AND that DDI correctly shifts the component that changes
```

#### Tier 4: Mechanistic Sanity Checks

```
Invariants that must hold:
  - IV F = 1.0 always
  - CYP3A4 inhibitor increases AUC of CYP3A4 substrate (never decreases)
  - CYP inducer decreases AUC (never increases)
  - Higher dose of same drug does not change clearance (for linear PK drugs)
  - fu × CL_total = fu × CLint for low-extraction drugs
  - Mass balance: drug_in = drug_absorbed + drug_in_GI_lumen at all times
```

### What Would Make the Model "Look Smart but Be Wrong"

1. **Overfitting fu:** If fu_predicted is wrong, ALL downstream PK is wrong (CL, Vd, tissue concentrations). A model can produce plausible-looking concentration curves with completely wrong fu.

2. **Ignoring gut wall metabolism (fg):** Treating F as a lookup value instead of computing fa × fg × fh means DDIs that alter gut CYP3A4 are invisible. The model predicts normal absorption when it should predict 5-fold increase.

3. **Using total [I] instead of unbound [I] for DDI:** The inhibitor concentration that matters is unbound. Using total [I] with a highly protein-bound inhibitor will massively overpredict DDI severity.

4. **Ignoring fm:** Predicting all CYP3A4 substrates will have the same DDI magnitude. A drug with fm_CYP3A4 = 0.3 has a 1.4-fold AUC change; fm = 0.9 has a 10-fold change. Ignoring fm destroys calibration.

5. **Static enzyme pools:** Not modeling enzyme turnover means mechanism-based inhibitors and inducers are wrong. Paroxetine (irreversible CYP2D6 inhibitor) and rifampin (CYP3A4 inducer) have multi-day onset/offset that static models miss.

---

## Layer 12 — Reporting & Interpretation

### Output Artifacts

```
SimulationReport {
  input_summary: drugs, doses, patient_profile

  pk_table: {
    per drug: AUC, Cmax, Tmax, t1/2, CL, Vd, F_computed
    alone vs with co-administered drugs
    fold-change and clinical significance
  }

  concentration_time_curves: {
    plasma total and unbound for each drug
    overlay: alone vs with interaction
    organ-level curves for liver, kidney, heart
  }

  ddi_summary: {
    mechanism: "CYP3A4 competitive inhibition"
    perpetrator: Drug B
    victim: Drug A
    AUC_ratio: 4.2
    severity: "Major"
    clinical_action: "Reduce Drug A dose by 75% or use alternative"
  }

  bioavailability_decomposition: {
    fa: 0.95, fg: 0.57, fh: 0.43
    fa_with_DDI: 0.95, fg_with_DDI: 0.85, fh_with_DDI: 0.72
    F_baseline: 0.23, F_with_DDI: 0.58
  }

  cardiac_safety: CiPA score (existing)

  confidence_flags: {
    per module: "literature-validated" | "ML-predicted" | "extrapolated"
    overall_confidence: high | medium | low
  }
}
```

---

# C. COMPUTATIONAL MODULE MAP

| #   | Module                    | Input                                  | Output                                  | Science                                      | Depends On      |
| --- | ------------------------- | -------------------------------------- | --------------------------------------- | -------------------------------------------- | --------------- |
| 1   | `DrugFactory`             | SMILES                                 | Drug entity with all properties         | RDKit + ML models                            | None            |
| 2   | `DoseScheduler`           | Drug, dose, route, timing              | DoseEvent sequence                      | Administration logic                         | DrugFactory     |
| 3   | `DissolutionEngine`       | Drug, GI_pH, formulation               | dissolved_amount(t)                     | Noyes-Whitney                                | DrugFactory     |
| 4   | `PermeabilityPredictor`   | Drug properties                        | Peff (cm/s)                             | ML + Caco-2 scaling                          | DrugFactory     |
| 5   | `GIAbsorptionModule`      | dissolved_amount, Peff, P-gp           | amount_absorbed(t) = fa                 | Compartmental GI transit                     | 3, 4            |
| 6   | `GutWallMetabolism`       | Drug, CYP3A4_gut, inhibitors           | fg                                      | Qgut model + Michaelis-Menten                | DrugFactory, 9  |
| 7   | `HepaticFirstPass`        | Drug, CLint, fu, Qh, inhibitors        | fh                                      | Well-stirred model                           | DrugFactory, 9  |
| 8   | `BioavailabilityComputer` | fa, fg, fh                             | F(t) = fa × fg × fh                     | Multiplicative chain                         | 5, 6, 7         |
| 9   | `EnzymeKineticsEngine`    | All drugs, enzyme states               | CLint per enzyme, E_active(t)           | Michaelis-Menten + inhibition/induction ODEs | DrugFactory     |
| 10  | `ProteinBindingModule`    | Drug properties                        | fu_plasma, fu_gut                       | ML prediction                                | DrugFactory     |
| 11  | `TissuePartitioner`       | Drug properties, tissue params         | Kp per tissue                           | Rodgers-Rowland                              | DrugFactory, 10 |
| 12  | `PBPKCompartmentModel`    | All above + physiology                 | C(t) per compartment per drug           | Coupled ODE system                           | 5-11            |
| 13  | `RenalClearanceModule`    | Drug, GFR, fu, transporters            | CL_renal                                | Filtration + secretion - reabsorption        | DrugFactory, 10 |
| 14  | `DDIResolver`             | All drugs, enzyme/transporter profiles | Inhibition/induction factors per enzyme | fm rule + [I]/Ki                             | DrugFactory, 9  |
| 15  | `MetaboliteTracker`       | Parent drug, enzyme, rate              | Metabolite concentration(t)             | Formation rate + own PK                      | 9, 12           |
| 16  | `CardiacSafetyModule`     | Drug, ion channel IC50s                | CiPA score, APD90, EAD risk             | Hodgkin-Huxley (existing)                    | DrugFactory     |
| 17  | `PDEffectModule`          | C_unbound at effect site, PD params    | Effect(t), receptor occupancy           | Emax model                                   | 12              |
| 18  | `SimulationOrchestrator`  | All modules, time span                 | Full SimulationState(t)                 | ODE integration (BDF)                        | ALL             |
| 19  | `ValidationEngine`        | Predicted PK, reference data           | Pass/fail per metric, fold-errors       | Statistical comparison                       | 18              |
| 20  | `ReportGenerator`         | SimulationState                        | PDF, JSON, visualizations               | Formatting + clinical interpretation         | 18, 19          |

---

# D. CORE ENTITY / DATA MODEL

```
── Molecular ──
Drug                    # Root entity: SMILES + all derived properties
Metabolite              # Child of Drug with formation pathway
Formulation             # Release characteristics (IR, ER, solution)

── Administration ──
DoseEvent               # Drug + dose + route + timing
DoseSchedule            # Ordered list of DoseEvents (multi-dose regimen)

── Patient ──
PatientProfile          # Weight, age, sex, height, BSA
OrganFunction           # GFR (CKD stage), Child-Pugh score, cardiac output
GeneticProfile          # CYP2D6 phenotype, CYP2C19 phenotype, UGT1A1 *28

── Physiology ──
OrganCompartment        # Volume, blood flow, tissue composition
PhysiologyModel         # Reference human (70kg male, 60kg female) + scaling

── Biochemistry ──
Enzyme                  # CYP3A4, CYP2D6, etc. — abundance, Km, Vmax, turnover rate
Transporter             # P-gp, OATP1B1, etc. — abundance, Km, IC50
Receptor                # Target + Kd + Emax + EC50

── Interaction ──
InhibitionEvent         # Drug → Enzyme, Ki, mechanism type
InductionEvent          # Drug → NuclearReceptor → Enzyme, Emax, EC50
TransporterInhibition   # Drug → Transporter, IC50

── Simulation ──
SimulationConfig        # Time span, step size, solver params, drugs, patient
SimulationState         # Full state vector at time t
ConcentrationTimeCurve  # C(t) for a drug in a compartment
PKMetrics               # AUC, Cmax, Tmax, t1/2, CL, Vd, F

── Output ──
DDIReport               # Perpetrator, victim, mechanism, AUC_ratio, severity
BioavailabilityReport   # fa, fg, fh decomposition with/without DDI
CardiacSafetyReport     # CiPA score, channel contributions
ValidationResult        # Predicted vs observed, fold-error, pass/fail
```

---

# E. PHASED IMPLEMENTATION ROADMAP

## Phase 0 — Scientific Foundation (2 weeks)

**What gets built:**

- Drug entity model with RDKit property computation
- Enzyme/transporter database (static JSON/YAML) for Big 5 CYPs + P-gp + OATP1B1
- Reference physiology model (70 kg adult, 14 compartments with volumes and blood flows)
- Literature parameter database: Km, Vmax, Ki for ~50 high-priority drugs

**What gets deferred:** Everything else.

**Enough for:** Nothing user-facing. Internal foundation.

## Phase 1 — Core PK Engine (4 weeks)

**What gets built:**

- Single-drug PBPK ODE system (14 compartments)
- IV route: direct plasma introduction → distribution → elimination
- Well-stirred hepatic clearance model
- Renal clearance (GFR × fu)
- Tissue partitioning via Rodgers-Rowland
- ODE solver integration (scipy BDF)
- Single-drug concentration-time curve output
- PK metric extraction (AUC, Cmax, Tmax, t1/2)

**What gets deferred:** Oral absorption, DDI, bioavailability, metabolites.

**Enough for:** Validate single-drug IV PK predictions against 20 reference drugs. Internal milestone.

## Phase 2 — Oral Absorption & Bioavailability (3 weeks)

**What gets built:**

- GI absorption module (3-segment: duodenum, jejunum, ileum)
- Dissolution engine (Noyes-Whitney for BCS Class II; instant for BCS I)
- Permeability prediction (ML-predicted Peff)
- Gut wall metabolism module (fg via Qgut model)
- Hepatic first-pass module (fh via well-stirred model)
- Bioavailability as emergent property: F = fa × fg × fh
- Oral dose administration pipeline

**What gets deferred:** Food effects, pH-dependent dissolution for all drugs, P-gp/CYP3A4 interplay.

**Enough for:** Validate oral PK predictions. Compare predicted F vs known F for 30 drugs. First publishable result if >70% within 2-fold.

## Phase 3 — DDI Mechanism Layer (4 weeks)

**What gets built:**

- Multi-drug simulation (2+ drugs in same ODE system)
- CYP competitive inhibition ([I]/Ki model)
- CYP mechanism-based inhibition (enzyme turnover ODE)
- CYP induction (Emax model + enzyme synthesis/degradation ODE)
- DDI AUC ratio prediction using fm rule
- DDI impact on bioavailability (fg and fh shift under inhibition)
- P-gp inhibition at gut wall
- OATP1B1 inhibition at liver
- DDI severity classification (contraindicated/major/moderate/minor)
- Multi-drug interaction resolver (3+ drugs competing)

**What gets deferred:** UGT DDIs, MAO DDIs, renal transporter DDIs, mixed induction+inhibition dynamics.

**Enough for:** **V1 LAUNCH.** Predict DDI outcomes for drug pairs. Validate against FDA Top 10 DDI pairs. This is the minimum viable scientific product.

**Required for credibility:** ≥60% of predicted DDI AUC ratios within 2-fold of observed clinical data.

## Phase 4 — PD Integration (3 weeks)

**What gets built:**

- Emax PD model for drugs with known parameters
- Receptor occupancy modeling for same-target drugs
- PD DDI classification (additive, synergistic, antagonistic)
- Time-to-effect and duration-of-effect curves
- Integration with PK: effect-site concentration drives PD

**What gets deferred:** Full receptor binding kinetics, downstream signaling, efficacy prediction.

**Enough for:** Predict clinical significance of DDIs (not just exposure change, but effect change).

## Phase 5 — Population & Patient Variability (3 weeks)

**What gets built:**

- CYP2D6 phenotype integration (PM/IM/EM/UM → enzyme activity multiplier)
- CYP2C19 phenotype integration
- Age scaling (pediatric: immature enzymes; elderly: reduced function)
- Renal impairment (GFR-based dose adjustment)
- Hepatic impairment (Child-Pugh-based CLint scaling)
- Monte Carlo population simulation (sample patient parameters)

**What gets deferred:** Pregnancy physiology, obesity scaling, ethnic-specific parameters.

## Phase 6 — Validation & Benchmarking (Ongoing)

**What gets built:**

- Automated benchmark suite: 50 single-drug PK + 30 DDI pairs + 10 bioavailability cases
- Comparison framework against Simcyp published results
- Confidence scoring per prediction
- Uncertainty quantification (parameter sensitivity analysis)

## Phase 7 — Platform & API (2 weeks)

**What gets built:**

- REST API wrapping simulation engine
- JSON input/output schema
- Batch simulation endpoint (screen N drug pairs)
- PDF report generation (existing NeoGen capability)
- Integration with existing NeoGen web frontend

---

# F. RECOMMENDED NEOGEN V1 ARCHITECTURE

## 1. Best First Architecture

**Deterministic whole-body PBPK with 14 compartments, coupled multi-drug ODE system, Michaelis-Menten enzyme kinetics with inhibition/induction dynamics.** This is the same architectural class as Simcyp and GastroPlus. It's FDA-accepted, scientifically defensible, and the standard for mechanistic DDI prediction.

Python + SciPy is sufficient for V1. No need for compiled languages until simulation speed becomes a bottleneck (>1000 simulations/day).

## 2. Best Level of Mechanistic Fidelity for V1

| Component                | Fidelity Level   | Rationale                              |
| ------------------------ | ---------------- | -------------------------------------- |
| Absorption (fa)          | Semi-mechanistic | ML-Peff + compartmental GI transit     |
| Gut wall metabolism (fg) | Mechanistic      | Qgut model — essential for oral DDI    |
| First-pass (fh)          | Mechanistic      | Well-stirred model — core of DDI       |
| Distribution             | Mechanistic      | Rodgers-Rowland Kp + PBPK compartments |
| CYP inhibition           | Mechanistic      | [I]/Ki — non-negotiable for DDI        |
| CYP induction            | Semi-mechanistic | Emax + enzyme turnover ODE             |
| Transporter DDI          | Semi-mechanistic | IC50-based for P-gp, OATP1B1           |
| Renal clearance          | Empirical        | GFR × fu × fe                          |
| PD                       | Empirical        | Emax model                             |
| Cardiac                  | Mechanistic      | CiPA (existing)                        |

## 3. Where Bioavailability Lives

**Bioavailability is NOT a stored parameter. It is computed at runtime as:**

```
F(t) = fa(t) × fg(t, enzyme_state, inhibitors) × fh(t, enzyme_state, inhibitors)
```

It lives at the intersection of Layers 3 (absorption), 5 (metabolism), and 7 (DDI resolution). Each sub-component (fa, fg, fh) is independently modifiable by co-administered drugs, patient physiology, and time-dependent enzyme states.

This is the **only correct architecture** for a DDI engine. Any system that stores F as a single number cannot model how one drug changes another's bioavailability.

## 4. How DDI Is Represented

DDI is **not a separate module** — it is an emergent property of the coupled ODE system. When Drug A and Drug B are both present:

- They share enzyme pools (competitive substrate/inhibitor)
- They share transporter pools (competitive substrate/inhibitor)
- Drug A's metabolic rate depends on Drug B's concentration, and vice versa
- This coupling is captured naturally in the ODE system

The DDI Resolver module computes the inhibition/induction factors that modify enzyme activity, but the actual DDI outcome emerges from solving the coupled system.

## 5. Build First vs Delay

### Build First (V1 — Required for Launch)

1. Drug entity from SMILES (RDKit + ML)
2. 14-compartment PBPK ODE system
3. Oral absorption with fa × fg × fh decomposition
4. CYP enzyme kinetics with inhibition (competitive + mechanism-based)
5. CYP induction with enzyme turnover
6. Multi-drug ODE coupling
7. P-gp and OATP1B1 transporter DDI
8. Validation against FDA DDI benchmark pairs
9. DDI severity classification
10. Report generation

### Delay (V2+)

- Metabolite tracking beyond primary metabolites
- UGT/Phase II DDIs
- Enterohepatic recirculation
- Food effects on absorption
- Population variability (Monte Carlo)
- Systems biology pathway integration
- Full receptor kinetics
- Sustained-release formulations

## 6. What Creates the Strongest Scientific Moat

Three things separate a toy from a defensible platform:

1. **Bioavailability as emergent property (fa × fg × fh):** No competitor in the structure-only prediction space does this. Simcyp requires measured parameters. NeoGen predicting all three components from SMILES alone and showing how DDIs shift each component is novel and publishable.

2. **CL/F contamination correction:** NeoGen already identified 4 cases where databases record apparent clearance as true clearance. Building the simulation on corrected data gives systematically better results than anyone training on contaminated data.

3. **Coupled multi-drug ODE with enzyme dynamics:** Not just static [I]/Ki calculation, but time-resolved enzyme pool states that capture mechanism-based inhibition onset, induction lag, and the interplay between them. This is what makes the rifampin effect, the ritonavir boosting strategy, and the grapefruit effect simulatable rather than lookup-table-able.

The moat is not in having a PBPK model (Simcyp has one). The moat is in **requiring only SMILES as input** while producing mechanistically grounded, DDI-aware, bioavailability-decomposed predictions that can be validated against clinical data.

---

**End of Architecture Document.**

_Total estimated implementation time for V1 (Phases 0-3): 13 weeks._
_Total estimated implementation time through V1 launch (including Phase 7 API): 15 weeks._
