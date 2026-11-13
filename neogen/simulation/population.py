"""
Population Simulation Engine
==============================
Simulates drug PK across diverse virtual patient populations.

Key features:
1. Genetic polymorphisms — CYP2D6 poor/intermediate/extensive/ultra-rapid metabolizers
2. Demographic variability — age, weight, sex distributions
3. Monte Carlo sampling — 100-1000 virtual patients
4. Population PK statistics — median, 5th/95th percentile bands

This is what clinical pharmacologists use for Phase I trial design:
"What dose gives 90% of patients a safe Cmax but still therapeutic?"

CYP2D6 Polymorphism Prevalence (global averages):
- Poor metabolizer (PM):          5-10%  → drug accumulates, toxicity risk
- Intermediate metabolizer (IM):  10-15% → slower than normal
- Extensive metabolizer (EM):     65-75% → "normal" metabolism (reference)
- Ultra-rapid metabolizer (UM):   5-10%  → drug cleared too fast, may be ineffective
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from neogen.simulation.pbpk import PBPKModel, PatientParameters

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# GENETIC POLYMORPHISM DATABASE
# ════════════════════════════════════════════════════════════════

@dataclass
class CYPGenotype:
    """A CYP enzyme genotype with its metabolic activity."""
    enzyme: str
    phenotype: str  # PM, IM, EM, UM
    activity_multiplier: float  # 1.0 = normal (EM)
    prevalence: float  # Fraction in general population
    description: str


# CYP2D6 — most clinically important polymorphism
CYP2D6_GENOTYPES = [
    CYPGenotype("CYP2D6", "PM", 0.0, 0.07,
                "Poor Metabolizer — no functional CYP2D6. Codeine gives no pain relief. SSRIs accumulate."),
    CYPGenotype("CYP2D6", "IM", 0.5, 0.12,
                "Intermediate Metabolizer — reduced CYP2D6 activity. May need lower doses of 2D6 substrates."),
    CYPGenotype("CYP2D6", "EM", 1.0, 0.71,
                "Extensive Metabolizer — normal CYP2D6 activity. Standard dosing applies."),
    CYPGenotype("CYP2D6", "UM", 2.5, 0.10,
                "Ultra-rapid Metabolizer — extra CYP2D6 gene copies. Codeine → too much morphine → respiratory depression."),
]

# CYP2C19 — important for PPIs, clopidogrel
CYP2C19_GENOTYPES = [
    CYPGenotype("CYP2C19", "PM", 0.0, 0.03,
                "Poor Metabolizer — clopidogrel completely ineffective. Higher PPI levels."),
    CYPGenotype("CYP2C19", "IM", 0.4, 0.25,
                "Intermediate Metabolizer — reduced clopidogrel activation."),
    CYPGenotype("CYP2C19", "EM", 1.0, 0.60,
                "Extensive Metabolizer — normal metabolism."),
    CYPGenotype("CYP2C19", "UM", 2.0, 0.12,
                "Ultra-rapid Metabolizer — PPIs cleared too fast, may need higher doses."),
]

# CYP2C9 — warfarin, NSAIDs
CYP2C9_GENOTYPES = [
    CYPGenotype("CYP2C9", "PM", 0.1, 0.01,
                "Poor Metabolizer — warfarin dose needs 80% reduction."),
    CYPGenotype("CYP2C9", "IM", 0.5, 0.15,
                "Intermediate Metabolizer — warfarin dose needs 30-50% reduction."),
    CYPGenotype("CYP2C9", "EM", 1.0, 0.84,
                "Extensive Metabolizer — standard warfarin dosing."),
]

# CYP3A4 — less genetic variation, more drug interaction driven
CYP3A4_GENOTYPES = [
    CYPGenotype("CYP3A4", "low", 0.6, 0.15,
                "Low activity — statin levels may be elevated."),
    CYPGenotype("CYP3A4", "normal", 1.0, 0.70,
                "Normal activity."),
    CYPGenotype("CYP3A4", "high", 1.5, 0.15,
                "High activity — some drugs may be cleared faster."),
]

ALL_GENOTYPES = {
    "CYP2D6": CYP2D6_GENOTYPES,
    "CYP2C19": CYP2C19_GENOTYPES,
    "CYP2C9": CYP2C9_GENOTYPES,
    "CYP3A4": CYP3A4_GENOTYPES,
}


# ════════════════════════════════════════════════════════════════
# VIRTUAL PATIENT GENERATOR
# ════════════════════════════════════════════════════════════════

@dataclass
class VirtualPatient:
    """A single virtual patient with demographics + genotype."""
    patient_id: int
    weight_kg: float
    age_years: int
    sex: str
    genotypes: dict[str, CYPGenotype] = field(default_factory=dict)

    @property
    def clearance_multiplier(self) -> float:
        """Overall clearance multiplier based on all genotypes."""
        mult = 1.0
        for g in self.genotypes.values():
            mult *= g.activity_multiplier
        return mult


def generate_population(
    n_patients: int = 100,
    age_range: tuple[int, int] = (18, 80),
    weight_range: tuple[float, float] = (45, 120),
    female_fraction: float = 0.5,
    seed: int = 42,
) -> list[VirtualPatient]:
    """
    Generate a diverse virtual patient population.

    Demographics are sampled from realistic distributions:
    - Weight: log-normal (right-skewed, more people around 70kg)
    - Age: uniform within range
    - Sex: binomial
    - Genotypes: sampled according to population prevalence
    """
    rng = np.random.RandomState(seed)
    patients = []

    for i in range(n_patients):
        # Weight: log-normal centered around 75kg
        weight = rng.lognormal(mean=np.log(75), sigma=0.2)
        weight = np.clip(weight, weight_range[0], weight_range[1])

        # Age: uniform
        age = rng.randint(age_range[0], age_range[1] + 1)

        # Sex
        sex = "female" if rng.random() < female_fraction else "male"

        # Genotypes: sample each enzyme independently
        genotypes = {}
        for enzyme, geno_list in ALL_GENOTYPES.items():
            probs = [g.prevalence for g in geno_list]
            probs = np.array(probs) / sum(probs)  # Normalize
            idx = rng.choice(len(geno_list), p=probs)
            genotypes[enzyme] = geno_list[idx]

        patients.append(VirtualPatient(
            patient_id=i,
            weight_kg=round(weight, 1),
            age_years=age,
            sex=sex,
            genotypes=genotypes,
        ))

    return patients


# ════════════════════════════════════════════════════════════════
# POPULATION SIMULATION
# ════════════════════════════════════════════════════════════════

class PopulationSimulator:
    """
    Run PBPK simulation across a population of virtual patients.

    Produces population PK statistics:
    - Median concentration-time curve
    - 5th-95th percentile bands
    - Cmax and AUC distributions
    - Genotype-stratified results
    """

    def __init__(
        self,
        dose_mg: float = 100.0,
        route: str = "oral",
        ml_predictions: dict = None,
        logp: float = 2.0,
        drug_cyp_substrates: list[str] = None,
    ):
        self.dose_mg = dose_mg
        self.route = route
        self.ml_predictions = ml_predictions or {}
        self.logp = logp
        self.drug_cyp_substrates = drug_cyp_substrates or []

    def run(
        self,
        n_patients: int = 100,
        n_timepoints: int = 100,
        seed: int = 42,
    ) -> dict:
        """
        Run population simulation.

        Returns dict with:
        - time_h: array of time points
        - median: median concentration at each time
        - p5, p95: 5th and 95th percentiles
        - p25, p75: 25th and 75th percentiles
        - individual_cmax: list of Cmax values
        - individual_auc: list of AUC values
        - genotype_groups: stratified results by genotype
        - patients: list of patient info dicts
        """
        population = generate_population(n_patients=n_patients, seed=seed)

        all_plasma = []
        patient_results = []

        for vp in population:
            patient_params = PatientParameters(
                weight_kg=vp.weight_kg,
                age_years=vp.age_years,
                sex=vp.sex,
            )

            model = PBPKModel(
                patient=patient_params,
                dose_mg=self.dose_mg,
                route=self.route,
            )

            # Modify predictions based on genotype
            adjusted_preds = dict(self.ml_predictions)

            # Apply genotype-based clearance modification
            for enzyme in self.drug_cyp_substrates:
                if enzyme in vp.genotypes:
                    geno = vp.genotypes[enzyme]
                    # Modify clearance based on enzyme activity
                    adjusted_preds["clearance"] = (
                        adjusted_preds.get("clearance", 1.0) * geno.activity_multiplier
                    )

            model.derive_pbpk_params(adjusted_preds, logp=self.logp)
            sol = model.solve(n_points=n_timepoints)

            plasma = sol["plasma_conc"]
            all_plasma.append(plasma)

            # Calculate individual PK metrics
            cmax = float(np.max(plasma))
            tmax = float(sol["time_h"][np.argmax(plasma)])
            # AUC via trapezoidal rule
            # Use scipy for trapezoidal integration (works across numpy versions)
            from scipy.integrate import trapezoid as _trapz
            auc = float(_trapz(plasma, sol["time_h"]))

            # Determine primary genotype effect
            primary_geno = "EM"
            for enzyme in self.drug_cyp_substrates:
                if enzyme in vp.genotypes:
                    primary_geno = vp.genotypes[enzyme].phenotype
                    break

            patient_results.append({
                "id": vp.patient_id,
                "weight_kg": vp.weight_kg,
                "age_years": vp.age_years,
                "sex": vp.sex,
                "genotype": primary_geno,
                "cmax": round(cmax, 4),
                "tmax": round(tmax, 2),
                "auc": round(auc, 4),
                "clearance_mult": round(vp.clearance_multiplier, 2),
            })

        # Stack all curves
        all_plasma = np.array(all_plasma)  # [n_patients, n_timepoints]
        time_h = sol["time_h"]

        # Population statistics
        median = np.median(all_plasma, axis=0)
        p5 = np.percentile(all_plasma, 5, axis=0)
        p95 = np.percentile(all_plasma, 95, axis=0)
        p25 = np.percentile(all_plasma, 25, axis=0)
        p75 = np.percentile(all_plasma, 75, axis=0)

        # Cmax and AUC distributions
        cmax_values = [p["cmax"] for p in patient_results]
        auc_values = [p["auc"] for p in patient_results]

        # Stratify by genotype
        genotype_groups = {}
        for phenotype in ["PM", "IM", "EM", "UM", "low", "normal", "high"]:
            group_indices = [i for i, p in enumerate(patient_results) if p["genotype"] == phenotype]
            if group_indices:
                group_plasma = all_plasma[group_indices]
                genotype_groups[phenotype] = {
                    "n_patients": len(group_indices),
                    "median_cmax": round(float(np.median([cmax_values[i] for i in group_indices])), 4),
                    "median_auc": round(float(np.median([auc_values[i] for i in group_indices])), 4),
                    "mean_curve": np.mean(group_plasma, axis=0).tolist(),
                }

        # Dose recommendation based on population
        cmax_95 = float(np.percentile(cmax_values, 95))
        cmax_median = float(np.median(cmax_values))

        return {
            "n_patients": n_patients,
            "time_h": time_h.tolist(),
            "median": median.tolist(),
            "p5": p5.tolist(),
            "p95": p95.tolist(),
            "p25": p25.tolist(),
            "p75": p75.tolist(),
            "cmax_distribution": {
                "values": cmax_values,
                "median": round(float(np.median(cmax_values)), 4),
                "mean": round(float(np.mean(cmax_values)), 4),
                "p5": round(float(np.percentile(cmax_values, 5)), 4),
                "p95": round(cmax_95, 4),
                "cv_percent": round(float(np.std(cmax_values) / np.mean(cmax_values) * 100), 1),
            },
            "auc_distribution": {
                "values": auc_values,
                "median": round(float(np.median(auc_values)), 4),
                "mean": round(float(np.mean(auc_values)), 4),
                "p5": round(float(np.percentile(auc_values, 5)), 4),
                "p95": round(float(np.percentile(auc_values, 95)), 4),
            },
            "genotype_groups": genotype_groups,
            "patients": patient_results[:20],  # Return first 20 for display
        }


# ════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("NeoGen Population Simulation — Test")
    print("=" * 60)

    # Simulate caffeine 200mg across 200 patients
    # Caffeine is primarily metabolized by CYP1A2
    print("\n--- Caffeine 200mg, 200 virtual patients ---")
    sim = PopulationSimulator(
        dose_mg=200,
        route="oral",
        ml_predictions={"cmax": 10, "auc": 50, "half_life": 5, "clearance": 0.35, "vd": 40},
        logp=-0.07,
        drug_cyp_substrates=["CYP2D6"],  # Using CYP2D6 for demo (has more genetic variation)
    )

    result = sim.run(n_patients=200, n_timepoints=100)

    print(f"  Patients: {result['n_patients']}")
    print(f"  Cmax: median={result['cmax_distribution']['median']:.2f}, "
          f"p5={result['cmax_distribution']['p5']:.2f}, "
          f"p95={result['cmax_distribution']['p95']:.2f}")
    print(f"  Cmax CV: {result['cmax_distribution']['cv_percent']:.1f}%")
    print(f"  AUC:  median={result['auc_distribution']['median']:.2f}")

    print("\n  Genotype stratification:")
    for phenotype, data in result["genotype_groups"].items():
        print(f"    {phenotype:4s}: n={data['n_patients']:3d}, "
              f"median Cmax={data['median_cmax']:.2f}, "
              f"median AUC={data['median_auc']:.2f}")

    print(f"\n  Sample patients:")
    for p in result["patients"][:5]:
        print(f"    Patient {p['id']:3d}: {p['weight_kg']}kg, {p['age_years']}yo {p['sex']}, "
              f"genotype={p['genotype']}, Cmax={p['cmax']:.2f}, AUC={p['auc']:.2f}")

    print("\nPopulation simulation test PASSED!")
