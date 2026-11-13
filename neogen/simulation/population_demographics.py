"""
Population Demographics Engine
================================
Generates realistic population cohorts based on real demographic data.

Data sources (all public, open-source):
- US Census Bureau (2020 Census, ACS)
- WHO Global Health Observatory
- CDC BRFSS (Behavioral Risk Factor Surveillance System)
- Seattle/King County Public Health data

Each region has:
- Total population
- Age distribution (5-year bins)
- Sex ratio
- Disease prevalence rates
- CYP polymorphism frequencies by ethnicity
- Socioeconomic health risk factors
"""
from __future__ import annotations

import random
import math
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# REAL DEMOGRAPHIC DATA (US Census 2020 + CDC + WHO)
# ════════════════════════════════════════════════════════════════

REGIONS = {
    "seattle": {
        "name": "Seattle Metropolitan Area",
        "total_population": 4_018_762,
        "country": "USA",
        "age_distribution": {
            "0-4": 0.055, "5-14": 0.105, "15-24": 0.125, "25-34": 0.175,
            "35-44": 0.155, "45-54": 0.125, "55-64": 0.115, "65-74": 0.085,
            "75-84": 0.040, "85+": 0.020,
        },
        "sex_ratio": {"male": 0.505, "female": 0.495},
        "ethnicity": {"white": 0.59, "asian": 0.17, "hispanic": 0.10, "black": 0.07, "other": 0.07},
        "median_age": 37.1,
        "median_bmi": 27.2,
        "disease_prevalence": {
            "hypertension": 0.28, "diabetes_t2": 0.09, "obesity": 0.27,
            "ckd": 0.06, "heart_failure": 0.02, "copd": 0.04,
            "depression": 0.18, "anxiety": 0.21, "asthma": 0.09,
            "liver_disease": 0.03, "cancer": 0.05, "hiv": 0.004,
        },
        "smoking_rate": 0.10,
        "alcohol_regular": 0.55,
    },
    "washington_state": {
        "name": "Washington State",
        "total_population": 7_705_281,
        "country": "USA",
        "age_distribution": {
            "0-4": 0.058, "5-14": 0.115, "15-24": 0.130, "25-34": 0.155,
            "35-44": 0.140, "45-54": 0.125, "55-64": 0.120, "65-74": 0.090,
            "75-84": 0.045, "85+": 0.022,
        },
        "sex_ratio": {"male": 0.502, "female": 0.498},
        "ethnicity": {"white": 0.64, "asian": 0.10, "hispanic": 0.13, "black": 0.04, "other": 0.09},
        "median_age": 37.7,
        "median_bmi": 28.0,
        "disease_prevalence": {
            "hypertension": 0.30, "diabetes_t2": 0.10, "obesity": 0.30,
            "ckd": 0.07, "heart_failure": 0.025, "copd": 0.05,
            "depression": 0.19, "anxiety": 0.20, "asthma": 0.10,
            "liver_disease": 0.035, "cancer": 0.055, "hiv": 0.003,
        },
        "smoking_rate": 0.12,
        "alcohol_regular": 0.52,
    },
    "usa": {
        "name": "United States of America",
        "total_population": 331_449_281,
        "country": "USA",
        "age_distribution": {
            "0-4": 0.059, "5-14": 0.122, "15-24": 0.132, "25-34": 0.140,
            "35-44": 0.128, "45-54": 0.125, "55-64": 0.130, "65-74": 0.095,
            "75-84": 0.047, "85+": 0.022,
        },
        "sex_ratio": {"male": 0.494, "female": 0.506},
        "ethnicity": {"white": 0.58, "asian": 0.06, "hispanic": 0.19, "black": 0.12, "other": 0.05},
        "median_age": 38.4,
        "median_bmi": 29.0,
        "disease_prevalence": {
            "hypertension": 0.47, "diabetes_t2": 0.13, "obesity": 0.42,
            "ckd": 0.15, "heart_failure": 0.06, "copd": 0.06,
            "depression": 0.21, "anxiety": 0.19, "asthma": 0.08,
            "liver_disease": 0.045, "cancer": 0.06, "hiv": 0.004,
        },
        "smoking_rate": 0.14,
        "alcohol_regular": 0.55,
    },
    "india": {
        "name": "India",
        "total_population": 1_428_627_663,
        "country": "India",
        "age_distribution": {
            "0-4": 0.085, "5-14": 0.175, "15-24": 0.175, "25-34": 0.155,
            "35-44": 0.130, "45-54": 0.105, "55-64": 0.080, "65-74": 0.055,
            "75-84": 0.025, "85+": 0.008,
        },
        "sex_ratio": {"male": 0.512, "female": 0.488},
        "ethnicity": {"south_asian": 0.95, "other": 0.05},
        "median_age": 28.4,
        "median_bmi": 23.5,
        "disease_prevalence": {
            "hypertension": 0.30, "diabetes_t2": 0.12, "obesity": 0.15,
            "ckd": 0.17, "heart_failure": 0.03, "copd": 0.04,
            "depression": 0.15, "anxiety": 0.12, "asthma": 0.03,
            "liver_disease": 0.04, "cancer": 0.03, "hiv": 0.002,
            "tuberculosis": 0.003, "malaria_endemic": True,
        },
        "smoking_rate": 0.20,
        "alcohol_regular": 0.30,
    },
    "europe": {
        "name": "European Union (27)",
        "total_population": 447_700_000,
        "country": "EU",
        "age_distribution": {
            "0-4": 0.046, "5-14": 0.100, "15-24": 0.108, "25-34": 0.120,
            "35-44": 0.130, "45-54": 0.145, "55-64": 0.135, "65-74": 0.110,
            "75-84": 0.070, "85+": 0.036,
        },
        "sex_ratio": {"male": 0.487, "female": 0.513},
        "ethnicity": {"white_european": 0.90, "other": 0.10},
        "median_age": 44.4,
        "median_bmi": 26.5,
        "disease_prevalence": {
            "hypertension": 0.35, "diabetes_t2": 0.09, "obesity": 0.23,
            "ckd": 0.10, "heart_failure": 0.04, "copd": 0.07,
            "depression": 0.17, "anxiety": 0.14, "asthma": 0.06,
            "liver_disease": 0.05, "cancer": 0.06, "hiv": 0.002,
        },
        "smoking_rate": 0.22,
        "alcohol_regular": 0.60,
    },
    "elderly": {
        "name": "Elderly Population (65+)",
        "total_population": 55_000_000,
        "country": "USA",
        "age_distribution": {
            "0-4": 0, "5-14": 0, "15-24": 0, "25-34": 0,
            "35-44": 0, "45-54": 0, "55-64": 0, "65-74": 0.55,
            "75-84": 0.30, "85+": 0.15,
        },
        "sex_ratio": {"male": 0.44, "female": 0.56},
        "ethnicity": {"white": 0.76, "black": 0.09, "hispanic": 0.09, "asian": 0.04, "other": 0.02},
        "median_age": 75,
        "median_bmi": 28.5,
        "disease_prevalence": {
            "hypertension": 0.75, "diabetes_t2": 0.27, "obesity": 0.35,
            "ckd": 0.38, "heart_failure": 0.14, "copd": 0.12,
            "depression": 0.20, "anxiety": 0.15, "asthma": 0.08,
            "liver_disease": 0.07, "cancer": 0.15, "hiv": 0.001,
            "polypharmacy": 0.40, "frailty": 0.15,
        },
        "smoking_rate": 0.08,
        "alcohol_regular": 0.40,
    },
    "cardiac_risk": {
        "name": "High Cardiac Risk Population",
        "total_population": 30_000_000,
        "country": "USA",
        "age_distribution": {
            "0-4": 0, "5-14": 0, "15-24": 0.01, "25-34": 0.03,
            "35-44": 0.08, "45-54": 0.20, "55-64": 0.30, "65-74": 0.25,
            "75-84": 0.10, "85+": 0.03,
        },
        "sex_ratio": {"male": 0.58, "female": 0.42},
        "ethnicity": {"white": 0.55, "black": 0.18, "hispanic": 0.15, "asian": 0.07, "other": 0.05},
        "median_age": 60,
        "median_bmi": 31.0,
        "disease_prevalence": {
            "hypertension": 0.85, "diabetes_t2": 0.40, "obesity": 0.55,
            "ckd": 0.30, "heart_failure": 0.25, "copd": 0.15,
            "atrial_fibrillation": 0.20, "prior_mi": 0.15,
            "hyperlipidemia": 0.70, "peripheral_vascular": 0.10,
        },
        "smoking_rate": 0.25,
        "alcohol_regular": 0.45,
    },
}

# CYP polymorphism frequencies by ethnicity (from PharmGKB + CPIC)
CYP_POLYMORPHISM_RATES = {
    "CYP2D6": {
        "white":      {"PM": 0.07, "IM": 0.10, "EM": 0.75, "UM": 0.08},
        "black":      {"PM": 0.02, "IM": 0.05, "EM": 0.63, "UM": 0.30},
        "asian":      {"PM": 0.01, "IM": 0.50, "EM": 0.48, "UM": 0.01},
        "hispanic":   {"PM": 0.03, "IM": 0.08, "EM": 0.79, "UM": 0.10},
        "south_asian": {"PM": 0.02, "IM": 0.35, "EM": 0.60, "UM": 0.03},
        "default":    {"PM": 0.05, "IM": 0.15, "EM": 0.72, "UM": 0.08},
    },
    "CYP2C19": {
        "white":      {"PM": 0.02, "IM": 0.25, "EM": 0.63, "RM": 0.10},
        "black":      {"PM": 0.04, "IM": 0.18, "EM": 0.58, "RM": 0.20},
        "asian":      {"PM": 0.15, "IM": 0.40, "EM": 0.43, "RM": 0.02},
        "hispanic":   {"PM": 0.02, "IM": 0.15, "EM": 0.68, "RM": 0.15},
        "south_asian": {"PM": 0.12, "IM": 0.35, "EM": 0.48, "RM": 0.05},
        "default":    {"PM": 0.05, "IM": 0.25, "EM": 0.58, "RM": 0.12},
    },
    "CYP2C9": {
        "white":      {"PM": 0.01, "IM": 0.15, "EM": 0.84},
        "black":      {"PM": 0.005, "IM": 0.05, "EM": 0.945},
        "asian":      {"PM": 0.005, "IM": 0.05, "EM": 0.945},
        "hispanic":   {"PM": 0.01, "IM": 0.10, "EM": 0.89},
        "default":    {"PM": 0.008, "IM": 0.10, "EM": 0.892},
    },
    "CYP3A4": {
        "default":    {"PM": 0.00, "IM": 0.05, "EM": 0.90, "UM": 0.05},
    },
}


# ════════════════════════════════════════════════════════════════
# COHORT GENERATION
# ════════════════════════════════════════════════════════════════

@dataclass
class PatientProfile:
    """Individual patient in a population cohort."""
    id: int = 0
    age: int = 40
    sex: str = "male"
    weight_kg: float = 70.0
    height_cm: float = 170.0
    bmi: float = 24.2
    ethnicity: str = "white"
    diseases: list = field(default_factory=list)
    cyp2d6: str = "EM"
    cyp2c19: str = "EM"
    cyp2c9: str = "EM"
    cyp3a4: str = "EM"
    smoker: bool = False
    alcohol: bool = False
    egfr: float = 90.0
    albumin: float = 40.0
    alt: float = 25.0
    risk_score: float = 0.0  # 0-100
    cluster: str = ""  # Assigned after simulation


@dataclass
class PopulationCohort:
    """A population cohort with demographic composition."""
    region: str
    region_name: str
    requested_size: int
    patients: list = field(default_factory=list)
    age_distribution: dict = field(default_factory=dict)
    sex_distribution: dict = field(default_factory=dict)
    disease_prevalence: dict = field(default_factory=dict)
    ethnicity_distribution: dict = field(default_factory=dict)
    cyp_distribution: dict = field(default_factory=dict)

    # Summary stats (computed after generation)
    mean_age: float = 0
    mean_bmi: float = 0
    pct_male: float = 0
    n_diseases_avg: float = 0


def generate_cohort(
    region: str = "usa",
    cohort_size: int = 100,
    seed: int = 42,
) -> PopulationCohort:
    """
    Generate a realistic population cohort based on regional demographics.

    Uses real demographic data to create a diverse patient population
    with age, sex, disease, and genetic distributions matching the region.
    """
    random.seed(seed)

    region_data = REGIONS.get(region, REGIONS["usa"])

    cohort = PopulationCohort(
        region=region,
        region_name=region_data["name"],
        requested_size=cohort_size,
        age_distribution=region_data["age_distribution"],
        sex_distribution=region_data["sex_ratio"],
        disease_prevalence=region_data.get("disease_prevalence", {}),
        ethnicity_distribution=region_data.get("ethnicity", {}),
    )

    patients = []
    age_bins = list(region_data["age_distribution"].items())
    ethnicities = list(region_data.get("ethnicity", {"default": 1.0}).items())
    diseases = region_data.get("disease_prevalence", {})

    for i in range(cohort_size):
        # Age
        age_bin = _weighted_choice(age_bins)
        age = _age_from_bin(age_bin)

        # Sex
        sex = "male" if random.random() < region_data["sex_ratio"]["male"] else "female"

        # Ethnicity
        ethnicity = _weighted_choice(ethnicities)

        # BMI
        base_bmi = region_data.get("median_bmi", 27)
        bmi = max(16, min(50, random.gauss(base_bmi, 5)))

        # Weight/Height from BMI
        height = random.gauss(175 if sex == "male" else 163, 8)
        weight = bmi * (height / 100) ** 2

        # Diseases (probability-based)
        patient_diseases = []
        for disease, prevalence in diseases.items():
            if isinstance(prevalence, bool):
                continue
            # Adjust prevalence by age
            age_factor = 1.0
            if age > 65:
                age_factor = 1.5
            elif age > 50:
                age_factor = 1.2
            elif age < 18:
                age_factor = 0.3

            if random.random() < prevalence * age_factor:
                patient_diseases.append(disease)

        # CYP genotypes
        eth_key = ethnicity if ethnicity in CYP_POLYMORPHISM_RATES.get("CYP2D6", {}) else "default"
        cyp2d6 = _weighted_choice(list(CYP_POLYMORPHISM_RATES["CYP2D6"].get(eth_key, CYP_POLYMORPHISM_RATES["CYP2D6"]["default"]).items()))
        cyp2c19 = _weighted_choice(list(CYP_POLYMORPHISM_RATES["CYP2C19"].get(eth_key, CYP_POLYMORPHISM_RATES["CYP2C19"]["default"]).items()))
        cyp2c9 = _weighted_choice(list(CYP_POLYMORPHISM_RATES["CYP2C9"].get(eth_key, CYP_POLYMORPHISM_RATES["CYP2C9"]["default"]).items()))
        cyp3a4 = _weighted_choice(list(CYP_POLYMORPHISM_RATES["CYP3A4"]["default"].items()))

        # Smoking/Alcohol
        smoker = random.random() < region_data.get("smoking_rate", 0.15)
        alcohol = random.random() < region_data.get("alcohol_regular", 0.50)

        # eGFR (declines with age)
        egfr = max(15, min(120, 120 - (age - 30) * 0.8 + random.gauss(0, 10)))
        if "ckd" in patient_diseases:
            egfr = max(15, egfr * 0.5)

        # Albumin
        albumin = max(20, min(50, random.gauss(40, 3)))
        if age > 75:
            albumin -= 5
        if "liver_disease" in patient_diseases:
            albumin -= 10

        # ALT
        alt = max(10, random.gauss(25, 10))
        if "liver_disease" in patient_diseases:
            alt *= 3

        # Risk score (composite)
        risk = 0
        risk += len(patient_diseases) * 10
        risk += max(0, (age - 50) * 0.5)
        risk += max(0, (bmi - 25) * 2)
        if smoker: risk += 10
        if egfr < 60: risk += 15
        if cyp2d6 == "PM": risk += 5
        risk = min(100, max(0, risk))

        patient = PatientProfile(
            id=i + 1,
            age=age,
            sex=sex,
            weight_kg=round(weight, 1),
            height_cm=round(height, 1),
            bmi=round(bmi, 1),
            ethnicity=ethnicity,
            diseases=patient_diseases,
            cyp2d6=cyp2d6,
            cyp2c19=cyp2c19,
            cyp2c9=cyp2c9,
            cyp3a4=cyp3a4,
            smoker=smoker,
            alcohol=alcohol,
            egfr=round(egfr, 1),
            albumin=round(albumin, 1),
            alt=round(alt, 1),
            risk_score=round(risk, 1),
        )
        patients.append(patient)

    cohort.patients = patients

    # Compute summary stats
    if patients:
        cohort.mean_age = round(sum(p.age for p in patients) / len(patients), 1)
        cohort.mean_bmi = round(sum(p.bmi for p in patients) / len(patients), 1)
        cohort.pct_male = round(100 * sum(1 for p in patients if p.sex == "male") / len(patients), 1)
        cohort.n_diseases_avg = round(sum(len(p.diseases) for p in patients) / len(patients), 1)

    # CYP distribution summary
    cyp_counts = {"CYP2D6": {}, "CYP2C19": {}}
    for p in patients:
        cyp_counts["CYP2D6"][p.cyp2d6] = cyp_counts["CYP2D6"].get(p.cyp2d6, 0) + 1
        cyp_counts["CYP2C19"][p.cyp2c19] = cyp_counts["CYP2C19"].get(p.cyp2c19, 0) + 1
    cohort.cyp_distribution = {
        k: {geno: round(100 * count / len(patients), 1) for geno, count in v.items()}
        for k, v in cyp_counts.items()
    }

    return cohort


def _weighted_choice(items: list) -> str:
    """Weighted random selection from (value, weight) pairs."""
    total = sum(w for _, w in items)
    r = random.random() * total
    cumulative = 0
    for value, weight in items:
        cumulative += weight
        if r <= cumulative:
            return value
    return items[-1][0]


def _age_from_bin(bin_label: str) -> int:
    """Convert age bin like '25-34' to a random age in that range."""
    if bin_label.endswith('+'):
        low = int(bin_label.replace('+', ''))
        return random.randint(low, low + 15)
    parts = bin_label.split('-')
    low, high = int(parts[0]), int(parts[1])
    return random.randint(low, high)


def get_available_regions() -> list[dict]:
    """Return list of available regions for the UI."""
    return [
        {
            "id": k,
            "name": v["name"],
            "population": v["total_population"],
            "country": v["country"],
            "median_age": v.get("median_age", 0),
        }
        for k, v in REGIONS.items()
    ]


def get_region_summary(region: str) -> dict:
    """Get demographic summary for a region."""
    data = REGIONS.get(region, REGIONS["usa"])
    return {
        "name": data["name"],
        "total_population": data["total_population"],
        "median_age": data.get("median_age", 0),
        "median_bmi": data.get("median_bmi", 27),
        "sex_ratio": data["sex_ratio"],
        "top_diseases": sorted(
            [(k, v) for k, v in data.get("disease_prevalence", {}).items() if isinstance(v, (int, float))],
            key=lambda x: -x[1]
        )[:8],
        "smoking_rate": data.get("smoking_rate", 0),
        "alcohol_rate": data.get("alcohol_regular", 0),
    }
