"""
Swarm Engine — Orchestrates Population-Level Patient Simulation
================================================================
Manages N patient agents, runs temporal simulation, aggregates results.

Architecture:
1. Generate N patient profiles (diverse population)
2. For each round (hour/day):
   a. Each patient decides: take dose, skip, etc.
   b. PBPK runs for each patient's actual drug exposure
   c. Side effects evaluated based on organ concentrations
   d. Social interactions between patients
3. Aggregate population-level results
4. Generate analytics (adherence, efficacy, safety, subgroups)
"""
from __future__ import annotations

import logging
import random
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from neogen.swarm.patient_agent import (
    PatientAgent, PatientProfile, PatientMemory,
    AdherenceProfile, HealthLiteracy, AnxietyLevel,
    MedicationAction,
)
from neogen.simulation.opus_predictor import OpusPKPredictor

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# POPULATION GENERATOR
# ════════════════════════════════════════════════════════════════

class PopulationGenerator:
    """Generate diverse patient populations based on demographics."""

    # CYP2D6 phenotype frequencies by ethnicity
    CYP2D6_FREQ = {
        "caucasian": {"EM": 0.70, "IM": 0.10, "PM": 0.07, "UM": 0.13},
        "african_american": {"EM": 0.65, "IM": 0.10, "PM": 0.03, "UM": 0.22},
        "asian": {"EM": 0.75, "IM": 0.15, "PM": 0.01, "UM": 0.09},
        "hispanic": {"EM": 0.72, "IM": 0.10, "PM": 0.05, "UM": 0.13},
    }

    CYP2C19_FREQ = {
        "caucasian": {"EM": 0.74, "IM": 0.24, "PM": 0.02},
        "african_american": {"EM": 0.78, "IM": 0.18, "PM": 0.04},
        "asian": {"EM": 0.50, "IM": 0.35, "PM": 0.15},
        "hispanic": {"EM": 0.70, "IM": 0.25, "PM": 0.05},
    }

    @staticmethod
    def generate_population(
        n_patients: int = 100,
        age_range: tuple = (18, 85),
        sex_ratio: float = 0.5,  # fraction male
        disease_prevalence: dict = None,
        ethnicity_distribution: dict = None,
        adherence_distribution: dict = None,
        seed: int = 42,
    ) -> list[PatientProfile]:
        """Generate a diverse population of patient profiles."""
        random.seed(seed)
        profiles = []

        if ethnicity_distribution is None:
            ethnicity_distribution = {"caucasian": 0.60, "african_american": 0.15,
                                       "asian": 0.10, "hispanic": 0.15}

        if adherence_distribution is None:
            adherence_distribution = {
                AdherenceProfile.PERFECT: 0.10,
                AdherenceProfile.GOOD: 0.40,
                AdherenceProfile.MODERATE: 0.30,
                AdherenceProfile.POOR: 0.15,
                AdherenceProfile.NON_ADHERENT: 0.05,
            }

        if disease_prevalence is None:
            disease_prevalence = {
                "hypertension": 0.30,
                "type2_diabetes": 0.15,
                "ckd": 0.10,
                "obesity": 0.25,
                "depression": 0.10,
                "copd": 0.05,
                "heart_failure": 0.03,
                "liver_disease": 0.03,
            }

        ethnicities = list(ethnicity_distribution.keys())
        eth_weights = list(ethnicity_distribution.values())

        adh_profiles = list(adherence_distribution.keys())
        adh_weights = list(adherence_distribution.values())

        for i in range(n_patients):
            age = random.randint(age_range[0], age_range[1])
            sex = "male" if random.random() < sex_ratio else "female"
            ethnicity = random.choices(ethnicities, weights=eth_weights, k=1)[0]

            # Weight based on age, sex, ethnicity
            if sex == "male":
                weight = random.gauss(82, 15)
            else:
                weight = random.gauss(70, 14)
            weight = max(40, min(weight, 150))

            # Height
            if sex == "male":
                height = random.gauss(176, 7)
            else:
                height = random.gauss(163, 6)

            # Genetics based on ethnicity
            cyp2d6_dist = PopulationGenerator.CYP2D6_FREQ.get(ethnicity, PopulationGenerator.CYP2D6_FREQ["caucasian"])
            cyp2d6 = random.choices(list(cyp2d6_dist.keys()), weights=list(cyp2d6_dist.values()), k=1)[0]

            cyp2c19_dist = PopulationGenerator.CYP2C19_FREQ.get(ethnicity, PopulationGenerator.CYP2C19_FREQ["caucasian"])
            cyp2c19 = random.choices(list(cyp2c19_dist.keys()), weights=list(cyp2c19_dist.values()), k=1)[0]

            # Diseases (age-adjusted prevalence)
            diseases = []
            for disease, base_prev in disease_prevalence.items():
                age_factor = 1.0 + max(0, (age - 40)) * 0.02  # Increases with age
                if random.random() < base_prev * age_factor:
                    diseases.append(disease)

            # eGFR based on age and CKD status
            egfr = max(15, 120 - (age - 25) * 0.8 + random.gauss(0, 10))
            if "ckd" in diseases:
                egfr = max(15, egfr * random.uniform(0.3, 0.7))

            # ALT based on liver disease
            alt = max(10, random.gauss(25, 8))
            if "liver_disease" in diseases:
                alt = random.gauss(80, 30)

            # Behavioral
            adherence = random.choices(adh_profiles, weights=adh_weights, k=1)[0]
            # Elderly and complex patients tend to have worse adherence
            if age > 75 and len(diseases) > 2:
                if random.random() < 0.3:
                    adherence = AdherenceProfile.POOR

            anxiety = random.choices(
                [AnxietyLevel.NONE, AnxietyLevel.MILD, AnxietyLevel.MODERATE, AnxietyLevel.SEVERE],
                weights=[0.30, 0.40, 0.20, 0.10], k=1
            )[0]

            health_lit = random.choices(
                [HealthLiteracy.HIGH, HealthLiteracy.MODERATE, HealthLiteracy.LOW],
                weights=[0.25, 0.50, 0.25], k=1
            )[0]

            profile = PatientProfile(
                patient_id=f"P{i+1:04d}",
                age=age,
                sex=sex,
                weight_kg=round(weight, 1),
                height_cm=round(height, 1),
                ethnicity=ethnicity,
                cyp2d6=cyp2d6,
                cyp2c19=cyp2c19,
                cyp2c9=random.choices(["EM", "IM", "PM"], weights=[0.85, 0.12, 0.03], k=1)[0],
                diseases=diseases,
                egfr=round(egfr, 1),
                alt=round(alt, 1),
                albumin=round(random.gauss(4.0, 0.3), 1),
                creatinine=round(max(0.5, 120 / max(egfr, 15) * 0.8), 2),
                adherence_profile=adherence,
                health_literacy=health_lit,
                anxiety_level=anxiety,
                smoker=random.random() < 0.15,
                alcohol_use=random.choices(["none", "occasional", "moderate", "heavy"],
                                           weights=[0.40, 0.35, 0.20, 0.05], k=1)[0],
                has_family_support=random.random() < 0.75,
                has_regular_doctor=random.random() < 0.85,
                meal_regularity=random.choices(["regular", "irregular", "skips_meals"],
                                               weights=[0.60, 0.30, 0.10], k=1)[0],
                sleep_quality=random.choices(["good", "fair", "poor"],
                                            weights=[0.50, 0.35, 0.15], k=1)[0],
                work_schedule=random.choices(["regular", "shift_work", "irregular"],
                                           weights=[0.70, 0.15, 0.15], k=1)[0],
                peer_influence_susceptibility=round(random.uniform(0.2, 0.8), 2),
            )
            profiles.append(profile)

        return profiles


# ════════════════════════════════════════════════════════════════
# SWARM ENGINE
# ════════════════════════════════════════════════════════════════

class SwarmEngine:
    """
    Orchestrates population-level patient simulation.

    Runs N agents through temporal simulation, connecting each
    patient's medication behavior to NeoGen's PBPK engine.
    """

    def __init__(self, drug_name: str, drug_smiles: str = "",
                 dose_mg: float = 100, frequency: str = "BID",
                 n_patients: int = 100, n_days: int = 7,
                 use_llm: bool = False):
        self.drug_name = drug_name
        self.drug_smiles = drug_smiles
        self.dose_mg = dose_mg
        self.frequency = frequency
        self.n_patients = n_patients
        self.n_days = n_days
        self.use_llm = use_llm
        self.agents: list[PatientAgent] = []
        self.rounds_completed = 0
        self.round_log: list[dict] = []
        self.population_results: dict = {}

        # Initialize Opus predictor (cache-only, no API key)
        self.pk_predictor = OpusPKPredictor(api_key="")
        self.drug_pk: dict = {}  # Populated on first prediction

        # Doses per day based on frequency
        self.doses_per_day = {"QD": 1, "BID": 2, "TID": 3, "QID": 4}.get(frequency, 2)
        self.dose_times = {
            "QD": ["morning"],
            "BID": ["morning", "evening"],
            "TID": ["morning", "afternoon", "evening"],
            "QID": ["morning", "noon", "afternoon", "evening"],
        }.get(frequency, ["morning", "evening"])

    def initialize(self, profiles: list[PatientProfile] = None, seed: int = 42):
        """Initialize all patient agents."""
        if profiles is None:
            profiles = PopulationGenerator.generate_population(
                n_patients=self.n_patients, seed=seed
            )

        self.agents = []
        for profile in profiles[:self.n_patients]:
            agent = PatientAgent(
                profile=profile,
                drug_name=self.drug_name,
                dose_mg=self.dose_mg,
                frequency=self.frequency,
                use_llm=self.use_llm,
            )
            self.agents.append(agent)

        logger.info(f"Initialized {len(self.agents)} patient agents for {self.drug_name} {self.dose_mg}mg {self.frequency}")

    def run_simulation(self, callback=None) -> dict:
        """
        Run the full temporal simulation.

        For each day, for each dose time:
        1. Each patient decides whether to take the dose
        2. PBPK updates based on actual drug taken
        3. Side effects assessed
        4. Social interactions occur
        """
        logger.info(f"Starting {self.n_days}-day simulation with {len(self.agents)} patients")

        for day in range(1, self.n_days + 1):
            for time_of_day in self.dose_times:
                round_results = self._run_round(day, time_of_day)
                self.round_log.append(round_results)
                self.rounds_completed += 1

                if callback:
                    callback(round_results)

            # End of day: social interactions
            self._social_interaction_round(day)

            # End of day: assess side effects for patients who took doses
            self._assess_side_effects(day)

            logger.info(f"Day {day}/{self.n_days}: {self._quick_stats()}")

        # Compute final results
        self.population_results = self._compute_population_results()
        return self.population_results

    def _get_drug_pk(self, weight_kg: float = 70) -> dict:
        """Get PK parameters for the drug from Opus cache (no API calls)."""
        if not self.drug_pk:
            result = self.pk_predictor.predict(
                smiles=self.drug_smiles,
                drug_name=self.drug_name,
                dose_mg=self.dose_mg,
                route="oral",
                weight_kg=weight_kg,
            )
            self.drug_pk = result
            logger.info(
                f"Opus PK for {self.drug_name}: Cmax={result.get('cmax')}, "
                f"t1/2={result.get('half_life')}h, CL={result.get('clearance')} L/h, "
                f"Vd={result.get('vd')} L, source={result.get('source')}"
            )
        return self.drug_pk

    def _compute_dose_increment(self, agent, dose_multiplier: float = 1.0) -> float:
        """
        Compute plasma concentration increment from a dose using Opus PK data.

        Uses one-compartment PK model:
          C_increment = (F * Dose) / Vd
        Adjusted for patient weight relative to reference (70 kg).
        """
        pk = self._get_drug_pk(weight_kg=agent.profile.weight_kg)
        cmax = pk.get("cmax", 0)
        vd = pk.get("vd", 0)
        F = pk.get("F", 0.5)

        if cmax > 0:
            # Use Cmax directly as the expected peak increment for standard dose.
            # Scale linearly by dose_multiplier and inversely by patient weight.
            weight_factor = 70.0 / max(agent.profile.weight_kg, 40)
            increment = cmax * dose_multiplier * weight_factor
        elif vd > 0:
            # Fallback: one-compartment: delta_C = (F * Dose) / Vd
            effective_dose = self.dose_mg * dose_multiplier
            increment = (F * effective_dose) / vd
        else:
            # Last resort: simplified PK (original behavior)
            increment = self.dose_mg * dose_multiplier * 0.01

        # CYP2D6 phenotype adjustment (poor metabolizers accumulate more)
        cyp_factor = {"PM": 1.5, "IM": 1.2, "EM": 1.0, "UM": 0.7}.get(
            agent.profile.cyp2d6, 1.0
        )
        increment *= cyp_factor

        return increment

    def _run_round(self, day: int, time_of_day: str) -> dict:
        """Run one dosing round for all patients using Opus PK predictions."""
        actions = []
        took_dose = 0
        skipped = 0
        discontinued = 0

        # Ensure drug PK is loaded (uses cache only, no API calls)
        pk = self._get_drug_pk()
        half_life = pk.get("half_life", 0)
        hours_between = 24.0 / self.doses_per_day

        for agent in self.agents:
            agent.current_day = day
            agent.current_round = self.rounds_completed

            # Store PK predictions on the agent for downstream use
            if not agent.pk_predictions:
                agent.pk_predictions = {
                    "cmax": pk.get("cmax", 0),
                    "auc": pk.get("auc", 0),
                    "half_life": pk.get("half_life", 0),
                    "clearance": pk.get("clearance", 0),
                    "vd": pk.get("vd", 0),
                    "F": pk.get("F", 0.5),
                    "source": pk.get("source", "unknown"),
                }

            # Get peer messages from recent conversations
            peer_msgs = agent.memory.peer_conversations[-3:] if agent.memory.peer_conversations else []

            # Patient decides
            action = agent.decide_action(time_of_day, peer_msgs)
            agent.record_action(action)

            if action["action"] == MedicationAction.TAKE_DOSE.value:
                took_dose += 1
                agent.current_plasma_conc += self._compute_dose_increment(agent, dose_multiplier=1.0)
            elif action["action"] == MedicationAction.TAKE_LATE.value:
                took_dose += 1
                # Late dose: slightly lower peak due to shifted absorption
                agent.current_plasma_conc += self._compute_dose_increment(agent, dose_multiplier=0.85)
            elif action["action"] == MedicationAction.TAKE_DOUBLE_DOSE.value:
                took_dose += 1
                agent.current_plasma_conc += self._compute_dose_increment(agent, dose_multiplier=2.0)
            elif action["action"] == MedicationAction.SKIP_DOSE.value:
                skipped += 1
            elif action["action"] == MedicationAction.DISCONTINUE.value:
                discontinued += 1

            actions.append(action)

            # Natural drug elimination between rounds using half-life
            if half_life > 0:
                agent.current_plasma_conc *= 0.5 ** (hours_between / max(half_life, 0.5))
            else:
                agent.current_plasma_conc *= 0.85  # Default 15% elimination per round

        active = sum(1 for a in self.agents if a.active)

        return {
            "day": day,
            "time": time_of_day,
            "round": self.rounds_completed,
            "took_dose": took_dose,
            "skipped": skipped,
            "discontinued": discontinued,
            "active_patients": active,
            "adherence_this_round": round(took_dose / max(active, 1), 3),
            "actions": actions,
        }

    def _social_interaction_round(self, day: int):
        """Simulate social interactions between patients."""
        active_agents = [a for a in self.agents if a.active]
        if len(active_agents) < 2:
            return

        # 10% of patients talk to a random peer each day
        n_interactions = max(1, len(active_agents) // 10)

        for _ in range(n_interactions):
            sender = random.choice(active_agents)
            receiver = random.choice(active_agents)
            if sender.profile.patient_id == receiver.profile.patient_id:
                continue

            # Sender's message based on their experience
            if sender.memory.has_experienced_side_effects:
                sentiment = "negative"
                message = f"I experienced {sender.memory.most_recent_side_effect.get('effect', 'side effects')} from {self.drug_name}"
            elif sender.memory.adherence_rate > 0.8:
                sentiment = "positive"
                message = f"{self.drug_name} is working well for me"
            else:
                sentiment = "neutral"
                message = f"I'm not sure about {self.drug_name}"

            receiver.receive_peer_message(
                from_patient=sender.profile.patient_id,
                message=message,
                sentiment=sentiment,
            )

    def _assess_side_effects(self, day: int):
        """Assess side effects based on drug concentration and patient risk."""
        for agent in self.agents:
            if not agent.active:
                continue
            if agent.memory.total_doses_taken == 0:
                continue

            # Side effect probability based on dose, age, genetics
            base_se_prob = 0.05  # 5% per day base rate

            # Higher risk for elderly
            if agent.profile.age > 70:
                base_se_prob += 0.03

            # Higher risk for CYP PM (drug accumulates)
            if agent.profile.cyp2d6 == "PM":
                base_se_prob += 0.05

            # Higher risk with multiple medications (DDI)
            if len(agent.profile.current_medications) > 3:
                base_se_prob += 0.03

            # Higher risk with kidney/liver disease
            if "ckd" in agent.profile.diseases:
                base_se_prob += 0.04
            if "liver_disease" in agent.profile.diseases:
                base_se_prob += 0.05

            if random.random() < base_se_prob:
                effects = [
                    ("nausea", "mild", "gi"),
                    ("headache", "mild", "cns"),
                    ("dizziness", "mild", "cns"),
                    ("fatigue", "mild", "cns"),
                    ("diarrhea", "mild", "gi"),
                    ("insomnia", "mild", "cns"),
                    ("rash", "moderate", "skin"),
                    ("elevated ALT", "moderate", "hepatic"),
                    ("elevated creatinine", "moderate", "renal"),
                    ("QT prolongation", "severe", "cardiovascular"),
                ]
                effect, severity, organ = random.choice(effects)
                agent.receive_side_effect(effect, severity, organ, day)

    def _quick_stats(self) -> str:
        """Quick stats string for logging."""
        active = sum(1 for a in self.agents if a.active)
        avg_adh = sum(a.memory.adherence_rate for a in self.agents) / max(len(self.agents), 1)
        se_count = sum(len(a.memory.side_effects_experienced) for a in self.agents)
        return f"Active={active}/{len(self.agents)}, Adherence={avg_adh:.1%}, SideEffects={se_count}"

    def _compute_population_results(self) -> dict:
        """Compute comprehensive population-level results."""
        n = len(self.agents)
        if n == 0:
            return {"error": "No patients in simulation"}

        # Adherence
        adherence_rates = [a.memory.adherence_rate for a in self.agents]
        active_count = sum(1 for a in self.agents if a.active)
        discontinued_count = n - active_count

        # Side effects
        all_side_effects = []
        for a in self.agents:
            all_side_effects.extend(a.memory.side_effects_experienced)

        se_by_type = {}
        for se in all_side_effects:
            effect = se.get("effect", "unknown")
            se_by_type[effect] = se_by_type.get(effect, 0) + 1

        # Subgroup analysis
        def subgroup_adherence(filter_fn, label):
            agents = [a for a in self.agents if filter_fn(a)]
            if not agents:
                return None
            return {
                "label": label,
                "n": len(agents),
                "mean_adherence": round(sum(a.memory.adherence_rate for a in agents) / len(agents), 3),
                "still_active_pct": round(sum(1 for a in agents if a.active) / len(agents), 3),
                "side_effect_rate": round(sum(1 for a in agents if a.memory.has_experienced_side_effects) / len(agents), 3),
            }

        subgroups = [
            subgroup_adherence(lambda a: a.profile.age < 40, "Age <40"),
            subgroup_adherence(lambda a: 40 <= a.profile.age <= 65, "Age 40-65"),
            subgroup_adherence(lambda a: a.profile.age > 65, "Age >65"),
            subgroup_adherence(lambda a: a.profile.sex == "male", "Male"),
            subgroup_adherence(lambda a: a.profile.sex == "female", "Female"),
            subgroup_adherence(lambda a: a.profile.cyp2d6 == "PM", "CYP2D6 PM"),
            subgroup_adherence(lambda a: a.profile.cyp2d6 == "UM", "CYP2D6 UM"),
            subgroup_adherence(lambda a: "ckd" in a.profile.diseases, "CKD patients"),
            subgroup_adherence(lambda a: "type2_diabetes" in a.profile.diseases, "Diabetic"),
            subgroup_adherence(lambda a: a.profile.has_family_support, "Family support"),
            subgroup_adherence(lambda a: not a.profile.has_family_support, "No family support"),
            subgroup_adherence(lambda a: a.profile.adherence_profile == AdherenceProfile.POOR, "Poor adherence profile"),
            subgroup_adherence(lambda a: a.profile.anxiety_level == AnxietyLevel.SEVERE, "Severe anxiety"),
        ]
        subgroups = [s for s in subgroups if s is not None]

        return {
            "drug": self.drug_name,
            "dose_mg": self.dose_mg,
            "frequency": self.frequency,
            "n_patients": n,
            "n_days": self.n_days,
            "total_rounds": self.rounds_completed,

            "adherence": {
                "mean": round(sum(adherence_rates) / n, 3),
                "median": round(sorted(adherence_rates)[n // 2], 3),
                "min": round(min(adherence_rates), 3),
                "max": round(max(adherence_rates), 3),
                "pct_above_80": round(sum(1 for r in adherence_rates if r >= 0.8) / n, 3),
                "pct_above_50": round(sum(1 for r in adherence_rates if r >= 0.5) / n, 3),
            },

            "retention": {
                "still_active": active_count,
                "discontinued": discontinued_count,
                "retention_rate": round(active_count / n, 3),
            },

            "safety": {
                "total_side_effects": len(all_side_effects),
                "patients_with_side_effects": sum(1 for a in self.agents if a.memory.has_experienced_side_effects),
                "side_effect_rate": round(sum(1 for a in self.agents if a.memory.has_experienced_side_effects) / n, 3),
                "by_type": dict(sorted(se_by_type.items(), key=lambda x: -x[1])),
            },

            "subgroup_analysis": subgroups,

            "social_dynamics": {
                "total_peer_conversations": sum(len(a.memory.peer_conversations) for a in self.agents),
                "patients_influenced_by_peers": sum(1 for a in self.agents if len(a.memory.peer_conversations) > 0),
            },

            "pharmacokinetics": self._compute_pk_statistics(),
        }

    def _compute_pk_statistics(self) -> dict:
        """Compute population-level PK statistics from simulation."""
        n = len(self.agents)
        if n == 0:
            return {}

        # Collect final plasma concentrations (proxy for steady-state)
        plasma_concs = [a.current_plasma_conc for a in self.agents]

        # Collect per-patient Cmax: the maximum plasma concentration observed
        # Use stored pk_predictions Cmax scaled by patient factors
        patient_cmax_values = []
        for a in self.agents:
            cmax_pred = a.pk_predictions.get("cmax", 0)
            if cmax_pred > 0:
                # Scale by CYP phenotype and weight
                cyp_factor = {"PM": 1.5, "IM": 1.2, "EM": 1.0, "UM": 0.7}.get(
                    a.profile.cyp2d6, 1.0
                )
                weight_factor = 70.0 / max(a.profile.weight_kg, 40)
                patient_cmax_values.append(cmax_pred * cyp_factor * weight_factor)
            elif a.current_plasma_conc > 0:
                patient_cmax_values.append(a.current_plasma_conc)

        # Therapeutic window estimation:
        # Assume therapeutic range is 0.3x to 3x of reference Cmax
        ref_cmax = self.drug_pk.get("cmax", 0) if self.drug_pk else 0
        therapeutic_low = ref_cmax * 0.3
        therapeutic_high = ref_cmax * 3.0

        if ref_cmax > 0 and plasma_concs:
            in_therapeutic = sum(
                1 for c in plasma_concs
                if therapeutic_low <= c <= therapeutic_high
            )
            pct_therapeutic = round(in_therapeutic / n, 3)
        else:
            pct_therapeutic = None

        mean_cmax = round(sum(patient_cmax_values) / len(patient_cmax_values), 4) if patient_cmax_values else 0
        mean_css = round(sum(plasma_concs) / n, 4) if plasma_concs else 0
        max_conc = round(max(plasma_concs), 4) if plasma_concs else 0
        min_conc = round(min(plasma_concs), 4) if plasma_concs else 0

        pk_stats = {
            "mean_cmax_mg_l": mean_cmax,
            "mean_steady_state_conc_mg_l": mean_css,
            "max_plasma_conc_mg_l": max_conc,
            "min_plasma_conc_mg_l": min_conc,
            "pct_in_therapeutic_range": pct_therapeutic,
            "reference_cmax_mg_l": round(ref_cmax, 4) if ref_cmax else None,
            "therapeutic_range_mg_l": (
                [round(therapeutic_low, 4), round(therapeutic_high, 4)]
                if ref_cmax > 0 else None
            ),
            "pk_source": self.drug_pk.get("source", "unknown") if self.drug_pk else "none",
            "predictor_stats": self.pk_predictor.get_stats(),
        }

        return pk_stats

    def get_patient_summaries(self) -> list[dict]:
        """Get individual patient summaries."""
        return [a.get_summary() for a in self.agents]
