"""
NeoGen Swarm Intelligence Engine (MiroFish-Inspired)
=====================================================
Simulates thousands of virtual patient agents, each with unique:
- Biology (age, weight, sex, genetics, diseases)
- Behavior (adherence, lifestyle, diet, alcohol, exercise)
- Social dynamics (shares experiences, influences others)

Each agent runs through NeoGen's PBPK simulation with their
specific parameters, then BEHAVES based on the drug effects
they experience (reports side effects, skips doses, etc.)

The emergent population behavior predicts real-world drug
effectiveness — not just clinical trial efficacy.

Inspired by MiroFish's swarm intelligence architecture
(https://github.com/666ghj/MiroFish) but adapted for
pharmacology simulation rather than social prediction.
"""
from __future__ import annotations

import logging
import random
import json
import math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# PATIENT AGENT
# ════════════════════════════════════════════════════════════

@dataclass
class PatientAgent:
    """A virtual patient with biology + behavior + social traits."""
    id: int
    name: str

    # Biology
    age: int = 45
    sex: str = "male"
    weight_kg: float = 70
    egfr: float = 90
    alt: float = 25
    albumin: float = 4.0
    cyp2d6: str = "normal"
    cyp2c19: str = "normal"
    diseases: list = field(default_factory=list)

    # Behavior
    adherence_rate: float = 0.85       # 0-1, probability of taking dose on time
    alcohol_drinks_per_week: int = 0
    smoking: bool = False
    exercise_hours_per_week: float = 0
    diet_quality: str = "average"       # poor/average/good
    takes_with_food: bool = True
    forgets_evening_dose: float = 0.1   # probability of forgetting PM dose
    stops_when_better: bool = False     # stops medication when symptoms improve
    health_literacy: str = "medium"     # low/medium/high

    # Social
    influence_score: float = 0.5       # 0-1, how much they influence others
    susceptibility: float = 0.5        # 0-1, how much they're influenced by others
    shares_side_effects: bool = True
    trusts_doctor: float = 0.8         # 0-1

    # State (changes during simulation)
    current_dose_mg: float = 0
    days_on_drug: int = 0
    doses_taken: int = 0
    doses_missed: int = 0
    side_effects_experienced: list = field(default_factory=list)
    severity_score: float = 0          # 0-10, overall side effect burden
    stopped_medication: bool = False
    stop_reason: str = ""
    posts: list = field(default_factory=list)    # social media posts
    pk_results: dict = field(default_factory=dict)

    @property
    def adherence_actual(self):
        if self.doses_taken + self.doses_missed == 0:
            return self.adherence_rate
        return self.doses_taken / (self.doses_taken + self.doses_missed)

    @property
    def effective_drug_level(self):
        """Estimated drug level based on adherence."""
        if self.stopped_medication:
            return 0
        return self.pk_results.get("cmax", 0) * self.adherence_actual


# ════════════════════════════════════════════════════════════
# AGENT GENERATOR
# ════════════════════════════════════════════════════════════

# Real US population statistics for generating realistic agents
AGE_DISTRIBUTION = [
    (18, 25, 0.12), (25, 35, 0.17), (35, 45, 0.16), (45, 55, 0.17),
    (55, 65, 0.16), (65, 75, 0.13), (75, 90, 0.09),
]

CYP2D6_DISTRIBUTION = {
    "ultra-rapid": 0.05, "normal": 0.72, "intermediate": 0.16, "poor": 0.07
}

CYP2C19_DISTRIBUTION = {
    "normal": 0.65, "intermediate": 0.27, "poor": 0.08
}

COMMON_DISEASES = [
    ("hypertension", 0.30), ("diabetes", 0.12), ("obesity_I", 0.20),
    ("obesity_II", 0.10), ("ckd_2", 0.08), ("ckd_3a", 0.04),
    ("liver_A", 0.03), ("hypothyroidism", 0.05),
]

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Daniel", "Lisa", "Matthew", "Nancy",
    "Anthony", "Betty", "Mark", "Helen", "Donald", "Sandra", "Steven", "Donna",
    "Paul", "Carol", "Andrew", "Ruth", "Joshua", "Sharon", "Kenneth", "Michelle",
    "Kevin", "Laura", "Brian", "Emily", "George", "Amanda", "Timothy", "Dorothy",
    "Ronald", "Melissa", "Edward", "Deborah", "Jason", "Stephanie", "Jeffrey", "Rebecca",
    "Ryan", "Angela", "Jacob", "Brenda", "Gary", "Nicole", "Nicholas", "Amy",
    "Eric", "Kathleen", "Jonathan", "Shirley", "Stephen", "Anna", "Larry", "Cynthia",
]


def generate_agent(agent_id: int) -> PatientAgent:
    """Generate a realistic virtual patient with randomized demographics."""
    # Age from distribution
    r = random.random()
    cumulative = 0
    age = 45
    for low, high, prob in AGE_DISTRIBUTION:
        cumulative += prob
        if r < cumulative:
            age = random.randint(low, high)
            break

    sex = random.choice(["male", "female"])
    name = random.choice(FIRST_NAMES)

    # Weight based on age/sex (normal distribution)
    if sex == "male":
        weight = random.gauss(82, 15)
    else:
        weight = random.gauss(70, 14)
    weight = max(45, min(150, weight))

    # Lab values
    egfr = max(15, min(120, random.gauss(85 - (age - 40) * 0.5, 15)))
    alt = max(10, min(200, random.gauss(25 + (age - 40) * 0.3, 10)))
    albumin = max(2.0, min(5.0, random.gauss(4.0 - (age - 40) * 0.01, 0.3)))

    # Genetics
    cyp2d6 = random.choices(list(CYP2D6_DISTRIBUTION.keys()),
                            weights=list(CYP2D6_DISTRIBUTION.values()))[0]
    cyp2c19 = random.choices(list(CYP2C19_DISTRIBUTION.keys()),
                             weights=list(CYP2C19_DISTRIBUTION.values()))[0]

    # Diseases (based on age and probability)
    diseases = []
    for disease, base_prob in COMMON_DISEASES:
        age_factor = max(0.5, age / 50)
        if random.random() < base_prob * age_factor:
            diseases.append(disease)

    # Behavior
    adherence = random.gauss(0.80, 0.15)
    adherence = max(0.3, min(1.0, adherence))

    alcohol = 0
    if age >= 21:
        if random.random() < 0.55:  # 55% drink
            alcohol = random.choices([1, 3, 5, 10, 15],
                                     weights=[0.4, 0.3, 0.15, 0.1, 0.05])[0]

    smoking = random.random() < 0.14  # 14% smoking rate

    exercise = random.choices([0, 1, 3, 5, 7],
                              weights=[0.25, 0.25, 0.25, 0.15, 0.10])[0]

    health_literacy = random.choices(["low", "medium", "high"],
                                     weights=[0.25, 0.50, 0.25])[0]

    return PatientAgent(
        id=agent_id,
        name=f"{name}_{agent_id}",
        age=age, sex=sex, weight_kg=round(weight, 1),
        egfr=round(egfr, 1), alt=round(alt, 1), albumin=round(albumin, 1),
        cyp2d6=cyp2d6, cyp2c19=cyp2c19,
        diseases=diseases,
        adherence_rate=round(adherence, 2),
        alcohol_drinks_per_week=alcohol,
        smoking=smoking,
        exercise_hours_per_week=exercise,
        diet_quality=random.choice(["poor", "average", "good"]),
        takes_with_food=random.random() < 0.7,
        forgets_evening_dose=round(random.random() * 0.3, 2),
        stops_when_better=random.random() < 0.15,
        health_literacy=health_literacy,
        influence_score=round(random.random(), 2),
        susceptibility=round(random.random(), 2),
        shares_side_effects=random.random() < 0.6,
        trusts_doctor=round(random.gauss(0.75, 0.15), 2),
    )


def generate_population(n: int, seed: int = 42) -> list[PatientAgent]:
    """Generate a population of N virtual patients."""
    random.seed(seed)
    return [generate_agent(i) for i in range(n)]


# ════════════════════════════════════════════════════════════
# SWARM SIMULATION ENGINE
# ════════════════════════════════════════════════════════════

@dataclass
class SwarmSimulationConfig:
    """Configuration for swarm simulation."""
    drug_smiles: str = ""
    drug_name: str = "Unknown Drug"
    dose_mg: float = 100
    frequency: str = "once_daily"        # once_daily, twice_daily, three_times_daily
    duration_days: int = 30
    n_agents: int = 100
    seed: int = 42
    side_effect_threshold: float = 0.5   # Fraction of Cmax that triggers side effects
    serious_threshold: float = 0.8       # Fraction of Cmax that triggers serious effects


@dataclass
class SwarmResult:
    """Results from swarm simulation."""
    config: dict = field(default_factory=dict)
    agents: list = field(default_factory=list)
    population_stats: dict = field(default_factory=dict)
    timeline: list = field(default_factory=list)
    social_posts: list = field(default_factory=list)
    adherence_report: dict = field(default_factory=dict)
    safety_signals: list = field(default_factory=list)
    outcome_prediction: dict = field(default_factory=dict)


def simulate_agent_day(agent: PatientAgent, day: int, config: SwarmSimulationConfig,
                       population_sentiment: float) -> dict:
    """Simulate one day for one agent."""
    events = []

    if agent.stopped_medication:
        return {"events": events, "took_dose": False}

    # Decide if agent takes dose today
    adherence_modifier = 1.0

    # Alcohol reduces adherence
    if agent.alcohol_drinks_per_week > 7:
        adherence_modifier *= 0.85

    # Social influence: if many people posting negative, reduces adherence
    if population_sentiment < 0.3:
        adherence_modifier *= (0.7 + agent.trusts_doctor * 0.3)

    # Forgets evening dose
    if config.frequency in ["twice_daily", "three_times_daily"]:
        if random.random() < agent.forgets_evening_dose:
            agent.doses_missed += 1
            events.append({"type": "missed_dose", "reason": "forgot_evening"})

    # Main dose adherence
    took_dose = random.random() < (agent.adherence_rate * adherence_modifier)

    if took_dose:
        agent.doses_taken += 1
        agent.days_on_drug += 1

        # Check for side effects based on PK
        cmax = agent.pk_results.get("cmax", 0)
        if cmax > 0:
            # Higher drug levels in poor metabolizers
            cyp_multiplier = 1.0
            if agent.cyp2d6 == "poor":
                cyp_multiplier = 3.0
            elif agent.cyp2d6 == "intermediate":
                cyp_multiplier = 1.5
            elif agent.cyp2d6 == "ultra-rapid":
                cyp_multiplier = 0.5

            effective_cmax = cmax * cyp_multiplier * (1 + agent.alcohol_drinks_per_week * 0.05)

            # Mild side effects (only 15-25% chance per day if above threshold)
            if effective_cmax > config.side_effect_threshold * cmax * 1.5 and random.random() < 0.08:
                mild_effects = random.choices(
                    ["nausea", "headache", "dizziness", "fatigue", "insomnia", "dry_mouth"],
                    k=1
                )
                for effect in mild_effects:
                    if effect not in agent.side_effects_experienced:
                        agent.side_effects_experienced.append(effect)
                        agent.severity_score += 0.5
                        events.append({"type": "side_effect", "effect": effect, "severity": "mild"})

            # Serious side effects (very rare — 1-2% chance)
            if effective_cmax > config.serious_threshold * cmax * 2 and random.random() < 0.015:
                serious_effects = random.choices(
                    ["severe_nausea", "arrhythmia", "liver_enzyme_elevation", "severe_dizziness",
                     "allergic_reaction", "bleeding"],
                    k=1
                )
                for effect in serious_effects:
                    agent.side_effects_experienced.append(effect)
                    agent.severity_score += 3
                    events.append({"type": "serious_side_effect", "effect": effect, "severity": "serious"})

        # Decide to stop medication (only if severity is really high)
        if agent.severity_score > 8 and random.random() < 0.15:
            agent.stopped_medication = True
            agent.stop_reason = "side_effects"
            events.append({"type": "discontinued", "reason": "side_effects",
                          "day": day, "effects": agent.side_effects_experienced})

        if agent.stops_when_better and day > 14 and random.random() < 0.1:
            agent.stopped_medication = True
            agent.stop_reason = "feeling_better"
            events.append({"type": "discontinued", "reason": "feeling_better", "day": day})

    else:
        agent.doses_missed += 1
        events.append({"type": "missed_dose", "reason": "non_adherent"})

    # Social behavior: post about experience
    if agent.shares_side_effects and len(agent.side_effects_experienced) > 0 and random.random() < 0.1:
        sentiment = "negative" if agent.severity_score > 3 else "neutral"
        post = {
            "agent_id": agent.id,
            "agent_name": agent.name,
            "day": day,
            "sentiment": sentiment,
            "content": _generate_post(agent, sentiment),
            "influence": agent.influence_score,
        }
        agent.posts.append(post)
        events.append({"type": "social_post", "sentiment": sentiment})

    if agent.stopped_medication and agent.shares_side_effects and random.random() < 0.3:
        post = {
            "agent_id": agent.id,
            "agent_name": agent.name,
            "day": day,
            "sentiment": "negative",
            "content": f"Stopped taking {config.drug_name}. Side effects were unbearable: {', '.join(agent.side_effects_experienced[:3])}",
            "influence": agent.influence_score,
        }
        agent.posts.append(post)

    return {"events": events, "took_dose": took_dose}


def _generate_post(agent: PatientAgent, sentiment: str) -> str:
    """Generate a social media post from an agent."""
    if sentiment == "negative":
        templates = [
            f"Day {agent.days_on_drug}: Experiencing {agent.side_effects_experienced[-1]}. Anyone else?",
            f"Not sure if I should continue. {agent.side_effects_experienced[-1]} is really bothering me.",
            f"Doctor said to keep taking it but the {agent.side_effects_experienced[-1]} is getting worse.",
            f"Week {agent.days_on_drug // 7}: Side effects include {', '.join(agent.side_effects_experienced[:2])}. Frustrating.",
        ]
    else:
        templates = [
            f"Day {agent.days_on_drug}: Some mild {agent.side_effects_experienced[-1]} but manageable.",
            f"Started new medication. Minor {agent.side_effects_experienced[-1]}, doctor says it's normal.",
            f"Anyone else on this? Mild side effects so far but seems to be working.",
        ]
    return random.choice(templates)


def run_swarm_simulation(config: SwarmSimulationConfig) -> SwarmResult:
    """
    Run the full swarm simulation.

    1. Generate N patient agents
    2. Get PBPK predictions for each unique biology
    3. Simulate day-by-day behavior for duration_days
    4. Track population-level outcomes
    5. Generate report
    """
    logger.info(f"Starting swarm simulation: {config.n_agents} agents, {config.duration_days} days")

    # Generate population
    agents = generate_population(config.n_agents, config.seed)

    # For now, assign PK results based on simplified predictions
    # In production, each agent would call /api/predict with their specific parameters
    for agent in agents:
        base_cmax = random.gauss(10, 3)  # Will be replaced by actual NeoGen prediction
        base_hl = random.gauss(6, 2)
        agent.pk_results = {
            "cmax": max(0.1, base_cmax * (70 / agent.weight_kg)),
            "half_life": max(0.5, base_hl),
            "auc": max(1, base_cmax * base_hl * 1.44),
        }
        agent.current_dose_mg = config.dose_mg

    # Day-by-day simulation
    timeline = []
    all_posts = []
    population_sentiment = 0.7  # Start positive

    for day in range(1, config.duration_days + 1):
        day_events = []
        day_doses_taken = 0
        day_doses_missed = 0
        day_discontinuations = 0
        day_side_effects = 0

        for agent in agents:
            result = simulate_agent_day(agent, day, config, population_sentiment)
            if result["took_dose"]:
                day_doses_taken += 1
            else:
                day_doses_missed += 1
            for event in result["events"]:
                day_events.append(event)
                if event["type"] == "discontinued":
                    day_discontinuations += 1
                if event["type"] in ["side_effect", "serious_side_effect"]:
                    day_side_effects += 1

        # Collect social posts
        for agent in agents:
            for post in agent.posts:
                if post.get("day") == day:
                    all_posts.append(post)

        # Update population sentiment based on social posts
        day_posts = [p for p in all_posts if p.get("day") == day]
        if day_posts:
            neg_ratio = sum(1 for p in day_posts if p["sentiment"] == "negative") / len(day_posts)
            population_sentiment = population_sentiment * 0.9 + (1 - neg_ratio) * 0.1

        active_agents = sum(1 for a in agents if not a.stopped_medication)
        adherent_agents = sum(1 for a in agents if not a.stopped_medication and a.adherence_actual > 0.8)

        timeline.append({
            "day": day,
            "active_patients": active_agents,
            "adherent_patients": adherent_agents,
            "doses_taken": day_doses_taken,
            "doses_missed": day_doses_missed,
            "new_discontinuations": day_discontinuations,
            "side_effects_reported": day_side_effects,
            "population_sentiment": round(population_sentiment, 3),
            "n_posts": len(day_posts),
        })

    # Calculate final statistics
    active_end = sum(1 for a in agents if not a.stopped_medication)
    adherent_end = sum(1 for a in agents if not a.stopped_medication and a.adherence_actual > 0.8)
    total_stopped = sum(1 for a in agents if a.stopped_medication)
    stopped_side_effects = sum(1 for a in agents if a.stop_reason == "side_effects")
    stopped_feeling_better = sum(1 for a in agents if a.stop_reason == "feeling_better")

    # Side effect frequency
    all_effects = []
    for a in agents:
        all_effects.extend(a.side_effects_experienced)
    effect_counts = {}
    for e in all_effects:
        effect_counts[e] = effect_counts.get(e, 0) + 1

    # Safety signals (effects reported by >5% of patients)
    safety_signals = []
    for effect, count in sorted(effect_counts.items(), key=lambda x: -x[1]):
        pct = count / config.n_agents * 100
        if pct > 5:
            safety_signals.append({
                "effect": effect,
                "count": count,
                "percent": round(pct, 1),
                "signal": "STRONG" if pct > 20 else "MODERATE" if pct > 10 else "WEAK",
            })

    # Agent summaries
    agent_summaries = []
    for a in agents:
        agent_summaries.append({
            "id": a.id, "name": a.name, "age": a.age, "sex": a.sex,
            "weight": a.weight_kg, "cyp2d6": a.cyp2d6,
            "diseases": a.diseases,
            "adherence_actual": round(a.adherence_actual, 2),
            "doses_taken": a.doses_taken, "doses_missed": a.doses_missed,
            "side_effects": a.side_effects_experienced,
            "severity_score": a.severity_score,
            "stopped": a.stopped_medication, "stop_reason": a.stop_reason,
            "n_posts": len(a.posts),
        })

    result = SwarmResult(
        config={
            "drug": config.drug_name, "smiles": config.drug_smiles,
            "dose_mg": config.dose_mg, "frequency": config.frequency,
            "duration_days": config.duration_days, "n_agents": config.n_agents,
        },
        agents=agent_summaries,
        population_stats={
            "total_patients": config.n_agents,
            "active_at_end": active_end,
            "adherent_at_end": adherent_end,
            "discontinued_total": total_stopped,
            "discontinued_side_effects": stopped_side_effects,
            "discontinued_feeling_better": stopped_feeling_better,
            "retention_rate": round(active_end / config.n_agents * 100, 1),
            "adherence_rate": round(adherent_end / config.n_agents * 100, 1),
            "mean_adherence": round(sum(a.adherence_actual for a in agents) / len(agents), 2),
            "mean_severity_score": round(sum(a.severity_score for a in agents) / len(agents), 1),
        },
        timeline=timeline,
        social_posts=all_posts[-50:],  # Last 50 posts
        adherence_report={
            "overall_adherence": round(sum(a.adherence_actual for a in agents) / len(agents) * 100, 1),
            "by_age_group": _adherence_by_group(agents, "age"),
            "by_sex": _adherence_by_group(agents, "sex"),
            "by_cyp2d6": _adherence_by_group(agents, "cyp2d6"),
        },
        safety_signals=safety_signals,
        outcome_prediction={
            "real_world_effectiveness": round(
                (active_end / config.n_agents) *
                (sum(a.adherence_actual for a in agents if not a.stopped_medication) /
                 max(active_end, 1)) * 100, 1
            ),
            "clinical_trial_efficacy": 85,  # Assumed from controlled trial
            "efficacy_effectiveness_gap": round(
                85 - (active_end / config.n_agents) *
                (sum(a.adherence_actual for a in agents if not a.stopped_medication) /
                 max(active_end, 1)) * 100, 1
            ),
            "interpretation": _generate_outcome_interpretation(agents, config),
        },
    )

    logger.info(f"Swarm simulation complete: retention={result.population_stats['retention_rate']}%, "
                f"adherence={result.population_stats['adherence_rate']}%")

    return result


def _adherence_by_group(agents: list, group_by: str) -> dict:
    """Calculate adherence statistics by demographic group."""
    groups = {}
    for a in agents:
        if group_by == "age":
            key = f"{(a.age // 10) * 10}-{(a.age // 10) * 10 + 9}"
        elif group_by == "sex":
            key = a.sex
        elif group_by == "cyp2d6":
            key = a.cyp2d6
        else:
            key = "all"

        if key not in groups:
            groups[key] = {"adherent": 0, "total": 0, "stopped": 0}
        groups[key]["total"] += 1
        if a.adherence_actual > 0.8 and not a.stopped_medication:
            groups[key]["adherent"] += 1
        if a.stopped_medication:
            groups[key]["stopped"] += 1

    result = {}
    for key, data in groups.items():
        result[key] = {
            "n": data["total"],
            "adherence_pct": round(data["adherent"] / max(data["total"], 1) * 100, 1),
            "discontinuation_pct": round(data["stopped"] / max(data["total"], 1) * 100, 1),
        }
    return result


def _generate_outcome_interpretation(agents: list, config: SwarmSimulationConfig) -> str:
    """Generate a text interpretation of the swarm results."""
    active = sum(1 for a in agents if not a.stopped_medication)
    stopped_se = sum(1 for a in agents if a.stop_reason == "side_effects")
    stopped_fb = sum(1 for a in agents if a.stop_reason == "feeling_better")
    poor_met = sum(1 for a in agents if a.cyp2d6 == "poor" and a.stopped_medication)
    poor_total = sum(1 for a in agents if a.cyp2d6 == "poor")

    lines = []
    lines.append(f"After {config.duration_days} days of {config.drug_name} {config.dose_mg}mg "
                 f"{config.frequency.replace('_', ' ')}:")
    lines.append(f"- {active}/{config.n_agents} patients ({active/config.n_agents*100:.0f}%) "
                 f"remained on medication")
    if stopped_se > 0:
        lines.append(f"- {stopped_se} patients discontinued due to side effects")
    if stopped_fb > 0:
        lines.append(f"- {stopped_fb} patients stopped because they felt better")
    if poor_total > 0 and poor_met > 0:
        lines.append(f"- CYP2D6 poor metabolizers: {poor_met}/{poor_total} "
                     f"({poor_met/poor_total*100:.0f}%) discontinued (vs "
                     f"{(config.n_agents-active-poor_met)/(config.n_agents-poor_total)*100:.0f}% overall)")

    return "\n".join(lines)
