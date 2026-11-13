"""
Patient Agent — The Core Unit of the Swarm
=============================================
Each PatientAgent represents one virtual patient with:
- Demographics (age, weight, sex, ethnicity)
- Genetics (CYP polymorphisms)
- Medical history (diseases, current medications)
- Behavioral traits (adherence, anxiety, health literacy)
- Social connections (family, peers, healthcare provider)
- Memory (past actions, side effects experienced)
- Decision-making (LLM-powered or rule-based)

Inspired by MiroFish's agent architecture but:
- No file system access
- No shell execution
- No social media posting
- Sandboxed to medical simulation only
"""
from __future__ import annotations

import random
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class AdherenceProfile(str, Enum):
    PERFECT = "perfect"           # Takes every dose exactly on time
    GOOD = "good"                 # Misses <5% of doses
    MODERATE = "moderate"         # Misses 10-20% of doses
    POOR = "poor"                 # Misses 30-50% of doses
    NON_ADHERENT = "non_adherent" # Misses >50% of doses


class HealthLiteracy(str, Enum):
    HIGH = "high"         # Understands medication well, follows instructions
    MODERATE = "moderate"  # Basic understanding, may misinterpret
    LOW = "low"           # Poor understanding, relies on others


class AnxietyLevel(str, Enum):
    NONE = "none"
    MILD = "mild"         # Occasionally worried about side effects
    MODERATE = "moderate"  # Frequently checks symptoms, considers stopping
    SEVERE = "severe"     # Very likely to discontinue at first side effect


@dataclass
class PatientProfile:
    """Complete patient profile for simulation."""
    # Identity
    patient_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    age: int = 40
    sex: str = "male"
    weight_kg: float = 70
    height_cm: float = 170
    ethnicity: str = "caucasian"

    # Genetics
    cyp2d6: str = "EM"    # EM, IM, PM, UM
    cyp2c19: str = "EM"
    cyp2c9: str = "EM"
    cyp3a4: str = "normal"

    # Medical history
    diseases: list[str] = field(default_factory=list)
    current_medications: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)

    # Labs
    egfr: float = 90
    alt: float = 25
    albumin: float = 4.0
    creatinine: float = 1.0

    # Behavioral
    adherence_profile: AdherenceProfile = AdherenceProfile.GOOD
    health_literacy: HealthLiteracy = HealthLiteracy.MODERATE
    anxiety_level: AnxietyLevel = AnxietyLevel.MILD
    smoker: bool = False
    alcohol_use: str = "none"  # none, occasional, moderate, heavy
    exercise_level: str = "moderate"  # sedentary, light, moderate, active

    # Social
    has_family_support: bool = True
    has_regular_doctor: bool = True
    insurance_status: str = "insured"
    peer_influence_susceptibility: float = 0.5  # 0=immune, 1=highly influenced

    # Lifestyle
    meal_regularity: str = "regular"  # regular, irregular, skips_meals
    sleep_quality: str = "good"  # good, fair, poor
    work_schedule: str = "regular"  # regular, shift_work, irregular


@dataclass
class PatientMemory:
    """Agent memory — tracks past actions and experiences."""
    doses_taken: list[dict] = field(default_factory=list)
    doses_missed: list[dict] = field(default_factory=list)
    side_effects_experienced: list[dict] = field(default_factory=list)
    doctor_visits: list[dict] = field(default_factory=list)
    peer_conversations: list[dict] = field(default_factory=list)
    mood_history: list[dict] = field(default_factory=list)
    adherence_streak: int = 0  # consecutive doses taken
    total_doses_prescribed: int = 0
    total_doses_taken: int = 0

    @property
    def adherence_rate(self) -> float:
        if self.total_doses_prescribed == 0:
            return 1.0
        return self.total_doses_taken / self.total_doses_prescribed

    @property
    def has_experienced_side_effects(self) -> bool:
        return len(self.side_effects_experienced) > 0

    @property
    def most_recent_side_effect(self) -> Optional[dict]:
        return self.side_effects_experienced[-1] if self.side_effects_experienced else None


class MedicationAction(str, Enum):
    """Possible actions a patient can take each round."""
    TAKE_DOSE = "take_dose"
    SKIP_DOSE = "skip_dose"
    TAKE_DOUBLE_DOSE = "take_double_dose"
    TAKE_LATE = "take_late"
    DISCONTINUE = "discontinue"
    CALL_DOCTOR = "call_doctor"
    VISIT_ER = "visit_er"
    TALK_TO_PEER = "talk_to_peer"
    SEARCH_INTERNET = "search_internet"
    TAKE_OTC_REMEDY = "take_otc_remedy"


class PatientAgent:
    """
    A single virtual patient agent in the swarm.

    Each agent:
    1. Has a unique profile (demographics, genetics, behavior)
    2. Makes decisions each round (take dose, skip, etc.)
    3. Experiences side effects based on PBPK simulation
    4. Remembers past experiences (affects future decisions)
    5. Can be influenced by other agents (social dynamics)
    """

    def __init__(self, profile: PatientProfile, drug_name: str = "",
                 dose_mg: float = 100, frequency: str = "BID",
                 use_llm: bool = False):
        self.profile = profile
        self.drug_name = drug_name
        self.dose_mg = dose_mg
        self.frequency = frequency  # QD, BID, TID, QID
        self.use_llm = use_llm
        self.memory = PatientMemory()
        self.active = True  # False if patient discontinues
        self.current_day = 0
        self.current_round = 0

        # PK state (updated by PBPK engine)
        self.current_plasma_conc = 0.0
        self.current_organ_concs = {}
        self.pk_predictions = {}
        self.toxicity_alerts = []

    def decide_action(self, time_of_day: str = "morning",
                      peer_messages: list[dict] = None) -> dict:
        """
        Decide what action to take this round.

        Uses rule-based decision-making (fast, deterministic)
        or LLM-based reasoning (slower, more nuanced).
        """
        if not self.active:
            return {"action": MedicationAction.DISCONTINUE, "reason": "Already discontinued"}

        if self.use_llm:
            return self._llm_decide(time_of_day, peer_messages)
        else:
            return self._rule_decide(time_of_day, peer_messages)

    def _rule_decide(self, time_of_day: str, peer_messages: list[dict] = None) -> dict:
        """Rule-based decision engine — fast, no API cost."""
        action = MedicationAction.TAKE_DOSE
        reason = "Scheduled dose"
        confidence = 1.0

        p = self.profile
        m = self.memory
        roll = random.random()

        # ── Adherence-based skipping ──
        skip_probability = {
            AdherenceProfile.PERFECT: 0.0,
            AdherenceProfile.GOOD: 0.05,
            AdherenceProfile.MODERATE: 0.15,
            AdherenceProfile.POOR: 0.35,
            AdherenceProfile.NON_ADHERENT: 0.60,
        }[p.adherence_profile]

        # Modify by factors
        if p.work_schedule == "shift_work" and time_of_day in ["morning", "night"]:
            skip_probability += 0.10
            reason = "Shift work schedule conflict"

        if p.meal_regularity == "skips_meals" and time_of_day == "morning":
            skip_probability += 0.08
            reason = "Skipped breakfast, forgot medication"

        if p.sleep_quality == "poor" and time_of_day == "morning":
            skip_probability += 0.05
            reason = "Poor sleep, overslept"

        if not p.has_family_support:
            skip_probability += 0.05

        # ── Side effect anxiety ──
        if m.has_experienced_side_effects:
            if p.anxiety_level == AnxietyLevel.SEVERE:
                skip_probability += 0.30
                reason = "Afraid of side effects recurring"
            elif p.anxiety_level == AnxietyLevel.MODERATE:
                skip_probability += 0.15
                reason = "Worried about side effects"
            elif p.anxiety_level == AnxietyLevel.MILD:
                skip_probability += 0.05

        # ── Peer influence ──
        if peer_messages:
            positive = sum(1 for msg in peer_messages if msg.get("sentiment") == "positive")
            negative = sum(1 for msg in peer_messages if msg.get("sentiment") == "negative")

            influence = p.peer_influence_susceptibility
            if negative > positive:
                skip_probability += 0.10 * influence
                reason = "Heard negative experiences from peers"
            elif positive > negative:
                skip_probability -= 0.10 * influence
                reason = "Encouraged by peers' positive experiences"

        # ── Adherence streak bonus ──
        if m.adherence_streak > 7:
            skip_probability *= 0.8  # 20% less likely to break a streak

        # ── Make decision ──
        if roll < skip_probability:
            action = MedicationAction.SKIP_DOSE
            if not reason or reason == "Scheduled dose":
                reason = "Forgot or chose to skip"
        else:
            action = MedicationAction.TAKE_DOSE
            reason = "Took scheduled dose"

            # Small chance of taking late
            if roll < skip_probability + 0.10:
                action = MedicationAction.TAKE_LATE
                late_hours = random.choice([0.5, 1, 2, 3, 4])
                reason = f"Took dose {late_hours}h late"

            # Very rare: double dose (forgot if they took it)
            if p.adherence_profile == AdherenceProfile.POOR and roll > 0.95:
                action = MedicationAction.TAKE_DOUBLE_DOSE
                reason = "Couldn't remember if dose was taken, took extra"

        # ── Severe side effect → call doctor or ER ──
        if m.has_experienced_side_effects:
            last_se = m.most_recent_side_effect
            if last_se and last_se.get("severity") == "severe":
                if roll < 0.30:
                    action = MedicationAction.CALL_DOCTOR
                    reason = f"Severe side effect: {last_se.get('effect', 'unknown')}"
                elif roll < 0.10:
                    action = MedicationAction.VISIT_ER
                    reason = f"Emergency: {last_se.get('effect', 'unknown')}"

        # ── Discontinuation decision ──
        if m.adherence_rate < 0.3 and self.current_day > 7:
            if roll < 0.20:
                action = MedicationAction.DISCONTINUE
                reason = "Patient decided to stop taking medication"
                self.active = False

        if m.side_effects_experienced and len(m.side_effects_experienced) >= 3:
            if p.anxiety_level in [AnxietyLevel.SEVERE, AnxietyLevel.MODERATE]:
                if roll < 0.15:
                    action = MedicationAction.DISCONTINUE
                    reason = "Multiple side effects — patient discontinued"
                    self.active = False

        return {
            "action": action.value,
            "reason": reason,
            "patient_id": self.profile.patient_id,
            "day": self.current_day,
            "time": time_of_day,
            "round": self.current_round,
            "skip_probability": round(skip_probability, 3),
        }

    def _llm_decide(self, time_of_day: str, peer_messages: list[dict] = None) -> dict:
        """LLM-based decision — more nuanced but costs API calls."""
        # This would call Claude Opus to reason about the patient's decision
        # For now, fall back to rule-based
        return self._rule_decide(time_of_day, peer_messages)

    def record_action(self, action_result: dict):
        """Record the action taken this round."""
        action = action_result["action"]

        if action == MedicationAction.TAKE_DOSE.value:
            self.memory.doses_taken.append(action_result)
            self.memory.total_doses_taken += 1
            self.memory.adherence_streak += 1
        elif action == MedicationAction.TAKE_LATE.value:
            self.memory.doses_taken.append(action_result)
            self.memory.total_doses_taken += 1
            self.memory.adherence_streak += 1
        elif action == MedicationAction.TAKE_DOUBLE_DOSE.value:
            self.memory.doses_taken.append(action_result)
            self.memory.total_doses_taken += 2  # Took double
            self.memory.adherence_streak = 0
        elif action == MedicationAction.SKIP_DOSE.value:
            self.memory.doses_missed.append(action_result)
            self.memory.adherence_streak = 0
        elif action == MedicationAction.CALL_DOCTOR.value:
            self.memory.doctor_visits.append(action_result)
        elif action == MedicationAction.DISCONTINUE.value:
            self.active = False

        self.memory.total_doses_prescribed += 1

    def receive_side_effect(self, effect: str, severity: str = "mild",
                           organ: str = "", day: int = 0):
        """Record a side effect experienced by this patient."""
        self.memory.side_effects_experienced.append({
            "effect": effect,
            "severity": severity,
            "organ": organ,
            "day": day,
            "round": self.current_round,
        })

    def receive_peer_message(self, from_patient: str, message: str,
                            sentiment: str = "neutral"):
        """Receive a message from another patient agent."""
        self.memory.peer_conversations.append({
            "from": from_patient,
            "message": message,
            "sentiment": sentiment,
            "day": self.current_day,
        })

    def get_summary(self) -> dict:
        """Get summary of this patient's simulation."""
        return {
            "patient_id": self.profile.patient_id,
            "age": self.profile.age,
            "sex": self.profile.sex,
            "weight_kg": self.profile.weight_kg,
            "diseases": self.profile.diseases,
            "cyp2d6": self.profile.cyp2d6,
            "adherence_profile": self.profile.adherence_profile.value,
            "anxiety_level": self.profile.anxiety_level.value,
            "adherence_rate": round(self.memory.adherence_rate, 3),
            "total_doses_taken": self.memory.total_doses_taken,
            "total_doses_prescribed": self.memory.total_doses_prescribed,
            "doses_missed": len(self.memory.doses_missed),
            "side_effects_count": len(self.memory.side_effects_experienced),
            "doctor_visits": len(self.memory.doctor_visits),
            "still_active": self.active,
            "current_plasma_conc": round(self.current_plasma_conc, 4),
            "pk_predictions": self.pk_predictions,
        }
