"""
Agent and Genome models — the core heritable unit of the evolutionary swarm.

Each agent has a TraitVector (genome) that determines its scientific reasoning
style, and this genome is heritable through crossover and mutation.
"""

import uuid
import random
from dataclasses import dataclass, field
from typing import Optional


# All heritable traits with ranges [0.0, 1.0]
TRAIT_NAMES = [
    "risk_tolerance",           # Willingness to propose bold claims
    "curiosity",                # Drive to explore novel angles
    "skepticism",               # Tendency to doubt and challenge
    "mathematical_rigor",       # Preference for formal math over prose
    "abstraction_tendency",     # Preference for high-level models vs details
    "empirical_bias",           # Weight given to data over theory
    "literature_reliance",      # How much an agent cites prior work
    "collaboration_tendency",   # Willingness to build on others' ideas
    "contrarianism",            # Tendency to oppose consensus
    "falsification_drive",      # Drive to find flaws in claims
    "mechanistic_preference",   # Preference for mechanistic explanations
    "reductionism_vs_systems",  # 0=reductionist, 1=systems-level
    "tolerance_for_uncertainty", # Comfort with ambiguity
    "memory_depth",             # How much prior context an agent recalls
    "decision_speed",           # Quick vs deliberate reasoning
    "confidence_calibration",   # How well confidence matches accuracy
    "environmental_sensitivity", # Sensitivity to feedback and scoring
]


@dataclass
class TraitVector:
    """Heritable genome of a scientist agent. Each trait is [0.0, 1.0]."""
    traits: dict[str, float] = field(default_factory=dict)

    @classmethod
    def random(cls, seed: Optional[int] = None) -> "TraitVector":
        """Generate a random trait vector."""
        rng = random.Random(seed)
        return cls(traits={name: round(rng.random(), 3) for name in TRAIT_NAMES})

    @classmethod
    def crossover(cls, parent_a: "TraitVector", parent_b: "TraitVector",
                  seed: Optional[int] = None) -> "TraitVector":
        """Produce offspring genome via uniform crossover."""
        rng = random.Random(seed)
        child_traits = {}
        for name in TRAIT_NAMES:
            if rng.random() < 0.5:
                child_traits[name] = parent_a.traits.get(name, 0.5)
            else:
                child_traits[name] = parent_b.traits.get(name, 0.5)
        return cls(traits=child_traits)

    def mutate(self, rate: float = 0.15, magnitude: float = 0.2,
               seed: Optional[int] = None) -> "TraitVector":
        """Return a mutated copy of this genome."""
        rng = random.Random(seed)
        new_traits = {}
        for name, value in self.traits.items():
            if rng.random() < rate:
                delta = rng.gauss(0, magnitude)
                new_traits[name] = round(max(0.0, min(1.0, value + delta)), 3)
            else:
                new_traits[name] = value
        return TraitVector(traits=new_traits)

    def distance(self, other: "TraitVector") -> float:
        """Euclidean distance between two genomes (for diversity measurement)."""
        total = 0.0
        for name in TRAIT_NAMES:
            a = self.traits.get(name, 0.5)
            b = other.traits.get(name, 0.5)
            total += (a - b) ** 2
        return total ** 0.5

    def dominant_traits(self, threshold: float = 0.7) -> list[str]:
        """Return traits with values above threshold."""
        return [name for name, val in self.traits.items() if val >= threshold]

    def to_personality_description(self) -> str:
        """Convert trait vector to a natural language personality summary."""
        desc_parts = []
        t = self.traits

        if t.get("mathematical_rigor", 0.5) > 0.7:
            desc_parts.append("highly mathematically rigorous")
        elif t.get("mathematical_rigor", 0.5) < 0.3:
            desc_parts.append("prefers intuitive reasoning over formal math")

        if t.get("skepticism", 0.5) > 0.7:
            desc_parts.append("deeply skeptical of unsubstantiated claims")
        if t.get("contrarianism", 0.5) > 0.7:
            desc_parts.append("tends to challenge consensus positions")
        if t.get("risk_tolerance", 0.5) > 0.7:
            desc_parts.append("willing to propose bold, unconventional hypotheses")
        if t.get("empirical_bias", 0.5) > 0.7:
            desc_parts.append("strongly data-driven, demands empirical evidence")
        if t.get("mechanistic_preference", 0.5) > 0.7:
            desc_parts.append("insists on mechanistic explanations")
        if t.get("reductionism_vs_systems", 0.5) > 0.7:
            desc_parts.append("thinks at the systems level, integrating across scales")
        elif t.get("reductionism_vs_systems", 0.5) < 0.3:
            desc_parts.append("reductionist, breaks problems into atomic components")
        if t.get("falsification_drive", 0.5) > 0.7:
            desc_parts.append("driven to falsify and stress-test every claim")
        if t.get("collaboration_tendency", 0.5) > 0.7:
            desc_parts.append("collaborative, builds on others' ideas constructively")
        if t.get("tolerance_for_uncertainty", 0.5) > 0.7:
            desc_parts.append("comfortable expressing uncertainty and partial knowledge")

        if not desc_parts:
            desc_parts.append("balanced reasoning style without strong biases")

        return "; ".join(desc_parts)


@dataclass
class AgentGenome:
    """Full heritable state of an agent, including traits and ancestry."""
    genome_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    traits: TraitVector = field(default_factory=TraitVector.random)
    parent_a_id: Optional[str] = None
    parent_b_id: Optional[str] = None
    generation: int = 0
    mutations_applied: int = 0


@dataclass
class ScientistAgent:
    """A single AI scientist agent in the evolutionary swarm."""

    # Identity
    agent_id: str = field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:8]}")
    name: str = ""
    generation: int = 0

    # Lineage
    lineage_id: str = ""
    parent_ids: list[str] = field(default_factory=list)

    # Role
    role: str = ""                          # e.g., "cardiac electrophysiologist"
    specialty: str = ""                     # More detailed specialty
    expertise_areas: list[str] = field(default_factory=list)
    knowledge_priorities: list[str] = field(default_factory=list)
    expected_contribution: str = ""

    # Genome
    genome: AgentGenome = field(default_factory=AgentGenome)

    # State
    memory: list[dict] = field(default_factory=list)
    hypotheses_proposed: list[str] = field(default_factory=list)
    critiques_given: list[str] = field(default_factory=list)

    # Performance
    fitness_history: list[float] = field(default_factory=list)
    total_fitness: float = 0.0
    debates_participated: int = 0
    claims_validated: int = 0
    claims_rejected: int = 0

    # Status
    alive: bool = True
    eliminated_at_generation: Optional[int] = None

    def personality_description(self) -> str:
        """Natural language description of this agent's reasoning style."""
        return self.genome.traits.to_personality_description()

    def to_system_prompt(self, scenario_context: str = "") -> str:
        """Build the LLM system prompt for this agent."""
        personality = self.personality_description()
        return f"""You are {self.name}, a {self.role} in an evolutionary scientific research swarm.

SPECIALTY: {self.specialty}
EXPERTISE: {', '.join(self.expertise_areas)}
EXPECTED CONTRIBUTION: {self.expected_contribution}

YOUR REASONING STYLE: {personality}

KNOWLEDGE PRIORITIES: {', '.join(self.knowledge_priorities)}

RULES:
1. You are a rigorous scientist. Every claim must have a rationale.
2. Use mathematical notation (equations, variables, relationships) when appropriate.
3. Cite mechanisms, not just correlations.
4. Express uncertainty honestly — state confidence levels for claims.
5. If you disagree with another agent, explain EXACTLY what is wrong and why.
6. Do not fabricate citations or data. Say "I believe" or "I hypothesize" for unsupported claims.
7. Propose TESTABLE predictions whenever possible.
8. Keep responses focused — no filler. Quality over quantity.

{f'SCIENTIFIC CONTEXT:{chr(10)}{scenario_context}' if scenario_context else ''}"""
