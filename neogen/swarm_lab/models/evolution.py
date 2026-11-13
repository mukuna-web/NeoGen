"""
Evolution tracking models — lineages, generations, and history.
"""

from dataclasses import dataclass, field
from typing import Optional

from .agent import ScientistAgent
from .fitness import FitnessScore, LineageFitness
from .hypothesis import Hypothesis


@dataclass
class Lineage:
    """A family line of agents across generations."""
    lineage_id: str = ""
    founder_id: str = ""                # Original ancestor agent_id
    founder_role: str = ""
    current_members: list[str] = field(default_factory=list)   # agent_ids
    all_members: list[str] = field(default_factory=list)       # all-time
    generations_survived: int = 0
    fitness: LineageFitness = field(default_factory=LineageFitness)


@dataclass
class Generation:
    """Snapshot of one generation of the swarm."""
    generation_number: int = 0
    agents: list[ScientistAgent] = field(default_factory=list)
    fitness_scores: list[FitnessScore] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    best_agent_id: str = ""
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    diversity_score: float = 0.0        # Average pairwise genome distance
    survivors: list[str] = field(default_factory=list)        # agent_ids
    eliminated: list[str] = field(default_factory=list)       # agent_ids
    offspring: list[str] = field(default_factory=list)        # new agent_ids


@dataclass
class EvolutionHistory:
    """Complete history of an evolutionary run."""
    run_id: str = ""
    scenario_id: str = ""
    generations: list[Generation] = field(default_factory=list)
    lineages: dict[str, Lineage] = field(default_factory=dict)
    best_hypotheses: list[Hypothesis] = field(default_factory=list)
    convergence_generation: Optional[int] = None  # When diversity dropped below threshold
    total_agents_created: int = 0
    total_agents_eliminated: int = 0
