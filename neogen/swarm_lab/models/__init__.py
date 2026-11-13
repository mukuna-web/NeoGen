"""Data models for the Evolutionary Scientist Swarm."""

from .agent import AgentGenome, ScientistAgent, TraitVector
from .scenario import Scenario, ValidationSource
from .debate import Claim, Critique, DebateRound, DebateTurn, DebateTrace
from .hypothesis import Hypothesis, MathModel, CausalRelation
from .fitness import FitnessScore, LineageFitness
from .evolution import Lineage, Generation, EvolutionHistory
