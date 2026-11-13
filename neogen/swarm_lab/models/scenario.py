"""
Scenario model — defines a scientific problem for the swarm to tackle.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValidationSource:
    """A source of ground truth for validating agent outputs."""
    source_type: str              # "literature" | "dataset" | "simulation" | "benchmark"
    name: str
    description: str
    data: dict = field(default_factory=dict)  # Structured reference data
    file_path: Optional[str] = None


@dataclass
class Scenario:
    """A scientific problem / experiment for the swarm."""

    scenario_id: str = ""
    title: str = ""
    domain: str = ""                          # "cardiac_physiology", "pharmacology", etc.
    question: str = ""                        # The core scientific question
    description: str = ""                     # Detailed problem description
    constraints: list[str] = field(default_factory=list)
    target_outputs: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)

    # What kinds of roles the swarm should generate
    suggested_specialties: list[str] = field(default_factory=list)
    num_specialties: int = 8

    # Validation
    validation_sources: list[ValidationSource] = field(default_factory=list)
    ground_truth: dict = field(default_factory=dict)  # Known correct answers

    # Context provided to agents
    background_knowledge: str = ""
    known_equations: list[str] = field(default_factory=list)
    known_variables: list[str] = field(default_factory=list)
    known_subsystems: list[str] = field(default_factory=list)

    # Metadata
    difficulty: str = "hard"  # "easy" | "medium" | "hard" | "research_frontier"
    tags: list[str] = field(default_factory=list)
