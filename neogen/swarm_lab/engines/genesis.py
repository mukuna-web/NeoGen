"""
Agent Genesis Engine — spawns scientist agents with heritable genomes.

Generates the initial population and offspring for subsequent generations.
"""

import uuid
import random
import logging
from typing import Optional

from ..models.agent import ScientistAgent, AgentGenome, TraitVector
from ..config import SwarmConfig

logger = logging.getLogger(__name__)

# Name pools for generated agents
FIRST_NAMES = [
    "Ada", "Alan", "Amir", "Anna", "Beatrice", "Carl", "Claire", "David",
    "Elena", "Felix", "Grace", "Hans", "Ingrid", "James", "Kenji", "Lena",
    "Marco", "Nadia", "Oscar", "Priya", "Ravi", "Sofia", "Thomas", "Uma",
    "Viktor", "Yuki", "Zara", "Emeka", "Fatima", "Chen",
]

LAST_NAMES = [
    "Chen", "Mueller", "Patel", "Kim", "Weber", "Okafor", "Tanaka", "Dubois",
    "Morrison", "Park", "Sharma", "Hassan", "Gonzalez", "Sterling", "Petrov",
    "Watson", "Zhang", "O'Brien", "Larsson", "Rossi", "Mansour", "Sato",
    "Al-Hassan", "Martinez", "Adeyemi", "Kowalski", "Nakamura",
]


class GenesisEngine:
    """Spawns and manages scientist agent populations."""

    def __init__(self, config: SwarmConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self._name_counter = 0

    def spawn_initial_population(self, roles: list[dict]) -> list[ScientistAgent]:
        """Spawn the generation-0 population from role definitions.

        Args:
            roles: List of role dicts with keys: role, specialty, expertise_areas,
                   knowledge_priorities, expected_contribution
        """
        agents = []
        for i, role_def in enumerate(roles):
            genome = AgentGenome(
                traits=TraitVector.random(seed=self.config.seed + i),
                generation=0,
            )
            lineage_id = f"lineage_{uuid.uuid4().hex[:6]}"

            agent = ScientistAgent(
                name=self._generate_name(),
                generation=0,
                lineage_id=lineage_id,
                role=role_def["role"],
                specialty=role_def.get("specialty", role_def["role"]),
                expertise_areas=role_def.get("expertise_areas", []),
                knowledge_priorities=role_def.get("knowledge_priorities", []),
                expected_contribution=role_def.get("expected_contribution", ""),
                genome=genome,
            )
            agents.append(agent)

        logger.info("Spawned %d agents for generation 0", len(agents))
        return agents

    def breed(self, parent_a: ScientistAgent, parent_b: ScientistAgent,
              generation: int) -> ScientistAgent:
        """Breed two parents to produce an offspring agent."""
        # Crossover genomes
        child_traits = TraitVector.crossover(
            parent_a.genome.traits,
            parent_b.genome.traits,
            seed=self.rng.randint(0, 100000),
        )

        # Mutate
        child_traits = child_traits.mutate(
            rate=self.config.mutation_rate,
            seed=self.rng.randint(0, 100000),
        )

        child_genome = AgentGenome(
            traits=child_traits,
            parent_a_id=parent_a.genome.genome_id,
            parent_b_id=parent_b.genome.genome_id,
            generation=generation,
            mutations_applied=1 if self.rng.random() < self.config.mutation_rate else 0,
        )

        # Inherit role from fitter parent with some variation
        child = ScientistAgent(
            name=self._generate_name(),
            generation=generation,
            lineage_id=parent_a.lineage_id,  # Inherit lineage from parent A
            parent_ids=[parent_a.agent_id, parent_b.agent_id],
            role=parent_a.role if parent_a.total_fitness >= parent_b.total_fitness else parent_b.role,
            specialty=parent_a.specialty if self.rng.random() < 0.6 else parent_b.specialty,
            expertise_areas=list(set(parent_a.expertise_areas[:3] + parent_b.expertise_areas[:3])),
            knowledge_priorities=parent_a.knowledge_priorities if self.rng.random() < 0.5 else parent_b.knowledge_priorities,
            expected_contribution=parent_a.expected_contribution,
            genome=child_genome,
        )

        return child

    def mutant(self, parent: ScientistAgent, generation: int) -> ScientistAgent:
        """Create a mutated clone of a single parent (asexual reproduction)."""
        mutated_traits = parent.genome.traits.mutate(
            rate=self.config.mutation_rate * 2,  # Higher mutation for clones
            magnitude=0.3,
            seed=self.rng.randint(0, 100000),
        )

        child_genome = AgentGenome(
            traits=mutated_traits,
            parent_a_id=parent.genome.genome_id,
            generation=generation,
            mutations_applied=1,
        )

        child = ScientistAgent(
            name=self._generate_name(),
            generation=generation,
            lineage_id=parent.lineage_id,
            parent_ids=[parent.agent_id],
            role=parent.role,
            specialty=parent.specialty,
            expertise_areas=parent.expertise_areas,
            knowledge_priorities=parent.knowledge_priorities,
            expected_contribution=parent.expected_contribution,
            genome=child_genome,
        )
        return child

    def _generate_name(self) -> str:
        """Generate a unique scientist name."""
        first = self.rng.choice(FIRST_NAMES)
        last = self.rng.choice(LAST_NAMES)
        self._name_counter += 1
        return f"Dr. {first} {last}"
