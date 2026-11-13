"""
Evolution / Breeding Engine — selection, crossover, mutation, lineage tracking.

Implements:
  - Tournament selection
  - Elitism (top N always survive)
  - Crossover breeding
  - Mutation
  - Diversity pressure (bonus for genomic diversity)
  - Safeguards against premature convergence and groupthink
"""

import random
import logging

from ..models.agent import ScientistAgent
from ..models.fitness import FitnessScore
from ..models.evolution import Generation, Lineage, EvolutionHistory
from ..config import SwarmConfig
from .genesis import GenesisEngine

logger = logging.getLogger(__name__)


class EvolutionEngine:
    """Manages the evolutionary cycle: selection, breeding, mutation."""

    def __init__(self, config: SwarmConfig, genesis: GenesisEngine):
        self.config = config
        self.genesis = genesis
        self.rng = random.Random(config.seed)
        self.history = EvolutionHistory()
        self.lineages: dict[str, Lineage] = {}

    def evolve_generation(self, agents: list[ScientistAgent],
                          scores: list[FitnessScore],
                          gen_number: int) -> tuple[list[ScientistAgent], Generation]:
        """Run one evolutionary cycle: select, breed, mutate.

        Returns:
            (new_population, generation_record)
        """
        target_size = self.config.population_size
        score_map = {s.agent_id: s for s in scores}

        # Apply diversity bonus
        self._apply_diversity_bonus(agents, scores)

        # Sort by fitness
        ranked = sorted(agents, key=lambda a: score_map.get(a.agent_id, FitnessScore()).weighted_total, reverse=True)

        # Record generation
        gen = Generation(
            generation_number=gen_number,
            agents=list(agents),
            fitness_scores=list(scores),
            best_agent_id=ranked[0].agent_id if ranked else "",
            best_fitness=score_map.get(ranked[0].agent_id, FitnessScore()).weighted_total if ranked else 0,
            avg_fitness=sum(s.weighted_total for s in scores) / max(len(scores), 1),
            diversity_score=self._compute_diversity(agents),
        )

        # --- SELECTION ---
        num_survivors = max(self.config.elitism_count, int(len(agents) * self.config.survival_rate))

        # Elite always survive
        elites = ranked[:self.config.elitism_count]
        survivors = list(elites)

        # Tournament selection for remaining slots
        remaining = ranked[self.config.elitism_count:]
        while len(survivors) < num_survivors and remaining:
            winner = self._tournament_select(remaining, score_map)
            if winner not in survivors:
                survivors.append(winner)
            remaining = [a for a in remaining if a.agent_id != winner.agent_id]

        # Mark eliminated agents
        survivor_ids = {a.agent_id for a in survivors}
        for agent in agents:
            if agent.agent_id not in survivor_ids:
                agent.alive = False
                agent.eliminated_at_generation = gen_number
                gen.eliminated.append(agent.agent_id)
            else:
                gen.survivors.append(agent.agent_id)

        # --- BREEDING ---
        new_population = list(survivors)
        next_gen = gen_number + 1

        while len(new_population) < target_size:
            if self.rng.random() < self.config.crossover_rate and len(survivors) >= 2:
                # Sexual reproduction: pick two parents
                parent_a = self._tournament_select(survivors, score_map)
                parent_b = self._tournament_select(
                    [s for s in survivors if s.agent_id != parent_a.agent_id],
                    score_map
                )
                child = self.genesis.breed(parent_a, parent_b, next_gen)
            else:
                # Asexual reproduction with higher mutation
                parent = self._tournament_select(survivors, score_map)
                child = self.genesis.mutant(parent, next_gen)

            new_population.append(child)
            gen.offspring.append(child.agent_id)

            # Track lineage
            self._update_lineage(child)

        # Update fitness history for survivors
        for agent in new_population:
            if agent.agent_id in score_map:
                agent.fitness_history.append(score_map[agent.agent_id].weighted_total)
                agent.total_fitness = sum(agent.fitness_history) / len(agent.fitness_history)

        logger.info(
            "Gen %d: best=%.3f, avg=%.3f, diversity=%.3f, survived=%d, bred=%d",
            gen_number, gen.best_fitness, gen.avg_fitness, gen.diversity_score,
            len(gen.survivors), len(gen.offspring),
        )

        self.history.generations.append(gen)
        return new_population, gen

    def _tournament_select(self, candidates: list[ScientistAgent],
                           score_map: dict, tournament_size: int = 3) -> ScientistAgent:
        """Tournament selection: pick random subset, return the fittest."""
        if len(candidates) <= tournament_size:
            tournament = list(candidates)
        else:
            tournament = self.rng.sample(candidates, tournament_size)

        return max(tournament, key=lambda a: score_map.get(a.agent_id, FitnessScore()).weighted_total)

    def _apply_diversity_bonus(self, agents: list[ScientistAgent],
                               scores: list[FitnessScore]) -> None:
        """Add diversity pressure: reward agents with unique genomes."""
        if self.config.diversity_pressure <= 0:
            return

        score_map = {s.agent_id: s for s in scores}
        for agent in agents:
            # Average distance to all other agents
            distances = []
            for other in agents:
                if other.agent_id != agent.agent_id:
                    dist = agent.genome.traits.distance(other.genome.traits)
                    distances.append(dist)
            if distances:
                avg_distance = sum(distances) / len(distances)
                bonus = avg_distance * self.config.diversity_pressure
                if agent.agent_id in score_map:
                    score_map[agent.agent_id].weighted_total += bonus

    def _compute_diversity(self, agents: list[ScientistAgent]) -> float:
        """Compute average pairwise genome distance (diversity metric)."""
        if len(agents) < 2:
            return 0.0
        total = 0.0
        count = 0
        for i, a in enumerate(agents):
            for b in agents[i + 1:]:
                total += a.genome.traits.distance(b.genome.traits)
                count += 1
        return total / max(count, 1)

    def _update_lineage(self, agent: ScientistAgent) -> None:
        """Track agent in its lineage."""
        lid = agent.lineage_id
        if lid not in self.lineages:
            self.lineages[lid] = Lineage(
                lineage_id=lid,
                founder_id=agent.agent_id,
                founder_role=agent.role,
            )
        lineage = self.lineages[lid]
        lineage.current_members.append(agent.agent_id)
        lineage.all_members.append(agent.agent_id)
        lineage.generations_survived = agent.generation + 1
