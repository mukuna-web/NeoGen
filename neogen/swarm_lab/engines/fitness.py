"""
Fitness Scoring Engine — multi-objective scoring for agents and lineages.
"""

import logging
from ..models.agent import ScientistAgent
from ..models.debate import DebateTrace
from ..models.fitness import FitnessScore
from ..config import SwarmConfig

logger = logging.getLogger(__name__)


class FitnessEngine:
    """Computes multi-objective fitness scores for agents."""

    def __init__(self, config: SwarmConfig):
        self.weights = config.fitness_weights

    def score_agent(self, agent: ScientistAgent, validation_results: list[dict],
                    debate_traces: list[DebateTrace],
                    generation: int) -> FitnessScore:
        """Compute fitness score for an agent based on validation and debate performance."""
        score = FitnessScore(agent_id=agent.agent_id, generation=generation)

        # --- Validation-derived scores ---
        if validation_results:
            lit_scores = [v.get("literature", {}) for v in validation_results]
            bench_scores = [v.get("benchmark", {}) for v in validation_results]
            integrity_scores = [v.get("integrity", {}) for v in validation_results]

            score.factual_accuracy = _avg([s.get("factual_accuracy", 0.5) for s in lit_scores])
            score.biological_plausibility = _avg([s.get("biological_plausibility", 0.5) for s in lit_scores])
            score.mathematical_consistency = _avg([s.get("mathematical_consistency", 0.5) for s in lit_scores])
            score.predictive_accuracy = _avg([s.get("predictive_accuracy", 0.5) for s in bench_scores if s])

            # Integrity penalty
            integrity_avg = _avg([s.get("integrity_score", 0.5) for s in integrity_scores])
            if integrity_avg < 0.3:
                # Heavy penalty for suspected fabrication
                score.factual_accuracy *= 0.5
                score.predictive_accuracy *= 0.5

        # --- Debate-derived scores ---
        if debate_traces:
            # Collaboration: did this agent's claims survive critique?
            survived = 0
            total = 0
            for trace in debate_traces:
                for round_ in trace.rounds:
                    for claim in round_.best_claims:
                        if claim.agent_id == agent.agent_id:
                            survived += 1
                    for claim in round_.rejected_claims:
                        if claim.agent_id == agent.agent_id:
                            total += 1
                    total += survived  # Total claims this agent made

            if total > 0:
                score.collaboration_quality = min(1.0, survived / max(total, 1))

            # Falsification drive: quality of critiques given
            critiques_given = len(agent.critiques_given)
            score.falsification_drive = min(1.0, critiques_given / max(len(debate_traces), 1))

        # --- Novelty score (genome-based) ---
        # Higher trait diversity → higher novelty
        trait_vals = list(agent.genome.traits.traits.values())
        if trait_vals:
            variance = _variance(trait_vals)
            score.novelty = min(1.0, variance * 4)  # Scale variance to 0-1

        # --- Explanatory usefulness (from trait vector) ---
        score.explanatory_usefulness = agent.genome.traits.traits.get("abstraction_tendency", 0.5)

        # Compute weighted total
        score.compute_weighted(self.weights)

        return score

    def rank_population(self, scores: list[FitnessScore]) -> list[FitnessScore]:
        """Rank agents by fitness within a generation."""
        sorted_scores = sorted(scores, key=lambda s: s.weighted_total, reverse=True)
        for rank, score in enumerate(sorted_scores):
            score.rank_in_generation = rank + 1
        return sorted_scores


def _avg(values: list[float]) -> float:
    if not values:
        return 0.5
    return sum(values) / len(values)


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)
