"""
Fitness scoring models — multi-objective scoring for agents and lineages.
"""

from dataclasses import dataclass, field


@dataclass
class FitnessScore:
    """Multi-objective fitness score for a single agent in one generation."""
    agent_id: str = ""
    generation: int = 0

    # Component scores (0.0 to 1.0)
    factual_accuracy: float = 0.0
    predictive_accuracy: float = 0.0
    mathematical_consistency: float = 0.0
    biological_plausibility: float = 0.0
    explanatory_usefulness: float = 0.0
    falsification_drive: float = 0.0
    collaboration_quality: float = 0.0
    novelty: float = 0.0

    # Computed
    weighted_total: float = 0.0
    rank_in_generation: int = 0

    def compute_weighted(self, weights: dict[str, float]) -> float:
        """Compute weighted total fitness from component scores."""
        self.weighted_total = sum(
            getattr(self, key, 0.0) * weight
            for key, weight in weights.items()
        )
        return self.weighted_total

    def to_dict(self) -> dict:
        """Return all component scores as a dict."""
        return {
            "factual_accuracy": self.factual_accuracy,
            "predictive_accuracy": self.predictive_accuracy,
            "mathematical_consistency": self.mathematical_consistency,
            "biological_plausibility": self.biological_plausibility,
            "explanatory_usefulness": self.explanatory_usefulness,
            "falsification_drive": self.falsification_drive,
            "collaboration_quality": self.collaboration_quality,
            "novelty": self.novelty,
            "weighted_total": self.weighted_total,
        }


@dataclass
class LineageFitness:
    """Aggregated fitness across a lineage's history."""
    lineage_id: str = ""
    generations_survived: int = 0
    avg_fitness: float = 0.0
    peak_fitness: float = 0.0
    fitness_trend: list[float] = field(default_factory=list)  # Per-generation averages
    total_offspring: int = 0
    notable_contributions: list[str] = field(default_factory=list)
