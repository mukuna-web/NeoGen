"""
Global configuration for the Evolutionary Scientist Swarm.
"""

from dataclasses import dataclass, field


@dataclass
class SwarmConfig:
    """Master configuration for an evolutionary swarm run."""

    # Population
    population_size: int = 12
    cluster_size: int = 3
    num_clusters: int = 4

    # Evolution
    num_generations: int = 5
    survival_rate: float = 0.4          # Top 40% survive
    mutation_rate: float = 0.15         # 15% trait mutation
    crossover_rate: float = 0.6         # 60% of offspring via crossover
    elitism_count: int = 2              # Top 2 always survive
    diversity_pressure: float = 0.1     # Bonus for diverse genomes

    # Debate
    debate_rounds: int = 3              # Rounds of claim → critique → defense
    max_hypotheses_per_agent: int = 2

    # LLM
    model: str = "claude-sonnet-4-20250514"
    max_tokens_per_call: int = 1500
    temperature: float = 0.7

    # Fitness weights (must sum to 1.0)
    fitness_weights: dict = field(default_factory=lambda: {
        "factual_accuracy": 0.15,
        "predictive_accuracy": 0.15,
        "mathematical_consistency": 0.15,
        "biological_plausibility": 0.15,
        "explanatory_usefulness": 0.10,
        "falsification_drive": 0.10,
        "collaboration_quality": 0.10,
        "novelty": 0.10,
    })

    # Reproducibility
    seed: int = 42
    run_id: str = ""

    # Output
    output_dir: str = "swarm_runs"
    verbose: bool = True
