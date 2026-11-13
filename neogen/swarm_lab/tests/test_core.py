"""
Core tests for the swarm — genome operations, agent creation, fitness scoring.
"""

import json
from neogen.swarm_lab.models.agent import TraitVector, AgentGenome, ScientistAgent, TRAIT_NAMES
from neogen.swarm_lab.models.fitness import FitnessScore
from neogen.swarm_lab.engines.genesis import GenesisEngine
from neogen.swarm_lab.engines.scenario_loader import load_scenario
from neogen.swarm_lab.config import SwarmConfig


def test_trait_vector_random():
    t = TraitVector.random(seed=42)
    assert len(t.traits) == len(TRAIT_NAMES)
    assert all(0.0 <= v <= 1.0 for v in t.traits.values())


def test_trait_vector_crossover():
    a = TraitVector.random(seed=1)
    b = TraitVector.random(seed=2)
    child = TraitVector.crossover(a, b, seed=3)
    assert len(child.traits) == len(TRAIT_NAMES)
    # Child should differ from both parents
    assert child.traits != a.traits
    assert child.traits != b.traits


def test_trait_vector_mutation():
    original = TraitVector.random(seed=42)
    mutated = original.mutate(rate=1.0, magnitude=0.3, seed=99)
    # With 100% rate, at least some traits should change
    changed = sum(1 for k in TRAIT_NAMES if original.traits[k] != mutated.traits[k])
    assert changed > 0
    # All values still in range
    assert all(0.0 <= v <= 1.0 for v in mutated.traits.values())


def test_trait_distance():
    a = TraitVector.random(seed=1)
    b = TraitVector.random(seed=2)
    dist = a.distance(b)
    assert dist > 0
    assert a.distance(a) == 0.0


def test_personality_description():
    t = TraitVector(traits={name: 0.8 for name in TRAIT_NAMES})
    desc = t.to_personality_description()
    assert isinstance(desc, str)
    assert len(desc) > 10


def test_genesis_spawn():
    config = SwarmConfig(population_size=4, seed=42)
    genesis = GenesisEngine(config)
    roles = [
        {"role": "modeler", "specialty": "math", "expertise_areas": ["ODE"]},
        {"role": "physiologist", "specialty": "cardiac"},
        {"role": "skeptic", "specialty": "falsification"},
        {"role": "biophysicist", "specialty": "mechanics"},
    ]
    agents = genesis.spawn_initial_population(roles)
    assert len(agents) == 4
    assert all(a.generation == 0 for a in agents)
    assert all(a.name.startswith("Dr.") for a in agents)
    # Each should have unique genome
    ids = {a.genome.genome_id for a in agents}
    assert len(ids) == 4


def test_genesis_breed():
    config = SwarmConfig(seed=42)
    genesis = GenesisEngine(config)
    roles = [{"role": "A"}, {"role": "B"}]
    parents = genesis.spawn_initial_population(roles)
    child = genesis.breed(parents[0], parents[1], generation=1)
    assert child.generation == 1
    assert len(child.parent_ids) == 2
    assert child.genome.parent_a_id is not None


def test_fitness_score_weighted():
    score = FitnessScore(
        factual_accuracy=0.8,
        predictive_accuracy=0.6,
        mathematical_consistency=0.7,
        biological_plausibility=0.9,
    )
    weights = {
        "factual_accuracy": 0.25,
        "predictive_accuracy": 0.25,
        "mathematical_consistency": 0.25,
        "biological_plausibility": 0.25,
    }
    total = score.compute_weighted(weights)
    expected = (0.8 + 0.6 + 0.7 + 0.9) * 0.25
    assert abs(total - expected) < 0.001


def test_scenario_load():
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scenarios", "heart_modeling.yaml")
    scenario = load_scenario(path)
    assert scenario.scenario_id == "heart_modeling_v1"
    assert scenario.domain == "cardiac_physiology"
    assert len(scenario.known_equations) > 0
    assert len(scenario.ground_truth) > 0
    assert len(scenario.suggested_specialties) >= 10


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
