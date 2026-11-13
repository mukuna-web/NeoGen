"""
Swarm Orchestrator — the main evolutionary loop.

Ties together all engines into the core cycle:
  spawn → cluster → debate → validate → score → select → breed → repeat
"""

import uuid
import random
import time
import json
import os
import logging
from dataclasses import asdict

from .config import SwarmConfig
from .models.scenario import Scenario
from .models.agent import ScientistAgent
from .models.debate import DebateTrace
from .models.fitness import FitnessScore
from .models.evolution import Generation
from .models.hypothesis import Hypothesis
from .llm.client import SwarmLLMClient
from .engines.genesis import GenesisEngine
from .engines.roles import RoleGenerator
from .engines.debate import DebateEngine
from .engines.validation import ValidationEngine
from .engines.fitness import FitnessEngine
from .engines.evolution import EvolutionEngine
from .engines.memory import MemoryStore, MemoryEntry
from .outputs.report import ReportEngine

logger = logging.getLogger(__name__)


class SwarmOrchestrator:
    """Orchestrates the full evolutionary scientist swarm cycle."""

    def __init__(self, config: SwarmConfig = None):
        self.config = config or SwarmConfig()

        if not self.config.run_id:
            self.config.run_id = f"run_{uuid.uuid4().hex[:8]}"

        # Initialize engines
        self.llm = SwarmLLMClient(model=self.config.model)
        self.genesis = GenesisEngine(self.config)
        self.role_gen = RoleGenerator(self.llm)
        self.debate = DebateEngine(self.llm, num_rounds=self.config.debate_rounds)
        self.validation = ValidationEngine(self.llm)
        self.fitness = FitnessEngine(self.config)
        self.evolution = EvolutionEngine(self.config, self.genesis)
        self.memory = MemoryStore(self.config.output_dir)
        self.report = ReportEngine(self.llm)

        # State
        self.population: list[ScientistAgent] = []
        self.all_debate_traces: list[DebateTrace] = []
        self.all_hypotheses: list[Hypothesis] = []
        self.generation_results: list[dict] = []

    def run(self, scenario: Scenario) -> dict:
        """Run the full evolutionary swarm on a scientific scenario.

        Returns a comprehensive results dict.
        """
        start_time = time.time()

        print(f"\n{'='*70}")
        print(f"  NEOGEN EVOLUTIONARY SCIENTIST SWARM")
        print(f"  Run ID: {self.config.run_id}")
        print(f"  Scenario: {scenario.title}")
        print(f"  Population: {self.config.population_size} | Generations: {self.config.num_generations}")
        print(f"  Model: {self.config.model}")
        print(f"{'='*70}\n")

        # === PHASE 1: ROLE GENERATION ===
        print("PHASE 1: Generating specialist roles...")
        roles = self.role_gen.generate_roles(scenario, num_roles=self.config.population_size)
        for i, role in enumerate(roles):
            print(f"  [{i+1}] {role['role']}")

        # === PHASE 2: SPAWN INITIAL POPULATION ===
        print(f"\nPHASE 2: Spawning {self.config.population_size} scientist agents...")
        self.population = self.genesis.spawn_initial_population(roles)
        for agent in self.population:
            dominant = agent.genome.traits.dominant_traits(0.7)
            print(f"  {agent.name}: {agent.role} | Traits: {', '.join(dominant[:3]) or 'balanced'}")

        # === PHASE 3: EVOLUTIONARY LOOP ===
        for gen in range(self.config.num_generations):
            print(f"\n{'─'*70}")
            print(f"GENERATION {gen + 1}/{self.config.num_generations}")
            print(f"{'─'*70}")

            gen_result = self._run_generation(gen, scenario)
            self.generation_results.append(gen_result)

            # Check for convergence (low diversity)
            if gen_result.get("diversity", 1.0) < 0.1:
                print("  WARNING: Diversity collapsed — injecting random mutants")
                self._inject_diversity(gen + 1, roles)

        # === PHASE 4: FINAL REPORTING ===
        print(f"\n{'='*70}")
        print("PHASE 4: Generating scientific report...")
        print(f"{'='*70}\n")

        elapsed = time.time() - start_time
        api_stats = self.llm.get_stats()

        # Generate final report
        report = self.report.generate_final_report(
            scenario=scenario,
            generations=self.generation_results,
            debate_traces=self.all_debate_traces,
            hypotheses=self.all_hypotheses,
            evolution_history=self.evolution.history,
            api_stats=api_stats,
            elapsed_time=elapsed,
        )

        # Save outputs
        self._save_outputs(scenario, report)

        print(f"\nRun complete in {elapsed:.1f}s")
        print(f"API calls: {api_stats['total_calls']} | Cost: ${api_stats['total_cost_usd']:.4f}")

        return report

    def _run_generation(self, gen_number: int, scenario: Scenario) -> dict:
        """Run one generation: cluster → debate → validate → score → evolve."""

        # Step 1: Form clusters
        clusters = self._form_clusters(self.population)
        print(f"  Formed {len(clusters)} debate clusters")

        # Step 2: Run debates in each cluster
        all_claims = []
        gen_traces = []
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i+1}: {', '.join(a.name for a in cluster)}")
            trace = self.debate.run_cluster_debate(cluster, scenario)
            gen_traces.append(trace)
            self.all_debate_traces.append(trace)

            # Collect all surviving claims
            for round_ in trace.rounds:
                all_claims.extend(round_.best_claims)

        # Step 3: Validate claims
        print(f"  Validating {len(all_claims)} claims...")
        agent_validations: dict[str, list[dict]] = {}
        for claim in all_claims:
            result = self.validation.validate_claim(claim, scenario)
            agent_validations.setdefault(claim.agent_id, []).append(result)

            # Store to memory
            self.memory.add_generation_memory(gen_number, MemoryEntry(
                entry_type="hypothesis",
                content=claim.statement,
                source_agent=claim.agent_id,
                source_generation=gen_number,
                fitness_score=result["aggregate"].get("factual_accuracy", 0.5),
                tags=list(claim.variables)[:3] if isinstance(claim.variables, list) else list(claim.variables.keys())[:3] if isinstance(claim.variables, dict) else [],
            ))

        # Step 4: Score agents
        scores = []
        for agent in self.population:
            validations = agent_validations.get(agent.agent_id, [])
            agent_traces = [t for t in gen_traces if agent.agent_id in t.agent_ids]
            score = self.fitness.score_agent(agent, validations, agent_traces, gen_number)
            scores.append(score)

        ranked = self.fitness.rank_population(scores)
        best = ranked[0] if ranked else None

        print(f"  Best agent: {best.agent_id} (fitness={best.weighted_total:.3f})" if best else "  No scores")
        print(f"  Score breakdown: {json.dumps({k: round(v, 2) for k, v in best.to_dict().items()})}" if best else "")

        # Step 5: Promote best to global memory
        self.memory.promote_best_to_global(gen_number)

        # Step 6: Evolve
        if gen_number < self.config.num_generations - 1:
            self.population, gen_record = self.evolution.evolve_generation(
                self.population, scores, gen_number
            )
        else:
            gen_record = Generation(generation_number=gen_number, agents=self.population, fitness_scores=scores)

        # Collect hypotheses from this generation
        for claim in all_claims:
            hyp = Hypothesis(
                hypothesis_id=claim.claim_id,
                title=claim.statement[:80],
                statement=claim.statement,
                testable_predictions=claim.testable_predictions,
                generation=gen_number,
                confidence=claim.confidence,
                proposed_by=[claim.agent_id],
            )
            self.all_hypotheses.append(hyp)

        return {
            "generation": gen_number,
            "best_fitness": best.weighted_total if best else 0,
            "avg_fitness": sum(s.weighted_total for s in scores) / max(len(scores), 1),
            "diversity": self.evolution._compute_diversity(self.population) if hasattr(self.evolution, '_compute_diversity') else 0,
            "claims_produced": len(all_claims),
            "agents_alive": len([a for a in self.population if a.alive]),
            "best_agent": best.agent_id if best else "",
            "top_scores": {s.agent_id: round(s.weighted_total, 3) for s in ranked[:3]},
        }

    def _form_clusters(self, agents: list[ScientistAgent]) -> list[list[ScientistAgent]]:
        """Form debate clusters with diverse agents."""
        rng = random.Random(self.config.seed)
        shuffled = list(agents)
        rng.shuffle(shuffled)

        cluster_size = self.config.cluster_size
        clusters = []
        for i in range(0, len(shuffled), cluster_size):
            cluster = shuffled[i:i + cluster_size]
            if len(cluster) >= 2:  # Minimum 2 for debate
                clusters.append(cluster)

        return clusters

    def _inject_diversity(self, generation: int, roles: list[dict]) -> None:
        """Inject random new agents to prevent convergence collapse."""
        rng = random.Random(self.config.seed + generation * 100)
        num_inject = max(2, self.config.population_size // 4)

        for _ in range(num_inject):
            role = rng.choice(roles)
            new_agents = self.genesis.spawn_initial_population([role])
            if new_agents:
                new_agent = new_agents[0]
                new_agent.generation = generation
                # Replace worst agent
                if self.population:
                    worst = min(self.population, key=lambda a: a.total_fitness)
                    self.population.remove(worst)
                    self.population.append(new_agent)

    def _save_outputs(self, scenario: Scenario, report: dict) -> None:
        """Save all outputs to disk."""
        out_dir = os.path.join(self.config.output_dir, self.config.run_id)
        os.makedirs(out_dir, exist_ok=True)

        # Save report
        with open(os.path.join(out_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Save memory
        self.memory.save(os.path.join(out_dir, "memory.json"))

        # Save config
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        # Save generation results
        with open(os.path.join(out_dir, "generations.json"), "w") as f:
            json.dump(self.generation_results, f, indent=2, default=str)

        print(f"  Outputs saved to: {out_dir}/")
