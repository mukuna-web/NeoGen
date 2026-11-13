"""
CLI entry point: python -m neogen.swarm_lab

Usage:
    python -m neogen.swarm_lab run scenarios/heart_modeling.yaml
    python -m neogen.swarm_lab run scenarios/heart_modeling.yaml --generations 3 --population 8
    python -m neogen.swarm_lab list-scenarios
    python -m neogen.swarm_lab demo
"""

import argparse
import logging
import os
import sys

from .config import SwarmConfig
from .orchestrator import SwarmOrchestrator
from .engines.scenario_loader import load_scenario, list_scenarios


def main():
    parser = argparse.ArgumentParser(
        description="NeoGen Evolutionary Scientist Swarm (Mirofish Lab)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m neogen.swarm_lab demo
  python -m neogen.swarm_lab run neogen/swarm_lab/scenarios/heart_modeling.yaml
  python -m neogen.swarm_lab run scenario.yaml --generations 5 --population 12
  python -m neogen.swarm_lab list-scenarios
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a scenario")
    run_parser.add_argument("scenario", help="Path to scenario YAML file")
    run_parser.add_argument("--generations", "-g", type=int, default=3, help="Number of generations")
    run_parser.add_argument("--population", "-p", type=int, default=8, help="Population size")
    run_parser.add_argument("--cluster-size", type=int, default=3, help="Agents per debate cluster")
    run_parser.add_argument("--debate-rounds", type=int, default=2, help="Debate rounds per cluster")
    run_parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="LLM model")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    run_parser.add_argument("--output-dir", type=str, default="swarm_runs", help="Output directory")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run the heart modeling demo")
    demo_parser.add_argument("--generations", "-g", type=int, default=2, help="Generations (default 2 for demo)")
    demo_parser.add_argument("--population", "-p", type=int, default=6, help="Population (default 6 for demo)")
    demo_parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")

    # List scenarios
    subparsers.add_parser("list-scenarios", help="List available scenarios")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Setup logging
    log_level = logging.DEBUG if getattr(args, 'verbose', False) else logging.INFO
    logging.basicConfig(level=log_level, format="%(name)s: %(message)s")

    if args.command == "list-scenarios":
        scenarios_dir = os.path.join(os.path.dirname(__file__), "scenarios")
        for s in list_scenarios(scenarios_dir):
            print(f"  {s['id']:25s} {s['title']:50s} [{s['difficulty']}]")
        return

    if args.command == "demo":
        scenario_path = os.path.join(os.path.dirname(__file__), "scenarios", "heart_modeling.yaml")
        config = SwarmConfig(
            population_size=args.population,
            num_generations=args.generations,
            cluster_size=3,
            debate_rounds=2,
            model=args.model,
            seed=42,
        )
    elif args.command == "run":
        scenario_path = args.scenario
        config = SwarmConfig(
            population_size=args.population,
            num_generations=args.generations,
            cluster_size=args.cluster_size,
            debate_rounds=args.debate_rounds,
            model=args.model,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    else:
        parser.print_help()
        return

    # Load scenario
    scenario = load_scenario(scenario_path)
    print(f"Loaded scenario: {scenario.title}")

    # Run
    orchestrator = SwarmOrchestrator(config)
    report = orchestrator.run(scenario)

    # Print report
    if report.get("report_text"):
        print(f"\n{'='*70}")
        print("SCIENTIFIC REPORT")
        print(f"{'='*70}")
        print(report["report_text"])

    # Print fitness curve
    if report.get("generation_fitness_curve"):
        print(f"\nFitness Curve:")
        for g in report["generation_fitness_curve"]:
            bar = "█" * int(g["best"] * 40)
            print(f"  Gen {g['gen']}: best={g['best']:.3f} avg={g['avg']:.3f} {bar}")


if __name__ == "__main__":
    main()
