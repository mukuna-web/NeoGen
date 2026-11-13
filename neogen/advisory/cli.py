"""
CLI interface for the NeoGen Advisory Board.

Usage:
    python -m neogen.advisory "How do I improve clearance prediction?"
    python -m neogen.advisory --review "Add pKa features to bioavailability model"
    python -m neogen.advisory --agent pk_lead "What's wrong with our IVIVE scaling?"
    python -m neogen.advisory --list-departments
    python -m neogen.advisory --list-agents
    python -m neogen.advisory --interactive
"""

import argparse
import sys

from .orchestrator import AdvisoryBoard


def print_header():
    print()
    print("=" * 60)
    print("  NeoGen Advisory Board")
    print("  Virtual Biotech Research Organization")
    print("  8 Departments | 30 Expert Agents | 3-Layer Review")
    print("=" * 60)
    print()


def print_consultation_result(result):
    print(f"\n{'='*60}")
    print("ADVISORY BOARD RESPONSE")
    print(f"{'='*60}")
    print(f"\nDepartments consulted: {', '.join(result.departments_consulted)}")
    print(f"Agents consulted: {len(result.responses)}")
    print()

    if len(result.responses) > 1:
        # Show synthesis first
        print("SYNTHESIZED RECOMMENDATION:")
        print("-" * 40)
        print(result.synthesis)

        print(f"\n{'─'*40}")
        print("INDIVIDUAL EXPERT OPINIONS:")
        print("─" * 40)
        for r in result.responses:
            print(f"\n  {r['agent_name']} — {r['title']} ({r['department']})")
            print(f"  {'·'*40}")
            # Indent each line
            for line in r["response"].split("\n"):
                print(f"  {line}")
    else:
        print(result.synthesis)

    stats = result.token_usage
    print(f"\n{'─'*40}")
    print(f"Cost: ${result.total_cost:.4f} | Tokens: {stats['input']}in / {stats['output']}out")


def interactive_mode(board: AdvisoryBoard):
    """Interactive consultation mode."""
    print_header()
    print("Interactive mode — type your questions, 'quit' to exit.\n")
    print("Commands:")
    print("  /departments  — List all departments")
    print("  /agents       — List all agents")
    print("  /agent <id>   — Talk to a specific agent")
    print("  /review       — Start a proposal review")
    print("  /stats        — Show API usage stats")
    print("  /modules      — List NeoGen modules")
    print("  quit          — Exit")
    print()

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        if query == "/departments":
            for d in board.list_departments():
                print(f"  {d['id']:25s} {d['name']:35s} ({d['agent_count']} agents)")
            continue

        if query == "/agents":
            for a in board.list_agents():
                print(f"  {a['id']:25s} {a['name']:20s} {a['title']}")
            continue

        if query.startswith("/agent "):
            parts = query.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /agent <agent_id> <question>")
                continue
            rest = parts[1]
            agent_parts = rest.split(maxsplit=1)
            agent_id = agent_parts[0]
            if len(agent_parts) > 1:
                agent_query = agent_parts[1]
            else:
                agent_query = input(f"  Question for {agent_id}: ").strip()

            result = board.ask_agent(agent_id, agent_query)
            if "error" in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"\n  {result['agent_name']} ({result['title']}):")
                print(f"  {'─'*40}")
                for line in result["response"].split("\n"):
                    print(f"  {line}")
                print(f"\n  Cost: ${result['cost']:.4f}")
            continue

        if query == "/review":
            title = input("  Proposal title: ").strip()
            description = input("  Description: ").strip()
            modules_str = input("  Affected modules (comma-separated, or Enter to auto-detect): ").strip()
            modules = [m.strip() for m in modules_str.split(",") if m.strip()] if modules_str else None
            risk = input("  Risk level (low/medium/high/critical) [medium]: ").strip() or "medium"

            result = board.review(title, description, modules, risk)
            print(f"\nFinal Verdict: {result.final_verdict}")
            print(f"Reasoning:\n{result.final_reasoning}")
            continue

        if query == "/stats":
            stats = board.get_session_stats()
            print(f"  Calls: {stats['total_calls']}")
            print(f"  Input tokens: {stats['total_input_tokens']:,}")
            print(f"  Output tokens: {stats['total_output_tokens']:,}")
            print(f"  Total cost: ${stats['total_cost_usd']:.4f}")
            continue

        if query == "/modules":
            for m in board.list_modules():
                print(f"  {m['package']:12s} {m['name']:35s} {m['lines']:5d} lines  {m['description'][:50]}")
            continue

        # Regular consultation
        result = board.consult(query)
        print_consultation_result(result)


def main():
    parser = argparse.ArgumentParser(
        description="NeoGen Advisory Board — Virtual Biotech Organization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m neogen.advisory "How do I improve clearance prediction?"
  python -m neogen.advisory --review "Add pKa features to bioavailability"
  python -m neogen.advisory --agent pk_lead "What's wrong with our IVIVE?"
  python -m neogen.advisory --interactive
  python -m neogen.advisory --list-departments
        """,
    )

    parser.add_argument("query", nargs="?", help="Question to ask the advisory board")
    parser.add_argument("--review", action="store_true", help="Run as a proposal review")
    parser.add_argument("--agent", type=str, help="Ask a specific agent by ID")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--list-departments", action="store_true", help="List departments")
    parser.add_argument("--list-agents", action="store_true", help="List agents")
    parser.add_argument("--list-modules", action="store_true", help="List NeoGen modules")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                        help="Claude model to use (default: sonnet)")
    parser.add_argument("--max-agents", type=int, default=3,
                        help="Max agents to consult (controls cost)")

    args = parser.parse_args()

    board = AdvisoryBoard(model=args.model)

    if args.list_departments:
        print_header()
        for d in board.list_departments():
            print(f"  {d['id']:25s} {d['name']:35s} ({d['agent_count']} agents)")
        return

    if args.list_agents:
        print_header()
        for a in board.list_agents():
            print(f"  {a['id']:25s} {a['name']:20s} {a['title']}")
        return

    if args.list_modules:
        for m in board.list_modules():
            print(f"  {m['package']:12s} {m['name']:35s} {m['lines']:5d} lines  {m['description'][:50]}")
        return

    if args.interactive:
        interactive_mode(board)
        return

    if not args.query:
        parser.print_help()
        return

    print_header()

    if args.agent:
        result = board.ask_agent(args.agent, args.query)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"{result['agent_name']} ({result['title']}):")
            print("-" * 40)
            print(result["response"])
            print(f"\nCost: ${result['cost']:.4f}")

    elif args.review:
        result = board.review(args.query, args.query)
        print(f"\nFinal Verdict: {result.final_verdict}")

    else:
        result = board.consult(args.query, max_agents=args.max_agents)
        print_consultation_result(result)


if __name__ == "__main__":
    main()
