"""
Scientific Report Engine — generates high-value research outputs from swarm runs.

Produces:
  - Ranked hypotheses
  - Best mathematical model
  - Research memo
  - Lineage evolution summary
  - Unresolved questions
  - Next experiments to run
"""

import logging
from ..llm.client import SwarmLLMClient

logger = logging.getLogger(__name__)

REPORT_PROMPT = """You are a scientific report writer for an evolutionary AI research swarm. Synthesize the results of {num_generations} generations of scientific debate into a comprehensive research memo.

SCENARIO: {scenario_title}
QUESTION: {scenario_question}
DOMAIN: {scenario_domain}

GENERATION RESULTS:
{generation_summaries}

TOP HYPOTHESES THAT SURVIVED EVOLUTION:
{top_hypotheses}

DEBATE HIGHLIGHTS:
{debate_highlights}

Write a research memo with these sections:

1. EXECUTIVE SUMMARY (3-5 sentences)
2. RANKED HYPOTHESES (numbered, with confidence scores)
3. BEST MATHEMATICAL MODEL (equations, variables, assumptions)
4. MECHANISTIC EXPLANATION (how the system works based on the debate)
5. DISAGREEMENT MAP (where agents disagreed and why)
6. REJECTED HYPOTHESES (what was tried and failed, and why)
7. LINEAGE EVOLUTION (which scientific approaches survived and thrived)
8. UNRESOLVED QUESTIONS (what remains unknown)
9. RECOMMENDED NEXT EXPERIMENTS (concrete next steps)
10. CONFIDENCE ASSESSMENT (how much to trust these results)

Write for a biotech founder / scientific lead. Be rigorous but readable."""


class ReportEngine:
    """Generates scientific reports from swarm evolution results."""

    def __init__(self, llm: SwarmLLMClient):
        self.llm = llm

    def generate_final_report(self, scenario, generations, debate_traces,
                               hypotheses, evolution_history, api_stats,
                               elapsed_time) -> dict:
        """Generate comprehensive final report."""

        # Summarize generations
        gen_summaries = []
        for g in generations:
            gen_summaries.append(
                f"Gen {g['generation']+1}: best_fitness={g['best_fitness']:.3f}, "
                f"avg={g['avg_fitness']:.3f}, claims={g['claims_produced']}, "
                f"diversity={g.get('diversity', 0):.3f}"
            )

        # Top hypotheses
        sorted_hyps = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)
        top_hyps = []
        for h in sorted_hyps[:10]:
            top_hyps.append(f"- [{h.confidence:.0%}] {h.statement}")

        # Debate highlights
        highlights = []
        for trace in debate_traces[-3:]:  # Last 3 debates
            for round_ in trace.rounds:
                for claim in round_.best_claims:
                    highlights.append(f"SURVIVED: {claim.statement[:150]} (by {claim.agent_name})")
                for claim in round_.rejected_claims[:2]:
                    highlights.append(f"REJECTED: {claim.statement[:150]} (by {claim.agent_name})")

        # Generate report via LLM
        prompt = REPORT_PROMPT.format(
            num_generations=len(generations),
            scenario_title=scenario.title,
            scenario_question=scenario.question,
            scenario_domain=scenario.domain,
            generation_summaries="\n".join(gen_summaries),
            top_hypotheses="\n".join(top_hyps[:10]),
            debate_highlights="\n".join(highlights[:15]),
        )

        response = self.llm.call(
            system_prompt="You are a scientific report writer. Be thorough and rigorous.",
            user_message=prompt,
            max_tokens=3000,
        )

        report = {
            "run_id": "",
            "scenario": {
                "title": scenario.title,
                "question": scenario.question,
                "domain": scenario.domain,
            },
            "summary": {
                "generations_run": len(generations),
                "total_hypotheses": len(hypotheses),
                "total_debates": len(debate_traces),
                "elapsed_seconds": round(elapsed_time, 1),
                "api_stats": api_stats,
            },
            "report_text": response.text,
            "ranked_hypotheses": [
                {"rank": i+1, "statement": h.statement, "confidence": h.confidence,
                 "generation": h.generation}
                for i, h in enumerate(sorted_hyps[:10])
            ],
            "generation_fitness_curve": [
                {"gen": g["generation"]+1, "best": g["best_fitness"], "avg": g["avg_fitness"]}
                for g in generations
            ],
        }

        return report
