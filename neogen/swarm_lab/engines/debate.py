"""
Debate Protocol Engine — structured scientific argumentation.

Implements two debate modes:
  1. Claim → Critique → Defense (primary)
  2. Red Team vs Blue Team (secondary)

Produces structured debate traces, not chat logs.
"""

import uuid
import logging

from ..llm.client import SwarmLLMClient
from ..models.agent import ScientistAgent
from ..models.debate import Claim, Critique, DebateTurn, DebateRound, DebateTrace
from ..models.scenario import Scenario

logger = logging.getLogger(__name__)


PROPOSAL_PROMPT = """You are participating in a scientific debate. Based on the question below, propose a specific scientific claim or model.

QUESTION: {question}

YOUR TASK: Propose a concrete claim, hypothesis, or mathematical model.

You MUST provide your response as JSON with these fields:
{{
    "statement": "Your core claim in 1-2 sentences",
    "assumptions": ["assumption 1", "assumption 2"],
    "variables": ["var1: description", "var2: description"],
    "equations": ["equation 1 in text/LaTeX form"],
    "supporting_rationale": "Why you believe this (3-5 sentences)",
    "evidence_cited": ["evidence 1", "evidence 2"],
    "confidence": 0.7,
    "testable_predictions": ["prediction 1", "prediction 2"]
}}

Be specific. Use mathematical notation where appropriate. Do NOT be vague."""


CRITIQUE_PROMPT = """You are a scientific critic reviewing a colleague's claim.

ORIGINAL CLAIM by {claimant}:
  Statement: {statement}
  Assumptions: {assumptions}
  Equations: {equations}
  Rationale: {rationale}
  Confidence: {confidence}

YOUR TASK: Critique this claim rigorously. Find weaknesses.

Respond as JSON:
{{
    "critique_type": "assumption_challenge|math_error|missing_variable|bio_implausible|insufficient_evidence",
    "critique_text": "Your detailed critique (3-5 sentences)",
    "specific_weakness": "The single most important flaw",
    "alternative_proposal": "What you would propose instead (1-2 sentences)",
    "severity": "low|medium|high|fatal",
    "confidence": 0.7
}}

Be rigorous but fair. Attack the idea, not the person."""


DEFENSE_PROMPT = """Your scientific claim has been critiqued. Defend or revise it.

YOUR ORIGINAL CLAIM: {statement}

CRITIQUE from {critic}:
  Type: {critique_type}
  Text: {critique_text}
  Weakness: {specific_weakness}
  Severity: {severity}

YOUR TASK: Either defend your claim against this critique or revise it.

Respond as JSON:
{{
    "action": "defend|revise|concede",
    "response": "Your defense or revision (3-5 sentences)",
    "revised_claim": "Your revised claim if applicable, or null",
    "revised_equations": ["revised equation if applicable"],
    "confidence_after": 0.6
}}"""


class DebateEngine:
    """Runs structured scientific debates within agent clusters."""

    def __init__(self, llm: SwarmLLMClient, num_rounds: int = 3):
        self.llm = llm
        self.num_rounds = num_rounds

    def run_cluster_debate(self, agents: list[ScientistAgent],
                           scenario: Scenario) -> DebateTrace:
        """Run a full debate within a cluster of agents."""
        trace = DebateTrace(
            cluster_id=f"cluster_{uuid.uuid4().hex[:6]}",
            agent_ids=[a.agent_id for a in agents],
            agent_names=[a.name for a in agents],
            scenario_id=scenario.scenario_id,
        )

        topic = scenario.question
        prior_context = ""

        for round_num in range(self.num_rounds):
            logger.info("  Debate round %d/%d", round_num + 1, self.num_rounds)
            debate_round = self._run_round(
                agents, scenario, round_num, topic, prior_context
            )
            trace.rounds.append(debate_round)

            # Build context for next round from this round's results
            prior_context = self._summarize_round(debate_round)

        # Extract final outcomes
        for r in trace.rounds:
            trace.emergent_hypotheses.extend(
                c.statement for c in r.best_claims
            )
            trace.final_disagreements.extend(
                t.unresolved_disagreements for t in r.turns
            )

        return trace

    def _run_round(self, agents: list[ScientistAgent], scenario: Scenario,
                   round_num: int, topic: str, prior_context: str) -> DebateRound:
        """Run one round of claim → critique → defense."""
        debate_round = DebateRound(round_number=round_num, topic=topic)

        # Phase 1: PROPOSAL — each agent proposes a claim
        proposal_turn = DebateTurn(turn_number=0, phase="proposal")
        for agent in agents:
            claim = self._get_proposal(agent, scenario, prior_context)
            proposal_turn.claims.append(claim)
        debate_round.turns.append(proposal_turn)

        # Phase 2: CRITIQUE — each agent critiques another's claim
        critique_turn = DebateTurn(turn_number=1, phase="critique")
        for i, agent in enumerate(agents):
            # Critique the next agent's claim (round-robin)
            target_claim = proposal_turn.claims[(i + 1) % len(agents)]
            critique = self._get_critique(agent, target_claim, scenario)
            critique_turn.critiques.append(critique)
        debate_round.turns.append(critique_turn)

        # Phase 3: DEFENSE — agents defend or revise based on critiques
        defense_turn = DebateTurn(turn_number=2, phase="defense")
        for i, agent in enumerate(agents):
            # Find the critique targeting this agent's claim
            my_claim = proposal_turn.claims[i]
            my_critiques = [c for c in critique_turn.critiques
                           if c.target_claim_id == my_claim.claim_id]
            if my_critiques:
                defense = self._get_defense(agent, my_claim, my_critiques[0], scenario)
                defense_turn.defenses.append(defense)
        debate_round.turns.append(defense_turn)

        # Identify best claims (those that survived critique)
        for claim in proposal_turn.claims:
            critiques_received = [c for c in critique_turn.critiques
                                  if c.target_claim_id == claim.claim_id]
            fatal_critiques = [c for c in critiques_received if c.severity == "fatal"]
            if not fatal_critiques:
                debate_round.best_claims.append(claim)
            else:
                debate_round.rejected_claims.append(claim)

        return debate_round

    def _get_proposal(self, agent: ScientistAgent, scenario: Scenario,
                      prior_context: str) -> Claim:
        """Get a scientific claim/proposal from an agent."""
        system_prompt = agent.to_system_prompt(scenario.background_knowledge)
        user_msg = PROPOSAL_PROMPT.format(question=scenario.question)
        if prior_context:
            user_msg += f"\n\nPRIOR DISCUSSION CONTEXT:\n{prior_context}"

        response = self.llm.call(system_prompt, user_msg, max_tokens=1000, json_mode=True)
        data = response.parsed

        claim = Claim(
            claim_id=f"claim_{uuid.uuid4().hex[:6]}",
            agent_id=agent.agent_id,
            agent_name=agent.name,
            statement=data.get("statement", response.text[:200]),
            assumptions=data.get("assumptions", []),
            variables=data.get("variables", []),
            equations=data.get("equations", []),
            supporting_rationale=data.get("supporting_rationale", ""),
            evidence_cited=data.get("evidence_cited", []),
            confidence=data.get("confidence", 0.5),
            testable_predictions=data.get("testable_predictions", []),
        )

        agent.hypotheses_proposed.append(claim.claim_id)
        return claim

    def _get_critique(self, agent: ScientistAgent, target_claim: Claim,
                      scenario: Scenario) -> Critique:
        """Get a structured critique from an agent."""
        system_prompt = agent.to_system_prompt(scenario.background_knowledge)
        user_msg = CRITIQUE_PROMPT.format(
            claimant=target_claim.agent_name,
            statement=target_claim.statement,
            assumptions=", ".join(target_claim.assumptions),
            equations=", ".join(target_claim.equations),
            rationale=target_claim.supporting_rationale,
            confidence=target_claim.confidence,
        )

        response = self.llm.call(system_prompt, user_msg, max_tokens=800, json_mode=True)
        data = response.parsed

        critique = Critique(
            critique_id=f"crit_{uuid.uuid4().hex[:6]}",
            agent_id=agent.agent_id,
            agent_name=agent.name,
            target_claim_id=target_claim.claim_id,
            critique_type=data.get("critique_type", "general"),
            critique_text=data.get("critique_text", response.text[:300]),
            specific_weakness=data.get("specific_weakness", ""),
            alternative_proposal=data.get("alternative_proposal", ""),
            severity=data.get("severity", "medium"),
            confidence=data.get("confidence", 0.5),
        )

        agent.critiques_given.append(critique.critique_id)
        return critique

    def _get_defense(self, agent: ScientistAgent, claim: Claim,
                     critique: Critique, scenario: Scenario) -> dict:
        """Get a defense or revision from an agent."""
        system_prompt = agent.to_system_prompt(scenario.background_knowledge)
        user_msg = DEFENSE_PROMPT.format(
            statement=claim.statement,
            critic=critique.agent_name,
            critique_type=critique.critique_type,
            critique_text=critique.critique_text,
            specific_weakness=critique.specific_weakness,
            severity=critique.severity,
        )

        response = self.llm.call(system_prompt, user_msg, max_tokens=600, json_mode=True)
        data = response.parsed

        return {
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "action": data.get("action", "defend"),
            "response": data.get("response", response.text[:300]),
            "revised_claim": data.get("revised_claim"),
            "confidence_after": data.get("confidence_after", claim.confidence),
        }

    def _summarize_round(self, debate_round: DebateRound) -> str:
        """Summarize a debate round for context in the next round."""
        parts = []
        for claim in debate_round.best_claims:
            parts.append(f"- SURVIVING CLAIM ({claim.agent_name}): {claim.statement}")
        for claim in debate_round.rejected_claims:
            parts.append(f"- REJECTED CLAIM ({claim.agent_name}): {claim.statement}")
        return "\n".join(parts)
