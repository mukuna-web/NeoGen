"""
Debate models — structured scientific argument traces, not chat logs.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Claim:
    """A scientific claim made by an agent."""
    claim_id: str = ""
    agent_id: str = ""
    agent_name: str = ""
    statement: str = ""
    assumptions: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    equations: list[str] = field(default_factory=list)
    supporting_rationale: str = ""
    evidence_cited: list[str] = field(default_factory=list)
    confidence: float = 0.5             # 0.0 to 1.0
    testable_predictions: list[str] = field(default_factory=list)


@dataclass
class Critique:
    """A structured critique of a claim by another agent."""
    critique_id: str = ""
    agent_id: str = ""
    agent_name: str = ""
    target_claim_id: str = ""
    critique_type: str = ""             # "assumption_challenge" | "math_error" | "missing_variable" | "bio_implausible" | "insufficient_evidence"
    critique_text: str = ""
    specific_weakness: str = ""
    alternative_proposal: str = ""
    severity: str = "medium"            # "low" | "medium" | "high" | "fatal"
    confidence: float = 0.5


@dataclass
class DebateTurn:
    """A single turn in a debate: claim + critiques + defense."""
    turn_number: int = 0
    phase: str = ""                     # "proposal" | "critique" | "defense" | "synthesis"
    claims: list[Claim] = field(default_factory=list)
    critiques: list[Critique] = field(default_factory=list)
    defenses: list[dict] = field(default_factory=list)  # [{agent_id, response, revised_claim}]
    consensus_points: list[str] = field(default_factory=list)
    unresolved_disagreements: list[str] = field(default_factory=list)


@dataclass
class DebateRound:
    """A full debate round within a cluster."""
    round_number: int = 0
    topic: str = ""
    turns: list[DebateTurn] = field(default_factory=list)
    best_claims: list[Claim] = field(default_factory=list)
    rejected_claims: list[Claim] = field(default_factory=list)


@dataclass
class DebateTrace:
    """Complete trace of a scientific discussion within a cluster."""
    cluster_id: str = ""
    agent_ids: list[str] = field(default_factory=list)
    agent_names: list[str] = field(default_factory=list)
    scenario_id: str = ""
    rounds: list[DebateRound] = field(default_factory=list)
    final_consensus: list[str] = field(default_factory=list)
    final_disagreements: list[str] = field(default_factory=list)
    emergent_hypotheses: list[str] = field(default_factory=list)
    quality_score: float = 0.0
