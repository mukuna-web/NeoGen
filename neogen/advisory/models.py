"""
Data models for the NeoGen Advisory Board.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class AgentPersona:
    """A single advisory board agent with domain expertise."""
    agent_id: str
    name: str
    title: str
    department_id: str
    role_level: str  # "specialist" | "lead" | "executive"
    expertise_areas: list[str]
    owned_modules: list[str]
    system_prompt: str
    communication_style: str  # "formal_academic" | "pragmatic" | "cautious" | "direct"
    risk_tolerance: str  # "conservative" | "moderate" | "aggressive"
    age_range: str = "35-55"
    background: str = ""


@dataclass
class Department:
    """A department within the virtual biotech organization."""
    dept_id: str
    name: str
    description: str
    agents: list[AgentPersona] = field(default_factory=list)
    lead_agent_id: str = ""
    owned_modules: list[str] = field(default_factory=list)
    routing_keywords: list[str] = field(default_factory=list)


@dataclass
class AgentReview:
    """A single agent's review of a proposal."""
    agent_id: str
    agent_name: str
    department_id: str
    verdict: str  # "approve" | "concern" | "reject"
    reasoning: str
    conditions: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Proposal:
    """A proposed change to NeoGen, reviewed by the advisory pipeline."""
    proposal_id: str
    title: str
    description: str
    affected_modules: list[str] = field(default_factory=list)
    affected_departments: list[str] = field(default_factory=list)
    risk_level: str = "medium"  # "low" | "medium" | "high" | "critical"
    status: str = "draft"  # "draft" | "dept_review" | "cross_review" | "exec_review" | "approved" | "rejected"
    reviews: list[AgentReview] = field(default_factory=list)
    created_by: str = "user"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    final_verdict: str = ""
    final_reasoning: str = ""


@dataclass
class ConsultationResult:
    """Result from a consultation query."""
    query: str
    departments_consulted: list[str]
    responses: list[dict] = field(default_factory=list)  # [{agent_id, agent_name, department, response}]
    synthesis: str = ""
    total_cost: float = 0.0
    token_usage: dict = field(default_factory=lambda: {"input": 0, "output": 0})


@dataclass
class PipelineResult:
    """Result from a full review pipeline run."""
    proposal: Proposal
    department_reviews: list[AgentReview] = field(default_factory=list)
    cross_department_reviews: list[AgentReview] = field(default_factory=list)
    executive_reviews: list[AgentReview] = field(default_factory=list)
    final_verdict: str = ""
    final_reasoning: str = ""
    total_cost: float = 0.0
