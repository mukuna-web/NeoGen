"""
Query Router — routes questions and proposals to the right departments.

Uses keyword matching against department routing_keywords defined in departments.yaml.
"""

import os
import logging
from typing import Optional

import yaml

from .models import Department, AgentPersona

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.dirname(__file__)


def load_departments() -> dict[str, Department]:
    """Load department definitions from YAML."""
    dept_path = os.path.join(CONFIG_DIR, "departments.yaml")
    with open(dept_path) as f:
        data = yaml.safe_load(f)

    departments = {}
    for dept_id, dept_data in data["departments"].items():
        departments[dept_id] = Department(
            dept_id=dept_id,
            name=dept_data["name"],
            description=dept_data["description"],
            lead_agent_id=dept_data.get("lead_agent_id", ""),
            owned_modules=dept_data.get("owned_modules", []),
            routing_keywords=dept_data.get("routing_keywords", []),
        )
    return departments


def load_personas() -> dict[str, AgentPersona]:
    """Load agent persona definitions from YAML."""
    persona_path = os.path.join(CONFIG_DIR, "personas.yaml")
    with open(persona_path) as f:
        data = yaml.safe_load(f)

    personas = {}
    for agent_id, agent_data in data["agents"].items():
        personas[agent_id] = AgentPersona(
            agent_id=agent_id,
            name=agent_data["name"],
            title=agent_data["title"],
            department_id=agent_data["department_id"],
            role_level=agent_data["role_level"],
            expertise_areas=agent_data["expertise_areas"],
            owned_modules=agent_data.get("owned_modules", []),
            system_prompt="",  # built dynamically
            communication_style=agent_data.get("communication_style", "pragmatic"),
            risk_tolerance=agent_data.get("risk_tolerance", "moderate"),
            background=agent_data.get("background", ""),
        )
    return personas


class QueryRouter:
    """Routes queries to relevant departments based on keyword matching."""

    def __init__(self):
        self.departments = load_departments()
        self.personas = load_personas()

        # Attach agents to departments
        for agent_id, persona in self.personas.items():
            dept_id = persona.department_id
            if dept_id in self.departments:
                self.departments[dept_id].agents.append(persona)

    def route(self, query: str, max_departments: int = 3) -> list[Department]:
        """Route a query to the most relevant departments."""
        query_lower = query.lower()
        scores: dict[str, int] = {}

        for dept_id, dept in self.departments.items():
            score = 0
            for keyword in dept.routing_keywords:
                if keyword in query_lower:
                    score += 1
                    # Bonus for exact module name matches
                    if keyword.endswith(".py"):
                        score += 2
            if score > 0:
                scores[dept_id] = score

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Return top N departments
        result = []
        for dept_id, score in ranked[:max_departments]:
            result.append(self.departments[dept_id])

        # If no matches, default to PK Core (most general)
        if not result:
            result.append(self.departments["pk_core"])

        return result

    def route_by_modules(self, module_names: list[str]) -> list[Department]:
        """Route by affected module names (for proposals)."""
        affected_depts = set()
        for dept_id, dept in self.departments.items():
            for module in module_names:
                if module in dept.owned_modules:
                    affected_depts.add(dept_id)

        return [self.departments[d] for d in affected_depts]

    def get_department(self, dept_id: str) -> Optional[Department]:
        """Get a specific department by ID."""
        return self.departments.get(dept_id)

    def get_agent(self, agent_id: str) -> Optional[AgentPersona]:
        """Get a specific agent by ID."""
        return self.personas.get(agent_id)

    def get_department_agents(self, dept_id: str) -> list[AgentPersona]:
        """Get all agents in a department."""
        return [p for p in self.personas.values() if p.department_id == dept_id]

    def get_executives(self) -> list[AgentPersona]:
        """Get executive committee agents."""
        return [p for p in self.personas.values() if p.role_level == "executive"]

    def list_departments(self) -> list[dict]:
        """List all departments with agent counts."""
        result = []
        for dept_id, dept in self.departments.items():
            result.append({
                "id": dept_id,
                "name": dept.name,
                "description": dept.description,
                "agent_count": len(dept.agents),
                "lead": dept.lead_agent_id,
            })
        return result
