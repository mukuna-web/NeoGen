"""
Dynamic Role Generator — generates appropriate specialist panels for any scientific problem.

Does NOT rely on a fixed set of roles. Instead, it analyzes the problem
and dynamically generates relevant scientific roles.
"""

import logging

from ..llm.client import SwarmLLMClient
from ..models.scenario import Scenario

logger = logging.getLogger(__name__)

ROLE_GENERATION_PROMPT = """You are a scientific team builder. Given a research problem, generate a panel of specialist scientist roles that would be needed to tackle it rigorously.

For each role, provide:
1. role: Short role title (e.g., "cardiac electrophysiologist")
2. specialty: More detailed specialty description
3. expertise_areas: List of 3-5 specific expertise areas
4. knowledge_priorities: What this role should focus on
5. expected_contribution: What unique value this role brings to the team

RULES:
- Generate exactly {num_roles} roles
- Include at least one "scientific skeptic / falsifier" role
- Include at least one "mathematical modeler" role
- Include diverse perspectives (reductionist AND systems-level thinkers)
- Roles should be complementary, not redundant
- Be specific to the problem domain

Return as JSON array of role objects."""


class RoleGenerator:
    """Dynamically generates scientist roles appropriate for a given problem."""

    def __init__(self, llm: SwarmLLMClient):
        self.llm = llm

    def generate_roles(self, scenario: Scenario, num_roles: int = 8) -> list[dict]:
        """Generate specialist roles for a scientific scenario.

        Returns list of role dicts with: role, specialty, expertise_areas,
        knowledge_priorities, expected_contribution
        """
        # If scenario has suggested specialties, use them as hints
        hints = ""
        if scenario.suggested_specialties:
            hints = f"\nSuggested specialties to consider: {', '.join(scenario.suggested_specialties)}"

        prompt = ROLE_GENERATION_PROMPT.format(num_roles=num_roles)

        user_msg = f"""PROBLEM: {scenario.question}

DOMAIN: {scenario.domain}

DESCRIPTION: {scenario.description}

TARGET OUTPUTS: {', '.join(scenario.target_outputs)}

KNOWN SUBSYSTEMS: {', '.join(scenario.known_subsystems)}
{hints}

Generate {num_roles} specialist roles as a JSON array."""

        response = self.llm.call(
            system_prompt=prompt,
            user_message=user_msg,
            max_tokens=2000,
            json_mode=True,
        )

        if response.parsed and "items" in response.parsed:
            roles = response.parsed["items"]
        elif response.parsed and isinstance(response.parsed.get("roles"), list):
            roles = response.parsed["roles"]
        elif response.is_mock:
            roles = self._fallback_roles(scenario, num_roles)
        else:
            # Try to parse from raw
            roles = self._fallback_roles(scenario, num_roles)

        # Ensure we have enough roles
        while len(roles) < num_roles:
            roles.append(self._generic_role(len(roles)))

        return roles[:num_roles]

    def _fallback_roles(self, scenario: Scenario, num_roles: int) -> list[dict]:
        """Fallback role generation when LLM is unavailable."""
        base_roles = [
            {
                "role": "mathematical modeler",
                "specialty": f"Mathematical modeling of {scenario.domain}",
                "expertise_areas": ["differential equations", "dynamical systems", "symbolic computation"],
                "knowledge_priorities": ["mathematical abstraction", "equation derivation"],
                "expected_contribution": "Translate biological mechanisms into precise mathematical form",
            },
            {
                "role": "domain physiologist",
                "specialty": f"Physiology of {scenario.domain}",
                "expertise_areas": ["organ physiology", "systems biology", "clinical science"],
                "knowledge_priorities": ["biological mechanisms", "physiological parameters"],
                "expected_contribution": "Ground truth about how the biological system actually works",
            },
            {
                "role": "systems biologist",
                "specialty": "Multi-scale systems integration",
                "expertise_areas": ["network modeling", "emergent behavior", "cross-scale coupling"],
                "knowledge_priorities": ["system-level behavior", "feedback loops"],
                "expected_contribution": "Integrate subsystem models into coherent whole-system understanding",
            },
            {
                "role": "biophysicist",
                "specialty": "Physical principles in biological systems",
                "expertise_areas": ["thermodynamics", "mechanics", "electrical signaling"],
                "knowledge_priorities": ["physical constraints", "energy conservation"],
                "expected_contribution": "Ensure models respect physical laws and constraints",
            },
            {
                "role": "computational scientist",
                "specialty": "Numerical methods and simulation",
                "expertise_areas": ["ODE solvers", "numerical stability", "parameter estimation"],
                "knowledge_priorities": ["computational tractability", "numerical accuracy"],
                "expected_contribution": "Assess whether proposed models are computationally feasible",
            },
            {
                "role": "scientific skeptic",
                "specialty": "Critical analysis and falsification",
                "expertise_areas": ["experimental design", "statistical rigor", "bias detection"],
                "knowledge_priorities": ["identifying weaknesses", "testing assumptions"],
                "expected_contribution": "Challenge every claim, find flaws, demand evidence",
            },
            {
                "role": "literature auditor",
                "specialty": "Scientific literature synthesis",
                "expertise_areas": ["literature review", "historical context", "competing models"],
                "knowledge_priorities": ["prior work", "known models", "open questions"],
                "expected_contribution": "Ensure the team doesn't reinvent the wheel or miss known results",
            },
            {
                "role": "control systems theorist",
                "specialty": "Feedback control and regulation",
                "expertise_areas": ["control theory", "stability analysis", "homeostasis modeling"],
                "knowledge_priorities": ["regulatory mechanisms", "feedback loops", "stability"],
                "expected_contribution": "Model regulatory and homeostatic mechanisms mathematically",
            },
        ]

        # Add domain-specific roles from scenario suggestions
        for i, spec in enumerate(scenario.suggested_specialties):
            if i + len(base_roles) < num_roles:
                base_roles.append({
                    "role": spec,
                    "specialty": spec,
                    "expertise_areas": [spec, scenario.domain],
                    "knowledge_priorities": [scenario.domain],
                    "expected_contribution": f"Specialized knowledge of {spec}",
                })

        return base_roles[:num_roles]

    def _generic_role(self, index: int) -> dict:
        return {
            "role": f"research scientist {index}",
            "specialty": "General scientific reasoning",
            "expertise_areas": ["scientific method", "critical thinking"],
            "knowledge_priorities": ["evidence-based reasoning"],
            "expected_contribution": "General scientific perspective",
        }
