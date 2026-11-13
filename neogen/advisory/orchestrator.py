"""
Advisory Board Orchestrator — Main entry point for the NeoGen virtual biotech org.

Three modes:
  1. consult(query) — Ask a question, get expert answers
  2. review(title, description) — Propose a change, run review pipeline
  3. ask_agent(agent_id, query) — Talk to a specific agent directly
"""

import logging

from .models import ConsultationResult, PipelineResult
from .llm_client import AdvisoryLLMClient
from .router import QueryRouter
from .code_reader import CodeReader
from .pipeline import ReviewPipeline
from .prompts.consultation import build_agent_system_prompt, build_synthesis_prompt

logger = logging.getLogger(__name__)


class AdvisoryBoard:
    """The NeoGen Virtual Biotech Advisory Board.

    Usage:
        board = AdvisoryBoard()
        result = board.consult("How do I improve clearance prediction?")
        print(result.synthesis)
    """

    def __init__(self, api_key: str = "", model: str = "claude-sonnet-4-20250514"):
        self.llm = AdvisoryLLMClient(api_key=api_key, model=model)
        self.router = QueryRouter()
        self.code_reader = CodeReader()
        self.pipeline = ReviewPipeline(self.llm, self.router, self.code_reader)

    def consult(self, query: str, departments: list[str] = None,
                max_agents: int = 3, include_code: bool = True) -> ConsultationResult:
        """Consult the advisory board on a question.

        Args:
            query: Your question
            departments: Specific department IDs to target (auto-routes if None)
            max_agents: Max agents to consult (controls cost)
            include_code: Whether to include module summaries in context
        """
        result = ConsultationResult(query=query, departments_consulted=[])

        # Route to departments
        if departments:
            target_depts = [self.router.get_department(d) for d in departments
                           if self.router.get_department(d)]
        else:
            target_depts = self.router.route(query)

        result.departments_consulted = [d.dept_id for d in target_depts]

        print(f"\nConsulting departments: {', '.join(d.name for d in target_depts)}")
        print("-" * 50)

        # Collect agent responses
        agents_consulted = 0
        for dept in target_depts:
            if agents_consulted >= max_agents:
                break

            # Pick the lead + 1 specialist if budget allows
            agents_to_ask = []
            lead = self.router.get_agent(dept.lead_agent_id)
            if lead:
                agents_to_ask.append(lead)
            for agent in dept.agents:
                if agent.agent_id != dept.lead_agent_id and len(agents_to_ask) < 2:
                    agents_to_ask.append(agent)

            for agent in agents_to_ask:
                if agents_consulted >= max_agents:
                    break

                print(f"  Asking: {agent.name} ({agent.title})")

                # Build code context
                module_context = ""
                if include_code:
                    for mod in agent.owned_modules[:2]:
                        summary = self.code_reader.read_module_summary(mod)
                        if not summary.startswith("Module '"):
                            module_context += summary + "\n\n"

                persona_dict = {
                    "name": agent.name,
                    "title": agent.title,
                    "department_name": dept.name,
                    "expertise_areas": agent.expertise_areas,
                    "owned_modules": agent.owned_modules,
                    "communication_style": agent.communication_style,
                    "risk_tolerance": agent.risk_tolerance,
                    "background": agent.background,
                }

                system_prompt = build_agent_system_prompt(persona_dict, module_context)
                response = self.llm.call(system_prompt, query, max_tokens=800)

                result.responses.append({
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "title": agent.title,
                    "department": dept.name,
                    "response": response.text,
                    "cost": response.cost,
                })
                agents_consulted += 1

        # Synthesize if multiple responses
        if len(result.responses) > 1:
            print(f"\n  Synthesizing {len(result.responses)} expert opinions...")
            synth_prompt = build_synthesis_prompt(query, result.responses)
            synth_response = self.llm.call(
                system_prompt="You are the NeoGen Advisory Board synthesis engine.",
                user_message=synth_prompt,
                max_tokens=1000,
            )
            result.synthesis = synth_response.text
        elif result.responses:
            result.synthesis = result.responses[0]["response"]

        result.total_cost = self.llm.total_cost
        result.token_usage = {
            "input": self.llm.total_input_tokens,
            "output": self.llm.total_output_tokens,
        }

        return result

    def review(self, title: str, description: str,
               affected_modules: list[str] = None,
               risk_level: str = "medium") -> PipelineResult:
        """Submit a change proposal for multi-layer review.

        Args:
            title: Short title of the proposed change
            description: Detailed description of what and why
            affected_modules: List of NeoGen module filenames affected
            risk_level: "low" | "medium" | "high" | "critical"
        """
        proposal = self.pipeline.create_proposal(
            title=title,
            description=description,
            affected_modules=affected_modules,
            risk_level=risk_level,
        )
        return self.pipeline.run_full_pipeline(proposal)

    def ask_agent(self, agent_id: str, query: str,
                  include_code: bool = True) -> dict:
        """Ask a specific agent directly.

        Args:
            agent_id: The agent ID (e.g., "pk_lead", "metabolism_lead")
            query: Your question
        """
        agent = self.router.get_agent(agent_id)
        if not agent:
            return {"error": f"Agent '{agent_id}' not found. Use list_agents() to see available agents."}

        dept = self.router.get_department(agent.department_id)

        module_context = ""
        if include_code:
            for mod in agent.owned_modules[:2]:
                summary = self.code_reader.read_module_summary(mod)
                if not summary.startswith("Module '"):
                    module_context += summary + "\n\n"

        persona_dict = {
            "name": agent.name,
            "title": agent.title,
            "department_name": dept.name if dept else agent.department_id,
            "expertise_areas": agent.expertise_areas,
            "owned_modules": agent.owned_modules,
            "communication_style": agent.communication_style,
            "risk_tolerance": agent.risk_tolerance,
            "background": agent.background,
        }

        system_prompt = build_agent_system_prompt(persona_dict, module_context)
        response = self.llm.call(system_prompt, query, max_tokens=1000)

        return {
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "title": agent.title,
            "department": dept.name if dept else agent.department_id,
            "response": response.text,
            "cost": response.cost,
        }

    def list_departments(self) -> list[dict]:
        """List all departments."""
        return self.router.list_departments()

    def list_agents(self) -> list[dict]:
        """List all agents."""
        return [
            {
                "id": a.agent_id,
                "name": a.name,
                "title": a.title,
                "department": a.department_id,
                "role": a.role_level,
                "expertise": a.expertise_areas[:3],
            }
            for a in self.router.personas.values()
        ]

    def list_modules(self) -> list[dict]:
        """List all NeoGen modules the advisory board can read."""
        return self.code_reader.list_modules()

    def get_session_stats(self) -> dict:
        """Get API usage stats for this session."""
        return self.llm.get_session_stats()
