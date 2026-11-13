"""
Review Pipeline — Multi-layer approval system for proposed NeoGen changes.

Layer 1: Department review (specialists evaluate within their domain)
Layer 2: Cross-department impact review (affected departments weigh in)
Layer 3: Executive review (CSO/CMO make final call)
"""

import logging
import uuid
from dataclasses import asdict
from datetime import datetime

from .models import Proposal, AgentReview, PipelineResult
from .llm_client import AdvisoryLLMClient
from .router import QueryRouter
from .code_reader import CodeReader
from .prompts.review import (
    build_department_review_prompt,
    build_cross_department_prompt,
    build_executive_review_prompt,
)

logger = logging.getLogger(__name__)


class ReviewPipeline:
    """Multi-layer review pipeline for NeoGen change proposals."""

    def __init__(self, llm_client: AdvisoryLLMClient, router: QueryRouter,
                 code_reader: CodeReader):
        self.llm = llm_client
        self.router = router
        self.code_reader = code_reader

    def create_proposal(self, title: str, description: str,
                        affected_modules: list[str] = None,
                        risk_level: str = "medium") -> Proposal:
        """Create a new change proposal."""
        proposal = Proposal(
            proposal_id=f"prop_{uuid.uuid4().hex[:8]}",
            title=title,
            description=description,
            affected_modules=affected_modules or [],
            risk_level=risk_level,
        )

        # Auto-detect affected departments from modules
        if affected_modules:
            affected_depts = self.router.route_by_modules(affected_modules)
            proposal.affected_departments = [d.dept_id for d in affected_depts]
        else:
            # Route by description keywords
            routed = self.router.route(description)
            proposal.affected_departments = [d.dept_id for d in routed]

        return proposal

    def run_full_pipeline(self, proposal: Proposal) -> PipelineResult:
        """Run the complete 3-layer review pipeline."""
        result = PipelineResult(proposal=proposal)

        print(f"\n{'='*60}")
        print(f"REVIEW PIPELINE: {proposal.title}")
        print(f"Risk Level: {proposal.risk_level}")
        print(f"Affected Departments: {', '.join(proposal.affected_departments)}")
        print(f"{'='*60}\n")

        # Layer 1: Department Review
        print("LAYER 1: Department Review")
        print("-" * 40)
        proposal.status = "dept_review"
        result.department_reviews = self._run_department_review(proposal)

        # Layer 2: Cross-Department Impact
        print("\nLAYER 2: Cross-Department Impact Review")
        print("-" * 40)
        proposal.status = "cross_review"
        result.cross_department_reviews = self._run_cross_department_review(
            proposal, result.department_reviews
        )

        # Layer 3: Executive Review
        print("\nLAYER 3: Executive Review")
        print("-" * 40)
        proposal.status = "exec_review"
        result.executive_reviews = self._run_executive_review(
            proposal, result.department_reviews + result.cross_department_reviews
        )

        # Determine final verdict
        exec_verdicts = [r.verdict for r in result.executive_reviews]
        if all(v == "approve" for v in exec_verdicts):
            result.final_verdict = "APPROVED"
        elif any(v == "reject" for v in exec_verdicts):
            result.final_verdict = "REJECTED"
        else:
            result.final_verdict = "REVISE"

        result.final_reasoning = "\n".join(
            f"  {r.agent_name}: {r.reasoning[:200]}" for r in result.executive_reviews
        )

        proposal.status = "approved" if result.final_verdict == "APPROVED" else "rejected"
        proposal.final_verdict = result.final_verdict
        result.total_cost = self.llm.total_cost

        print(f"\n{'='*60}")
        print(f"FINAL VERDICT: {result.final_verdict}")
        print(f"Total API Cost: ${result.total_cost:.4f}")
        print(f"{'='*60}\n")

        return result

    def _run_department_review(self, proposal: Proposal) -> list[AgentReview]:
        """Layer 1: Primary department agents review."""
        reviews = []

        for dept_id in proposal.affected_departments:
            agents = self.router.get_department_agents(dept_id)
            dept = self.router.get_department(dept_id)

            for agent in agents[:2]:  # Max 2 agents per department for cost
                print(f"  Reviewing: {agent.name} ({agent.title})")

                # Get relevant module context
                module_context = ""
                relevant_modules = [m for m in agent.owned_modules
                                    if m in proposal.affected_modules]
                for mod in relevant_modules[:2]:  # Max 2 modules for token control
                    module_context += self.code_reader.read_module_summary(mod) + "\n\n"

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

                proposal_dict = {
                    "title": proposal.title,
                    "description": proposal.description,
                    "affected_modules": proposal.affected_modules,
                    "risk_level": proposal.risk_level,
                }

                prompt = build_department_review_prompt(
                    persona_dict, proposal_dict, module_context
                )

                response = self.llm.call(
                    system_prompt=prompt,
                    user_message=f"Please review this proposal: {proposal.title}\n\n{proposal.description}",
                    max_tokens=800,
                )

                # Parse verdict from response
                verdict = self._extract_verdict(response.text)

                review = AgentReview(
                    agent_id=agent.agent_id,
                    agent_name=agent.name,
                    department_id=agent.department_id,
                    verdict=verdict,
                    reasoning=response.text,
                )
                reviews.append(review)
                print(f"    Verdict: {verdict.upper()}")

        return reviews

    def _run_cross_department_review(self, proposal: Proposal,
                                      dept_reviews: list[AgentReview]) -> list[AgentReview]:
        """Layer 2: Other affected departments review for downstream impact."""
        reviews = []

        # Find departments NOT in the primary review
        all_routed = self.router.route(proposal.description, max_departments=5)
        cross_depts = [d for d in all_routed
                       if d.dept_id not in proposal.affected_departments]

        for dept in cross_depts[:2]:  # Max 2 cross-department reviews
            lead = self.router.get_agent(dept.lead_agent_id)
            if not lead:
                continue

            print(f"  Cross-review: {lead.name} ({dept.name})")

            persona_dict = {
                "name": lead.name,
                "title": lead.title,
                "department_name": dept.name,
                "expertise_areas": lead.expertise_areas,
                "owned_modules": lead.owned_modules,
                "communication_style": lead.communication_style,
                "risk_tolerance": lead.risk_tolerance,
                "background": lead.background,
            }

            reviews_summary = [
                {"agent_name": r.agent_name, "verdict": r.verdict,
                 "reasoning": r.reasoning[:300]}
                for r in dept_reviews
            ]

            prompt = build_cross_department_prompt(
                persona_dict,
                {"title": proposal.title, "description": proposal.description,
                 "affected_modules": proposal.affected_modules},
                reviews_summary,
            )

            response = self.llm.call(
                system_prompt=prompt,
                user_message=f"Assess cross-department impact of: {proposal.title}",
                max_tokens=600,
            )

            verdict = self._extract_verdict(response.text)
            review = AgentReview(
                agent_id=lead.agent_id,
                agent_name=lead.name,
                department_id=dept.dept_id,
                verdict=verdict,
                reasoning=response.text,
            )
            reviews.append(review)
            print(f"    Verdict: {verdict.upper()}")

        return reviews

    def _run_executive_review(self, proposal: Proposal,
                               all_reviews: list[AgentReview]) -> list[AgentReview]:
        """Layer 3: Executive committee makes final decision."""
        reviews = []
        executives = self.router.get_executives()

        reviews_summary = [
            {"agent_name": r.agent_name, "department": r.department_id,
             "verdict": r.verdict, "conditions": r.conditions}
            for r in all_reviews
        ]

        for exec_agent in executives:
            print(f"  Executive: {exec_agent.name} ({exec_agent.title})")

            prompt = build_executive_review_prompt(
                exec_agent.title,
                {"title": proposal.title, "description": proposal.description,
                 "risk_level": proposal.risk_level},
                reviews_summary,
            )

            response = self.llm.call(
                system_prompt=prompt,
                user_message=f"Make your final decision on: {proposal.title}",
                max_tokens=600,
            )

            verdict = self._extract_verdict(response.text)
            review = AgentReview(
                agent_id=exec_agent.agent_id,
                agent_name=exec_agent.name,
                department_id="executive",
                verdict=verdict,
                reasoning=response.text,
            )
            reviews.append(review)
            print(f"    Final: {verdict.upper()}")

        return reviews

    def _extract_verdict(self, text: str) -> str:
        """Extract verdict from agent response text."""
        text_lower = text.lower()
        if "reject" in text_lower[:200]:
            return "reject"
        elif "concern" in text_lower[:200] or "revise" in text_lower[:200]:
            return "concern"
        else:
            return "approve"
