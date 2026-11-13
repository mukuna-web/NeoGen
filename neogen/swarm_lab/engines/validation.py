"""
Reality Validation Engine — validates agent outputs against ground truth.

Pluggable validator modules:
  A. Literature-grounded validation
  B. Dataset-grounded validation
  C. Simulation-grounded validation
  D. Benchmark / known reality validation
"""

import logging
from abc import ABC, abstractmethod

from ..llm.client import SwarmLLMClient
from ..models.debate import Claim
from ..models.hypothesis import Hypothesis
from ..models.scenario import Scenario, ValidationSource

logger = logging.getLogger(__name__)


class Validator(ABC):
    """Base class for all validators."""

    @abstractmethod
    def validate_claim(self, claim: Claim, scenario: Scenario) -> dict:
        """Validate a claim. Returns dict with scores."""
        ...


class LiteratureValidator(Validator):
    """Validates claims against scientific literature knowledge."""

    def __init__(self, llm: SwarmLLMClient):
        self.llm = llm

    def validate_claim(self, claim: Claim, scenario: Scenario) -> dict:
        prompt = f"""You are a scientific literature validator. Assess whether this claim is consistent with established scientific knowledge.

CLAIM: {claim.statement}
EQUATIONS: {', '.join(claim.equations)}
ASSUMPTIONS: {', '.join(claim.assumptions)}
DOMAIN: {scenario.domain}

KNOWN FACTS FROM SCENARIO:
{scenario.background_knowledge[:2000]}

KNOWN EQUATIONS: {', '.join(scenario.known_equations[:5])}

Score each dimension 0.0-1.0 and respond as JSON:
{{
    "factual_accuracy": 0.0-1.0,
    "biological_plausibility": 0.0-1.0,
    "mathematical_consistency": 0.0-1.0,
    "evidence_coverage": 0.0-1.0,
    "contradictions_found": ["list of contradictions"],
    "unsupported_assertions": ["list of unsupported claims"],
    "reasoning": "Brief explanation of scores"
}}"""

        response = self.llm.call(
            system_prompt="You are a rigorous scientific validator. Be honest and critical.",
            user_message=prompt,
            max_tokens=800,
            json_mode=True,
        )

        data = response.parsed
        return {
            "factual_accuracy": float(data.get("factual_accuracy", 0.5)),
            "biological_plausibility": float(data.get("biological_plausibility", 0.5)),
            "mathematical_consistency": float(data.get("mathematical_consistency", 0.5)),
            "evidence_coverage": float(data.get("evidence_coverage", 0.5)),
            "contradictions": data.get("contradictions_found", []),
            "unsupported": data.get("unsupported_assertions", []),
            "reasoning": data.get("reasoning", ""),
            "validator": "literature",
        }


class BenchmarkValidator(Validator):
    """Validates claims against known ground truth data."""

    def __init__(self, llm: SwarmLLMClient):
        self.llm = llm

    def validate_claim(self, claim: Claim, scenario: Scenario) -> dict:
        if not scenario.ground_truth:
            return {"validator": "benchmark", "skipped": True, "reason": "No ground truth"}

        prompt = f"""You are a benchmark validator. Compare this claim against known ground truth.

CLAIM: {claim.statement}
EQUATIONS: {', '.join(claim.equations)}

GROUND TRUTH REFERENCE:
{_format_ground_truth(scenario.ground_truth)}

KNOWN EQUATIONS: {', '.join(scenario.known_equations)}

Score and respond as JSON:
{{
    "predictive_accuracy": 0.0-1.0,
    "equation_correctness": 0.0-1.0,
    "variable_completeness": 0.0-1.0,
    "parameter_accuracy": 0.0-1.0,
    "errors_found": ["list of specific errors"],
    "reasoning": "Brief explanation"
}}"""

        response = self.llm.call(
            system_prompt="You are a benchmark validator comparing claims to known results.",
            user_message=prompt,
            max_tokens=600,
            json_mode=True,
        )

        data = response.parsed
        return {
            "predictive_accuracy": float(data.get("predictive_accuracy", 0.5)),
            "equation_correctness": float(data.get("equation_correctness", 0.5)),
            "variable_completeness": float(data.get("variable_completeness", 0.5)),
            "errors": data.get("errors_found", []),
            "reasoning": data.get("reasoning", ""),
            "validator": "benchmark",
        }


class IntegrityChecker:
    """Scientific integrity guardrails — prevents junk science."""

    def __init__(self, llm: SwarmLLMClient):
        self.llm = llm

    def check(self, claim: Claim) -> dict:
        """Check for fabricated citations, impossible mechanisms, etc."""
        prompt = f"""Assess this scientific claim for integrity issues.

CLAIM: {claim.statement}
EVIDENCE CITED: {', '.join(claim.evidence_cited)}
EQUATIONS: {', '.join(claim.equations)}
CONFIDENCE: {claim.confidence}

Flag any of these issues:
1. Fabricated or unverifiable citations
2. Unsupported certainty (high confidence without evidence)
3. Mathematically invalid equations
4. Biologically impossible mechanisms
5. Persuasive nonsense (sounds good but is meaningless)
6. Consensus claimed without evidence

Respond as JSON:
{{
    "integrity_score": 0.0-1.0,
    "issues_found": ["issue 1", "issue 2"],
    "is_fabrication_suspected": false,
    "confidence_calibration": 0.0-1.0,
    "reasoning": "explanation"
}}"""

        response = self.llm.call(
            system_prompt="You are a scientific integrity auditor. Be ruthlessly honest.",
            user_message=prompt,
            max_tokens=500,
            json_mode=True,
        )

        data = response.parsed
        return {
            "integrity_score": float(data.get("integrity_score", 0.5)),
            "issues": data.get("issues_found", []),
            "fabrication_suspected": data.get("is_fabrication_suspected", False),
            "confidence_calibration": float(data.get("confidence_calibration", 0.5)),
            "validator": "integrity",
        }


class ValidationEngine:
    """Orchestrates all validators for comprehensive reality checking."""

    def __init__(self, llm: SwarmLLMClient):
        self.literature = LiteratureValidator(llm)
        self.benchmark = BenchmarkValidator(llm)
        self.integrity = IntegrityChecker(llm)

    def validate_claim(self, claim: Claim, scenario: Scenario) -> dict:
        """Run all applicable validators on a claim."""
        results = {}

        # Always run literature and integrity
        results["literature"] = self.literature.validate_claim(claim, scenario)
        results["integrity"] = self.integrity.check(claim)

        # Run benchmark if ground truth available
        if scenario.ground_truth:
            results["benchmark"] = self.benchmark.validate_claim(claim, scenario)

        # Aggregate scores
        scores = {
            "factual_accuracy": results["literature"].get("factual_accuracy", 0.5),
            "biological_plausibility": results["literature"].get("biological_plausibility", 0.5),
            "mathematical_consistency": results["literature"].get("mathematical_consistency", 0.5),
            "predictive_accuracy": results.get("benchmark", {}).get("predictive_accuracy", 0.5),
            "integrity_score": results["integrity"].get("integrity_score", 0.5),
        }

        results["aggregate"] = scores
        return results


def _format_ground_truth(gt: dict) -> str:
    """Format ground truth dict into readable text."""
    parts = []
    for key, value in gt.items():
        if isinstance(value, dict):
            parts.append(f"{key}:")
            for k, v in value.items():
                parts.append(f"  {k}: {v}")
        else:
            parts.append(f"{key}: {value}")
    return "\n".join(parts)
