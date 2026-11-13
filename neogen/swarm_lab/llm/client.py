"""
LLM Client for the Evolutionary Scientist Swarm.

Wraps Anthropic API with structured output parsing, cost tracking,
and retry logic. Designed for high-throughput agent calls.
"""

import os
import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
}


@dataclass
class LLMCall:
    """Result of a single LLM call."""
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    model: str = ""
    is_mock: bool = False
    parsed: dict = field(default_factory=dict)  # If JSON was requested


class SwarmLLMClient:
    """High-throughput LLM client for swarm agent reasoning."""

    def __init__(self, api_key: str = "", model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self.total_calls = 0
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        if not self.api_key:
            logger.warning("No ANTHROPIC_API_KEY. Swarm will use mock responses.")

    def call(self, system_prompt: str, user_message: str,
             max_tokens: int = 1500, json_mode: bool = False) -> LLMCall:
        """Call the LLM and return structured result."""
        if not self.api_key:
            return self._mock(user_message, json_mode)

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            messages = [{"role": "user", "content": user_message}]

            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
            )

            text = response.content[0].text
            inp = response.usage.input_tokens
            out = response.usage.output_tokens
            pricing = PRICING.get(self.model, PRICING["claude-sonnet-4-20250514"])
            cost = (inp * pricing["input"] + out * pricing["output"]) / 1_000_000

            self.total_calls += 1
            self.total_cost += cost
            self.total_input_tokens += inp
            self.total_output_tokens += out

            result = LLMCall(text=text, input_tokens=inp, output_tokens=out,
                             cost=cost, model=self.model)

            if json_mode:
                result.parsed = self._extract_json(text)

            return result

        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return LLMCall(text=f"Error: {e}", is_mock=True)

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from LLM response text."""
        import re
        # Try to find JSON block
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try array
        arr_match = re.search(r'\[[\s\S]*\]', text)
        if arr_match:
            try:
                return {"items": json.loads(arr_match.group())}
            except json.JSONDecodeError:
                pass

        return {"raw": text}

    def _mock(self, user_message: str, json_mode: bool) -> LLMCall:
        """Generate mock response for testing without API key."""
        if json_mode:
            mock_data = {
                "claims": [{"statement": "Mock claim for testing", "confidence": 0.5}],
                "equations": ["dV/dt = -V/tau"],
                "variables": {"V": "membrane voltage", "tau": "time constant"},
            }
            return LLMCall(text=json.dumps(mock_data), parsed=mock_data, is_mock=True)
        return LLMCall(
            text="[Mock Mode] This is a placeholder response. Set ANTHROPIC_API_KEY for real agent reasoning.",
            is_mock=True,
        )

    def get_stats(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_cost_usd": round(self.total_cost, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
