"""
LLM Client for the NeoGen Advisory Board.

Wraps the Anthropic API with cost tracking. Mirrors the pattern from ai_brain.py
but uses Sonnet (5x cheaper) and tracks token usage for budget awareness.
"""

import os
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Claude Sonnet 4 pricing (per million tokens)
PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
}


@dataclass
class LLMResponse:
    """Response from an LLM call with metadata."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    model: str = ""
    is_mock: bool = False


@dataclass
class AdvisoryLLMClient:
    """Anthropic API client with cost tracking for advisory board."""

    api_key: str = ""
    model: str = "claude-sonnet-4-20250514"

    # Session-level tracking
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    total_calls: int = 0

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            logger.warning("No ANTHROPIC_API_KEY found. Advisory Board will use mock responses.")

    def call(self, system_prompt: str, user_message: str, max_tokens: int = 1500) -> LLMResponse:
        """Call Claude API and track costs."""
        if not self.api_key:
            return LLMResponse(
                text=self._mock_response(user_message),
                is_mock=True,
                model="mock"
            )

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            text = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            # Calculate cost
            pricing = PRICING.get(self.model, PRICING["claude-sonnet-4-20250514"])
            cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

            # Track session totals
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost
            self.total_calls += 1

            return LLMResponse(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                model=self.model,
            )

        except Exception as e:
            logger.error("Claude API call failed: %s", e)
            return LLMResponse(
                text=f"API Error: {e}\n\nSet ANTHROPIC_API_KEY in your environment.",
                is_mock=True,
                model="error"
            )

    def get_session_stats(self) -> dict:
        """Return session-level usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "model": self.model,
        }

    def _mock_response(self, user_message: str) -> str:
        return (
            "**Advisory Board (Mock Mode — No API Key)**\n\n"
            "Set ANTHROPIC_API_KEY in your environment to enable AI advisory responses.\n\n"
            "Get your key at: https://console.anthropic.com\n"
        )
