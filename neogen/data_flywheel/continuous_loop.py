"""Bounded continuous-loop state recorder for the data flywheel.

This does NOT spin autonomously. It records each cycle's outcome (objective,
gap addressed, artifacts, status) and enforces a hard per-session cycle bound so
the loop stays checkpointed and observable. State is replayable JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
LOOP_STATE_PATH = _REPO_ROOT / "data" / "flywheel" / "loop_state.json"
DEFAULT_MAX_CYCLES = 3


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class LoopCycle:
    cycle_n: int
    objective: str
    gap_id: str | None
    status: str  # complete | blocked
    artifacts: tuple[str, ...] = field(default_factory=tuple)
    tests_passed: int = 0
    started_at: str = ""
    completed_at: str = ""
    next_action: str | None = None

    def __post_init__(self) -> None:
        if self.cycle_n < 1:
            raise ValueError("cycle_n must be >= 1")
        if self.status not in ("complete", "blocked"):
            raise ValueError("status must be 'complete' or 'blocked'")


@dataclass
class LoopState:
    task_id: str
    max_cycles: int = DEFAULT_MAX_CYCLES
    cycles: list[LoopCycle] = field(default_factory=list)

    def record_cycle(self, cycle: LoopCycle) -> None:
        if len(self.cycles) >= self.max_cycles:
            raise RuntimeError(
                f"loop bound reached ({self.max_cycles}); refusing to record more"
            )
        self.cycles.append(cycle)

    def should_continue(self) -> bool:
        """True only if under the bound AND no cycle is blocked."""
        if len(self.cycles) >= self.max_cycles:
            return False
        return not any(c.status == "blocked" for c in self.cycles)

    def summary(self) -> dict:
        return {
            "task_id": self.task_id,
            "max_cycles": self.max_cycles,
            "cycles_run": len(self.cycles),
            "should_continue": self.should_continue(),
            "completed": sum(1 for c in self.cycles if c.status == "complete"),
            "blocked": sum(1 for c in self.cycles if c.status == "blocked"),
            "total_tests_passed": sum(c.tests_passed for c in self.cycles),
            "cycles": [asdict(c) for c in self.cycles],
        }

    def save(self, path: Path | str | None = None) -> str:
        p = Path(path) if path else LOOP_STATE_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"generated_at": _now(), **self.summary()}
        p.write_text(json.dumps(payload, indent=2, sort_keys=True, default=list))
        return str(p)
