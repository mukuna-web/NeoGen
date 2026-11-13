"""Tests for the bounded continuous-loop state recorder."""

from __future__ import annotations

import pytest

from neogen.data_flywheel.continuous_loop import LoopCycle, LoopState


def _cyc(n: int, status: str = "complete", tests: int = 0) -> LoopCycle:
    return LoopCycle(
        cycle_n=n,
        objective=f"cycle {n}",
        gap_id=f"GAP-00{n}",
        status=status,
        artifacts=(f"art{n}.json",),
        tests_passed=tests,
        started_at="2026-05-29T00:00:00Z",
        completed_at="2026-05-29T00:01:00Z",
    )


@pytest.mark.unit
def test_record_and_summary() -> None:
    s = LoopState(task_id="t")
    s.record_cycle(_cyc(1, tests=6))
    s.record_cycle(_cyc(2, tests=6))
    out = s.summary()
    assert out["cycles_run"] == 2
    assert out["completed"] == 2
    assert out["total_tests_passed"] == 12


@pytest.mark.unit
def test_max_cycle_bound_enforced() -> None:
    s = LoopState(task_id="t", max_cycles=3)
    for n in (1, 2, 3):
        s.record_cycle(_cyc(n))
    assert s.should_continue() is False
    with pytest.raises(RuntimeError, match="loop bound reached"):
        s.record_cycle(_cyc(4))


@pytest.mark.unit
def test_blocked_cycle_stops_loop() -> None:
    s = LoopState(task_id="t")
    s.record_cycle(_cyc(1, status="blocked"))
    assert s.should_continue() is False


@pytest.mark.unit
def test_invalid_cycle_rejected() -> None:
    with pytest.raises(ValueError):
        LoopCycle(cycle_n=0, objective="x", gap_id=None, status="complete")
    with pytest.raises(ValueError):
        LoopCycle(cycle_n=1, objective="x", gap_id=None, status="banana")


@pytest.mark.unit
def test_save_writes_replayable_json(tmp_path) -> None:
    s = LoopState(task_id="t")
    s.record_cycle(_cyc(1, tests=7))
    p = s.save(tmp_path / "loop_state.json")
    import json

    data = json.loads(open(p).read())
    assert data["task_id"] == "t"
    assert data["cycles_run"] == 1
    assert data["total_tests_passed"] == 7
