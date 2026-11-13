from neogen.events.store import clear_events, list_events
from neogen.validation.mvp_gate import run_mvp_gate, stable_result_hash


PASS_METRICS = {
    "tier1_production_pct": 74.0,
    "lookup_blocked_tier1_pct": 57.0,
    "chiral_directionality_pct": 88.9,
    "chiral_magnitude_pct": 66.7,
    "replay_parity_pct": 100.0,
    "feature_determinism_pct": 100.0,
}


def test_mvp_gate_returns_five_criterion_result():
    clear_events()
    result = run_mvp_gate(baseline_metrics=PASS_METRICS, persist=False)

    assert result.overall_pass is True
    assert len(result.criterion_results) == 5
    assert set(result.criterion_results) == {
        "tier1_production",
        "lookup_blocked_tier1",
        "chiral_directionality",
        "replay_parity",
        "feature_determinism",
    }


def test_mvp_gate_short_circuits_on_first_failure():
    result = run_mvp_gate(
        baseline_metrics={**PASS_METRICS, "tier1_production_pct": 70.0},
        persist=False,
        short_circuit=True,
    )

    assert result.overall_pass is False
    assert list(result.criterion_results) == ["tier1_production"]
    assert result.recommendation == "pivot"


def test_mvp_gate_emits_completed_event_with_verdict():
    clear_events()
    result = run_mvp_gate(baseline_metrics=PASS_METRICS, persist=False)
    events = list_events()

    assert events
    assert events[-1].event_type == "mvp.gate.completed"
    assert events[-1].verdict == result.verdict
    assert events[-1].result_hash == result.result_hash


def test_mvp_gate_replay_hash_is_byte_identical():
    first = run_mvp_gate(baseline_metrics=PASS_METRICS, persist=False)
    second = run_mvp_gate(baseline_metrics=PASS_METRICS, persist=False)

    assert first.to_dict() == second.to_dict()
    assert first.result_hash == second.result_hash
    assert first.result_hash == stable_result_hash(first.to_dict())
