"""Tests for the NeoGen V2 FastAPI surface.

Uses :class:`fastapi.testclient.TestClient` — no real HTTP server
is started. Each test exercises one endpoint end-to-end against
the V1 engine, so these tests also serve as an integration check
that every router correctly wires schemas ↔ engine ↔ response.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _engine_available() -> bool:
    try:
        from neogen.engine.orchestrator import (  # noqa: F401
            SimulationOrchestrator,
        )
        from neogen.engine.drug_factory import DrugFactory  # noqa: F401
    except Exception:
        return False
    return True


def _fastapi_available() -> bool:
    try:
        from fastapi.testclient import TestClient  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not (_engine_available() and _fastapi_available()),
    reason="V1 engine or FastAPI TestClient not available",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from neogen.api.v2.app import create_v2_app

    app = create_v2_app()
    with TestClient(app) as c:
        yield c


# ════════════════════════════════════════════════════════════════
# HEALTH + INFO
# ════════════════════════════════════════════════════════════════


class TestHealthEndpoints:
    def test_root(self, client) -> None:
        r = client.get("/v2/")
        assert r.status_code == 200
        body = r.json()
        assert body["service"] == "neogen-v2-api"
        assert body["engine_version"].startswith("v1.")
        assert body["available_drugs_count"] >= 20

    def test_health_ok(self, client) -> None:
        r = client.get("/v2/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["imports_ok"] is True

    def test_openapi_exposes_v2_routes(self, client) -> None:
        r = client.get("/v2/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        paths = set(schema.get("paths", {}).keys())
        # Spot-check that the big endpoints are all documented
        assert "/v2/" in paths
        assert "/v2/health" in paths
        assert "/v2/drugs" in paths
        assert "/v2/simulate" in paths
        assert "/v2/ddi" in paths
        assert "/v2/population" in paths
        assert "/v2/validation/tier4" in paths


# ════════════════════════════════════════════════════════════════
# DRUGS
# ════════════════════════════════════════════════════════════════


class TestDrugsEndpoints:
    def test_list_drugs(self, client) -> None:
        r = client.get("/v2/drugs")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == len(body["drugs"])
        assert "midazolam" in body["drugs"]
        assert "ketoconazole" in body["drugs"]

    def test_get_known_drug(self, client) -> None:
        r = client.get("/v2/drugs/midazolam")
        assert r.status_code == 200
        body = r.json()
        assert body.get("molecular_weight") == pytest.approx(325.8, rel=1e-3)

    def test_get_unknown_drug_404(self, client) -> None:
        r = client.get("/v2/drugs/not_a_real_drug_zzzzz")
        assert r.status_code == 404


# ════════════════════════════════════════════════════════════════
# SIMULATE
# ════════════════════════════════════════════════════════════════


class TestSimulateEndpoint:
    def test_iv_bolus_midazolam(self, client) -> None:
        r = client.post(
            "/v2/simulate",
            json={
                "drug_name": "midazolam",
                "dose_mg": 5.0,
                "route": "iv_bolus",
                "return_curves": False,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["drug_name"] == "midazolam"
        assert body["dose_mg"] == 5.0
        assert body["pk_metrics"]["cmax_mg_per_L"] > 0
        assert body["pk_metrics"]["auc_0_inf_mg_h_per_L"] > 0
        assert body["pk_metrics"]["half_life_h"] > 0
        assert body["engine_version"].startswith("v1.")
        # Curves off by default
        assert body["time_h"] is None
        assert body["plasma_conc_mg_per_L"] is None

    def test_simulate_with_curves(self, client) -> None:
        r = client.post(
            "/v2/simulate",
            json={
                "drug_name": "caffeine",
                "dose_mg": 100.0,
                "route": "iv_bolus",
                "return_curves": True,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["time_h"] is not None
        assert body["plasma_conc_mg_per_L"] is not None
        assert len(body["time_h"]) == len(body["plasma_conc_mg_per_L"])
        assert len(body["time_h"]) > 10
        assert body["organ_profiles"] is not None
        assert "liver" in body["organ_profiles"]

    def test_simulate_with_patient_override(self, client) -> None:
        r = client.post(
            "/v2/simulate",
            json={
                "drug_name": "midazolam",
                "dose_mg": 5.0,
                "route": "iv_bolus",
                "patient": {
                    "weight_kg": 100.0,
                    "age_years": 60.0,
                    "sex": "male",
                    "organ_function": {
                        "hepatic": 0.7,
                        "renal": 0.8,
                        "cardiac": 1.0,
                        "pulmonary": 1.0,
                    },
                    "genetics": {
                        "cyp3a4": "poor_metabolizer",
                        "cyp2d6": "normal_metabolizer",
                        "cyp2c9": "normal_metabolizer",
                        "cyp2c19": "normal_metabolizer",
                        "cyp1a2": "normal_metabolizer",
                        "cyp2b6": "normal_metabolizer",
                    },
                },
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["pk_metrics"]["half_life_h"] > 0

    def test_unknown_drug_404(self, client) -> None:
        r = client.post(
            "/v2/simulate",
            json={"drug_name": "bogus_drug_xxx", "dose_mg": 5.0},
        )
        assert r.status_code in (404, 500)

    def test_bad_dose_rejected(self, client) -> None:
        r = client.post(
            "/v2/simulate",
            json={"drug_name": "midazolam", "dose_mg": -1.0},
        )
        assert r.status_code == 422  # pydantic validation


# ════════════════════════════════════════════════════════════════
# DDI
# ════════════════════════════════════════════════════════════════


class TestDDIEndpoint:
    def test_ketoconazole_midazolam_reversible(self, client) -> None:
        r = client.post(
            "/v2/ddi",
            json={
                "victim_name": "midazolam",
                "victim_dose_mg": 5.0,
                "victim_route": "iv_bolus",
                "perpetrator_name": "ketoconazole",
                "perpetrator_iu_um": 0.5,
                "mode": "reversible",
                "duration_on_h": 500.0,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["aucr_predicted"] > 1.0  # inhibition raises AUC
        assert body["auc_baseline_mg_h_per_L"] > 0
        assert body["auc_ddi_mg_h_per_L"] > body["auc_baseline_mg_h_per_L"]
        # Reference pair match → clinical_aucr echoed
        assert body["clinical_aucr"] == 5.0
        assert body["aucr_predicted_within_2_fold_of_clinical"] is True

    def test_mbi_pair_runs(self, client) -> None:
        r = client.post(
            "/v2/ddi",
            json={
                "victim_name": "midazolam",
                "victim_dose_mg": 5.0,
                "victim_route": "iv_bolus",
                "perpetrator_name": "clarithromycin",
                "perpetrator_iu_um": 1.0,
                "mode": "mbi",
                "duration_on_h": 336.0,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["aucr_predicted"] > 1.0
        assert body["mode"] == "mbi"

    def test_unknown_victim_404(self, client) -> None:
        r = client.post(
            "/v2/ddi",
            json={
                "victim_name": "bogus_drug_xxx",
                "victim_dose_mg": 5.0,
                "perpetrator_name": "ketoconazole",
                "perpetrator_iu_um": 0.5,
                "mode": "reversible",
            },
        )
        assert r.status_code in (404, 500)


# ════════════════════════════════════════════════════════════════
# POPULATION
# ════════════════════════════════════════════════════════════════


class TestPopulationEndpoint:
    def test_small_monte_carlo(self, client) -> None:
        r = client.post(
            "/v2/population",
            json={
                "drug_name": "midazolam",
                "dose_mg": 5.0,
                "route": "iv_bolus",
                "n_patients": 5,
                "seed": 42,
                "label": "api_smoke",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["n_requested"] == 5
        assert body["n_completed"] >= 3
        assert body["seed"] == 42
        assert body["label"] == "api_smoke"
        assert "cmax_mg_per_L" in body["summaries"]
        assert body["summaries"]["cmax_mg_per_L"]["n"] >= 3
        assert body["demographics"]["n"] == 5

    def test_n_patients_cap(self, client) -> None:
        r = client.post(
            "/v2/population",
            json={
                "drug_name": "midazolam",
                "dose_mg": 5.0,
                "route": "iv_bolus",
                "n_patients": 10_000,  # above 500 cap
            },
        )
        assert r.status_code == 422


# ════════════════════════════════════════════════════════════════
# VALIDATION
# ════════════════════════════════════════════════════════════════


class TestValidationEndpoints:
    @pytest.mark.xfail(
        reason="KNOWN_GAPS.md #1: the organ-integrated mass-balance "
               "invariant closes to 4.02% against a 2% tolerance, so the "
               "Tier-4 gate reports overall_pass=False (7/8 invariants pass)",
        strict=False,
    )
    def test_tier4_runs(self, client) -> None:
        r = client.post(
            "/v2/validation/tier4",
            json={"save_report": False},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["overall_pass"] is True  # gate is advisory-aware
        assert body["report"]["tier4"]["n_total"] == 8

    def test_latest_report_before_save_fails_gracefully(self, client) -> None:
        # May or may not exist depending on order — both are acceptable
        r = client.get("/v2/validation/latest")
        assert r.status_code in (200, 404)

    def test_full_async_tier4_only(self, client) -> None:
        """Async Tier-4-only run should complete within a few seconds.

        The endpoint returns immediately with a task_id; the test
        polls the status endpoint until the task finishes (or fails
        after 30 seconds to keep the suite fast).
        """
        import time

        kick = client.post(
            "/v2/validation/full/async",
            json={
                "include_tier1": False,
                "include_tier2": False,
                "include_tier3": False,
                "include_tier4": True,
                "save_report": False,
            },
        )
        assert kick.status_code == 200
        start_body = kick.json()
        task_id = start_body["task_id"]
        assert start_body["status"] in ("pending", "running", "done")
        assert start_body["status_url"].endswith(task_id)

        # Poll for up to 30 s.
        deadline = time.time() + 30.0
        final = None
        while time.time() < deadline:
            status_resp = client.get(f"/v2/validation/status/{task_id}")
            assert status_resp.status_code == 200
            body = status_resp.json()
            if body["status"] in ("done", "failed"):
                final = body
                break
            time.sleep(0.1)

        assert final is not None, "task did not complete within 30 s"
        assert final["status"] == "done", f"task failed: {final.get('error')}"
        assert final["result"] is not None
        assert final["result"]["report"]["tier4"]["n_total"] == 8

    def test_status_unknown_task_404(self, client) -> None:
        r = client.get("/v2/validation/status/nonexistent_task_id")
        assert r.status_code == 404
