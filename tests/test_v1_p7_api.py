"""V1 P7 — FastAPI simulation API tests.

Uses FastAPI's ``TestClient`` (synchronous) against an app built with
an explicit CORS allow-list so the unit log stays clean. The test
module is intentionally small: every route gets one happy-path check,
plus a sprinkle of validation failure cases so the OpenAPI contract
actually means something.

These tests run the real V1 engine — they are not mocked. That is
intentional: the API layer is a thin translator, and the scariest
failure mode is that a schema change drifts away from the engine
dataclass shape. Running the engine end-to-end catches that drift.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from neogen.api.v1.app import build_app
from neogen.engine.orchestrator_v2 import ENGINE_VERSION


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Module-scoped client so the engine is constructed once."""
    app = build_app(cors_origins=("http://localhost:3000",))
    return TestClient(app)


# ════════════════════════════════════════════════════════════════
# META ENDPOINTS
# ════════════════════════════════════════════════════════════════


class TestHealth:
    def test_health_returns_ok_and_engine_version(self, client: TestClient) -> None:
        response = client.get("/v1/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["engine_version"] == ENGINE_VERSION


class TestDrugsList:
    def test_lists_literature_drugs(self, client: TestClient) -> None:
        response = client.get("/v1/drugs")
        assert response.status_code == 200
        payload = response.json()
        assert payload["count"] > 0
        drugs = payload["drugs"]
        # A handful of reference drugs we know are in the literature DB.
        for name in ("midazolam", "ketoconazole", "rifampicin", "caffeine"):
            assert name in drugs, f"expected {name!r} in drug list"
        # Sorted ascending — the API guarantees this.
        assert drugs == sorted(drugs)


# ════════════════════════════════════════════════════════════════
# /simulate  (MONOTHERAPY)
# ════════════════════════════════════════════════════════════════


class TestSimulate:
    def test_midazolam_iv_returns_pk_metrics(self, client: TestClient) -> None:
        response = client.post(
            "/v1/simulate",
            json={
                "drug_name": "midazolam",
                "dose_mg": 5.0,
                "route": "iv",
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["engine_version"] == ENGINE_VERSION
        assert payload["drug_name"] == "midazolam"
        assert payload["dose_mg"] == 5.0
        assert payload["route"] == "iv"
        assert payload["ddi_applied"] is False
        assert payload["n_perpetrators"] == 0
        assert payload["time_series"] is None

        pk = payload["pk_metrics"]
        # Sanity envelope from the benchmark baseline — not a tight
        # numeric check because we only want to prove the pipe plumbs
        # real values end-to-end.
        assert pk["cmax_mg_per_L"] > 0
        assert pk["auc_0_inf_mg_h_per_L"] > 0
        assert pk["half_life_h"] > 0
        assert pk["tmax_h"] >= 0

        clearance = payload["clearance"]
        assert clearance["hepatic_L_per_h"] > 0
        assert clearance["total_L_per_h"] >= clearance["hepatic_L_per_h"]

    def test_caffeine_oral_time_series_included_on_request(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/v1/simulate",
            json={
                "drug_name": "caffeine",
                "dose_mg": 200.0,
                "route": "oral",
                "include_time_series": True,
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        ts = payload["time_series"]
        assert ts is not None
        assert len(ts["time_h"]) == len(ts["plasma_conc_mg_per_L"])
        assert len(ts["time_h"]) > 5
        assert ts["time_h"][0] == pytest.approx(0.0, abs=1e-6)

    def test_route_case_insensitive(self, client: TestClient) -> None:
        response = client.post(
            "/v1/simulate",
            json={
                "drug_name": "metformin",
                "dose_mg": 500.0,
                "route": "ORAL",
            },
        )
        assert response.status_code == 200
        assert response.json()["route"] == "oral"

    def test_unknown_drug_rejected_with_400(self, client: TestClient) -> None:
        response = client.post(
            "/v1/simulate",
            json={
                "drug_name": "not_a_real_drug",
                "dose_mg": 10.0,
                "route": "iv",
            },
        )
        assert response.status_code == 400
        payload = response.json()
        assert payload["error"].startswith("HTTP")
        assert "not_a_real_drug" in payload["detail"]

    def test_bad_route_rejected_with_422(self, client: TestClient) -> None:
        response = client.post(
            "/v1/simulate",
            json={
                "drug_name": "midazolam",
                "dose_mg": 5.0,
                "route": "intramuscular",
            },
        )
        assert response.status_code == 422
        assert response.json()["error"] == "ValidationError"

    def test_negative_dose_rejected_with_422(self, client: TestClient) -> None:
        response = client.post(
            "/v1/simulate",
            json={
                "drug_name": "midazolam",
                "dose_mg": -1.0,
                "route": "iv",
            },
        )
        assert response.status_code == 422

    def test_patient_spec_applied_without_error(self, client: TestClient) -> None:
        """Patient adjustments — CKD renal scaling and CYP2D6 PM — should
        route through :func:`_patient_spec_to_profile` and the engine's
        population adapter without raising."""
        response = client.post(
            "/v1/simulate",
            json={
                "drug_name": "metoprolol",
                "dose_mg": 50.0,
                "route": "oral",
                "patient": {
                    "age_years": 72.0,
                    "weight_kg": 65.0,
                    "gfr_ml_per_min": 45.0,
                    "cyp_phenotype": "CYP2D6_PM",
                },
            },
        )
        assert response.status_code == 200, response.text
        pk = response.json()["pk_metrics"]
        assert pk["auc_0_inf_mg_h_per_L"] > 0


# ════════════════════════════════════════════════════════════════
# /simulate_ddi
# ════════════════════════════════════════════════════════════════


class TestSimulateDDI:
    def test_ketoconazole_plus_midazolam_auc_rises(
        self, client: TestClient
    ) -> None:
        """Canonical CYP3A4 DDI: ketoconazole should raise midazolam
        AUC substantially. We compare monotherapy vs DDI through the
        API and assert the expected direction + order of magnitude."""

        mono = client.post(
            "/v1/simulate",
            json={
                "drug_name": "midazolam",
                "dose_mg": 5.0,
                "route": "iv",
            },
        )
        assert mono.status_code == 200
        mono_auc = mono.json()["pk_metrics"]["auc_0_inf_mg_h_per_L"]

        ddi = client.post(
            "/v1/simulate_ddi",
            json={
                "victim_drug_name": "midazolam",
                "victim_dose_mg": 5.0,
                "victim_route": "iv",
                "perpetrators": [
                    {
                        "drug_name": "ketoconazole",
                        "steady_state_conc_um": 0.5,
                        "duration_on_h": 72.0,
                        "include_enzyme_pool": False,
                    }
                ],
            },
        )
        assert ddi.status_code == 200, ddi.text
        payload = ddi.json()
        assert payload["ddi_applied"] is True
        assert payload["n_perpetrators"] == 1
        ddi_auc = payload["pk_metrics"]["auc_0_inf_mg_h_per_L"]

        aucr = ddi_auc / mono_auc
        assert aucr > 2.0, (
            f"expected ketoconazole+midazolam AUCR > 2, got {aucr:.2f}"
        )

    def test_rifampicin_plus_midazolam_auc_drops(
        self, client: TestClient
    ) -> None:
        """Canonical CYP3A4 inducer: rifampicin should lower midazolam
        AUC. The 7-day default pretreatment in PerpetratorSpec makes
        this fire at the steady-state enzyme plateau."""

        mono = client.post(
            "/v1/simulate",
            json={
                "drug_name": "midazolam",
                "dose_mg": 5.0,
                "route": "iv",
            },
        )
        mono_auc = mono.json()["pk_metrics"]["auc_0_inf_mg_h_per_L"]

        ddi = client.post(
            "/v1/simulate_ddi",
            json={
                "victim_drug_name": "midazolam",
                "victim_dose_mg": 5.0,
                "victim_route": "iv",
                "perpetrators": [
                    {
                        "drug_name": "rifampicin",
                        "steady_state_conc_um": 0.2,
                        "duration_on_h": 72.0,
                        "include_enzyme_pool": True,
                    }
                ],
            },
        )
        assert ddi.status_code == 200, ddi.text
        ddi_auc = ddi.json()["pk_metrics"]["auc_0_inf_mg_h_per_L"]
        aucr = ddi_auc / mono_auc
        assert aucr < 1.0, (
            f"expected rifampicin+midazolam AUCR < 1.0, got {aucr:.2f}"
        )

    def test_empty_perpetrator_list_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/v1/simulate_ddi",
            json={
                "victim_drug_name": "midazolam",
                "victim_dose_mg": 5.0,
                "victim_route": "iv",
                "perpetrators": [],
            },
        )
        assert response.status_code == 422

    def test_unknown_perpetrator_rejected_with_400(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/v1/simulate_ddi",
            json={
                "victim_drug_name": "midazolam",
                "victim_dose_mg": 5.0,
                "victim_route": "iv",
                "perpetrators": [
                    {
                        "drug_name": "not_a_drug",
                        "steady_state_conc_um": 0.5,
                    }
                ],
            },
        )
        assert response.status_code == 400


# ════════════════════════════════════════════════════════════════
# OPENAPI SCHEMA
# ════════════════════════════════════════════════════════════════


class TestOpenAPIDocs:
    def test_openapi_json_is_well_formed(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        assert spec["info"]["version"] == ENGINE_VERSION
        paths = spec["paths"]
        assert "/v1/health" in paths
        assert "/v1/drugs" in paths
        assert "/v1/simulate" in paths
        assert "/v1/simulate_ddi" in paths
        assert "/v1/benchmarks/latest" in paths
        # Schemas registered for the response envelope.
        components = spec["components"]["schemas"]
        assert "SimulationResponse" in components
        assert "DDISimulationRequest" in components
        assert "ErrorResponse" in components
        # New enrichment sub-schemas from the P6 response expansion.
        assert "AbsorptionSummary" in components
        assert "DDISummaryResponse" in components
        assert "PDOutcomeResponse" in components
        assert "PerpetratorSummary" in components
        assert "ClinicalEffectResponse" in components
        # Benchmark endpoint models.
        assert "BenchmarkLatestResponse" in components
        assert "BenchmarkMetadata" in components
        assert "BenchmarkHeadline" in components
        assert "BenchmarkGate" in components


# ════════════════════════════════════════════════════════════════
# RESPONSE ENRICHMENT — absorption, DDI summary, PD outcome
# ════════════════════════════════════════════════════════════════


class TestAbsorptionSummary:
    """P2 absorption should be surfaced on oral /simulate responses,
    and absent on IV responses."""

    def test_oral_response_includes_absorption_summary(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/v1/simulate",
            json={
                "drug_name": "caffeine",
                "dose_mg": 200.0,
                "route": "oral",
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        absorption = payload["absorption"]
        assert absorption is not None
        assert absorption["drug_name"] == "caffeine"
        assert absorption["dose_mg"] == 200.0
        # Scalars must be well-formed numbers in their physical range.
        assert 0.0 <= absorption["fa"] <= 1.0 + 1e-6
        assert 0.0 <= absorption["fg"] <= 1.0 + 1e-6
        assert 0.0 <= absorption["f_portal"] <= 1.0 + 1e-6
        assert absorption["effective_ka_per_h"] >= 0.0
        assert absorption["tlag_h"] >= 0.0
        assert absorption["solver_success"] is True
        # BCS class is surfaced as the enum's string value.
        assert absorption["bcs_class"] is None or absorption["bcs_class"].startswith(
            "BCS_"
        )

    def test_iv_response_has_no_absorption_summary(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/v1/simulate",
            json={
                "drug_name": "midazolam",
                "dose_mg": 5.0,
                "route": "iv",
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        # IV schedules bypass the P2 engine entirely — the field should
        # serialise as null, not as an empty object.
        assert payload["absorption"] is None


class TestDDISummaryResponse:
    """/simulate_ddi should expose the perpetrator list, the mechanism
    label, and a dominant-mechanism summary. Monotherapy /simulate should
    not."""

    def test_ketoconazole_ddi_summary_labels_inhibition(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/v1/simulate_ddi",
            json={
                "victim_drug_name": "midazolam",
                "victim_dose_mg": 5.0,
                "victim_route": "iv",
                "perpetrators": [
                    {
                        "drug_name": "ketoconazole",
                        "steady_state_conc_um": 0.5,
                        "duration_on_h": 72.0,
                        "include_enzyme_pool": False,
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        summary = payload["ddi_summary"]
        assert summary is not None
        assert summary["n_perpetrators"] == 1
        perp = summary["perpetrators"][0]
        assert perp["drug_name"] == "ketoconazole"
        assert perp["steady_state_conc_um"] == 0.5
        assert perp["duration_on_h"] == 72.0
        # Ketoconazole is a reversible CYP3A4 inhibitor — not an MBI,
        # not an inducer.
        assert perp["mechanism"] in ("inhibition", "mbi")
        assert summary["dominant_mechanism"] == perp["mechanism"]

    def test_rifampicin_ddi_summary_labels_induction(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/v1/simulate_ddi",
            json={
                "victim_drug_name": "midazolam",
                "victim_dose_mg": 5.0,
                "victim_route": "iv",
                "perpetrators": [
                    {
                        "drug_name": "rifampicin",
                        "steady_state_conc_um": 0.2,
                        "duration_on_h": 72.0,
                        "include_enzyme_pool": True,
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text
        summary = response.json()["ddi_summary"]
        assert summary is not None
        assert summary["n_perpetrators"] == 1
        perp = summary["perpetrators"][0]
        assert perp["drug_name"] == "rifampicin"
        # Rifampicin is the canonical CYP3A4 inducer.
        assert perp["mechanism"] == "induction"
        assert summary["dominant_mechanism"] == "induction"

    def test_monotherapy_has_no_ddi_summary(self, client: TestClient) -> None:
        response = client.post(
            "/v1/simulate",
            json={
                "drug_name": "midazolam",
                "dose_mg": 5.0,
                "route": "iv",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        # Monotherapy must not synthesise a fake DDI envelope.
        assert payload["ddi_summary"] is None
        assert payload["ddi_applied"] is False
        assert payload["n_perpetrators"] == 0


class TestPDOutcomeResponse:
    """The PD envelope is populated only when the underlying drug has a
    PDProfile. Drugs without one must return ``pd_outcome: null``."""

    def test_drug_without_pd_profile_returns_null(
        self, client: TestClient
    ) -> None:
        # Caffeine is PK-only in the literature set — no PDProfile.
        response = client.post(
            "/v1/simulate",
            json={
                "drug_name": "caffeine",
                "dose_mg": 200.0,
                "route": "oral",
            },
        )
        assert response.status_code == 200
        assert response.json()["pd_outcome"] is None

    def test_pd_outcome_shape_when_present(self, client: TestClient) -> None:
        """Find a drug in the literature set that has a PDProfile and
        verify the outcome payload carries a verdict string. We discover
        the drug dynamically so this test stays green as PDProfiles are
        added/removed."""
        from neogen.core.constants.literature_params import LITERATURE_DRUGS
        from neogen.engine.drug_factory import DrugFactory

        factory = DrugFactory()
        pd_drug_name: str = ""
        for name in LITERATURE_DRUGS:
            drug = factory.create(name)
            if drug.pd_profile is not None and drug.pd_profile.has_any_pd_data:
                pd_drug_name = name
                break

        if not pd_drug_name:
            pytest.skip("No literature drug currently carries a PDProfile.")

        response = client.post(
            "/v1/simulate",
            json={
                "drug_name": pd_drug_name,
                "dose_mg": 5.0,
                "route": "iv",
            },
        )
        assert response.status_code == 200, response.text
        pd_payload = response.json()["pd_outcome"]
        assert pd_payload is not None
        assert isinstance(pd_payload["verdict"], str)
        assert len(pd_payload["verdict"]) > 0
        assert pd_payload["n_toxicity_effects"] >= 0
        # Optional scalars must either be None or numeric — never
        # serialised as empty strings.
        for key in (
            "peak_effect_fraction",
            "time_to_peak_effect_h",
            "hysteresis_lag_h",
            "effect_site_cmax_mg_per_L",
            "receptor_peak_occupancy",
            "receptor_time_above_therapeutic_h",
            "therapeutic_index",
        ):
            value = pd_payload[key]
            assert value is None or isinstance(value, (int, float))


# ════════════════════════════════════════════════════════════════
# clinical_report — plaintext passthrough
# ════════════════════════════════════════════════════════════════


class TestClinicalReport:
    """``include_clinical_report=True`` should echo the engine's layered
    plaintext summary lines on the response envelope. The default
    (``False``) path must keep the field null so existing callers see
    byte-identical payloads."""

    def test_default_omits_clinical_report(self, client: TestClient) -> None:
        response = client.post(
            "/v1/simulate",
            json={
                "drug_name": "midazolam",
                "dose_mg": 5.0,
                "route": "iv",
            },
        )
        assert response.status_code == 200
        assert response.json()["clinical_report"] is None

    def test_simulate_includes_clinical_report_when_requested(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/v1/simulate",
            json={
                "drug_name": "caffeine",
                "dose_mg": 200.0,
                "route": "oral",
                "include_clinical_report": True,
            },
        )
        assert response.status_code == 200, response.text
        lines = response.json()["clinical_report"]
        assert lines is not None
        assert isinstance(lines, list)
        assert len(lines) >= 2
        # The engine header always starts with "== <drug>".
        assert lines[0].startswith("== caffeine")
        # Caffeine has no PDProfile → engine emits the PK-only fallback.
        assert any("PK only" in line or "PK:" in line for line in lines)

    def test_simulate_ddi_includes_clinical_report_when_requested(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/v1/simulate_ddi",
            json={
                "victim_drug_name": "midazolam",
                "victim_dose_mg": 5.0,
                "victim_route": "iv",
                "include_clinical_report": True,
                "perpetrators": [
                    {
                        "drug_name": "ketoconazole",
                        "steady_state_conc_um": 0.5,
                        "duration_on_h": 72.0,
                        "include_enzyme_pool": False,
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text
        lines = response.json()["clinical_report"]
        assert lines is not None
        # The clinical report must at least name the victim drug.
        assert any("midazolam" in line.lower() for line in lines)


# ════════════════════════════════════════════════════════════════
# /benchmarks/latest
# ════════════════════════════════════════════════════════════════


class TestBenchmarksLatest:
    """The benchmark endpoint should surface the newest
    ``v1_benchmark_*.json`` artefact with headline + PK + DDI gate
    summaries. Tests write synthetic artefacts into a scratch directory
    that the endpoint resolves via ``NEOGEN_BENCHMARKS_DIR``.
    """

    @staticmethod
    def _write_artefact(directory, filename: str, payload: dict) -> None:
        import json

        (directory / filename).write_text(json.dumps(payload))

    @staticmethod
    def _sample_payload(*, combined_pct: float = 67.0) -> dict:
        return {
            "metadata": {
                "timestamp_utc": "2026-04-11T20:19:06+00:00",
                "engine_version": "v1.0.0-p5",
                "host": "test-host",
                "git_sha": "deadbeef",
                "notes": "",
            },
            "headline": {
                "total_scored": 119,
                "total_within_tolerance": 80,
                "combined_pct_within_tolerance": combined_pct,
                "passes_v1_gate": True,
            },
            "pk": {
                "engine_version": "v1.0.0-p5",
                "n_cases": 46,
                "n_successful": 46,
                "total_scored": 89,
                "total_within_tolerance": 60,
                "overall_pct_within_tolerance": 67.4,
                "overall_gmfe": 1.65,
                "passes_v1_gate": True,
                "per_metric": {},
                "cases": [],
            },
            "ddi": {
                "engine_version": "v1.0.0-p5",
                "n_cases": 15,
                "n_successful": 15,
                "passes_v1_gate": True,
                "aucr_summary": {},
                "cases": [],
            },
        }

    def test_returns_404_when_directory_is_empty(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("NEOGEN_BENCHMARKS_DIR", str(tmp_path))
        # Build a fresh client so the env var is picked up without any
        # module-level caching surprises.
        app = build_app(cors_origins=("http://localhost:3000",))
        fresh_client = TestClient(app)
        response = fresh_client.get("/v1/benchmarks/latest")
        assert response.status_code == 404
        assert "No benchmark artefacts" in response.json()["detail"]

    def test_returns_newest_artefact_by_filename(
        self, tmp_path, monkeypatch
    ) -> None:
        # Two artefacts — the ISO timestamp in the filename determines
        # which wins. The endpoint sorts lexicographically so the later
        # timestamp must be picked regardless of mtime.
        older = self._sample_payload(combined_pct=55.0)
        older["metadata"]["timestamp_utc"] = "2026-04-10T10:00:00+00:00"
        newer = self._sample_payload(combined_pct=67.2)
        newer["metadata"]["timestamp_utc"] = "2026-04-11T20:19:06+00:00"

        self._write_artefact(
            tmp_path, "v1_benchmark_2026-04-10T10-00-00Z.json", older
        )
        self._write_artefact(
            tmp_path, "v1_benchmark_2026-04-11T20-19-06Z.json", newer
        )

        monkeypatch.setenv("NEOGEN_BENCHMARKS_DIR", str(tmp_path))
        app = build_app(cors_origins=("http://localhost:3000",))
        fresh_client = TestClient(app)

        response = fresh_client.get("/v1/benchmarks/latest")
        assert response.status_code == 200, response.text
        payload = response.json()

        assert payload["filename"] == "v1_benchmark_2026-04-11T20-19-06Z.json"
        assert payload["metadata"]["timestamp_utc"] == "2026-04-11T20:19:06+00:00"
        assert payload["headline"]["combined_pct_within_tolerance"] == 67.2
        assert payload["headline"]["passes_v1_gate"] is True
        assert payload["pk"]["n_cases"] == 46
        assert payload["pk"]["overall_pct_within_tolerance"] == 67.4
        assert payload["ddi"]["n_cases"] == 15
        assert payload["ddi"]["passes_v1_gate"] is True

    def test_returns_500_when_artefact_is_corrupt(
        self, tmp_path, monkeypatch
    ) -> None:
        # A benchmark file that is not valid JSON should surface as a
        # 500 with a readable error rather than crash the handler.
        (tmp_path / "v1_benchmark_2026-04-11T99-99-99Z.json").write_text(
            "{not valid json"
        )
        monkeypatch.setenv("NEOGEN_BENCHMARKS_DIR", str(tmp_path))
        app = build_app(cors_origins=("http://localhost:3000",))
        fresh_client = TestClient(app)
        response = fresh_client.get("/v1/benchmarks/latest")
        assert response.status_code == 500
        assert "Could not read benchmark artefact" in response.json()["detail"]


# ════════════════════════════════════════════════════════════════
# /simulate_ddi — full enriched envelope integration
# ════════════════════════════════════════════════════════════════


class TestSimulateDDIFullEnvelope:
    """End-to-end smoke test that exercises every optional sub-object in
    a single DDI call: absorption (oral victim), ddi_summary, clinical
    report, and time series. pd_outcome is asserted to be either None
    or a valid dict depending on whether the victim has a PDProfile."""

    def test_ddi_response_shape_with_all_extensions(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/v1/simulate_ddi",
            json={
                "victim_drug_name": "midazolam",
                "victim_dose_mg": 7.5,
                "victim_route": "oral",
                "include_time_series": True,
                "include_clinical_report": True,
                "perpetrators": [
                    {
                        "drug_name": "ketoconazole",
                        "steady_state_conc_um": 0.5,
                        "duration_on_h": 72.0,
                        "include_enzyme_pool": False,
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()

        # Core PK envelope
        assert payload["drug_name"] == "midazolam"
        assert payload["route"] == "oral"
        assert payload["pk_metrics"]["cmax_mg_per_L"] > 0
        assert payload["pk_metrics"]["auc_0_inf_mg_h_per_L"] > 0

        # Absorption sub-object — oral path must populate it with the
        # canonical midazolam BCS class string (BCS_I).
        absorption = payload["absorption"]
        assert absorption is not None
        assert absorption["drug_name"] == "midazolam"
        assert 0.0 <= absorption["fa"] <= 1.0
        assert 0.0 <= absorption["fg"] <= 1.0
        assert 0.0 <= absorption["f_portal"] <= 1.0
        assert absorption["solver_success"] is True
        # BCS class may be None if the literature entry omits it, but
        # when present it must be a string starting with "BCS_".
        if absorption["bcs_class"] is not None:
            assert absorption["bcs_class"].startswith("BCS_")

        # DDI sub-object
        ddi = payload["ddi_summary"]
        assert ddi is not None
        assert ddi["n_perpetrators"] == 1
        assert ddi["perpetrators"][0]["drug_name"] == "ketoconazole"
        assert ddi["dominant_mechanism"] in ("inhibition", "mbi", "mixed")

        # Time series
        ts = payload["time_series"]
        assert ts is not None
        assert len(ts["time_h"]) == len(ts["plasma_conc_mg_per_L"])
        assert len(ts["time_h"]) > 10

        # Clinical report
        report = payload["clinical_report"]
        assert report is not None
        assert any("midazolam" in line.lower() for line in report)

        # PD outcome — null OR a dict with a verdict. We don't hard-code
        # which because the literature PDProfile for midazolam may toggle
        # between releases.
        pd_outcome = payload["pd_outcome"]
        assert pd_outcome is None or isinstance(pd_outcome.get("verdict"), str)
