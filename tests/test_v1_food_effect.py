"""V1 V2-extension — food-effect (fed vs fasted) tests.

These tests cover the first V2 engine extension: a fed-state GI tract
that the PK orchestrator selects when ``PatientProfile.is_fasted`` is
``False``. The fed tract ports well-established clinical physiology:

- delayed gastric emptying (0.25 h → 1.0 h)
- raised duodenum / upper-jejunum bile-salt concentration
- buffered gastric pH (1.5 → 4.5)
- roughly 30 % larger upper-SI luminal volumes

The assertions are intentionally *directional* — we do not pin absolute
Cmax values (those are tuned by the V1 freeze gate benchmark), we only
assert that the fed tract produces the clinically observed directional
change for each drug class:

- BCS I/II oral → slower ka, lower Cmax, similar AUC
- IV → food state has no effect at all

A small API-level check is included so the ``PatientSpec.is_fasted``
flag plumbs correctly through the /v1/simulate endpoint.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from neogen.api.v1.app import build_app
from neogen.core.constants.reference_physiology import build_reference_male
from neogen.core.entities.patient import PatientProfile
from neogen.engine.absorption.gut_transit import (
    build_fed_gi_tract,
    build_gi_tract_for_patient,
    build_reference_gi_tract,
)
from neogen.engine.dose_scheduler import DoseScheduler
from neogen.engine.drug_factory import DrugFactory
from neogen.engine.orchestrator_v2 import OrchestratorV2


# ════════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def factory() -> DrugFactory:
    return DrugFactory()


@pytest.fixture(scope="module")
def engine() -> OrchestratorV2:
    return OrchestratorV2()


@pytest.fixture(scope="module")
def scheduler() -> DoseScheduler:
    return DoseScheduler()


@pytest.fixture(scope="module")
def physiology():
    return build_reference_male()


@pytest.fixture(scope="module")
def api_client() -> TestClient:
    app = build_app(cors_origins=("http://localhost:3000",))
    return TestClient(app)


# ════════════════════════════════════════════════════════════════
# UNIT TESTS — gut_transit module
# ════════════════════════════════════════════════════════════════


class TestFedGITractConstruction:
    """Pure-function checks on the fed-state GI tract."""

    def test_fed_tract_has_buffered_gastric_ph(self) -> None:
        fed = build_fed_gi_tract()
        fasted = build_reference_gi_tract()
        assert fed.segments[0].pH > fasted.segments[0].pH
        # Physiological target: fed stomach sits around pH 4–5.
        assert 3.5 <= fed.segments[0].pH <= 6.0

    def test_fed_tract_slows_gastric_emptying(self) -> None:
        fed = build_fed_gi_tract()
        fasted = build_reference_gi_tract()
        assert fed.gastric_emptying_h > fasted.gastric_emptying_h
        # A meal should not completely freeze gastric transit — a few
        # hours is the clinical upper bound for homogenised liquid meals.
        assert 0.5 <= fed.gastric_emptying_h <= 3.0

    def test_fed_tract_raises_upper_si_bile_salts(self) -> None:
        fed = build_fed_gi_tract()
        fasted = build_reference_gi_tract()
        # Duodenum + upper jejunum pick up gallbladder-emptied bile.
        assert fed.segments[1].bile_salt_mM > fasted.segments[1].bile_salt_mM
        assert fed.segments[2].bile_salt_mM > fasted.segments[2].bile_salt_mM

    def test_fed_tract_expands_upper_si_volume(self) -> None:
        fed = build_fed_gi_tract()
        fasted = build_reference_gi_tract()
        assert fed.segments[1].volume_mL > fasted.segments[1].volume_mL
        assert fed.segments[2].volume_mL > fasted.segments[2].volume_mL

    def test_fed_tract_preserves_ileum_and_colon(self) -> None:
        fed = build_fed_gi_tract()
        fasted = build_reference_gi_tract()
        for idx in range(4, 9):  # Ileum_1 through Colon_trans
            assert fed.segments[idx] == fasted.segments[idx], (
                f"segment {idx} ({fasted.segments[idx].name}) should be "
                "untouched by the fed-state patch"
            )

    def test_dispatcher_fasted_flag(self) -> None:
        fasted = build_gi_tract_for_patient(is_fasted=True)
        fed = build_gi_tract_for_patient(is_fasted=False)
        assert fasted.segments[0].pH == build_reference_gi_tract().segments[0].pH
        assert fed.segments[0].pH == build_fed_gi_tract().segments[0].pH

    def test_fed_tract_is_not_reference(self) -> None:
        # The fed builder must return a fresh tract — never mutate the
        # module-level reference.
        fed = build_fed_gi_tract()
        ref = build_reference_gi_tract()
        assert fed is not ref
        assert fed.segments is not ref.segments


# ════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — orchestrator end-to-end
# ════════════════════════════════════════════════════════════════


class TestFoodEffectOralBCS2:
    """Ibuprofen is the canonical BCS II benchmark. Fed state should
    slow absorption (lower ka, longer tlag, lower Cmax) while AUC stays
    broadly similar — the textbook food-effect signature for an
    already-permeable immediate-release tablet."""

    def _simulate(self, *, engine, factory, scheduler, physiology, fasted: bool):
        drug = factory.create("ibuprofen")
        schedule = scheduler.single_oral("ibuprofen", dose_mg=400.0)
        patient = PatientProfile(is_fasted=fasted)
        return engine.run(drug, schedule, physiology, patient=patient)

    def test_fed_has_slower_effective_ka(
        self, engine, factory, scheduler, physiology
    ) -> None:
        fasted = self._simulate(
            engine=engine, factory=factory, scheduler=scheduler,
            physiology=physiology, fasted=True,
        )
        fed = self._simulate(
            engine=engine, factory=factory, scheduler=scheduler,
            physiology=physiology, fasted=False,
        )
        ka_fasted = fasted.absorption[0].effective_ka_per_h
        ka_fed = fed.absorption[0].effective_ka_per_h
        assert ka_fed < ka_fasted, (
            f"Fed ka ({ka_fed:.3f}/h) must be slower than fasted "
            f"({ka_fasted:.3f}/h) for a BCS II drug after a meal."
        )

    def test_fed_has_longer_lag(
        self, engine, factory, scheduler, physiology
    ) -> None:
        fasted = self._simulate(
            engine=engine, factory=factory, scheduler=scheduler,
            physiology=physiology, fasted=True,
        )
        fed = self._simulate(
            engine=engine, factory=factory, scheduler=scheduler,
            physiology=physiology, fasted=False,
        )
        assert fed.absorption[0].tlag_h > fasted.absorption[0].tlag_h

    def test_fed_cmax_is_reduced(
        self, engine, factory, scheduler, physiology
    ) -> None:
        fasted = self._simulate(
            engine=engine, factory=factory, scheduler=scheduler,
            physiology=physiology, fasted=True,
        )
        fed = self._simulate(
            engine=engine, factory=factory, scheduler=scheduler,
            physiology=physiology, fasted=False,
        )
        fasted_cmax = fasted.pk_result.pk_metrics.cmax_mg_per_L
        fed_cmax = fed.pk_result.pk_metrics.cmax_mg_per_L
        assert fed_cmax < fasted_cmax
        # Clinical fed-ibuprofen Cmax reduction is usually 15–35 %.
        ratio = fed_cmax / fasted_cmax
        assert 0.5 <= ratio <= 0.95, (
            f"fed/fasted Cmax ratio {ratio:.2f} outside "
            "physiological food-effect range [0.5, 0.95]."
        )

    def test_auc_is_broadly_preserved(
        self, engine, factory, scheduler, physiology
    ) -> None:
        # BCS II immediate-release formulations usually show <25 % AUC
        # change with food — the total amount absorbed is preserved,
        # only the rate drifts. We use a loose band so the test stays
        # robust as future calibration tweaks the fed tract.
        fasted = self._simulate(
            engine=engine, factory=factory, scheduler=scheduler,
            physiology=physiology, fasted=True,
        )
        fed = self._simulate(
            engine=engine, factory=factory, scheduler=scheduler,
            physiology=physiology, fasted=False,
        )
        a_fasted = fasted.pk_result.pk_metrics.auc_0_inf_mg_h_per_L
        a_fed = fed.pk_result.pk_metrics.auc_0_inf_mg_h_per_L
        ratio = a_fed / a_fasted
        assert 0.75 <= ratio <= 1.30, (
            f"fed/fasted AUC ratio {ratio:.2f} outside tolerance band."
        )


class TestFoodEffectOralBCS1:
    """Midazolam is the canonical BCS I high-permeability benchmark.
    Food should slow ka and cut Cmax but the Fa ceiling is already 1.0
    so the AUC signal is small."""

    def _simulate(self, *, engine, factory, scheduler, physiology, fasted: bool):
        drug = factory.create("midazolam")
        schedule = scheduler.single_oral("midazolam", dose_mg=7.5)
        patient = PatientProfile(is_fasted=fasted)
        return engine.run(drug, schedule, physiology, patient=patient)

    def test_fed_cmax_drops(
        self, engine, factory, scheduler, physiology
    ) -> None:
        fasted = self._simulate(
            engine=engine, factory=factory, scheduler=scheduler,
            physiology=physiology, fasted=True,
        )
        fed = self._simulate(
            engine=engine, factory=factory, scheduler=scheduler,
            physiology=physiology, fasted=False,
        )
        assert (
            fed.pk_result.pk_metrics.cmax_mg_per_L
            < fasted.pk_result.pk_metrics.cmax_mg_per_L
        )

    def test_fa_near_unity_both_states(
        self, engine, factory, scheduler, physiology
    ) -> None:
        # High-permeability, highly soluble — Fa saturates at the
        # ka-absorbed ceiling in both states. The fed tract must not
        # somehow reduce Fa below ~0.9.
        for fasted in (True, False):
            result = self._simulate(
                engine=engine, factory=factory, scheduler=scheduler,
                physiology=physiology, fasted=fasted,
            )
            assert result.absorption[0].fa >= 0.90


class TestIVRouteIgnoresFoodState:
    """IV bolus doses must be completely unaffected by fed/fasted state —
    absorption is bypassed entirely for the IV path."""

    def test_iv_bolus_cmax_independent_of_meal(
        self, engine, factory, scheduler, physiology
    ) -> None:
        drug = factory.create("midazolam")
        schedule = scheduler.iv_bolus("midazolam", dose_mg=5.0)

        fasted_result = engine.run(
            drug, schedule, physiology,
            patient=PatientProfile(is_fasted=True),
        )
        fed_result = engine.run(
            drug, schedule, physiology,
            patient=PatientProfile(is_fasted=False),
        )

        assert not fasted_result.absorption_profiles
        assert not fed_result.absorption_profiles

        # Byte-identical PK metrics on the IV path.
        assert (
            fasted_result.pk_result.pk_metrics.cmax_mg_per_L
            == pytest.approx(fed_result.pk_result.pk_metrics.cmax_mg_per_L)
        )
        assert (
            fasted_result.pk_result.pk_metrics.auc_0_inf_mg_h_per_L
            == pytest.approx(fed_result.pk_result.pk_metrics.auc_0_inf_mg_h_per_L)
        )


# ════════════════════════════════════════════════════════════════
# API TESTS — PatientSpec.is_fasted plumbing
# ════════════════════════════════════════════════════════════════


class TestAPIFoodEffect:
    """End-to-end API check: submitting ``patient.is_fasted=False``
    through /v1/simulate must drive the Cmax down relative to the
    fasted default for an oral BCS II drug."""

    def _simulate(self, client: TestClient, *, is_fasted: bool | None):
        body = {
            "drug_name": "ibuprofen",
            "dose_mg": 400.0,
            "route": "oral",
        }
        if is_fasted is not None:
            body["patient"] = {"is_fasted": is_fasted}
        response = client.post("/v1/simulate", json=body)
        assert response.status_code == 200, response.text
        return response.json()

    def test_default_matches_explicit_fasted(
        self, api_client: TestClient
    ) -> None:
        default = self._simulate(api_client, is_fasted=None)
        explicit = self._simulate(api_client, is_fasted=True)
        # Same PK metrics — default state is fasted.
        assert default["pk_metrics"]["cmax_mg_per_L"] == pytest.approx(
            explicit["pk_metrics"]["cmax_mg_per_L"]
        )

    def test_fed_lowers_cmax_versus_fasted(
        self, api_client: TestClient
    ) -> None:
        fasted = self._simulate(api_client, is_fasted=True)
        fed = self._simulate(api_client, is_fasted=False)
        assert (
            fed["pk_metrics"]["cmax_mg_per_L"]
            < fasted["pk_metrics"]["cmax_mg_per_L"]
        )
        # Absorption summary should reflect the slower ka as well.
        assert (
            fed["absorption"]["effective_ka_per_h"]
            < fasted["absorption"]["effective_ka_per_h"]
        )
