from fastapi import FastAPI
from fastapi.testclient import TestClient

from neogen.api.routes_features import router as feature_router
from neogen.api.routes_molecules import reset_molecule_route_state, router as molecule_router


def test_schema_mismatched_model_rejection_via_api():
    reset_molecule_route_state()
    app = FastAPI()
    app.include_router(molecule_router)
    app.include_router(feature_router)
    api = TestClient(app)

    normalized = api.post(
        "/v1/molecules/normalize",
        json={"input": "(R)-warfarin", "input_type": "name"},
    )
    assert normalized.status_code == 200, normalized.text
    resolved = api.post(
        "/v1/molecules/resolve-stereo",
        json={"identity_id": normalized.json()["molecular_identity"]["identity_id"]},
    )
    stereoisomer_id = resolved.json()["stereoisomer"]["stereoisomer_id"]
    vector = api.post(
        "/v1/features/compute",
        json={
            "entity_id": stereoisomer_id,
            "entity_type": "stereoisomer",
            "schema_version": "legacy_2d_v1",
        },
    )
    assert vector.status_code == 200, vector.text
    registered = api.post(
        "/v1/features/register-model",
        json={"model_id": "mdl_pk_stereo2d_v1", "schema_version": "stereo_2d_v1"},
    )
    assert registered.status_code == 200, registered.text

    check = api.post(
        "/v1/features/validate-compatible",
        json={
            "feature_vector_id": vector.json()["feature_vector_id"],
            "model_id": "mdl_pk_stereo2d_v1",
        },
    )

    assert check.status_code == 200, check.text
    assert check.json()["compatible"] is False
    assert "legacy_2d_v1" in check.json()["reason"]
    assert check.json()["events"][0]["event_type"] == "FeatureSchemaMismatch"
