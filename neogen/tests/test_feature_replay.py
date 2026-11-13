from fastapi import FastAPI
from fastapi.testclient import TestClient

from neogen.api.routes_features import router as feature_router
from neogen.api.routes_molecules import reset_molecule_route_state, router as molecule_router


def test_feature_vector_replay_matches_by_hash_and_cache_key():
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
    assert resolved.status_code == 200, resolved.text
    stereoisomer_id = resolved.json()["stereoisomer"]["stereoisomer_id"]
    body = {
        "entity_id": stereoisomer_id,
        "entity_type": "stereoisomer",
        "schema_version": "stereo_2d_v1",
    }

    first = api.post("/v1/features/compute", json=body)
    second = api.post("/v1/features/compute", json=body)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["vector_hash"] == second.json()["vector_hash"]
    assert first.json()["feature_vector_id"] == second.json()["feature_vector_id"]
    assert first.json()["events"][1]["event_type"] == "FeatureVectorComputed"
    assert first.json()["events"][1]["seed"] == 0
    assert first.json()["events"][1]["tool_version"]
