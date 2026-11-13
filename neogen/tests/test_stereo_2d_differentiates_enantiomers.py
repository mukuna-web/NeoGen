from fastapi import FastAPI
from fastapi.testclient import TestClient

from neogen.api.routes_features import router as feature_router
from neogen.api.routes_molecules import reset_molecule_route_state, router as molecule_router


def client() -> TestClient:
    reset_molecule_route_state()
    app = FastAPI()
    app.include_router(molecule_router)
    app.include_router(feature_router)
    return TestClient(app)


def _stereoisomer_id(api: TestClient, name: str) -> str:
    normalized = api.post(
        "/v1/molecules/normalize",
        json={"input": name, "input_type": "name"},
    )
    assert normalized.status_code == 200, normalized.text
    resolved = api.post(
        "/v1/molecules/resolve-stereo",
        json={"identity_id": normalized.json()["molecular_identity"]["identity_id"]},
    )
    assert resolved.status_code == 200, resolved.text
    return resolved.json()["stereoisomer"]["stereoisomer_id"]


def _compute(api: TestClient, stereoisomer_id: str) -> dict:
    response = api.post(
        "/v1/features/compute",
        json={
            "entity_id": stereoisomer_id,
            "entity_type": "stereoisomer",
            "schema_version": "stereo_2d_v1",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_stereo_2d_v1_differentiates_warfarin_enantiomers_by_vector_hash():
    api = client()
    r_vector = _compute(api, _stereoisomer_id(api, "(R)-warfarin"))
    s_vector = _compute(api, _stereoisomer_id(api, "(S)-warfarin"))

    assert r_vector["dimensions"] == 1054
    assert s_vector["dimensions"] == 1054
    assert r_vector["vector_hash"] != s_vector["vector_hash"]


def test_stereo_2d_v1_is_deterministic_for_same_isomeric_smiles():
    first = client()
    second = client()
    first_vector = _compute(first, _stereoisomer_id(first, "(R)-warfarin"))
    second_vector = _compute(second, _stereoisomer_id(second, "(R)-warfarin"))

    assert first_vector["vector_hash"] == second_vector["vector_hash"]
    assert first_vector["feature_vector_id"] == second_vector["feature_vector_id"]
