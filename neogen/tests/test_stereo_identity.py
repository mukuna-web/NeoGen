from fastapi import FastAPI
from fastapi.testclient import TestClient

from neogen.api.routes_molecules import reset_molecule_route_state, router
from neogen.chem.standardize import WARFARIN_FLAT_SMILES, WARFARIN_R_SMILES, WARFARIN_S_SMILES


def client() -> TestClient:
    reset_molecule_route_state()
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def normalize(api: TestClient, value: str, input_type: str = "smiles") -> dict:
    response = api.post(
        "/v1/molecules/normalize",
        json={"input": value, "input_type": input_type},
    )
    assert response.status_code == 200, response.text
    return response.json()


def resolve(api: TestClient, identity_id: str) -> dict:
    response = api.post(
        "/v1/molecules/resolve-stereo",
        json={"identity_id": identity_id},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_warfarin_enantiomers_have_distinct_isomers_and_shared_parent():
    api = client()
    r_identity = normalize(api, "(R)-warfarin", "name")
    s_identity = normalize(api, "(S)-warfarin", "name")

    r_stereo = resolve(api, r_identity["molecular_identity"]["identity_id"])["stereoisomer"]
    s_stereo = resolve(api, s_identity["molecular_identity"]["identity_id"])["stereoisomer"]

    assert r_stereo["parent_id"] == s_stereo["parent_id"]
    assert r_stereo["stereoisomer_id"] != s_stereo["stereoisomer_id"]


def test_racemate_warfarin_resolves_to_two_component_racemic_mixture():
    api = client()
    identity = normalize(api, "warfarin", "name")
    resolved = resolve(api, identity["molecular_identity"]["identity_id"])

    assert resolved["stereo_mixture"]["mixture_type"] == "racemic"
    assert len(resolved["stereo_mixture"]["component_stereoisomer_ids"]) == 2
    assert len(resolved["component_stereoisomers"]) == 2


def test_unspecified_stereo_is_not_fully_specified():
    api = client()
    identity = normalize(api, WARFARIN_FLAT_SMILES, "smiles")
    resolved = resolve(api, identity["molecular_identity"]["identity_id"])

    assert resolved["stereoisomer"]["stereo_ambiguity_level"] in {"partial", "unknown"}
    assert resolved["stereoisomer"]["stereo_ambiguity_level"] != "fully_specified"


def test_identity_hashes_are_identical_across_repeated_runs():
    triples = []
    for _ in range(10):
        api = client()
        normalized = normalize(api, WARFARIN_R_SMILES, "isomeric_smiles")
        resolved = resolve(api, normalized["molecular_identity"]["identity_id"])
        triples.append(
            (
                normalized["molecule_parent"]["parent_id"],
                normalized["molecular_identity"]["identity_id"],
                resolved["stereoisomer"]["stereoisomer_id"],
            )
        )

    assert len(set(triples)) == 1


def test_known_drug_warfarin_defaults_to_racemic_mixture():
    api = client()
    identity = normalize(api, "warfarin", "known_drug_id")
    resolved = resolve(api, identity["molecular_identity"]["identity_id"])

    assert resolved["stereo_mixture"]["mixture_type"] == "racemic"
    assert resolved["stereoisomer"] is None


def test_smiles_with_more_than_80_heavy_atoms_returns_http_400():
    api = client()
    response = api.post(
        "/v1/molecules/normalize",
        json={"input": "C" * 81, "input_type": "smiles"},
    )
    assert response.status_code == 400


def test_enumerate_stereo_respects_64_isomer_cap():
    api = client()
    normalized = normalize(api, "c1ccccc1[C@H](C)N", "isomeric_smiles")
    response = api.post(
        "/v1/molecules/enumerate-stereo",
        json={"identity_id": normalized["molecular_identity"]["identity_id"], "max_isomers": 64},
    )

    assert response.status_code == 200, response.text
    assert len(response.json()["stereoisomers"]) <= 64
