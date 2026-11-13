import hashlib
import json
from pathlib import Path

import numpy as np

from neogen.ml._features import compute_features_1044, compute_feature_vector
from neogen.ml.feature_schemas.conformer_3d_v1 import DIMENSIONS as CONFORMER_3D_DIMS
from neogen.ml.feature_schemas.conformer_3d_v1 import FEATURE_NAMES as CONFORMER_3D_NAMES


GOLDEN = Path(__file__).with_name("feature_legacy_100_golden.json")


def _sha_float64(values: np.ndarray) -> str:
    return hashlib.sha256(values.astype("<f8", copy=False).tobytes()).hexdigest()


def test_legacy_2d_v1_is_byte_identical_to_pre_refactor_100_drug_golden():
    records = json.loads(GOLDEN.read_text())
    assert len(records) == 100
    for record in records:
        values = compute_features_1044(record["smiles"])
        assert values is not None
        assert values.shape == (1044,)
        assert _sha_float64(values) == record["sha256_float64_le"]


def test_dispatcher_legacy_matches_public_legacy_function():
    smiles = json.loads(GOLDEN.read_text())[0]["smiles"]
    public = compute_features_1044(smiles)
    dispatched = compute_feature_vector(
        entity_id="fixture_0",
        entity_type="stereoisomer",
        schema_version="legacy_2d_v1",
        smiles=smiles,
    )
    assert public is not None
    np.testing.assert_array_equal(dispatched.values, public)


def test_conformer_3d_v1_schema_slot_remains_stable_after_week3_activation():
    assert CONFORMER_3D_DIMS == 1086
    assert len(CONFORMER_3D_NAMES) == 1086
    assert CONFORMER_3D_NAMES == tuple(sorted(CONFORMER_3D_NAMES))
