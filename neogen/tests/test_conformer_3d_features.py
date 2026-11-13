from neogen.chem.conformers import ConformerConfig, generate_conformer_set
from neogen.chem.standardize import WARFARIN_R_SMILES, molecular_identity_from_input
from neogen.chem.stereo import resolve_stereoisomer
from neogen.ml._features import vector_hash
from neogen.ml.feature_schemas import conformer_3d_v1


def _isomer(smiles: str, input_type: str = "isomeric_smiles"):
    _parent, identity, _mol = molecular_identity_from_input(smiles, input_type)
    return resolve_stereoisomer(identity)


def test_conformer_3d_compute_returns_1086_dimensions(tmp_path):
    isomer = _isomer(WARFARIN_R_SMILES)
    conformer_set = generate_conformer_set(
        isomer,
        ConformerConfig(seed=42, max_conformers=4, num_candidates=12),
        cache_root=tmp_path,
    )

    values = conformer_3d_v1.compute(conformer_set, isomer)

    assert values is not None
    assert values.shape == (1086,)


def test_caffeine_boltzmann_3d_descriptors_are_in_smoke_ballparks(tmp_path):
    isomer = _isomer("Cn1cnc2c1c(=O)n(C)c(=O)n2C", "smiles")
    conformer_set = generate_conformer_set(
        isomer,
        ConformerConfig(seed=42, max_conformers=10, num_candidates=50, prune_rmsd=0.0),
        cache_root=tmp_path,
    )

    tail = conformer_3d_v1.compute_3d_tail(conformer_set, isomer)
    asphericity_mean = tail[0]
    radius_gyration_mean = tail[16]
    psa_mean = tail[18]
    sasa_mean = tail[20]
    n_conformers = tail[22]

    assert 0.0 <= asphericity_mean <= 1.0
    assert 1.0 <= radius_gyration_mean <= 6.0
    assert 40.0 <= psa_mean <= 100.0
    assert 80.0 <= sasa_mean <= 250.0
    assert 1 <= n_conformers <= 10


def test_conformer_3d_replay_vector_hash_is_byte_identical(tmp_path):
    isomer = _isomer(WARFARIN_R_SMILES)
    config = ConformerConfig(seed=42, max_conformers=4, num_candidates=12)
    first_set = generate_conformer_set(isomer, config, cache_root=tmp_path)
    second_set = generate_conformer_set(isomer, config, cache_root=tmp_path)

    first = conformer_3d_v1.compute(first_set, isomer)
    second = conformer_3d_v1.compute(second_set, isomer)

    assert vector_hash(first) == vector_hash(second)
