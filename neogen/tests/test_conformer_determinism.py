import time

from neogen.chem import artifacts
from neogen.chem.conformers import ConformerConfig, generate_conformer_set
from neogen.chem.standardize import WARFARIN_R_SMILES, WARFARIN_S_SMILES, molecular_identity_from_input
from neogen.chem.stereo import resolve_stereoisomer
from neogen.events.store import clear_events, list_events


def _isomer(smiles: str):
    _parent, identity, _mol = molecular_identity_from_input(smiles, "isomeric_smiles")
    return resolve_stereoisomer(identity)


def test_same_stereoisomer_seed_replays_identical_conformer_set(tmp_path):
    isomer = _isomer(WARFARIN_R_SMILES)
    config = ConformerConfig(seed=42, max_conformers=10, num_candidates=20)

    first = generate_conformer_set(isomer, config, cache_root=tmp_path)
    second = generate_conformer_set(isomer, config, cache_root=tmp_path)

    assert first.conformer_set_id == second.conformer_set_id
    assert first.config_hash == second.config_hash
    assert [c.coordinate_hash for c in first.conformers] == [c.coordinate_hash for c in second.conformers]


def test_warfarin_enantiomers_have_different_conformer_sets(tmp_path):
    config = ConformerConfig(seed=42, max_conformers=5, num_candidates=12)

    r_set = generate_conformer_set(_isomer(WARFARIN_R_SMILES), config, cache_root=tmp_path)
    s_set = generate_conformer_set(_isomer(WARFARIN_S_SMILES), config, cache_root=tmp_path)

    assert r_set.stereoisomer_id != s_set.stereoisomer_id
    assert r_set.conformer_set_id != s_set.conformer_set_id


def test_energy_ordering_and_rmsd_cap_are_stored_by_rank(tmp_path):
    isomer = _isomer(WARFARIN_R_SMILES)
    config = ConformerConfig(seed=42, max_conformers=4, num_candidates=16)

    conformer_set = generate_conformer_set(isomer, config, cache_root=tmp_path)

    energies = [item.energy_kcal_mol for item in conformer_set.conformers]
    assert energies == sorted(energies)
    assert [item.rank for item in conformer_set.conformers] == list(range(len(conformer_set.conformers)))
    assert len(conformer_set.conformers) <= config.max_conformers


def test_cache_hit_on_second_request_is_fast(tmp_path):
    isomer = _isomer(WARFARIN_R_SMILES)
    config = ConformerConfig(seed=42, max_conformers=3, num_candidates=10)

    first = generate_conformer_set(isomer, config, cache_root=tmp_path)
    start = time.perf_counter()
    second = generate_conformer_set(isomer, config, cache_root=tmp_path)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert second.cache_hit is True
    assert second.conformer_set_id == first.conformer_set_id
    assert elapsed_ms < 50


def test_embedding_failure_emits_failed_event(monkeypatch, tmp_path):
    from rdkit.Chem import AllChem

    isomer = _isomer(WARFARIN_R_SMILES)
    clear_events()
    monkeypatch.setattr(AllChem, "EmbedMultipleConfs", lambda *args, **kwargs: [])

    try:
        generate_conformer_set(isomer, ConformerConfig(), cache_root=tmp_path)
    except Exception:
        pass

    failed = [event for event in list_events() if getattr(event, "event_type", "") == "ConformerGenerationFailed"]
    assert failed
    assert failed[-1].status == "failed"
    assert "zero embedded conformers" in failed[-1].warnings[0]


def test_caffeine_cache_size_validation(tmp_path):
    _parent, identity, _mol = molecular_identity_from_input(
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "smiles",
    )
    isomer = resolve_stereoisomer(identity)
    conformer_set = generate_conformer_set(
        isomer,
        ConformerConfig(seed=42, max_conformers=10, num_candidates=50, prune_rmsd=0.0),
        cache_root=tmp_path,
    )
    size = artifacts.cache_size_bytes(isomer.stereoisomer_id, conformer_set.config_hash, tmp_path)

    assert 5 <= len(conformer_set.conformers) <= 10
    assert 20_000 <= size <= 50_000
