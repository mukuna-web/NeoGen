from neogen.chem.conformers import ConformerConfig, generate_conformer_set
from neogen.chem.geometry import signed_chiral_volume
from neogen.chem.standardize import WARFARIN_R_SMILES, WARFARIN_S_SMILES, molecular_identity_from_input
from neogen.chem.stereo import resolve_stereoisomer
from neogen.ml.feature_schemas.conformer_3d_v1 import _load_mol


def _isomer(smiles: str):
    _parent, identity, _mol = molecular_identity_from_input(smiles, "isomeric_smiles")
    return resolve_stereoisomer(identity)


def test_warfarin_enantiomers_have_flipped_signed_chiral_volume(tmp_path):
    config = ConformerConfig(seed=42, max_conformers=1, num_candidates=8)
    r_iso = _isomer(WARFARIN_R_SMILES)
    s_iso = _isomer(WARFARIN_S_SMILES)
    r_set = generate_conformer_set(r_iso, config, cache_root=tmp_path)
    s_set = generate_conformer_set(s_iso, config, cache_root=tmp_path)

    r_mol = _load_mol(r_set.conformers[0])
    s_mol = _load_mol(s_set.conformers[0])
    r_volume = signed_chiral_volume(r_mol, r_iso, 0)
    s_volume = signed_chiral_volume(s_mol, s_iso, 0)

    assert abs(r_volume) > 0.01
    assert abs(s_volume) > 0.01
    assert r_volume * s_volume < 0.0
