from dataclasses import asdict

from neogen.chem.standardize import WARFARIN_R_SMILES, molecular_identity_from_input
from neogen.events.molecule_events import EventRef, molecule_event


def event_chain_for_input(value: str) -> tuple[str, str, str, str]:
    parent, identity, _mol = molecular_identity_from_input(value, "isomeric_smiles")
    submitted = molecule_event(
        event_type="MoleculeSubmitted",
        sequence_number=1,
        output_payload={"input": value, "input_type": "isomeric_smiles"},
        output_refs=[EventRef(type="molecular_input", id=identity.identity_id)],
    )
    normalized = molecule_event(
        event_type="MoleculeNormalized",
        sequence_number=2,
        parent_event_id=submitted.event_id,
        input_refs=[EventRef(type="molecular_input", id=identity.identity_id)],
        output_refs=[
            EventRef(type="molecular_identity", id=identity.identity_id),
            EventRef(type="molecule_parent", id=parent.parent_id),
        ],
        output_payload={
            "molecular_identity": asdict(identity),
            "molecule_parent": asdict(parent),
        },
    )
    return (
        parent.parent_id,
        identity.identity_id,
        submitted.output_hash,
        normalized.output_hash,
    )


def test_molecule_submitted_to_normalized_replay_matches_by_hash():
    first = event_chain_for_input(WARFARIN_R_SMILES)
    second = event_chain_for_input(WARFARIN_R_SMILES)

    assert first == second
