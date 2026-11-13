def test_rdkit_version_matches_standardization_version():
    from rdkit import __version__ as rdkit_version

    from neogen.chem.identity import STANDARDIZATION_VERSION

    # The stdz salt is versioned text; this test catches runtime drift from that text.
    expected = STANDARDIZATION_VERSION.split("rdkit_")[1]
    assert rdkit_version.startswith(expected), (
        f"RDKit version drift: {rdkit_version} vs expected {expected} "
        f"(stdz={STANDARDIZATION_VERSION})"
    )
