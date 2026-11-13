"""
NeoGen Test Suite — Shared Fixtures
====================================
Provides reusable fixtures for API testing, sample molecules,
and the ADME predictor instance.
"""
import os
import sys
import pytest

# Ensure project root is on the path so imports resolve
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Change working directory to project root so model paths resolve
os.chdir(PROJECT_ROOT)


# ── Sample SMILES for well-known drugs ──────────────────────────────
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(O)=O"
IBUPROFEN_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
METFORMIN_SMILES = "CN(C)C(=N)NC(=N)N"
PARACETAMOL_SMILES = "CC(=O)NC1=CC=C(C=C1)O"
INVALID_SMILES = "THIS_IS_NOT_A_MOLECULE"
GARBAGE_SMILES = "XYZXYZ[[[]]]]"


@pytest.fixture
def aspirin_smiles():
    return ASPIRIN_SMILES


@pytest.fixture
def ibuprofen_smiles():
    return IBUPROFEN_SMILES


@pytest.fixture
def metformin_smiles():
    return METFORMIN_SMILES


@pytest.fixture
def invalid_smiles():
    return INVALID_SMILES


# ── FastAPI Test Client ─────────────────────────────────────────────
@pytest.fixture(scope="session")
def client():
    """Create a TestClient for the legacy ``/api/predict`` FastAPI app.

    The legacy app depends on RDKit-backed feature code, so the fixture
    skips gracefully when RDKit is not installed. The V1 PBPK API has
    its own TestClient wiring in ``test_v1_p7_api.py`` that does not
    touch this fixture.
    """
    pytest.importorskip(
        "rdkit",
        reason="RDKit not installed — legacy /api/predict client unavailable.",
    )
    from fastapi.testclient import TestClient
    from neogen.api.predict import app  # noqa: WPS433 (lazy import is intentional)
    with TestClient(app) as c:
        yield c


# ── ADME Predictor Instance ────────────────────────────────────────
@pytest.fixture(scope="session")
def adme_predictor():
    """Load the ModularADMEPredictor once for the entire test session.

    Skips gracefully when RDKit is not installed; the predictor's
    feature pipeline calls RDKit at construction time.
    """
    pytest.importorskip(
        "rdkit",
        reason="RDKit not installed — ModularADMEPredictor unavailable.",
    )
    from neogen.simulation.adme_pipeline import ModularADMEPredictor  # noqa: WPS433
    predictor = ModularADMEPredictor()
    return predictor
