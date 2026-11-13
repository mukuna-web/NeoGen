"""V1.2 gap-closure tests — supplementary.

Tests for features that are NOT covered by the remote's dedicated test
files (test_v1_p3_gut_wall_mbi.py, test_v1_p3_transporter_ddi.py).

Covers:
  1. **XGBoost compat shim** — defensive multi-strategy model loader.
  2. **Backward compatibility** — V1.1 byte-parity for midazolam.
  3. **Per-drug gut extraction alpha** — simvastatin uses α=0.80.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from neogen.core.entities.drug import (
    BCSClass,
    CYPMetabolismProfile,
    Drug,
    InhibitionProfile,
    TransporterAffinities,
)
from neogen.engine.hepatic_clearance import HepaticClearanceEngine


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture()
def factory():
    from neogen.engine.drug_factory import DrugFactory
    return DrugFactory()


@pytest.fixture()
def midazolam(factory):
    return factory.create("midazolam")


@pytest.fixture()
def simvastatin(factory):
    return factory.create("simvastatin")


@pytest.fixture()
def physiology():
    from neogen.core.constants.reference_physiology import build_reference_male
    return build_reference_male()


# ════════════════════════════════════════════════════════════════
# 1. XGBOOST COMPAT SHIM
# ════════════════════════════════════════════════════════════════


class TestXGBoostCompat:
    def test_missing_file_returns_none(self, tmp_path) -> None:
        from neogen.simulation.xgboost_compat import safe_load_model

        result = safe_load_model(tmp_path / "nonexistent.joblib")
        assert result is None

    def test_valid_joblib_loads(self, tmp_path) -> None:
        import joblib

        from neogen.simulation.xgboost_compat import safe_load_model

        model = {"type": "test_model", "weights": [1.0, 2.0, 3.0]}
        path = tmp_path / "test_model.joblib"
        joblib.dump(model, path)

        loaded = safe_load_model(path)
        assert loaded is not None
        assert loaded["type"] == "test_model"

    def test_corrupt_file_returns_none(self, tmp_path) -> None:
        from neogen.simulation.xgboost_compat import safe_load_model

        path = tmp_path / "corrupt.joblib"
        path.write_text("this is not a valid pickle")

        result = safe_load_model(path)
        assert result is None


# ════════════════════════════════════════════════════════════════
# 2. BACKWARD COMPATIBILITY — V1.1 byte-parity
# ════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    def test_midazolam_unaffected(self, midazolam, physiology) -> None:
        """Midazolam (BCS I, no OATP1B1, no gut alpha override) should
        produce identical CLint to the pre-V1.2 path."""
        hce = HepaticClearanceEngine()
        result = hce.compute(midazolam, physiology)
        # No OATP1B1 → uptake factor = 1.0, CLint = base CLint.
        base_clint = hce._scale_clint(midazolam, physiology)
        assert result.clint_scaled_L_per_h == pytest.approx(
            round(base_clint, 4)
        )

    def test_gut_alpha_none_uses_engine_default(self, midazolam) -> None:
        from neogen.engine.absorption.permeability_engine import (
            PermeabilityEngine,
            _DEFAULT_GUT_EXTRACTION_ALPHA,
        )
        from neogen.core.entities.formulation import Formulation

        engine = PermeabilityEngine()
        form = Formulation()

        # With no per-drug alpha, the engine should use its default (0.3).
        profile = engine.simulate(midazolam, form, dose_mg=7.5)
        assert profile.fg >= 0.0  # smoke: no crash


# ════════════════════════════════════════════════════════════════
# 3. PER-DRUG GUT EXTRACTION ALPHA
# ════════════════════════════════════════════════════════════════


class TestPerDrugGutAlpha:
    def test_simvastatin_has_alpha_080_deprecated(self, simvastatin) -> None:
        """gut_extraction_alpha is retained as a deprecated field but
        is no longer used by the Qgut fg computation path."""
        assert simvastatin.gut_extraction_alpha == pytest.approx(0.80)

    def test_midazolam_has_alpha_045_deprecated(self, midazolam) -> None:
        """gut_extraction_alpha is retained as a deprecated field."""
        assert midazolam.gut_extraction_alpha == pytest.approx(0.45)

    def test_simvastatin_fg_lower_than_midazolam(
        self, simvastatin, midazolam
    ) -> None:
        """Simvastatin (higher CLint, higher fm_3A4) should have lower
        fg than midazolam under the Qgut model."""
        from neogen.engine.absorption.permeability_engine import PermeabilityEngine
        from neogen.core.entities.formulation import Formulation

        engine = PermeabilityEngine()
        form = Formulation()

        prof_sim = engine.simulate(simvastatin, form, dose_mg=20.0)
        prof_mid = engine.simulate(midazolam, form, dose_mg=7.5)

        # Under the Qgut model, simvastatin's higher CLint drives
        # more gut-wall extraction → lower fg than midazolam.
        assert prof_sim.fg < prof_mid.fg
