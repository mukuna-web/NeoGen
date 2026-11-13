"""V1 P2 Step 2 — gut_transit tests.

Exercises the 9-segment CAT-style GI transit model: segment construction
and validation, aggregate geometry/timing helpers, bolus-time lookups, and
immutability guarantees. No absorption physics here — dissolution and
permeation live in Steps 3 and 4.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from neogen.engine.absorption import (
    GISegment,
    GutTransit,
    REFERENCE_GI_SEGMENTS,
    build_reference_gi_tract,
)


# ────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────


@pytest.fixture
def tract() -> GutTransit:
    return build_reference_gi_tract()


@pytest.fixture
def tiny_tract() -> GutTransit:
    """Minimal 3-segment tract for cleaner unit-style tests."""
    return GutTransit(
        segments=(
            GISegment(
                name="Stomach",
                pH=1.5,
                volume_mL=250.0,
                surface_area_cm2=3.5,
                transit_time_h=0.25,
                radius_cm=5.0,
            ),
            GISegment(
                name="Jejunum_1",
                pH=6.2,
                volume_mL=120.0,
                surface_area_cm2=550.0,
                transit_time_h=0.5,
                radius_cm=1.75,
                has_cyp3a4=True,
                cyp3a4_activity=1.0,
            ),
            GISegment(
                name="Colon_asc",
                pH=7.8,
                volume_mL=150.0,
                surface_area_cm2=50.0,
                transit_time_h=4.0,
                radius_cm=3.0,
            ),
        )
    )


# ────────────────────────────────────────────────────────────────
# GISegment validation
# ────────────────────────────────────────────────────────────────


class TestGISegmentValidation:
    def test_basic_construction(self) -> None:
        seg = GISegment(
            name="Jejunum_1",
            pH=6.2,
            volume_mL=120.0,
            surface_area_cm2=550.0,
            transit_time_h=0.47,
            radius_cm=1.75,
            has_cyp3a4=True,
            cyp3a4_activity=1.0,
        )
        assert seg.is_small_intestine is True
        assert seg.is_stomach is False
        assert seg.is_colon is False

    @pytest.mark.parametrize(
        "field,value",
        [
            ("pH", 0.0),
            ("volume_mL", 0.0),
            ("surface_area_cm2", -1.0),
            ("transit_time_h", 0.0),
            ("radius_cm", -0.1),
        ],
    )
    def test_rejects_nonpositive_geometry(
        self, field: str, value: float
    ) -> None:
        kwargs = dict(
            name="x",
            pH=6.5,
            volume_mL=100.0,
            surface_area_cm2=100.0,
            transit_time_h=0.5,
            radius_cm=1.0,
        )
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            GISegment(**kwargs)  # type: ignore[arg-type]

    def test_rejects_out_of_range_cyp_activity(self) -> None:
        with pytest.raises(ValueError, match="cyp3a4_activity"):
            GISegment(
                name="x",
                pH=6.5,
                volume_mL=100.0,
                surface_area_cm2=100.0,
                transit_time_h=0.5,
                radius_cm=1.0,
                cyp3a4_activity=1.5,
            )

    def test_rejects_negative_bile_salt(self) -> None:
        with pytest.raises(ValueError, match="bile_salt_mM"):
            GISegment(
                name="x",
                pH=6.5,
                volume_mL=100.0,
                surface_area_cm2=100.0,
                transit_time_h=0.5,
                radius_cm=1.0,
                bile_salt_mM=-0.1,
            )

    def test_frozen(self) -> None:
        seg = GISegment(
            name="Stomach",
            pH=1.5,
            volume_mL=250.0,
            surface_area_cm2=3.5,
            transit_time_h=0.25,
            radius_cm=5.0,
        )
        with pytest.raises(FrozenInstanceError):
            seg.pH = 2.0  # type: ignore[misc]

    def test_region_flags(self) -> None:
        stomach = GISegment(
            name="Stomach",
            pH=1.5,
            volume_mL=250.0,
            surface_area_cm2=3.5,
            transit_time_h=0.25,
            radius_cm=5.0,
        )
        jejunum = GISegment(
            name="Jejunum_1",
            pH=6.2,
            volume_mL=120.0,
            surface_area_cm2=550.0,
            transit_time_h=0.47,
            radius_cm=1.75,
        )
        ileum = GISegment(
            name="Ileum_3",
            pH=7.2,
            volume_mL=120.0,
            surface_area_cm2=200.0,
            transit_time_h=0.47,
            radius_cm=1.5,
        )
        colon = GISegment(
            name="Colon_asc",
            pH=7.8,
            volume_mL=150.0,
            surface_area_cm2=50.0,
            transit_time_h=4.0,
            radius_cm=3.0,
        )
        assert stomach.is_stomach and not stomach.is_small_intestine
        assert jejunum.is_small_intestine
        assert ileum.is_small_intestine
        assert colon.is_colon


# ────────────────────────────────────────────────────────────────
# GutTransit — structure & invariants
# ────────────────────────────────────────────────────────────────


class TestGutTransitStructure:
    def test_reference_tract_has_nine_segments(
        self, tract: GutTransit
    ) -> None:
        assert len(tract.segments) == 9

    def test_reference_segment_order(self, tract: GutTransit) -> None:
        expected = [
            "Stomach",
            "Duodenum",
            "Jejunum_1",
            "Jejunum_2",
            "Ileum_1",
            "Ileum_2",
            "Ileum_3",
            "Colon_asc",
            "Colon_trans",
        ]
        assert [s.name for s in tract.segments] == expected

    def test_rejects_empty_tract(self) -> None:
        with pytest.raises(ValueError, match="at least one segment"):
            GutTransit(segments=())

    def test_rejects_duplicate_names(self) -> None:
        seg = GISegment(
            name="Jejunum_1",
            pH=6.2,
            volume_mL=120.0,
            surface_area_cm2=550.0,
            transit_time_h=0.47,
            radius_cm=1.75,
        )
        with pytest.raises(ValueError, match="unique"):
            GutTransit(segments=(seg, seg))

    def test_frozen(self, tract: GutTransit) -> None:
        with pytest.raises(FrozenInstanceError):
            tract.segments = ()  # type: ignore[misc]

    def test_reference_segments_tuple_is_immutable(self) -> None:
        assert isinstance(REFERENCE_GI_SEGMENTS, tuple)


# ────────────────────────────────────────────────────────────────
# GutTransit — physiological invariants
# ────────────────────────────────────────────────────────────────


class TestPhysiologyInvariants:
    def test_stomach_is_most_acidic(self, tract: GutTransit) -> None:
        stomach_pH = tract.segment_by_name("Stomach").pH
        others = [s.pH for s in tract.segments if not s.is_stomach]
        assert stomach_pH < min(others)
        assert stomach_pH < 3.0  # strongly acidic fasted state

    def test_small_intestine_ph_bounds(self, tract: GutTransit) -> None:
        for seg in tract.segments:
            if seg.is_small_intestine:
                assert 5.5 <= seg.pH <= 7.5

    def test_colonic_ph_above_neutral(self, tract: GutTransit) -> None:
        for seg in tract.segments:
            if seg.is_colon:
                assert seg.pH >= 7.0

    def test_small_intestine_ph_monotone_non_decreasing(
        self, tract: GutTransit
    ) -> None:
        """Within the SI, pH should rise (or stay flat) duodenum→ileum."""
        si_ph = [s.pH for s in tract.segments if s.is_small_intestine]
        assert si_ph == sorted(si_ph)

    def test_small_intestine_transit_in_physiological_window(
        self, tract: GutTransit
    ) -> None:
        """SI residence should be 2-5 h in fasted state."""
        assert 2.0 <= tract.small_intestine_transit_h <= 5.0

    def test_total_transit_dominated_by_colon(
        self, tract: GutTransit
    ) -> None:
        # Colon is slow — 8-12 h typical; total 9-segment ≈ 12.9 h.
        assert tract.colonic_transit_h >= 8.0
        assert 8.0 <= tract.total_transit_time_h <= 16.0

    def test_gastric_emptying_fast(self, tract: GutTransit) -> None:
        # Fasted stomach empties in ~10-30 min.
        assert 0.1 <= tract.gastric_emptying_h <= 0.5

    def test_si_dominates_absorptive_surface(
        self, tract: GutTransit
    ) -> None:
        """Small intestine must carry the overwhelming majority of SA."""
        si_sa = tract.small_intestine_surface_cm2
        total_sa = tract.total_absorptive_surface_cm2
        assert si_sa / total_sa > 0.9  # >90% of absorption surface

    def test_cyp3a4_localised_in_small_intestine(
        self, tract: GutTransit
    ) -> None:
        cyp_segments = [s for s in tract.segments if s.has_cyp3a4]
        assert len(cyp_segments) > 0
        for seg in cyp_segments:
            assert seg.is_small_intestine
        # Peak jejunal activity == 1.0
        jejunum_1 = tract.segment_by_name("Jejunum_1")
        assert jejunum_1.cyp3a4_activity == pytest.approx(1.0)


# ────────────────────────────────────────────────────────────────
# GutTransit — timing helpers
# ────────────────────────────────────────────────────────────────


class TestTimingHelpers:
    def test_cumulative_entry_times_start_at_zero(
        self, tiny_tract: GutTransit
    ) -> None:
        entries = tiny_tract.cumulative_entry_times_h
        assert entries[0] == pytest.approx(0.0)

    def test_cumulative_entry_times_are_monotonic(
        self, tract: GutTransit
    ) -> None:
        entries = tract.cumulative_entry_times_h
        assert list(entries) == sorted(entries)
        assert len(entries) == len(tract.segments)

    def test_exit_equals_entry_plus_transit(
        self, tiny_tract: GutTransit
    ) -> None:
        entries = tiny_tract.cumulative_entry_times_h
        exits = tiny_tract.cumulative_exit_times_h
        for entry, leave, seg in zip(entries, exits, tiny_tract.segments):
            assert leave == pytest.approx(entry + seg.transit_time_h)

    def test_final_exit_equals_total_transit(
        self, tract: GutTransit
    ) -> None:
        exits = tract.cumulative_exit_times_h
        assert exits[-1] == pytest.approx(tract.total_transit_time_h)

    def test_segment_for_time_in_stomach(
        self, tiny_tract: GutTransit
    ) -> None:
        seg = tiny_tract.segment_for_time(0.1)
        assert seg is not None
        assert seg.name == "Stomach"

    def test_segment_for_time_in_jejunum(
        self, tiny_tract: GutTransit
    ) -> None:
        # Stomach 0-0.25, Jejunum_1 0.25-0.75
        seg = tiny_tract.segment_for_time(0.5)
        assert seg is not None
        assert seg.name == "Jejunum_1"

    def test_segment_for_time_in_colon(
        self, tiny_tract: GutTransit
    ) -> None:
        # Colon_asc 0.75-4.75
        seg = tiny_tract.segment_for_time(2.0)
        assert seg is not None
        assert seg.name == "Colon_asc"

    def test_segment_for_time_past_end_returns_none(
        self, tiny_tract: GutTransit
    ) -> None:
        total = tiny_tract.total_transit_time_h
        assert tiny_tract.segment_for_time(total + 0.1) is None

    def test_segment_for_time_negative_returns_none(
        self, tiny_tract: GutTransit
    ) -> None:
        assert tiny_tract.segment_for_time(-0.1) is None

    def test_segment_for_time_boundary_belongs_to_next(
        self, tiny_tract: GutTransit
    ) -> None:
        # Boundary t=0.25 is the start of Jejunum_1, not end of Stomach.
        seg = tiny_tract.segment_for_time(0.25)
        assert seg is not None
        assert seg.name == "Jejunum_1"


# ────────────────────────────────────────────────────────────────
# Aggregate geometry
# ────────────────────────────────────────────────────────────────


class TestAggregateGeometry:
    def test_fluid_volume_sum(self, tract: GutTransit) -> None:
        expected = sum(s.volume_mL for s in tract.segments)
        assert tract.total_fluid_volume_mL == pytest.approx(expected)

    def test_surface_area_sum(self, tract: GutTransit) -> None:
        expected = sum(s.surface_area_cm2 for s in tract.segments)
        assert tract.total_absorptive_surface_cm2 == pytest.approx(expected)

    def test_segment_by_name_lookup(self, tract: GutTransit) -> None:
        seg = tract.segment_by_name("Duodenum")
        assert seg.pH == pytest.approx(6.0)
        assert seg.has_cyp3a4 is True

    def test_segment_by_name_missing_raises(
        self, tract: GutTransit
    ) -> None:
        with pytest.raises(KeyError):
            tract.segment_by_name("Esophagus")
