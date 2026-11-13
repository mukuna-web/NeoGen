"""Tests for the data-flywheel source registry loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from neogen.data_flywheel.source_registry import (
    REQUIRED_FIELDS,
    Registry,
    SourceRegistryError,
    load_registry,
)


@pytest.mark.unit
def test_real_registry_loads() -> None:
    reg = load_registry()
    assert isinstance(reg, Registry)
    assert len(reg.sources) >= 70  # 75 expected
    assert len(reg.categories) == 10


@pytest.mark.unit
def test_every_source_has_required_fields() -> None:
    reg = load_registry()
    for s in reg.sources:
        for field in REQUIRED_FIELDS:
            assert field in s, f"{s.get('source_id')} missing {field}"


@pytest.mark.unit
def test_source_ids_unique() -> None:
    reg = load_registry()
    ids = [s["source_id"] for s in reg.sources]
    assert len(ids) == len(set(ids))


@pytest.mark.unit
def test_p0_pilot_sources_present() -> None:
    reg = load_registry()
    p0 = {s["source_id"] for s in reg.by_priority("P0")}
    # A representative subset of the declared P0 pilot set must be present.
    for sid in ("rxnorm", "dailymed_spl", "openfda_labels", "drugcentral",
                "unichem", "pubchem", "chembl", "pk_db",
                "fda_cyp_transporter_tables", "ddinter", "pubmed"):
        assert sid in p0, f"expected P0 source {sid} missing"


@pytest.mark.unit
def test_categories_all_declared() -> None:
    reg = load_registry()
    declared = set(reg.categories)
    for s in reg.sources:
        assert s["category"] in declared


@pytest.mark.unit
def test_counts_shape() -> None:
    reg = load_registry()
    c = reg.counts()
    assert c["total"] == len(reg.sources)
    assert sum(c.get(p, 0) for p in ("P0", "P1", "P2", "P3")) == len(reg.sources)


@pytest.mark.unit
def test_missing_index_raises() -> None:
    with pytest.raises(SourceRegistryError):
        load_registry("/nonexistent/path/source_registry.yaml")


@pytest.mark.unit
def test_malformed_source_raises(tmp_path: Path) -> None:
    # Build a tiny malformed registry: a source missing required fields.
    idx = tmp_path / "source_registry.yaml"
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "g.yaml").write_text(
        "sources:\n"
        "  - source_id: bad\n"
        "    name: Bad Source\n"
        "    category: drug_identity_spine\n"  # missing most required fields
    )
    idx.write_text(
        "registry_version: '0.0.1'\n"
        "created_at: '2026-05-28'\n"
        "task_id: 'test'\n"
        "categories: [drug_identity_spine]\n"
        "includes: [sources/g.yaml]\n"
    )
    with pytest.raises(SourceRegistryError, match="missing required fields"):
        load_registry(idx)


@pytest.mark.unit
def test_bad_enum_raises(tmp_path: Path) -> None:
    idx = tmp_path / "source_registry.yaml"
    sdir = tmp_path / "sources"
    sdir.mkdir()
    (sdir / "g.yaml").write_text(
        "sources:\n"
        "  - source_id: bad\n"
        "    name: Bad\n"
        "    category: drug_identity_spine\n"
        "    source_url: http://x\n"
        "    access_method: telepathy\n"  # invalid enum
        "    license_status: open\n"
        "    robots_tos_status: open\n"
        "    update_frequency: daily\n"
        "    expected_data_types: [x]\n"
        "    key_identifiers: [x]\n"
        "    joins_to_identity_spine: x\n"
        "    value_for_moat: x\n"
        "    ingestion_priority: P0\n"
        "    crawl_allowed: true\n"
        "    requires_api_key: false\n"
        "    estimated_size: small\n"
        "    notes: x\n"
        "    first_ingestion_status: not_started\n"
    )
    idx.write_text(
        "registry_version: '0.0.1'\n"
        "created_at: '2026-05-28'\n"
        "task_id: 'test'\n"
        "categories: [drug_identity_spine]\n"
        "includes: [sources/g.yaml]\n"
    )
    with pytest.raises(SourceRegistryError, match="invalid access_method"):
        load_registry(idx)
