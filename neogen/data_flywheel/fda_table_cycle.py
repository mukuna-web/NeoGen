"""FDA structured-table ingestion cycle: orchestration + adversarial checks (GAP-014).

The replayable entry point that turns the prior cycle's directionality *gaps*
into cited mechanism assertions using the FDA clinical CYP/transporter matrix.
Composes the tested units (sources -> parser -> joiner) and runs nine adversarial
checks against the *real* produced data before anything is written.

``run_adversarial_checks`` is pure and unit-tested. ``run_cycle`` is pure
(returns serializable artifacts in memory). ``main`` does the live fetch + disk
writes so the module stays importable/offline-testable.

No FDA approval/clearance/clinical-validity claim. No severity. Engine untouched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from neogen.data_flywheel.fda_ddi_table_parser import (
    parse_clinical_matrix_html,
    parse_invitro_inhibitors,
)
from neogen.data_flywheel.fda_ddi_table_schema import FDADDITableRecord, classify_system
from neogen.data_flywheel.fda_mechanism_joiner import (
    build_role_index,
    join_gaps_to_fda_roles,
)
from neogen.data_flywheel.mechanism_assertion import MechanismAssertion


# ---------------------------------------------------------------------------
# Adversarial checks (pure; run against the real produced data)
# ---------------------------------------------------------------------------

def _check(name: str, passed: bool, detail: str) -> dict:
    return {"check": name, "passed": bool(passed), "detail": detail}


def run_adversarial_checks(
    *,
    records: list[FDADDITableRecord],
    invitro_records: list[FDADDITableRecord],
    assertions: list[MechanismAssertion],
    gaps: list[dict],
    source_url: str,
) -> list[dict]:
    """Nine traps the cycle must survive. Returns one result dict per check."""
    results: list[dict] = []

    # 1. Pair-order trap: swapping drug_a/drug_b in every gap must yield the
    #    identical assertion set (direction comes from roles, not ordering).
    swapped = [{"drug_a": g.get("drug_b"), "drug_b": g.get("drug_a")} for g in gaps]
    re_res = join_gaps_to_fda_roles(gaps=swapped, fda_records=records, source_url=source_url)
    h_orig = {a.provenance_hash for a in assertions}
    h_swap = {a.provenance_hash for a in re_res.assertions}
    results.append(_check(
        "pair_order_trap", h_orig == h_swap,
        f"orig={len(h_orig)} swapped={len(h_swap)} identical={h_orig == h_swap}",
    ))

    # 2. Table-context / marker-reaction trap: section-header strings must never
    #    appear as drugs, and no record may carry a marker_reaction role.
    drugs = {r.normalized_drug_name for r in records}
    leaked = drugs & {"p-gp", "bcrp", "oatp1b1/oatp1b3", "oat1/oat3",
                      "oct2/mate", "drug or other substance"}
    markers = [r for r in records if r.role_class == "marker_reaction"]
    results.append(_check(
        "table_context_marker_reaction_trap", not leaked and not markers,
        f"leaked_header_drugs={sorted(leaked) or 'none'} marker_records={len(markers)}",
    ))

    # 3. Backwards-direction (role) trap: a drug that is ONLY ever a substrate
    #    must never be emitted as a perpetrator.
    role_union: dict[str, set] = {}
    for r in records:
        role_union.setdefault(r.normalized_drug_name, set()).add(r.role_class)
    victim_only = {d for d, rs in role_union.items() if rs == {"substrate"}}
    backwards = sorted({a.perpetrator for a in assertions if a.perpetrator in victim_only})
    results.append(_check(
        "backwards_direction_trap", not backwards,
        f"victim_only_drugs={len(victim_only)} emitted_as_perpetrator={backwards or 'none'}",
    ))

    # 4. In-vitro-vs-clinical trap: in-vitro selective-inhibitor evidence must
    #    not enter the clinical role index used for assertions.
    clinical_idx = build_role_index(records + invitro_records, clinical_only=True)
    invitro_drugs = {r.normalized_drug_name for r in invitro_records}
    invitro_leaked = sorted({
        d for d in invitro_drugs if d in clinical_idx and d not in drugs
    })
    results.append(_check(
        "in_vitro_vs_clinical_trap", not invitro_leaked,
        f"invitro_drugs={len(invitro_drugs)} leaked_into_clinical={invitro_leaked or 'none'}",
    ))

    # 5. Stereo trap: racemic names (e.g. warfarin) must not gain an invented
    #    enantiomer descriptor in assertion text or drug fields.
    stereo_bad = [
        a for a in assertions
        if any(tok in a.evidence_text.lower() for tok in ("(s)-", "(r)-"))
    ]
    results.append(_check(
        "stereo_trap", not stereo_bad,
        f"assertions_claiming_stereospecificity={len(stereo_bad)}",
    ))

    # 6. Transporter trap: CYP enzymes -> cyp_* mechanism; transporters ->
    #    transporter_* mechanism. No cross-wiring.
    mism = []
    for a in assertions:
        sys = classify_system(a.enzyme_or_transporter or "")
        if sys == "CYP" and not a.mechanism.startswith("cyp_"):
            mism.append((a.enzyme_or_transporter, a.mechanism))
        if sys == "transporter" and not a.mechanism.startswith("transporter_"):
            mism.append((a.enzyme_or_transporter, a.mechanism))
    results.append(_check(
        "transporter_trap", not mism, f"mechanism_system_mismatches={mism or 'none'}",
    ))

    # 7. Strength-as-severity trap: no assertion may carry a severity field/key.
    has_sev = [a for a in assertions if hasattr(a, "severity")
               or "severity" in a.to_dict()]
    results.append(_check(
        "strength_severity_trap", not has_sev,
        f"assertions_with_severity={len(has_sev)} (strength stays classification text)",
    ))

    # 8. Duplicate trap: provenance hashes AND (pair,enzyme,perp,mechanism)
    #    tuples must be unique.
    hashes = [a.provenance_hash for a in assertions]
    keys = [(a.canonical_pair_id, a.enzyme_or_transporter, a.perpetrator, a.mechanism)
            for a in assertions]
    results.append(_check(
        "duplicate_trap",
        len(hashes) == len(set(hashes)) and len(keys) == len(set(keys)),
        f"assertions={len(hashes)} unique_hashes={len(set(hashes))} unique_keys={len(set(keys))}",
    ))

    # 9. Citation trap: every assertion must cite BOTH drugs and have both roles
    #    grounded in the FDA records (perpetrator role + substrate role exist).
    idx = build_role_index(records, clinical_only=True)
    uncited = []
    for a in assertions:
        txt = a.evidence_text.lower()
        cites_both = a.drug_a in txt and a.drug_b in txt and bool(a.evidence_text.strip())
        ent = a.enzyme_or_transporter
        perp_ok = a.perpetrator in idx and ent in idx[a.perpetrator] \
            and idx[a.perpetrator][ent].is_perpetrator
        vic_ok = a.victim in idx and ent in idx[a.victim] \
            and idx[a.victim][ent].is_victim
        if not (cites_both and perp_ok and vic_ok):
            uncited.append(a.canonical_pair_id)
    results.append(_check(
        "citation_trap", not uncited,
        f"assertions={len(assertions)} ungrounded={uncited or 'none'}",
    ))

    return results


# ---------------------------------------------------------------------------
# Cycle assembly (pure)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CycleResult:
    records: list[FDADDITableRecord]
    invitro_records: list[FDADDITableRecord]
    assertions: list[MechanismAssertion]
    residual_gaps: list
    resolved_pair_ids: list[str]
    join_stats: dict
    adversarial: list[dict]
    rxcui_coverage: dict


def run_cycle(
    *,
    clinical_html: str,
    design_html: str,
    gaps: list[dict],
    name_to_rxcui: dict[str, str],
    source_url_clinical: str,
    source_url_design: str,
    retrieved_at: str,
) -> CycleResult:
    """Parse FDA pages, join to gaps, run adversarial checks (no disk I/O)."""
    records = parse_clinical_matrix_html(
        clinical_html, source_url=source_url_clinical, retrieved_at=retrieved_at,
    )
    invitro = parse_invitro_inhibitors(
        design_html, source_url=source_url_design, retrieved_at=retrieved_at,
    )
    join = join_gaps_to_fda_roles(
        gaps=gaps, fda_records=records, source_url=source_url_clinical,
    )
    checks = run_adversarial_checks(
        records=records, invitro_records=invitro,
        assertions=list(join.assertions), gaps=gaps, source_url=source_url_clinical,
    )
    # Identity-spine mapping for the drugs that appear in assertions.
    assertion_drugs = sorted({d for a in join.assertions for d in (a.perpetrator, a.victim)})
    rxcui_coverage = {
        d: name_to_rxcui.get(d) for d in assertion_drugs
    }
    return CycleResult(
        records=records, invitro_records=invitro,
        assertions=list(join.assertions), residual_gaps=list(join.gaps),
        resolved_pair_ids=list(join.resolved_pair_ids), join_stats=join.stats,
        adversarial=checks, rxcui_coverage=rxcui_coverage,
    )


def all_checks_passed(checks: list[dict]) -> bool:
    return all(c["passed"] for c in checks)


# ---------------------------------------------------------------------------
# Live driver (disk I/O)
# ---------------------------------------------------------------------------

def main() -> int:  # pragma: no cover - exercised via the cycle run, not unit tests
    import os
    from datetime import datetime, timezone

    from neogen.data_flywheel.fda_ddi_table_sources import (
        fetch_clinical_examples, fetch_design_tables,
    )

    base = os.environ.get("NEOGEN_ROOT", os.getcwd())
    flow = os.path.join(base, "data", "flywheel")
    norm = os.path.join(flow, "normalized")
    raw = os.path.join(flow, "raw", "fda_ddi_tables")
    mani = os.path.join(flow, "manifests")
    gp = os.path.join(flow, "graph_patches")
    for d in (norm, raw, mani, gp):
        os.makedirs(d, exist_ok=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    clinical = fetch_clinical_examples(retrieved_at=now)
    design = fetch_design_tables(retrieved_at=now)

    gaps = json.load(open(os.path.join(norm, "directionality_gaps_pilot.json")))["gaps"]
    spine = json.load(open(os.path.join(norm, "identity_spine_2026-05-29.json")))
    name_to_rxcui = spine.get("name_to_rxcui", {})

    res = run_cycle(
        clinical_html=clinical.html, design_html=design.html, gaps=gaps,
        name_to_rxcui=name_to_rxcui,
        source_url_clinical=clinical.source_url, source_url_design=design.source_url,
        retrieved_at=now,
    )
    ok = all_checks_passed(res.adversarial)
    cycle_id = "fda-structured-ddi-table-cycle-2026-05-29"

    def _w(path: str, obj) -> None:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)

    # 1. meta + raw bodies (replayable; FDA pages are public-domain US-gov data)
    with open(os.path.join(raw, "clinical_examples.html"), "w") as f:
        f.write(clinical.html)
    with open(os.path.join(raw, "design_tables.html"), "w") as f:
        f.write(design.html)
    _w(os.path.join(raw, "fda_ddi_tables_meta.json"), {
        "cycle_id": cycle_id, "retrieved_at": now,
        "license_status": "open-us-gov-data",
        "note": "FDA pages are public-domain US-government data. Mechanistic "
                "classification resource, not severity labels.",
        "sources": [clinical.meta(), design.meta()],
        "table_headings": [
            "CYP/Transporter Clinical Substrates, Inhibitors, or Inducers (per-drug matrix)",
            "In vitro marker reactions (CYP-mediated metabolism)",
            "In vitro selective inhibitors for CYP-mediated metabolism",
        ],
    })

    # 2. table records (clinical matrix + in-vitro), reviewer_status preserved
    _w(os.path.join(norm, "fda_ddi_table_records.json"), {
        "cycle_id": cycle_id, "generated_at": now,
        "source_url_clinical": clinical.source_url,
        "source_url_design": design.source_url,
        "clinical_record_count": len(res.records),
        "invitro_record_count": len(res.invitro_records),
        "records": [r.to_dict() for r in res.records],
        "invitro_records": [r.to_dict() for r in res.invitro_records],
    })

    # 3. mechanism assertions (cited, fda_table confidence) + rxcui annotation
    _w(os.path.join(norm, "fda_mechanism_assertions_from_tables.json"), {
        "cycle_id": cycle_id, "generated_at": now,
        "confidence": "fda_table", "assertion_count": len(res.assertions),
        "rxcui_coverage": res.rxcui_coverage,
        "assertions": [
            {**a.to_dict(),
             "perpetrator_rxcui": name_to_rxcui.get(a.perpetrator),
             "victim_rxcui": name_to_rxcui.get(a.victim),
             "reviewer_status": "needs_review"}
            for a in res.assertions
        ],
    })

    # 4. gap update (resolved + residual); original gaps file left intact
    _w(os.path.join(norm, "fda_table_directionality_gap_updates.json"), {
        "cycle_id": cycle_id, "generated_at": now,
        "input_gap_count": len(gaps),
        "pairs_resolved": len(res.resolved_pair_ids),
        "resolved_pair_ids": res.resolved_pair_ids,
        "residual_gap_count": len(res.residual_gaps),
        "residual_gaps": [g.to_dict() for g in res.residual_gaps],
        "join_stats": res.join_stats,
    })

    # 5. manifest (counts + adversarial results + provenance hashes)
    _w(os.path.join(mani, "fda_structured_ddi_table_cycle_2026-05-29.json"), {
        "cycle_id": cycle_id, "started_at": now, "completed_at": now,
        "tool_used": "fda_ddi_table_sources+parser+joiner",
        "sources": [clinical.meta(), design.meta()],
        "clinical_records": len(res.records),
        "invitro_records": len(res.invitro_records),
        "input_gaps": len(gaps),
        "assertions_created": len(res.assertions),
        "pairs_resolved": len(res.resolved_pair_ids),
        "residual_gaps": len(res.residual_gaps),
        "join_stats": res.join_stats,
        "adversarial_checks": res.adversarial,
        "adversarial_all_pass": ok,
        "rxcui_coverage": res.rxcui_coverage,
        "vortext_endpoint": "/api/vortext/flywheel",
        "note": "FDA tables = mechanistic classification, not severity. "
                "No FDA approval/clearance/clinical-validity claim. Engine untouched.",
        "next_action": "Human review of fda_table assertions; pair PK values to "
                       "these victims (GAP-015); extend matrix beyond pilot drugs.",
    })

    # 6. graph patch (perpetrator -> victim edges via enzyme/transporter)
    _w(os.path.join(gp, "fda_structured_ddi_table_graph_patch_2026-05-29.json"), {
        "cycle_id": cycle_id, "generated_at": now,
        "patch_type": "mechanism_directionality",
        "confidence": "fda_table", "reviewer_status": "needs_review",
        "edges": [
            {
                "canonical_pair_id": a.canonical_pair_id,
                "perpetrator": a.perpetrator, "victim": a.victim,
                "perpetrator_rxcui": name_to_rxcui.get(a.perpetrator),
                "victim_rxcui": name_to_rxcui.get(a.victim),
                "mechanism": a.mechanism,
                "enzyme_or_transporter": a.enzyme_or_transporter,
                "evidence_text": a.evidence_text,
                "source_url": a.source_url,
                "provenance_hash": a.provenance_hash,
            }
            for a in res.assertions
        ],
    })

    print(json.dumps({
        "records": len(res.records), "invitro": len(res.invitro_records),
        "assertions": len(res.assertions), "resolved": len(res.resolved_pair_ids),
        "residual_gaps": len(res.residual_gaps), "adversarial_all_pass": ok,
    }, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
