"""Biological-logic benchmark against the current data-flywheel artifacts.

Checks that NeoGen's mechanistic outputs obey biological rules, using the FDA
table records as the standard:

- CYP/transporter inhibitor|inducer + substrate joins carry the correct
  perpetrator(=inhibitor/inducer)/victim(=substrate) direction;
- in-vitro evidence is not promoted into a clinical assertion;
- marker-reaction probes never become interaction assertions;
- inhibitor *strength* (strong/moderate/weak) is never mapped to severity;
- enantiomer-sensitive structures are not collapsed and do not silently use the
  stereo-blind legacy_2d_v1 schema;
- PK AUC/Cmax values are tied to exposure language, not adverse-event percents.

Research/prototype sanity only — not clinical validation. Robust to missing
artifacts (each absent dimension degrades to not_applicable).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from neogen.accuracy.contracts import AccuracyFinding, BiologicalBenchmarkResult
from neogen.data_flywheel.fda_ddi_table_schema import STRENGTHS, classify_system

_ROOT = Path(__file__).resolve().parents[2]
NORM = _ROOT / "data" / "flywheel" / "normalized"

_MARKER_TOKENS = ("hydroxylation", "demethylation", "deethylation", "dealkylation", "oxidation")
_EXPOSURE_TOKENS = ("auc", "cmax", "cmin", "exposure", "concentration", "fold", "-fold", "ratio")


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _bf(check_id, sev, status, expl, **kw) -> AccuracyFinding:
    return AccuracyFinding(
        check_id=check_id, check_type="biological", severity=sev, status=status,
        explanation=expl, **kw,
    )


def _role_index(records: list[dict]) -> dict:
    idx: dict = defaultdict(lambda: defaultdict(set))
    for r in records:
        d = (r.get("normalized_drug_name") or "").lower()
        e = r.get("entity")
        role = r.get("role_class")
        if d and e and role in ("substrate", "inhibitor", "inducer"):
            idx[d][e].add(role)
    return idx


def _result(dim, checked, findings) -> BiologicalBenchmarkResult:
    failed = sum(1 for f in findings if f.status in ("fail", "blocked"))
    warned = sum(1 for f in findings if f.status == "warning")
    if checked == 0:
        status = "not_applicable"
    elif failed:
        status = "fail"
    elif warned:
        status = "warning"
    else:
        status = "pass"
    return BiologicalBenchmarkResult(
        dimension=dim, checked=checked, passed=checked - failed - warned,
        failed=failed + warned, status=status,
        details=f"{checked} checked, {failed} failed, {warned} warned",
        findings=tuple(findings),
    )


def run_biological_benchmark(norm: Path = NORM) -> list[BiologicalBenchmarkResult]:
    tbl = _load(norm / "fda_ddi_table_records.json") or {}
    clinical = tbl.get("records", []) if isinstance(tbl, dict) else []
    invitro = tbl.get("invitro_records", []) if isinstance(tbl, dict) else []
    asr = _load(norm / "fda_mechanism_assertions_from_tables.json") or {}
    assertions = asr.get("assertions", []) if isinstance(asr, dict) else []
    clin_idx = _role_index(clinical)
    inv_idx = _role_index(invitro)

    cyp_f, trans_f, invitro_f = [], [], []
    cyp_n = trans_n = 0
    for a in assertions:
        e = a.get("enzyme_or_transporter") or ""
        perp = (a.get("perpetrator") or "").lower()
        vic = (a.get("victim") or "").lower()
        mech = a.get("mechanism") or ""
        want_perp = "inhibitor" if "inhibition" in mech else ("inducer" if "induction" in mech else None)
        sys = classify_system(e)
        bucket = cyp_f if sys == "CYP" else trans_f
        if sys == "CYP":
            cyp_n += 1
        elif sys == "transporter":
            trans_n += 1
        perp_roles = clin_idx.get(perp, {}).get(e, set())
        vic_roles = clin_idx.get(vic, {}).get(e, set())
        if want_perp and want_perp not in perp_roles:
            bucket.append(_bf(
                f"bio::direction::{a.get('canonical_pair_id')}::{e}", "high", "fail",
                f"{perp} is not a {want_perp} of {e} in the FDA clinical table "
                f"(roles={sorted(perp_roles) or 'none'}) — direction unsupported",
                actual=str(sorted(perp_roles)), expected=want_perp,
            ))
        elif "substrate" not in vic_roles:
            bucket.append(_bf(
                f"bio::victim::{a.get('canonical_pair_id')}::{e}", "high", "fail",
                f"{vic} is not a substrate of {e} in the FDA clinical table — victim unsupported",
                actual=str(sorted(vic_roles)), expected="substrate",
            ))
        # in-vitro must not be the sole basis
        if perp not in clin_idx and perp in inv_idx:
            invitro_f.append(_bf(
                f"bio::invitro::{a.get('canonical_pair_id')}", "high", "fail",
                f"{perp} role for {e} exists only in in-vitro data, promoted to a clinical assertion",
            ))

    # marker reaction not promoted
    marker_f = []
    markers = [r for r in clinical + invitro if r.get("role_class") == "marker_reaction"]
    leaked = [r for r in invitro if any(t in (r.get("normalized_drug_name") or "") for t in _MARKER_TOKENS)]
    if markers:
        marker_f.append(_bf("bio::marker::role", "high", "fail",
                            f"{len(markers)} record(s) carry role_class=marker_reaction (probe, not a drug)"))
    if leaked:
        marker_f.append(_bf("bio::marker::leak", "high", "fail",
                            f"{len(leaked)} in-vitro inhibitor name(s) look like marker reactions"))
    marker_checked = len(clinical) + len(invitro)

    # strength is never severity
    strength_f = []
    str_checked = len(clinical) + len(assertions)
    for a in assertions:
        if "severity" in a:
            strength_f.append(_bf("bio::strength::assertion_severity", "critical", "fail",
                                  f"assertion {a.get('canonical_pair_id')} carries a severity field"))
    bad_strength = [r for r in clinical if r.get("strength") not in STRENGTHS]
    if bad_strength:
        strength_f.append(_bf("bio::strength::vocab", "low", "warning",
                              f"{len(bad_strength)} record(s) use a strength outside the allowed vocabulary"))

    # stereochemistry handling
    stereo_f = []
    sc = _load(norm / "structure_stereo_cache_2026-05-29.json") or {}
    screcs = sc.get("records", []) if isinstance(sc, dict) else []
    for r in screcs:
        if r.get("stereo_relevance") == "enantiomer_sensitive":
            if not r.get("stereochemistry_status"):
                stereo_f.append(_bf(f"bio::stereo::status::{r.get('drug')}", "medium", "fail",
                                    f"{r.get('drug')} is enantiomer-sensitive but has no stereochemistry_status"))
            if r.get("required_feature_schema") == "legacy_2d_v1":
                stereo_f.append(_bf(f"bio::stereo::schema::{r.get('drug')}", "medium", "warning",
                                    f"{r.get('drug')} is enantiomer-sensitive but routed to stereo-blind legacy_2d_v1",
                                    remediation="Use stereo_2d_v1 or conformer_3d_v1 for chiral-sensitive drugs."))
    cf = _load(norm / "isomer_counterfactuals_pilot.json") or {}
    cfs = cf.get("counterfactuals", []) if isinstance(cf, dict) else []
    for c in cfs:
        if c.get("different_stereochemistry") and c.get("same_connectivity"):
            if not c.get("legacy_2d_expected_collision") or not c.get("stereo_2d_expected_separation"):
                stereo_f.append(_bf(f"bio::stereo::cf::{c.get('parent_name')}", "medium", "warning",
                                    f"{c.get('parent_name')} counterfactual: stereo isomers but collision/"
                                    "separation expectation not recorded"))
    stereo_checked = len(screcs) + len(cfs)

    # PK extraction sanity
    pk_f = []
    pk = _load(norm / "pk_effect_records_pilot.json") or {}
    pkrecs = pk.get("records", []) if isinstance(pk, dict) else []
    for r in pkrecs:
        has_val = r.get("auc_ratio") not in (None, "") or r.get("cmax_ratio") not in (None, "")
        cite = (r.get("citation_span") or "").lower()
        if has_val:
            if not cite:
                pk_f.append(_bf(f"bio::pk::cite::{r.get('canonical_pair_id')}", "medium", "fail",
                                f"PK record {r.get('canonical_pair_id')} has AUC/Cmax but no citation_span"))
            elif not any(t in cite for t in _EXPOSURE_TOKENS):
                pk_f.append(_bf(f"bio::pk::exposure::{r.get('canonical_pair_id')}", "medium", "warning",
                                f"PK record {r.get('canonical_pair_id')} citation lacks exposure language "
                                "(AUC/Cmax/fold) — risk of an AE-rate percentage misread as PK",
                                actual=cite[:120]))

    return [
        _result("cyp_directionality", cyp_n, cyp_f),
        _result("transporter_directionality", trans_n, trans_f),
        _result("in_vitro_not_promoted", len(assertions), invitro_f),
        _result("marker_reaction_excluded", marker_checked, marker_f),
        _result("strength_not_severity", str_checked, strength_f),
        _result("stereochemistry_handling", stereo_checked, stereo_f),
        _result("pk_extraction_sanity", len(pkrecs), pk_f),
    ]


def findings_from(results: list[BiologicalBenchmarkResult]) -> list[AccuracyFinding]:
    return [f for r in results for f in r.findings]
