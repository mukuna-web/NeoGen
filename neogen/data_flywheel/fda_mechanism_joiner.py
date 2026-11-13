"""Join FDA CYP/transporter table roles to DDI pairs -> cited mechanism assertions.

GAP-014. Converts directionality *gaps* (pairs where label prose did not yield an
explicit perpetrator/victim) into structured ``MechanismAssertion`` records when —
and only when — the FDA clinical table explicitly classifies BOTH drugs:

    drug P is an inhibitor/inducer of enzyme/transporter E, and
    drug V is a substrate of the SAME E
    => perpetrator = P, victim = V, mechanism = {cyp,transporter}_{inhibition,induction}

Direction is derived from the *roles*, never from drug ordering. Strength
(strong/moderate/weak) is preserved as classification text only — it is NOT
severity and is never promoted to a clinical-risk claim. If a drug holds both a
perpetrator and a substrate role for the same E (dual role), or roles do not
complement, the pair stays a gap.

This module is pure (no I/O): feed it parsed records + gaps, get assertions +
residual gaps + stats back. No NeoGen engine code is touched.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from neogen.data_flywheel.fda_ddi_table_schema import FDADDITableRecord, classify_system
from neogen.data_flywheel.mechanism_assertion import DirectionalityGap, MechanismAssertion
from neogen.data_flywheel.provenance import content_hash
from neogen.engine.ddi.pair_record import make_pair_id

# Evidence contexts that may back a *clinical* directionality assertion.
# In-vitro selective-inhibitor evidence is deliberately excluded here (it may
# corroborate but must not silently become a clinical claim).
_CLINICAL_CONTEXTS = {
    "clinical_example", "clinical_index", "healthcare_professional_example",
}
_PERP_ROLES = {"inhibitor", "inducer"}

# Exact synonyms (same substance / same RxCUI) so a gap that names one variant
# still resolves against the FDA table's preferred name. Not enantiomer
# collapse: these are identical entities, so no information is lost.
_SYNONYMS = {
    "rifampicin": "rifampin",
    "grapefruit": "grapefruit juice",
}


def _canon_drug(name: str) -> str:
    n = (name or "").strip().lower()
    return _SYNONYMS.get(n, n)


@dataclass(frozen=True)
class _DrugEntityRole:
    """Collapsed role view for one (drug, entity): which roles + their strengths."""

    is_perpetrator: bool
    is_victim: bool
    perp_role: str | None        # "inhibitor" | "inducer" | None
    perp_strength: str
    victim_strength: str

    @property
    def is_dual(self) -> bool:
        """Both perpetrator and substrate role for the same entity -> ambiguous."""
        return self.is_perpetrator and self.is_victim


@dataclass(frozen=True)
class JoinResult:
    assertions: tuple[MechanismAssertion, ...]
    gaps: tuple[DirectionalityGap, ...]
    resolved_pair_ids: tuple[str, ...]
    stats: dict


def build_role_index(
    records: list[FDADDITableRecord],
    *,
    clinical_only: bool = True,
) -> dict[str, dict[str, _DrugEntityRole]]:
    """Index records as ``drug -> entity -> _DrugEntityRole`` (collapsed roles)."""
    # drug -> entity -> {role_class -> strength}
    raw: dict[str, dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    for r in records:
        if clinical_only and r.evidence_context not in _CLINICAL_CONTEXTS:
            continue
        if r.role_class not in ("substrate", "inhibitor", "inducer"):
            continue
        raw[r.normalized_drug_name][r.entity][r.role_class] = r.strength

    index: dict[str, dict[str, _DrugEntityRole]] = {}
    for drug, ents in raw.items():
        index[drug] = {}
        for ent, roles in ents.items():
            perp_role = "inhibitor" if "inhibitor" in roles else (
                "inducer" if "inducer" in roles else None
            )
            index[drug][ent] = _DrugEntityRole(
                is_perpetrator=perp_role is not None,
                is_victim="substrate" in roles,
                perp_role=perp_role,
                perp_strength=roles.get(perp_role or "", "unknown"),
                victim_strength=roles.get("substrate", "unknown"),
            )
    return index


def _mechanism_for(entity: str, perp_role: str) -> str:
    system = classify_system(entity)
    if system == "CYP":
        return "cyp_inhibition" if perp_role == "inhibitor" else "cyp_induction"
    if system == "transporter":
        return "transporter_inhibition" if perp_role == "inhibitor" else "transporter_induction"
    return "unknown"


def _shared_entities(a_roles: dict, b_roles: dict) -> list[str]:
    # Deterministic order so output + provenance hashes are stable.
    return sorted(set(a_roles) & set(b_roles))


def join_gaps_to_fda_roles(
    *,
    gaps: list[dict],
    fda_records: list[FDADDITableRecord],
    source_url: str,
    clinical_only: bool = True,
) -> JoinResult:
    """Resolve directionality gaps using FDA table roles.

    ``gaps`` are dicts with at least ``drug_a``, ``drug_b`` (the prior-cycle
    directionality_gaps). For each gap we look up both drugs in the FDA role
    index and emit a cited ``MechanismAssertion`` per shared enzyme/transporter
    where roles complement. Otherwise the pair remains a ``DirectionalityGap``
    with an explicit reason.
    """
    index = build_role_index(fda_records, clinical_only=clinical_only)
    assertions: list[MechanismAssertion] = []
    residual_gaps: list[DirectionalityGap] = []
    resolved: set[str] = set()
    seen: set[tuple] = set()
    reason_counts: dict[str, int] = defaultdict(int)

    for gap in gaps:
        a = _canon_drug(gap.get("drug_a"))
        b = _canon_drug(gap.get("drug_b"))
        if not a or not b:
            continue
        pid = make_pair_id(a, b)
        a_roles = index.get(a, {})
        b_roles = index.get(b, {})

        if not a_roles or not b_roles:
            missing = a if not a_roles else b
            if not a_roles and not b_roles:
                missing = f"{a} and {b}"
            residual_gaps.append(DirectionalityGap(
                drug_a=a, drug_b=b, canonical_pair_id=pid,
                reason=f"no FDA clinical-table role for {missing}",
                evidence_text=(gap.get("evidence_text") or "")[:300],
                source_url=source_url,
            ))
            reason_counts["missing_fda_role"] += 1
            continue

        made_for_pair = False
        dual_flagged = False
        for ent in _shared_entities(a_roles, b_roles):
            ar = a_roles[ent]
            br = b_roles[ent]
            # Role trap: a drug that is BOTH perpetrator and substrate of E.
            if ar.is_dual or br.is_dual:
                dual_flagged = True
                continue
            # Determine clean complementary direction for THIS entity.
            if ar.is_perpetrator and br.is_victim and not br.is_perpetrator:
                perp, vic, perp_dr = a, b, ar
            elif br.is_perpetrator and ar.is_victim and not ar.is_perpetrator:
                perp, vic, perp_dr = b, a, br
            else:
                continue  # both perpetrators, both victims, or no complement

            mechanism = _mechanism_for(ent, perp_dr.perp_role or "inhibitor")
            key = (pid, ent, perp, mechanism)
            if key in seen:
                continue  # duplicate trap
            seen.add(key)

            vic_strength = (a_roles if vic == a else b_roles)[ent].victim_strength
            # Citation span cites BOTH sides explicitly; strength is shown as
            # classification text, never as severity.
            span = (
                f"FDA CYP/transporter table: {perp} = {ent} "
                f"{perp_dr.perp_strength} {perp_dr.perp_role}; "
                f"{vic} = {ent} {vic_strength} substrate."
            )
            assertions.append(MechanismAssertion(
                drug_a=a, drug_b=b, canonical_pair_id=pid,
                perpetrator=perp, victim=vic,
                mechanism=mechanism, enzyme_or_transporter=ent,
                evidence_text=span, source_url=source_url,
                confidence="fda_table",
                provenance_hash=content_hash(
                    f"{pid}|{perp}|{vic}|{mechanism}|{ent}|fda_table"
                ),
            ))
            made_for_pair = True

        if made_for_pair:
            resolved.add(pid)
        else:
            reason = (
                "dual perpetrator+substrate role for shared enzyme (ambiguous)"
                if dual_flagged else
                "FDA roles do not complement (no perpetrator/victim pairing)"
            )
            residual_gaps.append(DirectionalityGap(
                drug_a=a, drug_b=b, canonical_pair_id=pid, reason=reason,
                evidence_text=(gap.get("evidence_text") or "")[:300],
                source_url=source_url,
            ))
            reason_counts["dual_role" if dual_flagged else "no_complement"] += 1

    stats = {
        "input_gaps": len(gaps),
        "assertions_created": len(assertions),
        "pairs_resolved": len(resolved),
        "residual_gaps": len(residual_gaps),
        "fda_drugs_indexed": len(index),
        "reason_counts": dict(reason_counts),
    }
    return JoinResult(
        assertions=tuple(assertions),
        gaps=tuple(residual_gaps),
        resolved_pair_ids=tuple(sorted(resolved)),
        stats=stats,
    )
