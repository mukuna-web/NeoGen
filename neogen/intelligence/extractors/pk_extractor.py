"""
Stage 4: Biomedical PK/DDI/Organ Effect Extraction
=====================================================
Extracts structured pharmacokinetic data from FDA label text.
Uses regex + rule-based extraction (no LLM needed for most patterns).

Handles:
- PK parameters (Cmax, AUC, t1/2, CL, Vd, F, fu)
- DDI information (perpetrator/victim, fold changes)
- CYP enzyme identification
- Special population PK changes
- Organ effects and safety signals
- Study context (dose, population, design)
"""
from __future__ import annotations

import re
import logging
from typing import Optional
from neogen.intelligence.schemas.drug_schema import (
    PKValue, PKProfile, DDIRecord, OrganEffect, SpecialPopulationPK,
    SafetySignal, StudyContext, ExtractionTrace,
    EvidenceType, ConfidenceLevel, ExtractionMethod,
    InteractionSeverity, InteractionMechanism,
)

logger = logging.getLogger(__name__)


class PKExtractor:
    """Extract pharmacokinetic parameters from FDA label text."""

    # ── PK PARAMETER PATTERNS ──
    PK_PATTERNS = {
        "cmax": [
            r'[Cc]max\s*(?:was|of|=|:|\()?\s*(?:approximately\s+)?(\d+\.?\d*)\s*[±+/-]*\s*\d*\.?\d*\s*(ng/mL|ug/mL|mg/L|mcg/mL|ng/ml|ug/ml|mg/ml|nmol/L|uM|nM|mcg/L|pg/mL)',
            r'(?:peak|maximum)\s+(?:plasma\s+)?concentration[s]?\s*(?:was|of|=|:)?\s*(?:approximately\s+)?(\d+\.?\d*)\s*[±+/-]*\s*\d*\.?\d*\s*(ng/mL|ug/mL|mg/L)',
        ],
        "tmax": [
            r'[Tt]max\s*(?:was|of|=|:|\()?\s*(?:approximately\s+)?(\d+\.?\d*)\s*[±+/-]*\s*\d*\.?\d*\s*(hours?|h\b|hr)',
            r'(?:time\s+to\s+(?:peak|maximum)\s+concentration)\s*(?:was|of|=|:)?\s*(?:approximately\s+)?(\d+\.?\d*)\s*(hours?|h\b)',
        ],
        "auc": [
            r'AUC[0-9]*(?:[_,]?\s*(?:0-)?(?:inf|INF|infinity|∞))?\s*(?:was|of|=|:|\()?\s*(?:approximately\s+)?(\d+[,.]?\d*)\s*[±+/-]*\s*\d*\.?\d*\s*(ng[·\.\s]*h(?:r)?/mL|ug[·\.\s]*h/mL|mg[·\.\s]*h/L|h[·\.\s]*ng/mL|ng\.h/ml|mcg[·\.\s]*h/mL|nmol[·\.\s]*h/L)',
            r'AUC\s*(?:was|of|=|:)?\s*(\d+[,.]?\d*)\s*[±+/-]*\s*\d*\.?\d*\s*(ng[·\.\s]*h/mL|ug[·\.\s]*h/mL)',
        ],
        "half_life": [
            r'(?:terminal\s+)?(?:elimination\s+)?(?:half-life|t[½1]/?2|t1/2)\s*(?:was|of|=|:|\()?\s*(?:approximately\s+)?(\d+\.?\d*)\s*[±+/-]*\s*\d*\.?\d*\s*(hours?|h\b|hr|days?|d\b)',
            r'(?:half-life|t½)\s*(?:\([^)]*\))?\s*(?:was|of|=|:)?\s*(?:approximately\s+)?(\d+\.?\d*)\s*(hours?|h\b)',
        ],
        "clearance": [
            r'(?:apparent\s+)?(?:oral\s+)?(?:total\s+)?[Cc]learance\s*(?:\(CL/?F?\))?\s*(?:was|of|=|:|is)?\s*(?:approximately\s+)?(\d+\.?\d*)\s*[±+/-]*\s*\d*\.?\d*\s*(L/h(?:r)?|mL/min|mL/min/kg|L/h/kg|ml/min)',
        ],
        "vd": [
            r'(?:apparent\s+)?[Vv](?:olume\s+of\s+)?d(?:istribution)?\s*(?:\(V[dz]/?F?\))?\s*(?:was|of|=|:|is)?\s*(?:approximately\s+)?(\d+\.?\d*)\s*[±+/-]*\s*\d*\.?\d*\s*(L\b|liters?|L/kg)',
        ],
        "bioavailability": [
            r'(?:absolute\s+)?(?:oral\s+)?[Bb]ioavailability\s*(?:\(F\))?\s*(?:was|of|=|:|is)?\s*(?:approximately\s+)?(\d+\.?\d*)\s*(%)',
        ],
        "protein_binding": [
            r'(?:plasma\s+)?[Pp]rotein\s+binding\s*(?:was|of|=|:|is)?\s*(?:approximately\s+)?(\d+\.?\d*)\s*(%)',
            r'(\d+\.?\d*)%\s*(?:bound|protein[- ]bound)',
        ],
    }

    # ── CYP PATTERNS ──
    CYP_PATTERN = re.compile(
        r'(?:CYP|cyp)\s*([123]\w\d+)',
        re.IGNORECASE
    )

    CYP_SUBSTRATE_PATTERNS = [
        r'(?:primarily|mainly|predominantly)\s+(?:metabolized|cleared)\s+(?:by|via|through)\s+(?:CYP|cyp)\s*(\w+)',
        r'(?:CYP|cyp)\s*(\w+)\s+(?:is|are)\s+(?:the\s+)?(?:primary|major|main)\s+(?:enzyme|isoform)',
        r'(?:substrate|substrates)\s+(?:of|for)\s+(?:CYP|cyp)\s*(\w+)',
    ]

    # ── DDI PATTERNS ──
    DDI_FOLD_PATTERNS = [
        r'(\w+(?:\s+\w+)?)\s+(?:increased|raised|elevated)\s+(?:the\s+)?(?:AUC|Cmax|exposure)\s+(?:of\s+)?(\w+(?:\s+\w+)?)\s+(?:by\s+)?(?:approximately\s+)?(\d+\.?\d*)[- ]?fold',
        r'(?:co-?administration|concomitant\s+(?:use|administration))\s+(?:of|with)\s+(\w+)\s+(?:increased|decreased)\s+(\w+)\s+(?:AUC|Cmax)\s+(?:by\s+)?(\d+\.?\d*)[- ]?fold',
        r'(\w+)\s+(?:is\s+)?(?:a\s+)?(?:strong|moderate|weak)\s+(?:inhibitor|inducer)\s+of\s+(?:CYP|cyp)\s*(\w+)',
    ]

    # ── ORGAN EFFECT PATTERNS ──
    ORGAN_PATTERNS = {
        "cardiovascular": [
            r'(?:QT|QTc)\s*(?:interval\s+)?(?:prolongation|prolonged)',
            r'(?:cardiac|cardiovascular)\s+(?:arrest|failure|death|toxicity|events)',
            r'(?:hypertension|hypotension|tachycardia|bradycardia|arrhythmia)',
            r'(?:myocardial\s+infarction|stroke|thromboembolism)',
        ],
        "hepatic": [
            r'(?:hepato|liver)\s*(?:toxicity|damage|failure|injury|necrosis)',
            r'(?:ALT|AST|bilirubin|transaminase)\s+(?:elevation|increased|rise)',
            r'(?:jaundice|hepatitis|cholestasis)',
            r'Drug.?[Ii]nduced\s+[Ll]iver\s+[Ii]njury',
        ],
        "renal": [
            r'(?:nephro|renal|kidney)\s*(?:toxicity|damage|failure|injury|impairment)',
            r'(?:creatinine|BUN|GFR)\s+(?:increase|elevation|decreased)',
            r'(?:acute\s+kidney\s+injury|interstitial\s+nephritis)',
        ],
        "cns": [
            r'(?:seizure|convulsion|epilepsy)',
            r'(?:somnolence|drowsiness|sedation|dizziness)',
            r'(?:suicidal|depression|psychosis|hallucination)',
            r'(?:serotonin\s+syndrome|neuroleptic\s+malignant)',
        ],
        "gi": [
            r'(?:GI|gastrointestinal)\s+(?:bleeding|ulcer|perforation)',
            r'(?:nausea|vomiting|diarrhea|constipation|pancreatitis)',
        ],
        "respiratory": [
            r'(?:respiratory\s+depression|bronchospasm|pneumonitis)',
            r'(?:pulmonary\s+(?:embolism|fibrosis|edema))',
        ],
        "hematological": [
            r'(?:neutropenia|thrombocytopenia|agranulocytosis|pancytopenia)',
            r'(?:bleeding|hemorrhage|coagulopathy)',
            r'(?:anemia|bone\s+marrow\s+suppression)',
        ],
        "immune": [
            r'(?:anaphylaxis|angioedema|hypersensitivity)',
            r'(?:Stevens.?Johnson|toxic\s+epidermal)',
            r'(?:immunosuppression|opportunistic\s+infection)',
        ],
    }

    def __init__(self):
        self.extracted_count = 0

    def extract_pk_values(self, text: str, drug_name: str = "",
                          source_url: str = "", section: str = "") -> list[PKValue]:
        """Extract all PK parameter values from text."""
        results = []

        for param, patterns in self.PK_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    try:
                        value = float(match.group(1).replace(',', ''))
                        unit = match.group(2)

                        # Skip obviously wrong values
                        if value <= 0 or value > 1e8:
                            continue

                        # For protein binding, convert to fu
                        actual_param = param
                        if param == "protein_binding":
                            actual_param = "protein_binding_pct"

                        # For bioavailability, keep as percentage
                        if param == "bioavailability":
                            actual_param = "bioavailability_pct"

                        pk_val = PKValue(
                            parameter=actual_param,
                            value=value,
                            unit=unit,
                            trace=ExtractionTrace(
                                source_url=source_url,
                                source_type="openfda_label",
                                section_title=section or "clinical_pharmacology",
                                extraction_method=ExtractionMethod.REGEX,
                                confidence=ConfidenceLevel.HIGH,
                                evidence_type=EvidenceType.FDA_LABEL,
                                raw_text=match.group(0)[:200],
                            ),
                        )
                        results.append(pk_val)
                        self.extracted_count += 1
                    except (ValueError, IndexError):
                        continue

        return results

    def extract_cyp_enzymes(self, text: str) -> tuple[list[str], list[str]]:
        """Extract CYP enzymes involved in metabolism."""
        primary = []
        secondary = []

        # Find all CYP mentions
        all_cyps = set()
        for m in self.CYP_PATTERN.finditer(text):
            cyp = "CYP" + m.group(1).upper()
            all_cyps.add(cyp)

        # Find primary CYP
        for pattern in self.CYP_SUBSTRATE_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                cyp = "CYP" + m.group(1).upper()
                if cyp not in primary:
                    primary.append(cyp)

        # Everything else is secondary
        for cyp in all_cyps:
            if cyp not in primary and cyp not in secondary:
                secondary.append(cyp)

        return primary, secondary

    def extract_ddi(self, text: str, drug_name: str = "",
                    source_url: str = "") -> list[DDIRecord]:
        """Extract drug-drug interaction information."""
        results = []

        # Pattern: "X increased the AUC of Y by N-fold"
        for pattern in self.DDI_FOLD_PATTERNS[:2]:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    perp = m.group(1).strip()
                    victim = m.group(2).strip()
                    fold = float(m.group(3))

                    ddi = DDIRecord(
                        perpetrator_drug=perp,
                        victim_drug=victim if victim != drug_name else drug_name,
                        auc_ratio=fold,
                        mechanism=InteractionMechanism.CYP_INHIBITION,
                        severity=InteractionSeverity.MAJOR if fold >= 5 else
                                 InteractionSeverity.MODERATE if fold >= 2 else
                                 InteractionSeverity.MINOR,
                        trace=ExtractionTrace(
                            source_url=source_url,
                            source_type="openfda_label",
                            section_title="drug_interactions",
                            extraction_method=ExtractionMethod.REGEX,
                            confidence=ConfidenceLevel.HIGH,
                            evidence_type=EvidenceType.FDA_LABEL,
                            raw_text=m.group(0)[:200],
                        ),
                    )
                    results.append(ddi)
                except (ValueError, IndexError):
                    continue

        # Pattern: "X is a strong/moderate/weak inhibitor/inducer of CYP..."
        for m in re.finditer(self.DDI_FOLD_PATTERNS[2], text, re.IGNORECASE):
            drug = m.group(1).strip()
            cyp = "CYP" + m.group(2).upper()

            # Determine if inhibitor or inducer
            context = text[max(0, m.start()-50):m.end()+50].lower()
            is_inhibitor = "inhibitor" in context
            is_inducer = "inducer" in context

            strength = "unknown"
            if "strong" in context: strength = "strong"
            elif "moderate" in context: strength = "moderate"
            elif "weak" in context: strength = "weak"

            ddi = DDIRecord(
                perpetrator_drug=drug,
                victim_drug=f"{cyp} substrates",
                mechanism=InteractionMechanism.CYP_INHIBITION if is_inhibitor else InteractionMechanism.CYP_INDUCTION,
                mechanism_detail=f"{drug} is a {strength} {'inhibitor' if is_inhibitor else 'inducer'} of {cyp}",
                cyp_involved=cyp,
                severity=InteractionSeverity.MAJOR if strength == "strong" else
                         InteractionSeverity.MODERATE if strength == "moderate" else
                         InteractionSeverity.MINOR,
                trace=ExtractionTrace(
                    source_url=source_url,
                    source_type="openfda_label",
                    section_title="drug_interactions",
                    extraction_method=ExtractionMethod.REGEX,
                    confidence=ConfidenceLevel.HIGH,
                    evidence_type=EvidenceType.FDA_LABEL,
                    raw_text=m.group(0)[:200],
                ),
            )
            results.append(ddi)

        # Look for contraindicated combinations
        contra_pattern = r'(?:contraindicated|must\s+not\s+be\s+(?:used|administered|given))\s+(?:with|in\s+combination\s+with)\s+(\w+(?:\s+\w+)?)'
        for m in re.finditer(contra_pattern, text, re.IGNORECASE):
            contra_drug = m.group(1).strip()
            ddi = DDIRecord(
                perpetrator_drug=contra_drug,
                victim_drug=drug_name,
                severity=InteractionSeverity.CONTRAINDICATED,
                contraindicated=True,
                clinical_significance="Contraindicated combination",
                trace=ExtractionTrace(
                    source_url=source_url,
                    source_type="openfda_label",
                    section_title="contraindications",
                    extraction_method=ExtractionMethod.REGEX,
                    confidence=ConfidenceLevel.VALIDATED,
                    evidence_type=EvidenceType.FDA_LABEL,
                    raw_text=m.group(0)[:200],
                ),
            )
            results.append(ddi)

        return results

    def extract_organ_effects(self, text: str, source_url: str = "") -> list[OrganEffect]:
        """Extract organ-specific effects from warnings/adverse reactions text."""
        results = []

        for organ, patterns in self.ORGAN_PATTERNS.items():
            for pattern in patterns:
                for m in re.finditer(pattern, text, re.IGNORECASE):
                    # Get surrounding context
                    start = max(0, m.start() - 100)
                    end = min(len(text), m.end() + 100)
                    context = text[start:end]

                    effect = OrganEffect(
                        organ_system=organ,
                        effect_type="warning",
                        description=m.group(0),
                        trace=ExtractionTrace(
                            source_url=source_url,
                            source_type="openfda_label",
                            section_title="warnings_and_precautions",
                            extraction_method=ExtractionMethod.REGEX,
                            confidence=ConfidenceLevel.HIGH,
                            evidence_type=EvidenceType.FDA_LABEL,
                            raw_text=context[:300],
                        ),
                    )
                    results.append(effect)

        return results

    def extract_special_populations(self, text: str, source_url: str = "") -> list[SpecialPopulationPK]:
        """Extract special population PK changes."""
        results = []

        pop_patterns = [
            (r'(?:renal\s+impairment|renal\s+insufficiency|CKD)', "renal_impairment"),
            (r'(?:hepatic\s+impairment|liver\s+(?:disease|impairment)|Child.?Pugh)', "hepatic_impairment"),
            (r'(?:elderly|geriatric|older\s+(?:adults|patients)|age\s*>\s*65)', "elderly"),
            (r'(?:pediatric|children|adolescents?|age\s*<\s*18)', "pediatric"),
            (r'(?:poor\s+metabolizer|CYP2D6\s+PM|slow\s+metabolizer)', "cyp2d6_pm"),
            (r'(?:pregnancy|pregnant|lactation|breast.?feeding)', "pregnancy"),
        ]

        for pattern, pop_name in pop_patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                # Look for fold changes near this mention
                context = text[max(0, m.start()-200):min(len(text), m.end()+300)]

                fold_match = re.search(r'(\d+\.?\d*)[- ]?fold\s+(?:increase|higher|greater)', context, re.IGNORECASE)
                pct_match = re.search(r'(\d+\.?\d*)%\s+(?:increase|higher|greater|more)', context, re.IGNORECASE)
                decrease_match = re.search(r'(?:decrease|reduce|lower)\s+(?:by\s+)?(\d+\.?\d*)%', context, re.IGNORECASE)

                sp = SpecialPopulationPK(
                    population=pop_name,
                    parameter_affected="exposure",
                    fold_change=float(fold_match.group(1)) if fold_match else None,
                    pct_change=float(pct_match.group(1)) if pct_match else
                               -float(decrease_match.group(1)) if decrease_match else None,
                    trace=ExtractionTrace(
                        source_url=source_url,
                        source_type="openfda_label",
                        section_title="use_in_specific_populations",
                        extraction_method=ExtractionMethod.REGEX,
                        confidence=ConfidenceLevel.HIGH,
                        evidence_type=EvidenceType.FDA_LABEL,
                        raw_text=context[:300],
                    ),
                )
                results.append(sp)

        return results

    def extract_safety_signals(self, text: str, source_url: str = "") -> list[SafetySignal]:
        """Extract safety/warning signals."""
        results = []

        # Boxed warning
        if re.search(r'(?:WARNING|BOXED WARNING|BLACK BOX)', text, re.IGNORECASE):
            boxed_text = text[:500]  # First 500 chars likely contain boxed warning
            results.append(SafetySignal(
                signal_type="boxed_warning",
                description=boxed_text[:300],
                qt_prolongation="QT" in boxed_text.upper(),
                trace=ExtractionTrace(
                    source_url=source_url,
                    source_type="openfda_label",
                    section_title="boxed_warning",
                    extraction_method=ExtractionMethod.REGEX,
                    confidence=ConfidenceLevel.VALIDATED,
                    evidence_type=EvidenceType.FDA_LABEL,
                ),
            ))

        # QT prolongation
        if re.search(r'QT[c]?\s*(?:interval\s+)?(?:prolongation|prolonged)', text, re.IGNORECASE):
            results.append(SafetySignal(
                signal_type="warning",
                description="QT/QTc interval prolongation",
                organ_affected="cardiovascular",
                qt_prolongation=True,
                trace=ExtractionTrace(
                    source_url=source_url,
                    extraction_method=ExtractionMethod.REGEX,
                    confidence=ConfidenceLevel.VALIDATED,
                    evidence_type=EvidenceType.FDA_LABEL,
                ),
            ))

        # Hepatotoxicity
        if re.search(r'(?:hepatotoxicity|liver\s+(?:failure|injury|damage)|DILI)', text, re.IGNORECASE):
            results.append(SafetySignal(
                signal_type="warning",
                description="Hepatotoxicity risk",
                organ_affected="hepatic",
                hepatotoxicity=True,
                trace=ExtractionTrace(
                    source_url=source_url,
                    extraction_method=ExtractionMethod.REGEX,
                    confidence=ConfidenceLevel.VALIDATED,
                    evidence_type=EvidenceType.FDA_LABEL,
                ),
            ))

        return results

    def extract_all(self, drug_record: dict) -> dict:
        """Extract everything from a drug record (from openFDA)."""
        drug_name = drug_record.get("generic_name", "")
        source_url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:\"{drug_name}\""

        cp_text = drug_record.get("clinical_pharmacology_text", "")
        ddi_text = drug_record.get("drug_interactions_text", "")
        warnings_text = drug_record.get("warnings_text", "")
        adverse_text = drug_record.get("adverse_reactions_text", "")
        contra_text = drug_record.get("contraindications_text", "")
        mech_text = drug_record.get("mechanism_text", "")

        all_text = f"{cp_text} {ddi_text} {warnings_text} {adverse_text} {contra_text} {mech_text}"

        # Extract PK values
        pk_values = self.extract_pk_values(cp_text, drug_name, source_url, "clinical_pharmacology")

        # Extract CYP enzymes
        primary_cyp, secondary_cyp = self.extract_cyp_enzymes(all_text)

        # Extract DDI
        ddis = self.extract_ddi(ddi_text + " " + cp_text, drug_name, source_url)

        # Extract organ effects
        organ_effects = self.extract_organ_effects(warnings_text + " " + adverse_text, source_url)

        # Extract special populations
        special_pops = self.extract_special_populations(cp_text + " " + all_text, source_url)

        # Extract safety signals
        safety = self.extract_safety_signals(warnings_text + " " + contra_text, source_url)

        # Extract protein binding
        fu = None
        pb_pct = None
        for pk in pk_values:
            if pk.parameter == "protein_binding_pct":
                pb_pct = pk.value
                fu = 1 - pk.value / 100

        # Extract bioavailability
        F = None
        for pk in pk_values:
            if pk.parameter == "bioavailability_pct":
                F = pk.value / 100

        return {
            "drug_name": drug_name,
            "pk_values": [pv.model_dump() for pv in pk_values],
            "primary_cyp": primary_cyp,
            "secondary_cyp": secondary_cyp,
            "fu": fu,
            "protein_binding_pct": pb_pct,
            "bioavailability": F,
            "interactions": [d.model_dump() for d in ddis],
            "organ_effects": [o.model_dump() for o in organ_effects],
            "special_populations": [sp.model_dump() for sp in special_pops],
            "safety_signals": [s.model_dump() for s in safety],
            "has_data": bool(pk_values or ddis or organ_effects),
        }
