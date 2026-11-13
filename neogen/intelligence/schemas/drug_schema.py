"""
NeoGen FDA Drug Intelligence Schemas
======================================
Pydantic models for structured drug data extracted from FDA sources.
Every field supports source traceability and confidence scoring.

Design principles:
1. Every extracted value has a source (where it came from)
2. Every value has a confidence level (how reliable it is)
3. Every value has an evidence type (clinical trial vs label vs signal)
4. Schema is NeoGen-ready (maps directly to simulation parameters)
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ════════════════════════════════════════════════════════════════
# ENUMS
# ════════════════════════════════════════════════════════════════

class EvidenceType(str, Enum):
    CLINICAL_TRIAL = "clinical_trial"          # Phase I/II/III measured data
    FDA_LABEL = "fda_label"                    # Approved drug label
    CLINICAL_PHARMACOLOGY_REVIEW = "clin_pharm_review"  # FDA reviewer analysis
    IN_VITRO = "in_vitro"                      # Lab study
    POPULATION_PK = "pop_pk"                   # PopPK analysis
    POST_MARKET = "post_market"                # FAERS / post-market signal
    CASE_REPORT = "case_report"                # Published case report
    INFERRED = "inferred"                      # Computationally inferred

class ConfidenceLevel(str, Enum):
    VALIDATED = "validated"        # >90% — published clinical data
    HIGH = "high"                  # 70-90% — FDA label or strong evidence
    MODERATE = "moderate"          # 50-70% — consistent secondary sources
    LOW = "low"                    # 30-50% — limited evidence
    SPECULATIVE = "speculative"    # <30% — signal only

class ExtractionMethod(str, Enum):
    REGEX = "regex"
    TABLE_PARSE = "table_parse"
    LLM = "llm"
    NER = "ner"
    MANUAL = "manual"
    API = "api"

class InteractionSeverity(str, Enum):
    CONTRAINDICATED = "contraindicated"
    MAJOR = "major"
    MODERATE = "moderate"
    MINOR = "minor"
    UNKNOWN = "unknown"

class InteractionMechanism(str, Enum):
    CYP_INHIBITION = "cyp_inhibition"
    CYP_INDUCTION = "cyp_induction"
    TRANSPORTER = "transporter"
    PROTEIN_DISPLACEMENT = "protein_displacement"
    PD_SYNERGY = "pd_synergy"
    PD_ANTAGONISM = "pd_antagonism"
    RENAL_COMPETITION = "renal_competition"
    PH_DEPENDENT = "ph_dependent"
    CHELATION = "chelation"
    OTHER = "other"
    UNKNOWN = "unknown"


# ════════════════════════════════════════════════════════════════
# SOURCE TRACEABILITY
# ════════════════════════════════════════════════════════════════

class ExtractionTrace(BaseModel):
    """Every extracted value must have this attached."""
    source_url: str = ""
    source_type: str = ""             # "drugs_at_fda", "dailymed", "openfda", "faers"
    document_name: str = ""
    section_title: str = ""
    page_number: Optional[int] = None
    extraction_method: ExtractionMethod = ExtractionMethod.REGEX
    confidence: ConfidenceLevel = ConfidenceLevel.MODERATE
    evidence_type: EvidenceType = EvidenceType.FDA_LABEL
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    raw_text: str = ""                # The original text this was extracted from
    notes: str = ""


# ════════════════════════════════════════════════════════════════
# DRUG IDENTITY
# ════════════════════════════════════════════════════════════════

class DrugIdentity(BaseModel):
    """Core drug identification."""
    generic_name: str
    brand_names: list[str] = []
    active_ingredients: list[str] = []
    smiles: str = ""
    inchi: str = ""
    inchi_key: str = ""
    nda_number: str = ""
    bla_number: str = ""
    sponsor: str = ""
    approval_date: str = ""
    dosage_form: str = ""
    route: str = ""
    strengths: list[str] = []
    therapeutic_class: str = ""
    atc_code: str = ""
    rxnorm_cui: str = ""
    unii: str = ""
    is_combination: bool = False
    trace: ExtractionTrace = Field(default_factory=ExtractionTrace)


# ════════════════════════════════════════════════════════════════
# STUDY CONTEXT
# ════════════════════════════════════════════════════════════════

class StudyContext(BaseModel):
    """Context for any PK/PD measurement."""
    study_type: str = ""              # "Phase I", "Phase III", "BE", "DDI", "food effect"
    study_design: str = ""            # "randomized crossover", "parallel", "open-label"
    n_subjects: Optional[int] = None
    population: str = ""              # "healthy volunteers", "patients with renal impairment"
    dose_mg: Optional[float] = None
    dose_regimen: str = ""            # "single dose", "BID x 14 days", "QD steady state"
    route: str = "oral"
    formulation: str = ""
    fed_fasted: str = ""              # "fasted", "fed", "high-fat meal"
    single_or_multiple: str = ""      # "single", "multiple", "steady-state"
    trace: ExtractionTrace = Field(default_factory=ExtractionTrace)


# ════════════════════════════════════════════════════════════════
# PK PARAMETERS
# ════════════════════════════════════════════════════════════════

class PKValue(BaseModel):
    """A single PK parameter measurement with full context."""
    parameter: str                    # "cmax", "auc", "half_life", "clearance", "vd", etc.
    value: float
    unit: str
    sd: Optional[float] = None       # Standard deviation
    cv_pct: Optional[float] = None   # Coefficient of variation %
    ci_lower: Optional[float] = None # 90% CI lower
    ci_upper: Optional[float] = None # 90% CI upper
    geometric_mean: bool = False     # True if geometric mean (common for PK)
    study: StudyContext = Field(default_factory=StudyContext)
    trace: ExtractionTrace = Field(default_factory=ExtractionTrace)


class PKProfile(BaseModel):
    """Complete PK profile for a drug."""
    drug_name: str
    parameters: list[PKValue] = []

    # Absorption
    oral_absorption_pct: Optional[float] = None
    lag_time_h: Optional[float] = None
    food_effect: str = ""
    food_effect_cmax_ratio: Optional[float] = None
    food_effect_auc_ratio: Optional[float] = None

    # Distribution
    protein_binding_pct: Optional[float] = None
    fu: Optional[float] = None
    binding_protein: str = ""         # "albumin", "AAG", "both"
    bbb_penetration: str = ""
    rbc_plasma_ratio: Optional[float] = None
    tissue_distribution: str = ""

    # Metabolism
    primary_cyp: list[str] = []       # ["CYP3A4", "CYP2C9"]
    secondary_cyp: list[str] = []
    ugt_enzymes: list[str] = []
    other_enzymes: list[str] = []
    active_metabolites: list[str] = []
    major_metabolites: list[str] = []
    metabolic_pathway: str = ""

    # Excretion
    renal_excretion_pct: Optional[float] = None
    fecal_excretion_pct: Optional[float] = None
    unchanged_in_urine_pct: Optional[float] = None
    renal_clearance_lh: Optional[float] = None
    dialyzable: Optional[bool] = None

    # Linearity
    dose_proportional: Optional[bool] = None
    dose_proportional_range: str = ""

    trace: ExtractionTrace = Field(default_factory=ExtractionTrace)


# ════════════════════════════════════════════════════════════════
# DRUG-DRUG INTERACTIONS
# ════════════════════════════════════════════════════════════════

class DDIRecord(BaseModel):
    """A single drug-drug interaction finding."""
    perpetrator_drug: str
    victim_drug: str
    perpetrator_dose: str = ""
    victim_dose: str = ""
    mechanism: InteractionMechanism = InteractionMechanism.UNKNOWN
    mechanism_detail: str = ""        # "Ketoconazole inhibits CYP3A4"

    # Effect on victim PK
    cmax_ratio: Optional[float] = None    # GMR (e.g., 2.5 = 150% increase)
    auc_ratio: Optional[float] = None     # GMR
    cmin_ratio: Optional[float] = None
    clearance_ratio: Optional[float] = None

    severity: InteractionSeverity = InteractionSeverity.UNKNOWN
    clinical_significance: str = ""
    dose_adjustment: str = ""
    contraindicated: bool = False

    # CYP/transporter specifics
    cyp_involved: str = ""
    transporter_involved: str = ""
    inhibition_type: str = ""         # "competitive", "mechanism-based", "induction"
    ki_um: Optional[float] = None
    kinact_per_h: Optional[float] = None

    study: StudyContext = Field(default_factory=StudyContext)
    trace: ExtractionTrace = Field(default_factory=ExtractionTrace)


# ════════════════════════════════════════════════════════════════
# ORGAN EFFECTS (FOR NEOGEN SIMULATION)
# ════════════════════════════════════════════════════════════════

class OrganEffect(BaseModel):
    """Drug effect on a specific organ/system."""
    organ_system: str                 # "cardiovascular", "hepatic", "renal", "cns", "gi", etc.
    effect_type: str                  # "toxicity", "therapeutic", "side_effect", "warning"
    description: str
    severity: str = ""                # "mild", "moderate", "severe", "life-threatening"
    dose_dependent: Optional[bool] = None
    concentration_dependent: Optional[bool] = None
    reversible: Optional[bool] = None
    monitoring_required: str = ""
    incidence_pct: Optional[float] = None
    trace: ExtractionTrace = Field(default_factory=ExtractionTrace)


# ════════════════════════════════════════════════════════════════
# SPECIAL POPULATIONS
# ════════════════════════════════════════════════════════════════

class SpecialPopulationPK(BaseModel):
    """PK changes in special populations."""
    population: str                   # "renal_impairment_mild", "hepatic_child_pugh_b", "elderly", etc.
    parameter_affected: str           # "auc", "cmax", "clearance"
    fold_change: Optional[float] = None   # e.g., 1.5 = 50% increase
    pct_change: Optional[float] = None    # e.g., 50 = 50% increase
    dose_adjustment: str = ""
    contraindicated: bool = False
    clinical_recommendation: str = ""
    trace: ExtractionTrace = Field(default_factory=ExtractionTrace)


# ════════════════════════════════════════════════════════════════
# SAFETY
# ════════════════════════════════════════════════════════════════

class SafetySignal(BaseModel):
    """Safety/warning information."""
    signal_type: str                  # "boxed_warning", "contraindication", "warning", "adverse_reaction"
    description: str
    organ_affected: str = ""
    incidence_pct: Optional[float] = None
    dose_limiting: bool = False
    qt_prolongation: bool = False
    seizure_risk: bool = False
    hepatotoxicity: bool = False
    nephrotoxicity: bool = False
    monitoring: str = ""
    trace: ExtractionTrace = Field(default_factory=ExtractionTrace)


# ════════════════════════════════════════════════════════════════
# ADVERSE EVENT (FAERS)
# ════════════════════════════════════════════════════════════════

class AdverseEventSignal(BaseModel):
    """Post-market adverse event signal from FAERS."""
    drug_name: str
    event_term: str                   # MedDRA preferred term
    report_count: int = 0
    serious_count: int = 0
    death_count: int = 0
    prr: Optional[float] = None      # Proportional Reporting Ratio
    ror: Optional[float] = None       # Reporting Odds Ratio
    co_occurring_drugs: list[str] = []
    trace: ExtractionTrace = Field(default_factory=ExtractionTrace)


# ════════════════════════════════════════════════════════════════
# COMPLETE DRUG RECORD
# ════════════════════════════════════════════════════════════════

class FDADrugRecord(BaseModel):
    """Complete drug intelligence record for NeoGen."""
    identity: DrugIdentity
    pk_profile: PKProfile = Field(default_factory=lambda: PKProfile(drug_name=""))
    interactions: list[DDIRecord] = []
    organ_effects: list[OrganEffect] = []
    special_populations: list[SpecialPopulationPK] = []
    safety_signals: list[SafetySignal] = []
    adverse_events: list[AdverseEventSignal] = []

    # Metadata
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    data_completeness_pct: float = 0.0
    source_documents: list[str] = []

    def compute_completeness(self):
        """Calculate how complete this record is."""
        score = 0
        total = 0

        # Identity (10 points)
        total += 10
        if self.identity.generic_name: score += 2
        if self.identity.smiles: score += 3
        if self.identity.nda_number: score += 2
        if self.identity.therapeutic_class: score += 1
        if self.identity.route: score += 1
        if self.identity.approval_date: score += 1

        # PK (30 points)
        total += 30
        pk = self.pk_profile
        if pk.parameters: score += min(len(pk.parameters) * 3, 15)
        if pk.fu is not None: score += 3
        if pk.primary_cyp: score += 3
        if pk.renal_excretion_pct is not None: score += 3
        if pk.food_effect: score += 2
        if pk.active_metabolites: score += 2
        if pk.protein_binding_pct is not None: score += 2

        # DDI (20 points)
        total += 20
        if self.interactions: score += min(len(self.interactions) * 2, 20)

        # Safety (20 points)
        total += 20
        if self.safety_signals: score += min(len(self.safety_signals) * 2, 10)
        if self.organ_effects: score += min(len(self.organ_effects) * 2, 10)

        # Special populations (10 points)
        total += 10
        if self.special_populations: score += min(len(self.special_populations) * 2, 10)

        # Adverse events (10 points)
        total += 10
        if self.adverse_events: score += min(len(self.adverse_events), 10)

        self.data_completeness_pct = round(100 * score / total, 1)
        return self.data_completeness_pct
