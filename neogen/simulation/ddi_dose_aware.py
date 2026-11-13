"""Dose-aware DDI predictor — predicts AUCR magnitude, not just severity.

Mechanism:
    AUCR = 1 + (I / Ki) * fm

where
    I   = perpetrator unbound concentration at target tissue (µM)
    Ki  = perpetrator inhibition constant at the target enzyme (µM)
    fm  = fraction of victim cleared via the target enzyme

This is the standard mechanistic-static-model (MSM) from the FDA 2020
DDI guidance, simplified to a single enzyme. For multi-enzyme or
time-dependent inhibitors we fall back to conservative class-based
defaults.

Data sources for the Ki/fm tables:
    * FDA Drug Development and Drug Interactions tables (2023 update)
    * Varbanov et al. 2018, University of Washington Metabolism & Transport
      Drug Interaction Database ranges
    * Published label sections for the eight benchmark victims
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ─── perpetrator Ki (µM) at their dominant target enzyme ─────────────
# Values are order-of-magnitude typical unbound Ki from published
# in vitro CYP inhibition assays. Assumption marked where a canonical
# single number is not available.
# Geometric-mean Ki values across published ranges (not the most-potent
# literature extreme) — trades best-case predictions for calibration
# that matches clinical observation distributions.
PERPETRATOR_KI: dict[str, dict] = {
    "ketoconazole":   {"target": "cyp3a4", "ki_um": 0.15,  "mechanism": "competitive"},
    "itraconazole":   {"target": "cyp3a4", "ki_um": 0.03,  "mechanism": "competitive"},
    "clarithromycin": {"target": "cyp3a4", "ki_um": 1.5,   "mechanism": "MBI"},
    "erythromycin":   {"target": "cyp3a4", "ki_um": 8.0,   "mechanism": "MBI"},
    "diltiazem":      {"target": "cyp3a4", "ki_um": 5.0,   "mechanism": "MBI"},
    "verapamil":      {"target": "cyp3a4", "ki_um": 3.0,   "mechanism": "MBI"},
    "ritonavir":      {"target": "cyp3a4", "ki_um": 0.02,  "mechanism": "competitive"},
    "fluoxetine":     {"target": "cyp2d6", "ki_um": 0.3,   "mechanism": "competitive"},
    "paroxetine":     {"target": "cyp2d6", "ki_um": 0.05,  "mechanism": "MBI"},
    "cimetidine":     {"target": "cyp3a4", "ki_um": 65.0,  "mechanism": "competitive"},
    "omeprazole":     {"target": "cyp2c19", "ki_um": 2.0,  "mechanism": "competitive"},
    "rifampicin":     {"target": "cyp3a4", "ki_um": None,  "mechanism": "inducer"},
    "rifampin":       {"target": "cyp3a4", "ki_um": None,  "mechanism": "inducer"},
}

# ─── victim fm (fraction cleared by target enzyme) ───────────────────
VICTIM_FM: dict[str, dict] = {
    "midazolam":     {"cyp3a4": 0.95},
    "simvastatin":   {"cyp3a4": 0.95},
    "atorvastatin":  {"cyp3a4": 0.70, "oatp1b1": 0.30},
    "nifedipine":    {"cyp3a4": 0.80},
    "amlodipine":    {"cyp3a4": 0.90},
    "diazepam":      {"cyp3a4": 0.50, "cyp2c19": 0.45},
    "metoprolol":    {"cyp2d6": 0.70},
    "warfarin":      {"cyp2c9": 0.85, "cyp3a4": 0.10},
    "ibuprofen":     {"cyp2c9": 0.70},
}

# Induction fold-decreases (AUCR < 1). Rifampicin is the canonical strong
# CYP3A4 inducer → victim AUC drops ~5× on full steady state.
INDUCTION_FOLD_AUC_DECREASE: dict[str, float] = {
    "rifampicin": 0.1,  # victim AUC × 0.1 → AUCR = 0.1
    "rifampin":   0.1,
}

SeverityLevel = Literal["none", "minor", "moderate", "major"]


def _severity_from_aucr(aucr: float) -> SeverityLevel:
    """FDA 2020 in vitro DDI guidance thresholds."""
    if aucr >= 5.0:
        return "major"
    if aucr >= 2.0:
        return "moderate"
    if aucr >= 1.25 or aucr <= 0.8:  # clinically relevant decrease
        return "minor"
    return "none"


@dataclass(frozen=True)
class DoseContext:
    """Dose / exposure context for the perpetrator + victim."""
    perpetrator_ss_conc_um: float | None = None
    perpetrator_duration_h: float | None = None
    perpetrator_dose_mg: float | None = None
    victim_dose_mg: float | None = None
    victim_route: str | None = None


@dataclass
class DosedDDIPrediction:
    perpetrator: str
    victim: str
    predicted_aucr: float
    predicted_severity: SeverityLevel
    confidence: float
    rationale: str
    target_enzyme: str | None = None
    used: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "perpetrator": self.perpetrator,
            "victim": self.victim,
            "predicted_aucr": round(self.predicted_aucr, 3),
            "predicted_severity": self.predicted_severity,
            "confidence": round(self.confidence, 3),
            "target_enzyme": self.target_enzyme,
            "rationale": self.rationale,
            "used": self.used,
        }


class DoseAwareDDIPredictor:
    """Predict AUCR using MSM: AUCR = 1 + (I/Ki) * fm."""

    def predict(
        self,
        perpetrator: str,
        victim: str,
        ctx: DoseContext | None = None,
    ) -> DosedDDIPrediction:
        p = perpetrator.lower().strip()
        v = victim.lower().strip()
        ctx = ctx or DoseContext()

        # Inducer short-circuit
        if p in INDUCTION_FOLD_AUC_DECREASE:
            aucr = INDUCTION_FOLD_AUC_DECREASE[p]
            return DosedDDIPrediction(
                perpetrator=p, victim=v,
                predicted_aucr=aucr,
                predicted_severity=_severity_from_aucr(aucr),
                confidence=0.70,
                rationale="CYP3A4 inducer — literature typical fold decrease",
                target_enzyme="cyp3a4",
                used={"mechanism": "induction"},
            )

        if p not in PERPETRATOR_KI:
            return DosedDDIPrediction(
                perpetrator=p, victim=v,
                predicted_aucr=1.0,
                predicted_severity="none",
                confidence=0.0,
                rationale=f"no Ki data for perpetrator {p}",
                used={},
            )

        perp = PERPETRATOR_KI[p]
        target = perp["target"]
        ki = perp["ki_um"]
        mechanism = perp["mechanism"]

        if ki is None:
            return DosedDDIPrediction(
                perpetrator=p, victim=v,
                predicted_aucr=1.0, predicted_severity="none",
                confidence=0.0,
                rationale=f"Ki unavailable for {p}",
                used={},
            )

        fm_record = VICTIM_FM.get(v, {})
        fm = fm_record.get(target, 0.0)
        if fm == 0.0:
            return DosedDDIPrediction(
                perpetrator=p, victim=v,
                predicted_aucr=1.0,
                predicted_severity="none",
                confidence=0.20,
                rationale=f"{v} has no documented fm on {target}",
                target_enzyme=target,
                used={"ki_um": ki, "fm": 0.0},
            )

        # Concentration — use supplied SS conc, else fall back to a
        # class-default estimate from dose (very coarse).
        i_um = ctx.perpetrator_ss_conc_um
        conf_conc = 1.0
        if i_um is None:
            if ctx.perpetrator_dose_mg:
                i_um = 0.05 * ctx.perpetrator_dose_mg  # coarse µM ~ 0.05 × mg
                conf_conc = 0.5
            else:
                i_um = ki  # fallback: assume I ≈ Ki → AUCR ≈ 1 + fm
                conf_conc = 0.25

        # Core MSM. For MBI, shift Ki down by 3× to approximate time-dependent
        # inactivation at steady state.
        effective_ki = ki / 3.0 if mechanism == "MBI" else ki
        aucr = 1.0 + (i_um / effective_ki) * fm

        # Apply duration attenuation for short exposures (<24h)
        if ctx.perpetrator_duration_h is not None and ctx.perpetrator_duration_h < 24:
            frac = ctx.perpetrator_duration_h / 24.0
            aucr = 1.0 + (aucr - 1.0) * frac

        # Confidence: product of data availability × concentration confidence
        fm_conf = 1.0 if fm >= 0.5 else 0.6
        confidence = 0.85 * conf_conc * fm_conf

        return DosedDDIPrediction(
            perpetrator=p, victim=v,
            predicted_aucr=aucr,
            predicted_severity=_severity_from_aucr(aucr),
            confidence=confidence,
            rationale=(
                f"MSM: AUCR = 1 + (I={i_um:.3g}/Ki={effective_ki:.3g}) × fm={fm:.2f} "
                f"[{mechanism}]"
            ),
            target_enzyme=target,
            used={
                "ki_um": ki,
                "effective_ki_um": effective_ki,
                "i_um": i_um,
                "fm": fm,
                "mechanism": mechanism,
                "duration_h": ctx.perpetrator_duration_h,
            },
        )
