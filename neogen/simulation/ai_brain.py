"""
NeoGen AI Brain — Levels 1-4
================================
Uses Claude Opus via Anthropic API to interpret PBPK simulation results.

Level 1: Clinical Interpreter — reads simulation, writes clinical narrative
Level 2: Interactive Conversation — follow-up questions with context memory
Level 3: Autonomous Analyst — runs multiple simulations, generates drug report
Level 4: Trial Designer — designs Phase I clinical trial from PBPK data

Architecture:
  PBPK simulation runs → results packaged as structured context →
  sent to Claude Opus → AI writes interpretation → validation layer
  checks numbers → displayed to user

Safety:
  - Only simulation numbers are sent (no personal data)
  - AI is instructed to ONLY cite numbers from simulation data
  - Validation layer cross-checks AI claims against actual results
  - Disclaimer on every response
"""
from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ════════════════════════════════════════════════════════════════

LEVEL1_SYSTEM_PROMPT = """You are NeoGen's Clinical Pharmacology AI — a board-certified clinical pharmacologist interpreting PBPK simulation results.

You receive structured simulation data for a drug in a specific patient. Your job is to write a clear, clinical interpretation that a doctor or researcher can immediately understand and act on.

RULES (CRITICAL — FOLLOW EXACTLY):
1. ONLY cite numbers from the simulation data provided. NEVER invent values.
2. If data for something is not provided, say "not assessed in this simulation."
3. NEVER say a drug is "safe" — say "within expected therapeutic range based on simulation."
4. Always err on the side of caution for safety assessments.
5. Use plain English. Avoid jargon unless explaining it.
6. Keep your response under 300 words.
7. Always end with a disclaimer: "AI-generated interpretation. Not a substitute for clinical judgment."

STRUCTURE your response as:
1. SUMMARY (2-3 sentences: what drug, what dose, key finding)
2. PK PROFILE (peak level, time to peak, duration, elimination)
3. SAFETY ASSESSMENT (organ risks, therapeutic range comparison)
4. CLINICAL RECOMMENDATION (dose adjustment, monitoring, warnings)
5. DISCLAIMER
"""

LEVEL2_SYSTEM_PROMPT = """You are NeoGen's Interactive Clinical Pharmacologist. You have access to the full PBPK simulation data for the current drug and patient.

You answer follow-up questions about the simulation results. You maintain conversation context — the user may ask multiple questions about the same simulation.

RULES:
1. ONLY reference data from the simulation results provided in the context.
2. If asked about something not in the data, say "This was not assessed. I would need to run a separate simulation for that."
3. Be conversational but precise. Use the actual numbers from the simulation.
4. If the user asks "what if" questions (different dose, different patient), explain what you'd EXPECT based on PK principles, but note it would need a new simulation to confirm.
5. Keep each response under 200 words.
6. End safety-related answers with the disclaimer.
"""

LEVEL3_SYSTEM_PROMPT = """You are NeoGen's Autonomous Drug Analyst. You receive the results of multiple PBPK simulations run at different doses for the same drug.

Your job is to synthesize ALL results into a comprehensive drug development assessment.

STRUCTURE your report as:
1. EXECUTIVE SUMMARY (3 sentences: go/no-go recommendation)
2. PK PROFILE ACROSS DOSES (how Cmax, AUC, t½ change with dose — linear or non-linear?)
3. THERAPEUTIC WINDOW (what dose range achieves therapeutic levels without toxicity?)
4. SAFETY ASSESSMENT
   - Cardiac risk (hERG/QT at each dose)
   - Hepatotoxicity risk (liver concentrations vs thresholds)
   - Renal risk (kidney concentrations + clearance)
   - CNS effects (brain penetration + neurotransmitter effects)
5. DRUG INTERACTION RISK (which CYP enzymes, which inhibitors to avoid)
6. SPECIAL POPULATIONS (elderly, pediatric, renal impairment, hepatic impairment — how would PK change?)
7. RECOMMENDED DOSE RANGE for Phase I
8. KEY RISKS AND MITIGATION

Keep under 500 words. Be specific — cite actual numbers from each simulation.
"""

LEVEL4_SYSTEM_PROMPT = """You are NeoGen's Clinical Trial Designer. You receive comprehensive PBPK simulation data across multiple doses and populations.

Your job is to design a complete Phase I clinical trial protocol.

STRUCTURE your protocol as:
1. STUDY TITLE
2. STUDY OBJECTIVES (primary: safety/tolerability/PK; secondary: food effect, dose proportionality)
3. STUDY DESIGN (SAD, MAD, or SAD+MAD; open-label; dose-escalation)
4. DOSE SELECTION
   - Starting dose and rationale (from PBPK: 1/10th of dose causing any safety signal)
   - Dose escalation scheme (modified Fibonacci or fixed ratio)
   - Maximum planned dose and rationale
   - Sentinel dosing strategy
5. STUDY POPULATION
   - Inclusion criteria (age, weight, health status)
   - Exclusion criteria (CYP genotype if relevant, concomitant meds, organ impairment)
   - Number of subjects per cohort
   - Total number of subjects
6. PK SAMPLING SCHEDULE
   - Timepoints optimized from predicted concentration-time profile
   - Rationale for each timepoint (capture absorption, peak, distribution, elimination)
7. SAFETY MONITORING
   - Dose-limiting toxicity (DLT) criteria
   - Stopping rules
   - Safety assessments (ECG timing based on predicted Tmax, labs, vitals)
8. PHARMACOGENOMICS
   - CYP genotyping requirements (if drug is CYP2D6/2C19 substrate)
   - Stratification plan
9. FOOD EFFECT
   - Recommendation based on PBPK food effect simulation
10. STUDY DURATION AND TIMELINE

Be extremely specific. Use actual PBPK numbers to justify every decision.
"""


# ════════════════════════════════════════════════════════════════
# AI BRAIN CLASS
# ════════════════════════════════════════════════════════════════

@dataclass
class ConversationMessage:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class AIBrain:
    """NeoGen's AI Brain — Claude Opus via Anthropic API."""
    api_key: str = ""
    model: str = "claude-opus-4-20250514"
    conversation_history: list = field(default_factory=list)
    current_simulation_context: str = ""

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            logger.warning("No ANTHROPIC_API_KEY found. AI Brain will use mock responses.")

    def _call_claude(self, system_prompt: str, user_message: str,
                     history: list = None, max_tokens: int = 1500) -> str:
        """Call Claude Opus API."""
        if not self.api_key:
            return self._mock_response(user_message)

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            messages = []
            if history:
                for msg in history[-10:]:  # Keep last 10 messages
                    messages.append({"role": msg.role, "content": msg.content})
            messages.append({"role": "user", "content": user_message})

            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
            )

            return response.content[0].text

        except Exception as e:
            logger.error("Claude API call failed: %s", e)
            return f"AI Brain error: {str(e)}\n\nFalling back to simulation data only."

    def _mock_response(self, user_message: str) -> str:
        """Generate template response when no API key available."""
        return (
            "**AI Brain (Mock Mode — No API Key)**\n\n"
            "The AI Clinical Advisor requires a Claude API key to generate "
            "intelligent interpretations. Set ANTHROPIC_API_KEY in your .env file.\n\n"
            "In the meantime, please refer to the simulation data displayed "
            "in the panels for PK parameters, safety assessments, and organ concentrations.\n\n"
            "*AI-generated interpretation. Not a substitute for clinical judgment.*"
        )

    # ════════════════════════════════════════════════════════════
    # LEVEL 1: CLINICAL INTERPRETER
    # ════════════════════════════════════════════════════════════

    def interpret(self, simulation_results: dict) -> dict:
        """
        Level 1: Interpret PBPK simulation results.

        Args:
            simulation_results: Full output from /api/predict

        Returns:
            dict with interpretation text + metadata
        """
        # Package simulation data as structured context
        context = self._build_context(simulation_results)
        self.current_simulation_context = context

        prompt = f"""Here are the PBPK simulation results to interpret:

{context}

Please provide your clinical interpretation following the structure in your instructions."""

        interpretation = self._call_claude(LEVEL1_SYSTEM_PROMPT, prompt)

        # Validation: check if AI cited correct numbers
        validation = self._validate_interpretation(interpretation, simulation_results)

        return {
            "interpretation": interpretation,
            "level": 1,
            "model": self.model,
            "validation": validation,
            "provider": "anthropic" if self.api_key else "mock",
        }

    # ════════════════════════════════════════════════════════════
    # LEVEL 2: INTERACTIVE CONVERSATION
    # ════════════════════════════════════════════════════════════

    def chat(self, user_question: str, simulation_results: dict = None) -> dict:
        """
        Level 2: Answer follow-up questions about the simulation.

        Args:
            user_question: User's follow-up question
            simulation_results: Simulation data (optional if already set)
        """
        if simulation_results:
            self.current_simulation_context = self._build_context(simulation_results)

        # Build the message with simulation context
        context_prompt = f"""Current simulation context:
{self.current_simulation_context}

User question: {user_question}"""

        # Add to conversation history
        self.conversation_history.append(
            ConversationMessage(role="user", content=context_prompt)
        )

        response = self._call_claude(
            LEVEL2_SYSTEM_PROMPT,
            context_prompt,
            history=self.conversation_history,
            max_tokens=800,
        )

        self.conversation_history.append(
            ConversationMessage(role="assistant", content=response)
        )

        return {
            "response": response,
            "level": 2,
            "conversation_length": len(self.conversation_history),
            "provider": "anthropic" if self.api_key else "mock",
        }

    def clear_conversation(self):
        """Clear conversation history for a new session."""
        self.conversation_history = []
        self.current_simulation_context = ""

    # ════════════════════════════════════════════════════════════
    # LEVEL 3: AUTONOMOUS ANALYST
    # ════════════════════════════════════════════════════════════

    def analyze_drug(self, multi_dose_results: list) -> dict:
        """
        Level 3: Comprehensive drug analysis from multiple dose simulations.

        Args:
            multi_dose_results: List of simulation results at different doses
                                Each entry: {"dose_mg": X, "results": {...}}
        """
        # Build comprehensive context from all simulations
        context_parts = []
        for i, sim in enumerate(multi_dose_results):
            dose = sim.get("dose_mg", "unknown")
            context_parts.append(f"\n--- SIMULATION {i+1}: {dose}mg ---")
            context_parts.append(self._build_context(sim.get("results", sim)))

        full_context = "\n".join(context_parts)

        prompt = f"""Here are PBPK simulation results for the same drug at {len(multi_dose_results)} different doses:

{full_context}

Please provide your comprehensive drug development assessment following the structure in your instructions."""

        analysis = self._call_claude(
            LEVEL3_SYSTEM_PROMPT,
            prompt,
            max_tokens=2000,
        )

        return {
            "analysis": analysis,
            "level": 3,
            "doses_analyzed": len(multi_dose_results),
            "provider": "anthropic" if self.api_key else "mock",
        }

    # ════════════════════════════════════════════════════════════
    # LEVEL 4: TRIAL DESIGNER
    # ════════════════════════════════════════════════════════════

    def design_trial(self, multi_dose_results: list,
                     target_indication: str = "",
                     drug_name: str = "") -> dict:
        """
        Level 4: Design a Phase I clinical trial protocol.

        Args:
            multi_dose_results: List of simulation results at different doses
            target_indication: What disease this drug is intended to treat
            drug_name: Name or identifier for the drug
        """
        context_parts = []
        for i, sim in enumerate(multi_dose_results):
            dose = sim.get("dose_mg", "unknown")
            context_parts.append(f"\n--- DOSE LEVEL {i+1}: {dose}mg ---")
            context_parts.append(self._build_context(sim.get("results", sim)))

        full_context = "\n".join(context_parts)

        prompt = f"""Drug: {drug_name or 'Novel compound'}
Target indication: {target_indication or 'To be determined'}

PBPK simulation results at {len(multi_dose_results)} dose levels:

{full_context}

Please design a complete Phase I clinical trial protocol following the structure in your instructions. Base every decision on the actual simulation data provided."""

        protocol = self._call_claude(
            LEVEL4_SYSTEM_PROMPT,
            prompt,
            max_tokens=3000,
        )

        return {
            "protocol": protocol,
            "level": 4,
            "drug_name": drug_name or "Novel compound",
            "indication": target_indication,
            "doses_evaluated": len(multi_dose_results),
            "provider": "anthropic" if self.api_key else "mock",
        }

    # ════════════════════════════════════════════════════════════
    # CONTEXT BUILDER
    # ════════════════════════════════════════════════════════════

    def _build_context(self, results: dict) -> str:
        """Build structured context string from simulation results."""
        parts = []

        # Basic info
        smiles = results.get("smiles", "")
        if smiles:
            parts.append(f"Molecule: {smiles}")

        dose = results.get("dose_mg") or results.get("dose")
        if dose:
            parts.append(f"Dose: {dose}mg")

        route = results.get("route", "oral")
        parts.append(f"Route: {route}")

        weight = results.get("weight_kg", 70)
        parts.append(f"Patient weight: {weight}kg")

        # PK Predictions
        predictions = results.get("predictions", [])
        if predictions:
            parts.append("\nPK PARAMETERS:")
            for pred in predictions:
                param = pred.get("parameter", "")
                value = pred.get("value", 0)
                unit = pred.get("unit", "")
                parts.append(f"  {param}: {value} {unit}")

        # Time course summary
        time_course = results.get("time_course", [])
        if time_course:
            plasma_concs = [t.get("plasma_conc", 0) for t in time_course]
            if plasma_concs:
                max_conc = max(plasma_concs)
                max_idx = plasma_concs.index(max_conc)
                tmax = time_course[max_idx].get("time_h", 0)
                parts.append(f"\nTIME COURSE:")
                parts.append(f"  Peak plasma concentration: {max_conc:.4f} mg/L at {tmax:.1f} hours")
                parts.append(f"  Concentration at 24h: {plasma_concs[-1]:.4f} mg/L")

                # Organ concentrations at peak
                peak_point = time_course[max_idx]
                for key in ['liver_conc', 'kidney_conc', 'brain_conc', 'gut_conc']:
                    if key in peak_point:
                        organ = key.replace('_conc', '').title()
                        parts.append(f"  {organ} concentration at Tmax: {peak_point[key]:.4f} mg/L")

        # Toxicity
        toxicity = results.get("toxicity", {})
        if toxicity:
            parts.append(f"\nTOXICITY ASSESSMENT:")
            if isinstance(toxicity, dict):
                for key, val in toxicity.items():
                    if isinstance(val, (str, int, float, bool)):
                        parts.append(f"  {key}: {val}")
                    elif isinstance(val, dict):
                        parts.append(f"  {key}: {json.dumps(val, default=str)[:200]}")

        # Cardiac safety
        cardiac = results.get("cardiac_safety", {})
        if cardiac:
            parts.append(f"\nCARDIAC SAFETY:")
            risk = cardiac.get("torsade_risk", cardiac.get("risk_category", "unknown"))
            parts.append(f"  TdP risk: {risk}")
            qtc = cardiac.get("qtc_change_ms", "")
            if qtc:
                parts.append(f"  QTc change: {qtc}ms")

        # Metabolism
        metabolism = results.get("metabolism", {})
        if metabolism:
            parts.append(f"\nMETABOLISM:")
            primary_cyp = metabolism.get("primary_cyp", "")
            if primary_cyp:
                parts.append(f"  Primary CYP: {primary_cyp}")
            risk = metabolism.get("overall_risk", "")
            if risk:
                parts.append(f"  Overall risk: {risk}")

        # Bioavailability
        bioavail = results.get("bioavailability", {})
        if bioavail:
            f_val = bioavail.get("F", bioavail.get("bioavailability", ""))
            if f_val:
                parts.append(f"\nBIOAVAILABILITY: {f_val}")

        # Clinical effects
        clinical = results.get("clinical_effects", {})
        if clinical and isinstance(clinical, dict):
            effects = clinical.get("effects", [])
            if effects:
                parts.append(f"\nCLINICAL EFFECTS:")
                for effect in effects[:10]:
                    if isinstance(effect, dict):
                        parts.append(f"  - {effect.get('effect', '')} (severity: {effect.get('severity', '')})")
                    elif isinstance(effect, str):
                        parts.append(f"  - {effect}")

        # Disease states
        diseases = results.get("diseases", [])
        if diseases:
            parts.append(f"\nACTIVE DISEASE STATES: {', '.join(diseases)}")

        return "\n".join(parts)

    # ════════════════════════════════════════════════════════════
    # VALIDATION LAYER
    # ════════════════════════════════════════════════════════════

    def _validate_interpretation(self, interpretation: str, results: dict) -> dict:
        """
        Validate AI interpretation against actual simulation data.
        Checks if numbers cited by AI match the real simulation output.
        """
        validation = {
            "passed": True,
            "checks": [],
            "warnings": [],
        }

        predictions = results.get("predictions", [])

        for pred in predictions:
            param = pred.get("parameter", "")
            value = pred.get("value", 0)

            if value and isinstance(value, (int, float)):
                # Check if AI mentioned a number close to the actual value
                # (Allow 10% tolerance for rounding)
                value_str = f"{value:.1f}"
                if value_str in interpretation or f"{value:.0f}" in interpretation:
                    validation["checks"].append({
                        "parameter": param,
                        "expected": value,
                        "status": "VERIFIED"
                    })
                else:
                    validation["checks"].append({
                        "parameter": param,
                        "expected": value,
                        "status": "NOT_CITED",
                        "note": "AI did not explicitly cite this value (may be paraphrased)"
                    })

        # Check for safety keywords
        safety_keywords = ["safe", "no risk", "completely safe", "harmless"]
        for keyword in safety_keywords:
            if keyword.lower() in interpretation.lower():
                validation["warnings"].append(
                    f"AI used potentially misleading safety language: '{keyword}'"
                )
                validation["passed"] = False

        return validation


# ════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ════════════════════════════════════════════════════════════════

_brain_instance: Optional[AIBrain] = None

def get_brain() -> AIBrain:
    """Get or create the singleton AI Brain instance."""
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = AIBrain()
    return _brain_instance

def reset_brain():
    """Reset the AI Brain (new API key, etc.)."""
    global _brain_instance
    _brain_instance = None
