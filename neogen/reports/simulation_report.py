"""
NeoGen Simulation Report Generator
=====================================
Generates professional PDF reports after each simulation.
Uses ReportLab for PDF generation.

Report structure:
1. Header (branding, timestamp, simulation ID)
2. Input summary (drug, dose, patient, disease)
3. PK parameters (table)
4. Organ concentration timeline (table)
5. AI clinical interpretation
6. Safety assessment
7. DDI findings (if applicable)
8. Population summary (if applicable)
9. Technical appendix (raw data)
10. Disclaimers
"""
from __future__ import annotations

import io
import uuid
import logging
from datetime import datetime
from typing import Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                 Table, TableStyle, Image)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT

logger = logging.getLogger(__name__)


def generate_simulation_report(
    simulation_data: dict,
    ai_interpretation: str = "",
    output_path: str = None,
) -> bytes:
    """
    Generate a professional PDF report from simulation results.

    Args:
        simulation_data: Full simulation response from /api/predict
        ai_interpretation: Claude Opus clinical interpretation text
        output_path: Optional file path to save PDF

    Returns:
        PDF bytes
    """
    buffer = io.BytesIO()
    sim_id = str(uuid.uuid4())[:8].upper()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            topMargin=0.5*inch, bottomMargin=0.5*inch,
                            leftMargin=0.7*inch, rightMargin=0.7*inch)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('RTitle', parent=styles['Title'], fontSize=22,
                              textColor=HexColor('#1a365d'), spaceAfter=4, alignment=TA_CENTER))
    styles.add(ParagraphStyle('RSub', parent=styles['Normal'], fontSize=11,
                              textColor=HexColor('#4a5568'), alignment=TA_CENTER, spaceAfter=4))
    styles.add(ParagraphStyle('RH1', parent=styles['Heading1'], fontSize=14,
                              textColor=HexColor('#1a365d'), spaceBefore=14, spaceAfter=6,
                              borderWidth=0.5, borderColor=HexColor('#2563EB'), borderPadding=3))
    styles.add(ParagraphStyle('RH2', parent=styles['Heading2'], fontSize=11,
                              textColor=HexColor('#2563EB'), spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle('RB', parent=styles['Normal'], fontSize=9.5,
                              leading=13, spaceAfter=3, alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle('RBL', parent=styles['Normal'], fontSize=9.5,
                              leading=13, leftIndent=14, spaceAfter=2))
    styles.add(ParagraphStyle('RAI', parent=styles['Normal'], fontSize=9.5,
                              leading=13, leftIndent=10, rightIndent=10, spaceAfter=3,
                              backColor=HexColor('#F0F5FA'), borderColor=HexColor('#D1E3F8'),
                              borderWidth=0.5, borderPadding=6))
    styles.add(ParagraphStyle('RWarn', parent=styles['Normal'], fontSize=9,
                              leading=12, leftIndent=10, textColor=HexColor('#CC3300'),
                              backColor=HexColor('#FFF5F5'), borderColor=HexColor('#FED7D7'),
                              borderWidth=0.5, borderPadding=6, spaceAfter=4))
    styles.add(ParagraphStyle('RFoot', parent=styles['Normal'], fontSize=7,
                              textColor=HexColor('#999999'), alignment=TA_CENTER))
    styles.add(ParagraphStyle('RMeta', parent=styles['Normal'], fontSize=8,
                              textColor=HexColor('#718096'), alignment=TA_RIGHT))

    story = []

    def make_table(data, col_widths=None):
        if not col_widths:
            col_widths = [7*inch/len(data[0])] * len(data[0])
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a365d')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
            ('FONTSIZE', (0, 0), (-1, -1), 8.5),
            ('GRID', (0, 0), (-1, -1), 0.4, HexColor('#E2E8F0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#FFFFFF'), HexColor('#F7FAFC')]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        return t

    # ═══ 1. HEADER ═══
    story.append(Spacer(1, 20))
    story.append(Paragraph("NEOGEN", styles['RTitle']))
    story.append(Paragraph("Computational Pharmacokinetic Simulation Report", styles['RSub']))
    story.append(Spacer(1, 8))
    story.append(make_table([
        ["Report ID", f"NEO-{sim_id}"],
        ["Generated", timestamp],
        ["Platform Version", "NeoGen v2.1"],
        ["Simulation Engine", "14-Compartment PBPK + Claude Opus PK Predictor"],
        ["Report Type", "Drug Simulation Summary"],
    ], [1.5*inch, 5.5*inch]))
    story.append(Spacer(1, 10))

    # ═══ 2. INPUT SUMMARY ═══
    story.append(Paragraph("1. SIMULATION INPUT PARAMETERS", styles['RH1']))

    predictions = simulation_data.get('predictions', [])
    smiles = simulation_data.get('smiles', 'N/A')
    dose = simulation_data.get('dose_mg', 'N/A')
    route = simulation_data.get('route', 'oral')
    weight = simulation_data.get('weight_kg', 70)
    diseases = simulation_data.get('diseases', [])

    input_data = [
        ["Parameter", "Value"],
        ["Drug / Compound", simulation_data.get('drug_name', smiles[:50] if smiles else 'Unknown')],
        ["SMILES", smiles[:60] + '...' if len(str(smiles)) > 60 else str(smiles)],
        ["Dose", f"{dose} mg"],
        ["Route", str(route).capitalize()],
        ["Patient Weight", f"{weight} kg"],
        ["Disease States", ', '.join(diseases) if diseases else 'None (healthy adult)'],
        ["Prediction Method", simulation_data.get('prediction_method', 'Opus + PBPK')],
    ]
    story.append(make_table(input_data, [2*inch, 5*inch]))

    # ═══ 3. PK PARAMETERS ═══
    story.append(Paragraph("2. PREDICTED PHARMACOKINETIC PARAMETERS", styles['RH1']))

    if predictions:
        pk_data = [["Parameter", "Value", "Unit", "Description"]]
        param_descriptions = {
            'cmax': ('C<sub>max</sub>', 'Peak plasma concentration'),
            'auc': ('AUC', 'Total drug exposure (area under curve)'),
            'half_life': ('t<sub>1/2</sub>', 'Elimination half-life'),
            'clearance': ('CL', 'Total body clearance'),
            'vd': ('V<sub>d</sub>', 'Apparent volume of distribution'),
        }
        for pred in predictions:
            param = pred.get('parameter', '')
            info = param_descriptions.get(param, (param, ''))
            pk_data.append([
                info[0] if isinstance(info, tuple) else param,
                f"{pred.get('value', 'N/A'):.4g}" if isinstance(pred.get('value'), (int, float)) else str(pred.get('value', 'N/A')),
                pred.get('unit', ''),
                info[1] if isinstance(info, tuple) else '',
            ])
        story.append(make_table(pk_data, [1.2*inch, 1.5*inch, 1.3*inch, 3*inch]))
    else:
        story.append(Paragraph("No PK parameters available for this simulation.", styles['RB']))

    # ═══ 4. ORGAN CONCENTRATIONS ═══
    story.append(Paragraph("3. ORGAN CONCENTRATION SUMMARY", styles['RH1']))

    time_course = simulation_data.get('time_course', [])
    if time_course and len(time_course) > 0:
        # Get peak concentrations per organ
        organs = ['plasma_conc', 'liver_conc', 'kidney_conc', 'brain_conc', 'muscle_conc', 'fat_conc', 'gut_conc']
        organ_names = ['Plasma', 'Liver', 'Kidney', 'Brain', 'Muscle', 'Fat', 'GI Tract']

        organ_data = [["Organ", "Peak Conc (mg/L)", "Time to Peak (h)", "24h Conc (mg/L)"]]
        for organ_key, organ_name in zip(organs, organ_names):
            values = [tp.get(organ_key, 0) for tp in time_course if organ_key in tp]
            if values:
                peak = max(values)
                peak_idx = values.index(peak)
                peak_time = time_course[peak_idx].get('time_h', 0) if peak_idx < len(time_course) else 0
                final = values[-1] if values else 0
                organ_data.append([
                    organ_name,
                    f"{peak:.4g}",
                    f"{peak_time:.1f}",
                    f"{final:.4g}",
                ])
        story.append(make_table(organ_data, [1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch]))
        story.append(Paragraph(f"<i>Based on {len(time_course)} time points over {time_course[-1].get('time_h', 24):.0f} hours.</i>", styles['RFoot']))
    else:
        story.append(Paragraph("No time-course data available.", styles['RB']))

    # ═══ 5. AI CLINICAL INTERPRETATION ═══
    story.append(Paragraph("4. AI CLINICAL INTERPRETATION", styles['RH1']))

    if ai_interpretation:
        # Clean up markdown
        clean_text = ai_interpretation.replace('**', '<b>').replace('##', '').replace('###', '')
        clean_text = clean_text.replace('\n\n', '<br/><br/>').replace('\n', '<br/>')
        # Truncate if too long
        if len(clean_text) > 3000:
            clean_text = clean_text[:3000] + '...<br/><i>[Report truncated for length]</i>'
        story.append(Paragraph(clean_text, styles['RAI']))
    else:
        story.append(Paragraph("AI interpretation was not generated for this simulation.", styles['RB']))

    story.append(Paragraph("<i>Generated by Claude Opus (Anthropic). AI-generated content requires human expert review.</i>", styles['RFoot']))

    # ═══ 6. SAFETY ASSESSMENT ═══
    story.append(Paragraph("5. SAFETY ASSESSMENT", styles['RH1']))

    toxicity = simulation_data.get('toxicity', {})
    clinical_effects = simulation_data.get('clinical_effects', [])
    herg = simulation_data.get('herg_assessment', {})

    if toxicity or clinical_effects:
        if isinstance(clinical_effects, list) and clinical_effects:
            safety_data = [["Effect", "Severity", "Organ", "Description"]]
            for eff in clinical_effects[:10]:
                if isinstance(eff, dict):
                    safety_data.append([
                        eff.get('effect', 'Unknown'),
                        eff.get('severity', 'Unknown'),
                        eff.get('organ', ''),
                        eff.get('description', '')[:80],
                    ])
            story.append(make_table(safety_data, [1.5*inch, 1*inch, 1*inch, 3.5*inch]))
        else:
            story.append(Paragraph("No clinical effects predicted at this dose.", styles['RB']))

        if herg and isinstance(herg, dict):
            margin = herg.get('safety_margin', 'N/A')
            risk = herg.get('risk_category', 'Unknown')
            story.append(Paragraph(f"<b>Cardiac Safety (hERG):</b> Safety margin = {margin}x. Risk category: {risk}.", styles['RB']))
    else:
        story.append(Paragraph("Safety assessment data not available for this simulation.", styles['RB']))

    # ═══ 7. PHARMACODYNAMIC EFFECTS ═══
    story.append(Paragraph("6. PHARMACODYNAMIC EFFECTS", styles['RH1']))

    pd_data = simulation_data.get('pharmacodynamics', {})
    if pd_data and isinstance(pd_data, dict):
        moa = pd_data.get('mechanism_of_action', '')
        drug_class = pd_data.get('drug_class', '')
        if moa:
            story.append(Paragraph(f"<b>Mechanism of Action:</b> {moa}", styles['RB']))
        if drug_class:
            story.append(Paragraph(f"<b>Drug Class:</b> {drug_class}", styles['RB']))

        organ_effects = pd_data.get('organ_effects', [])
        if organ_effects and isinstance(organ_effects, list):
            for eff in organ_effects[:8]:
                if isinstance(eff, dict):
                    story.append(Paragraph(
                        f"<bullet>&bull;</bullet> <b>{eff.get('organ', 'Unknown')}:</b> {eff.get('effect', '')} "
                        f"(intensity: {eff.get('intensity', 'N/A')})", styles['RBL']))
    else:
        story.append(Paragraph("Pharmacodynamic data not available.", styles['RB']))

    # ═══ 8. TECHNICAL APPENDIX ═══
    story.append(PageBreak())
    story.append(Paragraph("TECHNICAL APPENDIX", styles['RH1']))

    story.append(Paragraph("A. Simulation Methodology", styles['RH2']))
    story.append(Paragraph(
        "This simulation used a 14-compartment physiologically based pharmacokinetic (PBPK) model "
        "solved using the Runge-Kutta 4th/5th order (RK45) method implemented in SciPy. "
        "Tissue partition coefficients (Kp) were predicted using the Rodgers and Rowland method "
        "with tissue composition data from ICRP Publication 89. "
        "Hepatic clearance was estimated using the well-stirred liver model with IVIVE empirical "
        "scaling factors (Hallifax and Houston, 2009). "
        "PK parameters were predicted using Claude Opus (Anthropic) with cached clinical "
        "pharmacokinetic values for 200 drugs, falling back to XGBoost/Chemprop ML ensemble "
        "trained on 132,552 ChEMBL records.",
        styles['RB']
    ))

    story.append(Paragraph("B. Model Compartments", styles['RH2']))
    compartments = [
        "Gut lumen", "Gut wall", "Liver", "Kidney", "Brain",
        "Muscle", "Fat", "Plasma/Blood", "Lung", "Spleen",
        "Skin", "Bone marrow", "RBC", "Pancreas"
    ]
    story.append(Paragraph(", ".join(compartments), styles['RB']))

    story.append(Paragraph("C. Data Sources", styles['RH2']))
    sources = [
        "ChEMBL (European Bioinformatics Institute) — 132,552 PK training records",
        "FDA Drug Labels (DailyMed) — published clinical PK values",
        "Goodman and Gilman's The Pharmacological Basis of Therapeutics, 13th ed.",
        "ICRP Publication 89 — organ volumes and blood flow rates",
        "Rodgers and Rowland (2005/2006) — tissue partition coefficient method",
        "Hallifax and Houston (2009) — IVIVE empirical scaling factors",
    ]
    for src in sources:
        story.append(Paragraph(f"<bullet>&bull;</bullet> {src}", styles['RBL']))

    # ═══ 9. DISCLAIMERS ═══
    story.append(Spacer(1, 15))
    story.append(Paragraph("IMPORTANT DISCLAIMERS", styles['RH1']))
    story.append(Paragraph(
        "<b>FOR RESEARCH AND EDUCATIONAL USE ONLY.</b> This report is generated by NeoGen, "
        "a computational simulation platform. The predictions contained herein are based on "
        "mathematical models and machine learning algorithms, NOT clinical data for this specific "
        "compound/scenario. These results should NOT be used for clinical decision-making, "
        "patient care, regulatory submissions, or any purpose requiring validated pharmacokinetic data. "
        "All outputs are exploratory estimates and may differ significantly from actual clinical outcomes. "
        "Always consult with a qualified pharmacologist or clinical pharmacokineticist before "
        "making any decisions based on computational predictions.",
        styles['RWarn']
    ))

    story.append(Paragraph(
        "<b>AI-GENERATED CONTENT:</b> The clinical interpretation section was generated by "
        "Claude Opus (Anthropic) and has not been reviewed by a human expert. AI-generated "
        "medical content may contain errors, omissions, or hallucinated information. "
        "Independent verification is required.",
        styles['RWarn']
    ))

    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Report NEO-{sim_id} | Generated {timestamp} | NeoGen v2.1 | University of Washington", styles['RFoot']))
    story.append(Paragraph("CONFIDENTIAL — For Research Use Only", styles['RFoot']))

    # Build
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    # Save to file if path provided
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)
        logger.info(f"Report saved to {output_path}")

    return pdf_bytes
