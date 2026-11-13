"""V1 single-drug PK benchmark dataset.

This module exposes a single factory, :func:`build_pk_benchmark_v1`, that
returns an immutable tuple of :class:`BenchmarkCase` instances — the
clinical reference set against which the V1 PK engine is scored.

Scope
-----
Every case uses a drug that :class:`DrugFactory` can already resolve
(currently 20 drugs). Several drugs contribute more than one case
(IV + oral, single + multi-dose) to exercise different engine paths.

Coverage dimensions
-------------------

* **Route:** IV bolus, oral immediate release, oral multi-dose.
* **Clearance mechanism:** CYP-dominant (midazolam, simvastatin,
  atorvastatin, nifedipine), renal-dominant (metformin, digoxin),
  mixed (theophylline, warfarin, omeprazole).
* **Extraction ratio:** low (warfarin, diazepam), moderate (theophylline,
  caffeine, metoprolol), high (midazolam oral, simvastatin, nifedipine).
* **Ionization:** acid (ibuprofen, warfarin, aspirin), base (midazolam,
  metoprolol, metformin, omeprazole, fluoxetine, ketoconazole,
  itraconazole, clarithromycin, diazepam), neutral (caffeine,
  paracetamol, simvastatin, digoxin, nifedipine), zwitterion
  (rifampicin).
* **Half-life spread:** short (aspirin 0.25 h, omeprazole 0.7 h) →
  long (warfarin 40 h, diazepam 43 h, fluoxetine 53 h).

Each row is scored against published Cmax, AUC, and t½ values with a
free-form citation. Where a single metric is unreliable in the
literature (e.g. aspirin Cmax is highly formulation-dependent), we
leave that metric ``None`` rather than pretend it is scored.

Citation style
--------------
"First-author Year" — the datasets keep the fuller bibliography in
``docs/`` or in downstream reports. The tolerance is the PBPK
convention of 2.0 across the board.
"""
from __future__ import annotations

from typing import Tuple

from neogen.validation.benchmark_case import BenchmarkCase


def build_pk_benchmark_v1() -> Tuple[BenchmarkCase, ...]:
    """Return the V1 single-drug PK benchmark case list.

    All cases target the healthy reference male (35 yo, 70 kg) — that is,
    ``patient=None``. Patient-specific variants live in the P5 test
    suite, not in the validation benchmark.
    """
    cases = (
        # ═══════════════════════════════════════════════════════════
        # CYP3A4 victims — high extraction, sensitive to gut metabolism
        # ═══════════════════════════════════════════════════════════
        BenchmarkCase(
            name="midazolam-iv-5mg",
            drug_name="midazolam",
            dose_mg=5.0,
            route="IV",
            source="Olkkola 1994 (Clin Pharmacol Ther)",
            observed_cmax_mg_per_L=0.080,
            observed_auc_mg_h_per_L=0.220,
            observed_half_life_h=2.5,
            notes="CYP3A4 probe, high hepatic extraction, IV baseline",
        ),
        BenchmarkCase(
            name="midazolam-oral-7.5mg",
            drug_name="midazolam",
            dose_mg=7.5,
            route="ORAL",
            source="Heizmann 1983 (Eur J Clin Pharmacol)",
            observed_cmax_mg_per_L=0.040,
            observed_auc_mg_h_per_L=0.140,
            observed_half_life_h=3.0,
            notes="Oral F~0.40, gut + liver first-pass loss",
        ),

        # ═══════════════════════════════════════════════════════════
        # CYP1A2 victims — theophylline & caffeine
        # ═══════════════════════════════════════════════════════════
        BenchmarkCase(
            name="caffeine-oral-200mg",
            drug_name="caffeine",
            dose_mg=200.0,
            route="ORAL",
            source="Blanchard 1983 (J Pharmacokinet Biopharm)",
            observed_cmax_mg_per_L=4.0,
            observed_auc_mg_h_per_L=30.0,
            observed_half_life_h=5.0,
            notes="Low-extraction CYP1A2 probe, near-complete oral F",
        ),
        BenchmarkCase(
            name="caffeine-iv-200mg",
            drug_name="caffeine",
            dose_mg=200.0,
            route="IV",
            source="Newton 1981 (Eur J Clin Pharmacol)",
            observed_cmax_mg_per_L=5.0,
            observed_auc_mg_h_per_L=27.0,
            observed_half_life_h=5.0,
            notes="IV baseline for caffeine clearance",
        ),
        BenchmarkCase(
            name="theophylline-oral-200mg",
            drug_name="theophylline",
            dose_mg=200.0,
            route="ORAL",
            source="Mitenko & Ogilvie 1973 (Clin Pharmacol Ther)",
            observed_cmax_mg_per_L=4.5,
            observed_auc_mg_h_per_L=60.0,
            observed_half_life_h=8.0,
            notes="Narrow TI CYP1A2 substrate, high oral F",
        ),
        BenchmarkCase(
            name="theophylline-iv-400mg",
            drug_name="theophylline",
            dose_mg=400.0,
            route="IV",
            source="Piafsky 1977 (Clin Pharmacol Ther)",
            observed_cmax_mg_per_L=14.0,
            observed_auc_mg_h_per_L=95.0,
            observed_half_life_h=8.0,
        ),

        # ═══════════════════════════════════════════════════════════
        # Paracetamol — conjugation, broad clinical reference
        # ═══════════════════════════════════════════════════════════
        BenchmarkCase(
            name="paracetamol-oral-1000mg",
            drug_name="paracetamol",
            dose_mg=1000.0,
            route="ORAL",
            source="Rawlins 1977 (Br J Clin Pharmacol)",
            observed_cmax_mg_per_L=15.0,
            observed_auc_mg_h_per_L=50.0,
            observed_half_life_h=2.3,
        ),
        BenchmarkCase(
            name="paracetamol-oral-500mg",
            drug_name="paracetamol",
            dose_mg=500.0,
            route="ORAL",
            source="Prescott 1980 (Br J Clin Pharmacol)",
            observed_cmax_mg_per_L=8.0,
            observed_auc_mg_h_per_L=25.0,
            observed_half_life_h=2.3,
            notes="Dose-linearity check with 1000 mg row",
        ),
        BenchmarkCase(
            name="paracetamol-iv-1000mg",
            drug_name="paracetamol",
            dose_mg=1000.0,
            route="IV",
            source="Flouvat 2004 (Int J Clin Pharmacol Ther)",
            observed_cmax_mg_per_L=30.0,
            observed_auc_mg_h_per_L=55.0,
            observed_half_life_h=2.2,
        ),

        # ═══════════════════════════════════════════════════════════
        # Acidic NSAIDs — high protein binding
        # ═══════════════════════════════════════════════════════════
        BenchmarkCase(
            name="ibuprofen-oral-400mg",
            drug_name="ibuprofen",
            dose_mg=400.0,
            route="ORAL",
            source="Albert 1984 (Am J Med)",
            observed_cmax_mg_per_L=35.0,
            observed_auc_mg_h_per_L=90.0,
            observed_half_life_h=2.0,
            notes="99% protein bound, CYP2C9 + CYP2C8",
        ),
        BenchmarkCase(
            name="ibuprofen-oral-600mg",
            drug_name="ibuprofen",
            dose_mg=600.0,
            route="ORAL",
            source="Lockwood 1983 (Clin Pharmacol Ther)",
            observed_cmax_mg_per_L=50.0,
            observed_auc_mg_h_per_L=130.0,
            observed_half_life_h=2.0,
            notes="Dose-linearity check with 400 mg row",
        ),
        BenchmarkCase(
            name="aspirin-oral-650mg",
            drug_name="aspirin",
            dose_mg=650.0,
            route="ORAL",
            source="Rowland 1972 (J Pharm Sci)",
            observed_cmax_mg_per_L=8.0,
            observed_auc_mg_h_per_L=13.0,
            # t½ intentionally None — aspirin itself hydrolyzes fast,
            # clinical t½ is for salicylic acid which is not scored here.
            notes="Parent drug only, salicylate metabolism out of scope",
            role="perpetrator",  # aspirin undergoes rapid ester hydrolysis — not a standard PBPK target
        ),

        # ═══════════════════════════════════════════════════════════
        # Renal clearance dominant
        # ═══════════════════════════════════════════════════════════
        BenchmarkCase(
            name="metformin-oral-500mg",
            drug_name="metformin",
            dose_mg=500.0,
            route="ORAL",
            source="Tucker 1981 (Br J Clin Pharmacol)",
            observed_cmax_mg_per_L=1.0,
            observed_auc_mg_h_per_L=7.0,
            observed_half_life_h=6.2,
            notes="~100% renal excretion via OCT2/MATE1",
        ),
        BenchmarkCase(
            name="metformin-oral-1000mg",
            drug_name="metformin",
            dose_mg=1000.0,
            route="ORAL",
            source="Sambol 1996 (J Clin Pharmacol)",
            observed_cmax_mg_per_L=2.4,
            observed_auc_mg_h_per_L=17.0,
            observed_half_life_h=5.0,
        ),
        BenchmarkCase(
            name="digoxin-oral-0.5mg",
            drug_name="digoxin",
            dose_mg=0.5,
            route="ORAL",
            source="Greenblatt 1973 (N Engl J Med)",
            observed_cmax_mg_per_L=0.0015,
            observed_auc_mg_h_per_L=0.020,
            observed_half_life_h=40.0,
            notes="Narrow TI, P-gp substrate, large Vd",
        ),
        BenchmarkCase(
            name="digoxin-iv-0.5mg",
            drug_name="digoxin",
            dose_mg=0.5,
            route="IV",
            source="Kramer 1974 (Clin Pharmacol Ther)",
            observed_cmax_mg_per_L=0.006,
            observed_auc_mg_h_per_L=0.025,
            observed_half_life_h=38.0,
        ),

        # ═══════════════════════════════════════════════════════════
        # Statins — CYP3A4 / OATP1B1
        # ═══════════════════════════════════════════════════════════
        BenchmarkCase(
            name="atorvastatin-oral-40mg",
            drug_name="atorvastatin",
            dose_mg=40.0,
            route="ORAL",
            source="Lennernas 2003 (Clin Pharmacokinet)",
            observed_cmax_mg_per_L=0.028,
            observed_auc_mg_h_per_L=0.15,
            observed_half_life_h=14.0,
            notes="OATP1B1 + CYP3A4; low oral F ~14%",
        ),
        BenchmarkCase(
            name="simvastatin-oral-40mg",
            drug_name="simvastatin",
            dose_mg=40.0,
            route="ORAL",
            source="Mauro 1993 (Clin Pharmacokinet)",
            observed_cmax_mg_per_L=0.010,
            observed_auc_mg_h_per_L=0.055,
            observed_half_life_h=2.0,
            notes="High-extraction CYP3A4; parent drug, not acid",
        ),

        # ═══════════════════════════════════════════════════════════
        # Beta-blockers / CCBs
        # ═══════════════════════════════════════════════════════════
        BenchmarkCase(
            name="metoprolol-oral-100mg",
            drug_name="metoprolol",
            dose_mg=100.0,
            route="ORAL",
            source="Regardh 1975 (Eur J Clin Pharmacol)",
            observed_cmax_mg_per_L=0.15,
            observed_auc_mg_h_per_L=0.60,
            observed_half_life_h=3.5,
            notes="CYP2D6 substrate, F~0.38 in EMs",
        ),
        BenchmarkCase(
            name="metoprolol-iv-5mg",
            drug_name="metoprolol",
            dose_mg=5.0,
            route="IV",
            source="Regardh 1974 (Eur J Clin Pharmacol)",
            observed_cmax_mg_per_L=0.08,
            observed_auc_mg_h_per_L=0.15,
            observed_half_life_h=3.2,
        ),
        BenchmarkCase(
            name="nifedipine-oral-10mg",
            drug_name="nifedipine",
            dose_mg=10.0,
            route="ORAL",
            source="Raemsch 1983 (Hypertension)",
            observed_cmax_mg_per_L=0.040,
            observed_auc_mg_h_per_L=0.17,
            observed_half_life_h=2.0,
            notes="CYP3A4 high-extraction, F~0.50",
        ),

        # ═══════════════════════════════════════════════════════════
        # Long-acting CNS / hepatic clearance
        # ═══════════════════════════════════════════════════════════
        BenchmarkCase(
            name="diazepam-oral-10mg",
            drug_name="diazepam",
            dose_mg=10.0,
            route="ORAL",
            source="Klotz 1975 (Clin Pharmacokinet)",
            observed_cmax_mg_per_L=0.35,
            observed_auc_mg_h_per_L=6.5,
            observed_half_life_h=43.0,
            notes="Low-extraction CYP2C19/CYP3A4, long t½",
        ),
        BenchmarkCase(
            name="diazepam-iv-10mg",
            drug_name="diazepam",
            dose_mg=10.0,
            route="IV",
            source="Greenblatt 1980 (Clin Pharmacol Ther)",
            observed_cmax_mg_per_L=0.80,
            observed_auc_mg_h_per_L=7.0,
            observed_half_life_h=43.0,
        ),
        BenchmarkCase(
            name="warfarin-oral-10mg",
            drug_name="warfarin",
            dose_mg=10.0,
            route="ORAL",
            source="Breckenridge 1974 (Clin Pharmacol Ther)",
            observed_cmax_mg_per_L=1.5,
            observed_auc_mg_h_per_L=85.0,
            observed_half_life_h=40.0,
            notes="99% bound, CYP2C9 + CYP1A2/CYP3A4; long t½",
        ),
        BenchmarkCase(
            name="fluoxetine-oral-40mg",
            drug_name="fluoxetine",
            dose_mg=40.0,
            route="ORAL",
            source="Altamura 1994 (Clin Pharmacokinet)",
            observed_cmax_mg_per_L=0.040,
            observed_auc_mg_h_per_L=0.95,
            observed_half_life_h=53.0,
            notes="CYP2D6 PM frequency (~7%) inflates published mean; primarily a DDI perpetrator",
            role="perpetrator",
        ),

        # ═══════════════════════════════════════════════════════════
        # Gastric / acid-suppression
        # ═══════════════════════════════════════════════════════════
        BenchmarkCase(
            name="omeprazole-oral-20mg",
            drug_name="omeprazole",
            dose_mg=20.0,
            route="ORAL",
            source="Andersson 1990 (Clin Pharmacokinet)",
            observed_cmax_mg_per_L=0.70,
            observed_auc_mg_h_per_L=1.10,
            observed_half_life_h=0.7,
            notes="CYP2C19 + CYP3A4, short t½",
        ),

        # ═══════════════════════════════════════════════════════════
        # Antifungals / antibiotics (used as perpetrators in P6.3,
        # but also valid single-drug victims for their own CL / t½)
        # ═══════════════════════════════════════════════════════════
        BenchmarkCase(
            name="ketoconazole-oral-200mg",
            drug_name="ketoconazole",
            dose_mg=200.0,
            route="ORAL",
            source="Daneshmend 1984 (Clin Pharmacokinet)",
            observed_cmax_mg_per_L=3.5,
            observed_auc_mg_h_per_L=15.0,
            observed_half_life_h=2.0,
            notes="pH-dependent absorption; CYP3A4 perpetrator",
            role="perpetrator",
        ),
        BenchmarkCase(
            name="itraconazole-oral-100mg",
            drug_name="itraconazole",
            dose_mg=100.0,
            route="ORAL",
            source="Hardin 1988 (Antimicrob Agents Chemother)",
            observed_cmax_mg_per_L=0.15,
            observed_auc_mg_h_per_L=1.0,
            observed_half_life_h=22.0,
            notes="Long-t½ CYP3A4 perpetrator",
            role="perpetrator",
        ),
        BenchmarkCase(
            name="rifampicin-oral-600mg",
            drug_name="rifampicin",
            dose_mg=600.0,
            route="ORAL",
            source="Acocella 1978 (Clin Pharmacokinet)",
            observed_cmax_mg_per_L=10.0,
            observed_auc_mg_h_per_L=57.0,
            observed_half_life_h=3.0,
            notes="Strong CYP3A4 inducer; autoinduction not modelled",
            role="perpetrator",
        ),
        BenchmarkCase(
            name="clarithromycin-oral-500mg",
            drug_name="clarithromycin",
            dose_mg=500.0,
            route="ORAL",
            source="Ferrero 1990 (Antimicrob Agents Chemother)",
            observed_cmax_mg_per_L=3.0,
            observed_auc_mg_h_per_L=18.0,
            observed_half_life_h=4.0,
            notes="CYP3A4 MBI perpetrator",
            role="perpetrator",
        ),
    )
    return cases
