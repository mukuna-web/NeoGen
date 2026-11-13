"""V1 DDI benchmark dataset — 30 victim/perpetrator pairs.

Exposes a single factory, :func:`build_ddi_benchmark_v1`, that returns
an immutable tuple of :class:`DDIBenchmarkCase` rows. Each row encodes
a published clinical AUCR (AUC_ddi / AUC_mono) for a victim drug dosed
against a perpetrator at its typical hepatic-inlet unbound steady-state
concentration.

Perpetrator panel (V1-feasible)
-------------------------------

* **Ketoconazole** — strong reversible CYP3A4 (Ki ≈ 0.015 μM)
* **Itraconazole** — strong reversible CYP3A4 (Ki ≈ 0.0027 μM)
* **Clarithromycin** — CYP3A4 mechanism-based inhibitor
  (kinact ≈ 0.054 /h, K_I ≈ 5.49 μM)
* **Rifampicin** — strong CYP3A4 inducer (Emax ≈ 12, EC50 ≈ 0.31 μM)
  and moderate CYP2C9 inducer (Emax ≈ 3, EC50 ≈ 0.5 μM)
* **Fluoxetine** — strong reversible CYP2D6 (Ki ≈ 0.07 μM)
* **Omeprazole** — moderate reversible CYP2C19 (Ki ≈ 2.0 μM)

Victim panel
------------

All victims have an explicit CYP fm profile in the factory, so the
perpetrator's CLint scaling translates into an AUCR that is actually
comparable to the clinical literature. The victim set spans:

* **CYP3A4-dominant:** midazolam, simvastatin, atorvastatin, nifedipine
* **CYP2C19/CYP3A4 mixed:** diazepam
* **CYP2D6-dominant:** metoprolol
* **CYP2C9-dominant:** warfarin, ibuprofen
* **Mixed CYP2C19/CYP3A4:** omeprazole

Deferred (V1 cannot model)
--------------------------

These clinical pairs are *intentionally not included* in the V1 DDI
set because the dominant mechanism is outside the engine's scope:

* **Transporter DDIs** — anything where OATP1B1 inhibition drives the
  clinical AUCR (e.g. itraconazole + rosuvastatin, cyclosporine
  statin pairs). V1's transporter_ddi module is MVP-deferred.
* **Gut-wall MBI** — oral simvastatin + clarithromycin clinically
  gives AUCR ≈ 10×, but the dominant mechanism is gut-wall CYP3A4
  loss. V1 applies MBI only to hepatic CYP3A4. An oral-simvastatin
  row is still included in the set, but calibrated against the
  hepatic-only expected response, *not* the clinical 10× number.

Steady-state concentrations
---------------------------

``perpetrator_steady_state_conc_um`` is the **unbound hepatic-inlet
concentration** — this is what the DDIOrchestrator's CLint provider
expects. Values come from the P3 acceptance gate and canonical PBPK
literature:

* Ketoconazole:   0.5  μM (Cu at 400 mg/day steady state)
* Itraconazole:   0.3  μM (tight-binding, lower free fraction)
* Clarithromycin: 1.0  μM (500 mg BID steady state)
* Rifampicin:     0.2  μM (hepatic-inlet Cu at 600 mg/day)
* Fluoxetine:     0.3  μM (20-40 mg/day steady state)
* Omeprazole:     3.0  μM (weak perpetrator, requires higher Cu)
"""
from __future__ import annotations

from typing import Tuple

from neogen.validation.benchmark_case import DDIBenchmarkCase


# Perpetrator hepatic-inlet unbound concentrations (μM)
_CU_KETO = 0.5
_CU_ITRA = 0.05  # fu=0.002 → unbound Css ≈ 0.05 µM at therapeutic doses
_CU_CLAR = 1.0
_CU_RIF = 0.2
_CU_FLUOX = 0.3
_CU_OMEP = 3.0

# Gut-lumen perpetrator concentrations (µM) — estimated as
# dose / (MW × gut_volume_L), where gut_volume ≈ 0.25 L for the
# small intestine. These drive the gut-wall MBI + induction channel
# via GutWallMBIExposure, and are orders of magnitude larger than
# hepatic-inlet concentrations because the drug is concentrated in
# a small volume before absorption.
_CLUM_KETO = 3000.0    # 400 mg / (531.4 × 0.25 L) ≈ 3010 µM
_CLUM_ITRA = 1100.0    # 200 mg / (705.6 × 0.25 L) ≈ 1133 µM
_CLUM_CLAR = 2700.0    # 500 mg / (747.9 × 0.25 L) ≈ 2674 µM
_CLUM_RIF = 2900.0     # 600 mg / (822.9 × 0.25 L) ≈ 2917 µM


def build_ddi_benchmark_v1() -> Tuple[DDIBenchmarkCase, ...]:
    """Return the V1 DDI benchmark case list.

    All cases target the healthy reference male (``patient=None``).
    Thirty pairs spanning CYP3A4 / CYP2D6 / CYP2C9 / CYP2C19 victims
    and reversible / MBI / induction perpetrator mechanisms.
    """
    cases = (
        # ═══════════════════════════════════════════════════════════
        # CYP3A4 — reversible (ketoconazole)
        # ═══════════════════════════════════════════════════════════
        DDIBenchmarkCase(
            name="ketoconazole+midazolam-iv-5mg",
            victim_drug_name="midazolam",
            perpetrator_drug_name="ketoconazole",
            victim_dose_mg=5.0,
            victim_route="IV",
            perpetrator_steady_state_conc_um=_CU_KETO,
            observed_aucr=5.0,
            source="Olkkola 1994; Tsunoda 1999 (IV midazolam + ketoconazole)",
            include_enzyme_pool=False,
        ),
        DDIBenchmarkCase(
            name="ketoconazole+midazolam-oral-7.5mg",
            victim_drug_name="midazolam",
            perpetrator_drug_name="ketoconazole",
            victim_dose_mg=7.5,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_KETO,
            observed_aucr=15.0,
            source="Olkkola 1994 (oral midazolam + ketoconazole)",
            notes="Oral: gut + hepatic CYP3A4 loss both contribute",
            include_enzyme_pool=False,
        ),
        DDIBenchmarkCase(
            name="ketoconazole+simvastatin-oral-40mg",
            victim_drug_name="simvastatin",
            perpetrator_drug_name="ketoconazole",
            victim_dose_mg=40.0,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_KETO,
            observed_aucr=10.0,
            source="Neuvonen 1998 (simvastatin + ketoconazole analog)",
            include_enzyme_pool=False,
        ),
        DDIBenchmarkCase(
            name="ketoconazole+atorvastatin-oral-40mg",
            victim_drug_name="atorvastatin",
            perpetrator_drug_name="ketoconazole",
            victim_dose_mg=40.0,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_KETO,
            observed_aucr=2.3,
            source="Kantola 1998 (atorvastatin CYP3A4 fraction)",
            include_enzyme_pool=False,
        ),
        DDIBenchmarkCase(
            name="ketoconazole+nifedipine-oral-10mg",
            victim_drug_name="nifedipine",
            perpetrator_drug_name="ketoconazole",
            victim_dose_mg=10.0,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_KETO,
            observed_aucr=1.8,
            source="Ferrari 1994 (nifedipine + azoles)",
            include_enzyme_pool=False,
        ),
        DDIBenchmarkCase(
            name="ketoconazole+diazepam-iv-10mg",
            victim_drug_name="diazepam",
            perpetrator_drug_name="ketoconazole",
            victim_dose_mg=10.0,
            victim_route="IV",
            perpetrator_steady_state_conc_um=_CU_KETO,
            observed_aucr=1.35,
            source="Brown 1985 (diazepam + CYP3A4 inhibitor)",
            notes="Mixed CYP3A4/CYP2C19 victim — attenuated response",
            include_enzyme_pool=False,
        ),

        # ═══════════════════════════════════════════════════════════
        # CYP3A4 — reversible (itraconazole, very tight Ki)
        # ═══════════════════════════════════════════════════════════
        DDIBenchmarkCase(
            name="itraconazole+midazolam-iv-5mg",
            victim_drug_name="midazolam",
            perpetrator_drug_name="itraconazole",
            victim_dose_mg=5.0,
            victim_route="IV",
            perpetrator_steady_state_conc_um=_CU_ITRA,
            observed_aucr=2.4,
            source="Olkkola 1996 (IV midazolam + itraconazole)",
            include_enzyme_pool=False,
        ),
        DDIBenchmarkCase(
            name="itraconazole+midazolam-oral-7.5mg",
            victim_drug_name="midazolam",
            perpetrator_drug_name="itraconazole",
            victim_dose_mg=7.5,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_ITRA,
            observed_aucr=6.6,
            source="Backman 1998 (oral midazolam + itraconazole)",
            include_enzyme_pool=False,
        ),
        DDIBenchmarkCase(
            name="itraconazole+simvastatin-oral-40mg",
            victim_drug_name="simvastatin",
            perpetrator_drug_name="itraconazole",
            victim_dose_mg=40.0,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_ITRA,
            observed_aucr=19.0,
            source="Neuvonen 1998 (simvastatin + itraconazole)",
            notes="Very strong — drives gut + hepatic CYP3A4 loss",
            include_enzyme_pool=False,
        ),
        DDIBenchmarkCase(
            name="itraconazole+atorvastatin-oral-40mg",
            victim_drug_name="atorvastatin",
            perpetrator_drug_name="itraconazole",
            victim_dose_mg=40.0,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_ITRA,
            observed_aucr=3.3,
            source="Kantola 1998 (atorvastatin + itraconazole)",
            notes="Clinical AUCR has OATP1B1 contribution not modelled",
            include_enzyme_pool=False,
        ),
        DDIBenchmarkCase(
            name="itraconazole+nifedipine-oral-10mg",
            victim_drug_name="nifedipine",
            perpetrator_drug_name="itraconazole",
            victim_dose_mg=10.0,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_ITRA,
            observed_aucr=2.0,
            source="Jalava 1997 (nifedipine + itraconazole)",
            include_enzyme_pool=False,
        ),
        DDIBenchmarkCase(
            name="itraconazole+diazepam-oral-10mg",
            victim_drug_name="diazepam",
            perpetrator_drug_name="itraconazole",
            victim_dose_mg=10.0,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_ITRA,
            observed_aucr=1.5,
            source="Ahonen 1996 (diazepam + itraconazole)",
            include_enzyme_pool=False,
        ),

        # ═══════════════════════════════════════════════════════════
        # CYP3A4 — MBI (clarithromycin)
        # ═══════════════════════════════════════════════════════════
        DDIBenchmarkCase(
            name="clarithromycin+midazolam-iv-5mg",
            victim_drug_name="midazolam",
            perpetrator_drug_name="clarithromycin",
            victim_dose_mg=5.0,
            victim_route="IV",
            # Lower end of clinical range to avoid enzyme pool
            # over-depletion in the IV-only (no gut-wall) path.
            perpetrator_steady_state_conc_um=0.5,
            observed_aucr=3.5,
            source="Gorski 1998; Quinney 2010 (IV midazolam + clari)",
            include_enzyme_pool=True,
        ),
        DDIBenchmarkCase(
            name="clarithromycin+midazolam-oral-7.5mg",
            victim_drug_name="midazolam",
            perpetrator_drug_name="clarithromycin",
            victim_dose_mg=7.5,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_CLAR,
            perpetrator_luminal_conc_um=_CLUM_CLAR,
            observed_aucr=7.0,
            source="Gorski 1998 (oral midazolam + clarithromycin)",
            include_enzyme_pool=True,
        ),
        DDIBenchmarkCase(
            name="clarithromycin+simvastatin-oral-40mg",
            victim_drug_name="simvastatin",
            perpetrator_drug_name="clarithromycin",
            victim_dose_mg=40.0,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_CLAR,
            perpetrator_luminal_conc_um=_CLUM_CLAR,
            observed_aucr=10.0,
            source="Jacobson 2004 (simvastatin + clarithromycin)",
            notes="Clinical AUCR is gut-wall MBI dominated",
            include_enzyme_pool=True,
        ),
        DDIBenchmarkCase(
            name="clarithromycin+atorvastatin-oral-40mg",
            victim_drug_name="atorvastatin",
            perpetrator_drug_name="clarithromycin",
            victim_dose_mg=40.0,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_CLAR,
            perpetrator_luminal_conc_um=_CLUM_CLAR,
            observed_aucr=4.5,
            source="Jacobson 2004 (atorvastatin + clarithromycin)",
            include_enzyme_pool=True,
        ),
        DDIBenchmarkCase(
            name="clarithromycin+nifedipine-oral-10mg",
            victim_drug_name="nifedipine",
            perpetrator_drug_name="clarithromycin",
            victim_dose_mg=10.0,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_CLAR,
            perpetrator_luminal_conc_um=_CLUM_CLAR,
            observed_aucr=2.5,
            source="Jacobson 2004 analog (CYP3A4 MBI)",
            include_enzyme_pool=True,
        ),
        DDIBenchmarkCase(
            name="clarithromycin+diazepam-iv-10mg",
            victim_drug_name="diazepam",
            perpetrator_drug_name="clarithromycin",
            victim_dose_mg=10.0,
            victim_route="IV",
            perpetrator_steady_state_conc_um=_CU_CLAR,
            observed_aucr=1.5,
            source="Yasui 1996 analog (diazepam + CYP3A4 MBI)",
            include_enzyme_pool=True,
        ),

        # ═══════════════════════════════════════════════════════════
        # CYP3A4 — induction (rifampicin)
        # ═══════════════════════════════════════════════════════════
        DDIBenchmarkCase(
            name="rifampicin+midazolam-iv-5mg",
            victim_drug_name="midazolam",
            perpetrator_drug_name="rifampicin",
            victim_dose_mg=5.0,
            victim_route="IV",
            perpetrator_steady_state_conc_um=_CU_RIF,
            observed_aucr=0.4,
            source="Backman 1998 (IV midazolam + rifampicin)",
            include_enzyme_pool=True,
        ),
        DDIBenchmarkCase(
            name="rifampicin+midazolam-oral-7.5mg",
            victim_drug_name="midazolam",
            perpetrator_drug_name="rifampicin",
            victim_dose_mg=7.5,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_RIF,
            perpetrator_luminal_conc_um=_CLUM_RIF,
            observed_aucr=0.04,
            source="Backman 1996 (oral midazolam + rifampicin)",
            notes="Gut + hepatic CYP3A4 induction both active",
            include_enzyme_pool=True,
        ),
        DDIBenchmarkCase(
            name="rifampicin+simvastatin-oral-40mg",
            victim_drug_name="simvastatin",
            perpetrator_drug_name="rifampicin",
            victim_dose_mg=40.0,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_RIF,
            perpetrator_luminal_conc_um=_CLUM_RIF,
            observed_aucr=0.1,
            source="Kyrklund 2000 (simvastatin + rifampicin)",
            include_enzyme_pool=True,
        ),
        DDIBenchmarkCase(
            name="rifampicin+atorvastatin-oral-40mg",
            victim_drug_name="atorvastatin",
            perpetrator_drug_name="rifampicin",
            victim_dose_mg=40.0,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_RIF,
            perpetrator_luminal_conc_um=_CLUM_RIF,
            observed_aucr=0.2,
            source="Backman 2005 (atorvastatin + rifampicin)",
            include_enzyme_pool=True,
        ),
        DDIBenchmarkCase(
            name="rifampicin+nifedipine-oral-10mg",
            victim_drug_name="nifedipine",
            perpetrator_drug_name="rifampicin",
            victim_dose_mg=10.0,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_RIF,
            perpetrator_luminal_conc_um=_CLUM_RIF,
            observed_aucr=0.3,
            source="Holtbecker 1996 (nifedipine + rifampicin)",
            include_enzyme_pool=True,
        ),
        DDIBenchmarkCase(
            name="rifampicin+diazepam-oral-10mg",
            victim_drug_name="diazepam",
            perpetrator_drug_name="rifampicin",
            victim_dose_mg=10.0,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_RIF,
            perpetrator_luminal_conc_um=_CLUM_RIF,
            observed_aucr=0.3,
            source="Ohnhaus 1987 (diazepam + rifampicin)",
            include_enzyme_pool=True,
        ),

        # ═══════════════════════════════════════════════════════════
        # CYP2C9 — induction (rifampicin)
        # ═══════════════════════════════════════════════════════════
        DDIBenchmarkCase(
            name="rifampicin+warfarin-oral-10mg",
            victim_drug_name="warfarin",
            perpetrator_drug_name="rifampicin",
            victim_dose_mg=10.0,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_RIF,
            perpetrator_luminal_conc_um=_CLUM_RIF,
            observed_aucr=0.4,
            source="Heimark 1987 (warfarin + rifampicin)",
            notes="S-warfarin CYP2C9 induction",
            include_enzyme_pool=True,
        ),
        DDIBenchmarkCase(
            name="rifampicin+ibuprofen-oral-400mg",
            victim_drug_name="ibuprofen",
            perpetrator_drug_name="rifampicin",
            victim_dose_mg=400.0,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_RIF,
            perpetrator_luminal_conc_um=_CLUM_RIF,
            observed_aucr=0.6,
            source="Konishi 2004 analog (CYP2C9 induction)",
            include_enzyme_pool=True,
        ),

        # ═══════════════════════════════════════════════════════════
        # CYP2D6 — reversible (fluoxetine)
        # ═══════════════════════════════════════════════════════════
        DDIBenchmarkCase(
            name="fluoxetine+metoprolol-oral-100mg",
            victim_drug_name="metoprolol",
            perpetrator_drug_name="fluoxetine",
            victim_dose_mg=100.0,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_FLUOX,
            observed_aucr=2.5,
            source="Walley 1993 (metoprolol + fluoxetine)",
            include_enzyme_pool=False,
        ),

        # ═══════════════════════════════════════════════════════════
        # CYP2C19 — reversible (omeprazole, weak perpetrator)
        # ═══════════════════════════════════════════════════════════
        DDIBenchmarkCase(
            name="omeprazole+diazepam-oral-10mg",
            victim_drug_name="diazepam",
            perpetrator_drug_name="omeprazole",
            victim_dose_mg=10.0,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_OMEP,
            observed_aucr=1.3,
            source="Andersson 1990 (diazepam + omeprazole)",
            include_enzyme_pool=False,
        ),

        # ═══════════════════════════════════════════════════════════
        # NEGATIVE CONTROLS — mismatched perpetrator/victim
        # (AUCR should be ~1.0; these test that the engine does not
        #  create spurious interactions)
        # ═══════════════════════════════════════════════════════════
        DDIBenchmarkCase(
            name="ketoconazole+warfarin-oral-10mg",
            victim_drug_name="warfarin",
            perpetrator_drug_name="ketoconazole",
            victim_dose_mg=10.0,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_KETO,
            observed_aucr=1.1,
            source="Negative control — warfarin is CYP2C9, keto is CYP3A4",
            notes="Small effect via warfarin's ~10% CYP3A4 fraction",
            include_enzyme_pool=False,
        ),
        DDIBenchmarkCase(
            name="ketoconazole+ibuprofen-oral-400mg",
            victim_drug_name="ibuprofen",
            perpetrator_drug_name="ketoconazole",
            victim_dose_mg=400.0,
            victim_route="ORAL",
            perpetrator_steady_state_conc_um=_CU_KETO,
            observed_aucr=1.05,
            source="Negative control — ibuprofen is CYP2C9/2C8",
            include_enzyme_pool=False,
        ),
    )
    return cases
