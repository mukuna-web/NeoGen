# Data provenance

What is in `data/` and `models/`, where it came from, and what this repository
deliberately does not ship.

## There is no patient data here

This repository contains no patient-level data, de-identified or otherwise, and
never has. The "population" layer (`neogen/population/`) is a virtual-patient
Monte Carlo: it samples physiological parameters from published distributions to
produce simulated individuals. Nothing in it derives from a real person.

Training-set sizes reported in `models/baselines/model_summary.json` — 38,661
rows for half-life, 18,212 for clearance, 13,578 for exposure, 8,750 for peak
concentration, 6,989 for volume of distribution — are counts of **compound-level
pharmacokinetic measurement records**, not counts of people.

## What ships in this repository

| Path | Contents | Origin |
|---|---|---|
| `data/clinical/gold_standard_pk.json` | Observed clinical pharmacokinetic values for the benchmark cases | Curated by hand from published clinical pharmacology literature |
| `data/validation/ddi_benchmark_30.json`<br>`data/validation/ddi_reference_pairs.json` | The 30 interaction pairs and their observed exposure ratios | Curated from published interaction studies |
| `data/validation/blind_holdout_50.json`<br>`data/validation/blind_holdout_results.json` | 50-compound holdout set and its recorded results | Curated; held out of training |
| `data/validation/tier3_bioavailability.json` | Bioavailability validation cases | Curated from literature |
| `data/training_stereo/chiral_cohort_v1.json` | 9 enantiomer pairs with published direction of effect | Curated; every row carries a PMID/DOI in `chiral_cohort_v1_citations.md` |
| `data/fda/dailymed_pk_data.json` | Pharmacokinetic values extracted from drug labels | DailyMed / openFDA — US federal government, public domain |
| `data/flywheel/` | Normalised label records, identity resolution, provenance manifests | Derived from openFDA, DailyMed, RxNorm — public domain federal sources |
| `models/` | Trained model artifacts (joblib) | Produced by this project |

Curated reference values are measured quantities reported in the scientific
literature, cited where the source is recorded. They are facts about drugs, not
creative works, and are included so the benchmark is runnable and checkable.
Each is a small hand-built table, not a redistribution of anyone's database.

## What this repository does not ship

**Third-party ADME datasets.** Earlier working copies of this project kept
local dumps of datasets from Therapeutics Data Commons — Caco-2 permeability,
bioavailability, half-life, protein binding, lipophilicity, hepatocyte and
microsome clearance, intestinal absorption, aqueous solubility, volume of
distribution. Those files are **not** in this repository. Their redistribution
terms vary by dataset and by upstream source, and rather than assert terms that
have not been verified for each one, they are fetched on demand:

```bash
python3 scripts/fetch_datasets.py
```

That script pulls them through the `PyTDC` package from the upstream project,
which leaves each dataset under its own licence and the user's own obligation to
honour it. Nothing in this repository requires them: the full test suite (1,167
tests) and both benchmark suites run without them, because the shipped models
are already trained. They are needed only to retrain from scratch.

**DrugBank.** DrugBank's licence does not permit redistribution. No
DrugBank-derived data file is in this repository. Several modules reference
DrugBank identifiers or will read a local DrugBank extract if one is present,
and they degrade gracefully when it is absent. If you have your own licensed
copy you can supply it; this project will not distribute one.

**Generated research artifacts.** The private working repository accumulated
roughly 21,800 generated research-loop reports and 2,100 generated candidate
modules. None of that is here — it is machine-generated output, not source.

## Trained models

The artifacts in `models/` were trained on the compound-level measurement
records described above: public ADME datasets plus literature-curated values.
If you intend to redistribute the trained artifacts themselves, note that the
terms of the upstream training data may bear on derived works, and that
question has not been settled here one way or the other.

## Reproducing the data pipeline

`neogen/data_flywheel/` holds the ingestion and normalisation layer: label
selection, chemical identity resolution, stereochemistry handling, provenance
manifests and a review workflow. Its manifests in `data/flywheel/manifests/`
record each ingestion cycle, so the normalised outputs can be traced back to the
federal source records they came from.
