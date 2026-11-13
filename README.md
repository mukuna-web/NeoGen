# NeoGen

A physiologically-based pharmacokinetic (PBPK) engine with a drug-interaction
layer and machine-learned ADME property prediction, built to predict how a drug
behaves in the body from its molecular structure alone.

The engine simulates a 14-compartment body, predicts the absorption,
distribution, metabolism and excretion properties it needs from chemical
structure, and reports every result against a frozen benchmark that publishes
its own misses.

**Engine version `v1.2.0`.** Everything below is reproducible from this
repository — see [Reproducing the benchmark](#reproducing-the-benchmark). Two
checks fail on purpose and are written up in [KNOWN_GAPS.md](KNOWN_GAPS.md);
read that before taking any number here at face value.

## What it does

**Structure to properties.** RDKit fingerprints and physicochemical descriptors
feed gradient-boosted (XGBoost) ensembles that predict the ADME endpoints the
simulation needs: clearance and intrinsic clearance, volume of distribution,
fraction unbound, half-life, bioavailability, solubility, permeability,
intestinal absorption, renal clearance, acidic and basic pKa, and CYP substrate
classification. Low-confidence predictions fall back to Tanimoto
nearest-neighbour retrieval against a reference set rather than guessing.

**Properties to exposure.** A 14-compartment PBPK model (SciPy `solve_ivp`,
BDF) integrates the concentration-time course across organs, with dissolution
and gut-transit absorption modelling, tissue partitioning, and hepatic and
renal elimination.

**Interactions.** An enzyme-competition engine models CYP inhibition,
induction, mechanism-based inactivation, gut-wall first-pass effects and
transporter-mediated interactions, deriving the unbound inhibitor concentration
from simulated pharmacokinetics rather than hard-coded literals.

**Population spread.** A virtual-patient Monte Carlo layer samples physiological
variability to produce exposure distributions instead of single point estimates.
These are *simulated* individuals — this repository contains no patient data of
any kind.

## Results

From `scripts/run_v1_benchmark.py` at `v1.2.0`, on this commit:

| Suite | Metric | Within 2-fold | Average fold error |
|---|---|---|---|
| Pharmacokinetics | 72 measures over 30 cases | **56/72 — 77.8%** | 1.55× |
| — peak concentration | 24 | 19/24 — 79.2% | 1.51× |
| — exposure (AUC) | 24 | 20/24 — 83.3% | 1.47× |
| — half-life | 24 | 17/24 — 70.8% | 1.69× |
| Drug interactions | 30 pairs, exposure ratio | **24/30 — 80.0%** | 1.49× |
| Combined | 102 measures | **80/102 — 78.4%** | — |

The acceptance gate is 60% within 2-fold on every half of the run; both halves
clear it. "Within 2-fold" is the conventional bar in this field — a prediction
counts only if it lands between half and twice the observed clinical value.

The benchmark report lists every case, passing and failing, with predicted
against observed values. Failures are not aggregated away: the largest misses
are visible per-case in the generated report, and the interaction suite's six
failing pairs are named there.

Earlier runs of an older commit recorded 25/30 (83.3%) on the interaction
suite. This commit gives 24/30 (80.0%). The figure above is what the code in
this repository actually produces.

## Install

Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

RDKit is pinned to `2026.03.1`, because the feature standardisation version
(`stdz_v2_rdkit_2026.03.1`) is frozen against it — a different RDKit will
change descriptor values and therefore predictions. Torch and chemprop are
optional; the pipeline falls back to XGBoost when they are absent.

## Reproducing the benchmark

```bash
python3 scripts/run_v1_benchmark.py
```

Runs both suites in about ten seconds and writes a timestamped Markdown and
JSON report to `reports/benchmarks/`. The Markdown report carries the headline,
the per-metric breakdown and a per-case table; the JSON is for dashboards.

```bash
python3 scripts/run_v1_benchmark.py --pk-only
python3 scripts/run_v1_benchmark.py --ddi-only
```

## Tests

```bash
python3 -m pytest -q -rx
```

1,167 pass; 2 are expected failures. The `-rx` flag prints their reasons, which
point at [KNOWN_GAPS.md](KNOWN_GAPS.md). The suite covers the engine phase by
phase, the interaction orchestrator, the data pipeline, the validation
harnesses and the API, and includes acceptance-gate tests that fail the build
if headline accuracy regresses.

## API

```bash
uvicorn neogen.api.v2.app:create_v2_app --factory --reload
```

Serves prediction, simulation, population and validation endpoints with
OpenAPI docs at `/docs`.

## Layout

| Path | What lives there |
|---|---|
| `neogen/engine/` | PBPK solver, clearance engines, interaction orchestrator |
| `neogen/simulation/` | Absorption, dissolution, gut transit, partitioning |
| `neogen/ml/` | ADME predictors and the retrieval fallback |
| `neogen/population/` | Virtual-patient Monte Carlo |
| `neogen/validation/` | Benchmark runner, scorecards, tier harnesses |
| `neogen/data_flywheel/` | Label ingestion, identity resolution, provenance |
| `models/` | Trained model artifacts |
| `data/` | Benchmark cases and curated reference values — see [DATA.md](DATA.md) |
| `reports/benchmarks/` | Generated benchmark reports |

## Provenance and limitations

Benchmark observed values are curated from published clinical pharmacokinetic
literature; [DATA.md](DATA.md) records where each dataset came from and which
ones this repository does not redistribute.

This is a research project. It is not a clinical tool, it is not validated for
any clinical or regulatory purpose, and no output from it should be used to make
a decision about any person's medication. The engine's evaluation design was
reviewed in conversation with a pharmacometrics specialist; that review is not
an endorsement of its accuracy or its fitness for any purpose.

## Licence

MIT — see [LICENSE](LICENSE). The licence covers the code in this repository.
It does not relicense upstream datasets, whose terms are their own; see
[DATA.md](DATA.md).
