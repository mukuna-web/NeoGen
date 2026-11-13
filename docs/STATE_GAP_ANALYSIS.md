# NeoGen State Gap Analysis — Current Codebase vs V1 Blueprint

> **⚠ OBSOLETE** — this document is a snapshot from the Day-1 Phase 0 spike
> taken before P5, P6, P7, the V1.1 plasma mass-balance fix, the ML
> property layer, the v2 API mount, and the Tier 4 strict mass-balance
> upgrade. Most of the gaps enumerated here have been closed. Kept for
> historical audit only.
>
> **Current state lives in:**
>
> - `reports/validation/validation_*.md` — latest tier results
> - `neogen/validation/*.py` — live validation harness
> - `neogen/engine/orchestrator_v2.py` — engine version tag + DDI wiring
> - Session chat logs — what changed per phase

**Date:** 2026-04-10
**Scope:** Day-1 Phase 0 spike. Read-only audit. No code changes.
**Authoritative sources:**

- `RoadMap.pdf` (7-page condensed blueprint)
- `docs/NEOGEN_V1_ARCHITECTURE.md` (1107-line long-form — canonical when the two differ)

---

## 1. Blueprint Diff — PDF vs NEOGEN_V1_ARCHITECTURE.md

| Aspect               | PDF                            | Markdown                                                                                                                           | Authoritative |
| -------------------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| Layer count          | 12                             | 12                                                                                                                                 | agree         |
| 14-comp list         | not spelled out                | **Arterial, Venous, Lung, Heart, Brain, Liver, Gut, Kidney, Muscle, Adipose, Skin, Bone, Spleen, Rest-of-body**                    | **Markdown**  |
| Absorption equations | high-level                     | Noyes-Whitney with full pH profile, Qgut, well-stirred, Peff by segment                                                            | **Markdown**  |
| Enzyme kinetics      | Michaelis-Menten + inhibition  | Full MM + competitive inhibition + MBI (`dE_active/dt = k_synth − k_deg·E − k_inact·[I]·E/(KI+[I])`) + induction (Emax + turnover) | **Markdown**  |
| Validation cohort    | 50 drugs + 30 DDI pairs + 10 F | same                                                                                                                               | agree         |
| Module list          | 20 modules                     | same 20                                                                                                                            | agree         |

**Conclusion:** The markdown is the richer source and the canonical 14-compartment definition lives only there. Use it as the source of truth. PDF is the condensed internal/external version.

---

## 2. `neogen/simulation/pbpk.py` — Current State

**File:** 890 LOC, single file, no subpackaging.

### What it is

A **single-drug, 16-state ODE model** with scipy `solve_ivp` (RK45, BDF fallback). Class `PBPKModel(patient, dose_mg, route)`.

### State vector (16 elements)

```
[0]  A_gut        — gut lumen amount
[1]  C_plasma     — plasma (≈ venous pool, not a separate compartment)
[2]  C_liver      [3]  C_kidney    [4]  C_brain
[5]  C_muscle     [6]  C_fat       [7]  C_gut_tissue
[8]  A_bile       (EHR)
[9]  C_lung       [10] C_spleen    [11] C_skin
[12] C_bone       [13] C_rbc       [14] C_gut_wall (enterocyte)
[15] C_pancreas
```

### Compartment comparison vs blueprint

| Blueprint                                 | Current `pbpk.py`                                    | Status                              |
| ----------------------------------------- | ---------------------------------------------------- | ----------------------------------- |
| Arterial blood pool                       | **missing** — uses `C_lu / Kp_lung` as proxy         | ⚠ gap                               |
| Venous blood pool                         | **missing** — plasma collapses this role             | ⚠ gap                               |
| Lung                                      | ✓                                                    | match                               |
| **Heart**                                 | **missing**                                          | ⚠ gap                               |
| Brain                                     | ✓                                                    | match                               |
| Liver                                     | ✓                                                    | match                               |
| Gut (metabolically active, CYP3A4 + P-gp) | ✓ (gut_tissue + gut_wall enterocyte, split)          | OK, but 2 compartments instead of 1 |
| Kidney                                    | ✓                                                    | match                               |
| Muscle                                    | ✓                                                    | match                               |
| Adipose (fat)                             | ✓                                                    | match                               |
| Skin                                      | ✓                                                    | match                               |
| Bone                                      | ✓                                                    | match                               |
| Spleen                                    | ✓                                                    | match                               |
| Rest-of-body                              | **approximated** by `Q_rest × C_pl` in venous mixing | ⚠ gap — not a separate compartment  |
| —                                         | **extras: bile, RBC, pancreas, gut lumen**           | off-spec                            |

**Verdict:** Current model has 13 real organ compartments + 3 extras (bile, RBC, pancreas) + lumen. It is missing **arterial, venous, heart, and rest-of-body** as distinct pools. Calling this "14-compartment" is accurate by count but **not canonical by topology**.

### Critical blueprint violations in `pbpk.py`

1. **`self.F` is stored as a scalar.** Computed once at `derive_pbpk_params` time (line 246), then used at initial condition: `y0[0] = self.F * self.dose_mg`. This bakes bioavailability into the initial dose amount. **Under DDI, F should be time-varying and emerge from the ODE state.** Current code cannot represent this.

2. **Enzyme pools are not state variables.** `self.CL_hep` is a scalar assigned once. No `E_active`, no enzyme turnover, no `dE_active/dt`. The ODE cannot represent mechanism-based inhibition, induction lag, or time-resolved enzyme dynamics. This is the single biggest mechanistic gap.

3. **Single-drug only.** Class state is indexed on one drug. No coupling across drugs. Multi-drug DDI is structurally impossible without a state-vector rewrite.

4. **`fg` and `fh` are not computed here.** They are imported from `bioavailability.py` which uses static logP-based estimates, not Qgut or well-stirred. The gut wall enterocyte compartment exists but is not the site of mechanistic CYP3A4 metabolism.

5. **Tight coupling.** `derive_pbpk_params` imports from 8 sibling modules via try/except. Any refactor of the engine must either preserve these imports or break production paths.

6. **Docstring says "7-Compartment"** (line 1–13) but the implementation is 16-state. Documentation drift.

### What's good

- Rodgers-Rowland Kp prediction integrated (`kp_prediction.py`).
- ICRP 89 physiology scaling by weight/age/sex exists.
- Fallback layering (try/except) means production API stays up.
- ODE solver wiring (solve_ivp + BDF fallback) is sound.
- 14 physical organs (roughly) already modeled — the skeleton is mostly there.

---

## 3. `neogen/simulation/bioavailability.py` — Current State

**File:** 316 LOC.

### What it is

A **static F predictor**. `BioavailabilityEngine(smiles, logp, mw, tpsa, hbd)` returns scalar F, fa, fg, fh.

Two paths:

- **Known drug** (`BIOAVAILABILITY_DATABASE`): SMILES-keyed lookup → hardcoded fa/fg/fh/f_oral (Diazepam, Paracetamol, Caffeine, Omeprazole, Ibuprofen, Aspirin, Metformin currently).
- **Unknown drug**: empirical rules from logP/TPSA/MW/HBD (Lipinski-ish).

### Blueprint violations

1. **F is scalar, not runtime function.** `get_bioavailability() → float`. The blueprint requires `F(t, enzyme_state)`.
2. **`fg` is NOT Qgut model.** Currently: `fg = 0.60 if logp>4 else 0.80 if logp>2 else 0.95`. The blueprint specifies `fg = Qgut / (Qgut + fu_gut × CLint_gut)` with enterocyte CYP3A4 abundance.
3. **`fh` is NOT well-stirred model.** Currently: `fh = 0.40 if logp>4 and mw>300 else ...`. The blueprint specifies `fh = 1 − fu·CLint / (Qh + fu·CLint)`.
4. **No enzyme-state input.** Cannot be perturbed by a co-administered inhibitor.
5. **Known-drug lookup is the opposite of "structure-only input".** It's a small hardcoded table that can't extrapolate.

### What's good

- Decomposition `F = fa × fg × fh` is already the mental model.
- BCS classification pipeline is in place.
- `get_full_assessment()` returns a clean dict the API can consume.

### Verdict

**Must be rewritten as a runtime computation** that takes current enzyme state and fu from the ODE. Keep the BCS classification logic and the known-drug table as a validation source, not a prediction source.

---

## 4. `neogen/simulation/ddi_lookup.py` — Current State

**File:** 123 LOC.

### What it is

Pure name-based lookup over DDInter 2.0 (222K records). Returns `{level: Major/Moderate/Minor, source}`.

### Blueprint violations

1. Blueprint (PDF p.4): **"DDI is NOT a separate lookup — it is an emergent property of the coupled ODE system."**
2. Zero mechanism, zero AUC ratio, zero PK simulation. Just severity labels from a public database.
3. If this is on the DDI prediction path, it contradicts the entire moat statement ("structure-only input → mechanistic simulation").

### Recommended fate

**Do not delete.** Retain as:

- **Labeling/severity fallback** — when mechanism cannot resolve a pair (e.g., PD interactions, non-CYP mechanisms).
- **Validation cross-check** — mechanistic predictions should correlate with DDInter severity on known pairs.
- **Frontend UX** — "DDInter labels this as Moderate" is a useful context annotation, not a prediction.

**Must not be on the prediction path** for V1. Rename/reorganize to make its role explicit, e.g., `neogen/data/ddi_labels.py` or `neogen/intelligence/ddinter.py`.

---

## 5. `adme_pipeline.py` — Philosophical Inversion (summary)

Current production prediction path is **ML-first** (RDKit210 → ML regressors → mechanistic derivation of t½/AUC/Cmax from one-compartment analytical solution). 63–68.5% within 2-fold on held-out drugs.

Blueprint V1 is **ODE-first** (RDKit → property predictors feed Drug entity → ODE orchestrator solves full PBPK + DDI coupled system → AUC/Cmax/t½ are secondary post-hoc outputs from the curves).

These are two different architectures. The ADME pipeline must be **preserved and demoted to a property-predictor layer** inside the Drug factory (where it already earns its keep producing fu, CL, Vd), while the simulation orchestrator takes over output. They should not both be exposed as prediction endpoints.

---

## 6. Consolidated Gap Register

| #     | Gap                                                                        | Severity                   | Location                                         | P0 action                                                                      |
| ----- | -------------------------------------------------------------------------- | -------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------ |
| GA-1  | `self.F` stored as scalar; DDI cannot perturb it                           | **Critical**               | `pbpk.py:189, 246`                               | Rewrite F as `F(t, enzyme_state)` in new engine; do not modify `pbpk.py` in P0 |
| GA-2  | Enzyme pools not in ODE state vector                                       | **Critical**               | `pbpk.py:450`                                    | Add `E_active` per enzyme × organ to new `SimulationState`                     |
| GA-3  | Single-drug state — no multi-drug coupling                                 | **Critical**               | `pbpk.py` class                                  | New state vector indexed `(drug, compartment)` and `(enzyme, organ)`           |
| GA-4  | `fg` uses logP rule, not Qgut                                              | High                       | `bioavailability.py:201–208`                     | Build `GutWallMetabolism` module in P2                                         |
| GA-5  | `fh` uses logP rule, not well-stirred with CLint                           | High                       | `bioavailability.py:210–218`                     | Build `HepaticFirstPass` module in P2                                          |
| GA-6  | `ddi_lookup.py` is on prediction path (inferred from filename + existence) | High                       | `simulation/ddi_lookup.py`                       | Confirm call sites in P0 audit extension; move to labels-only                  |
| GA-7  | Missing arterial + venous + heart + rest-of-body compartments              | High                       | `pbpk.py:450`                                    | Canonical state in new `engine/state.py` (P0)                                  |
| GA-8  | Docstring says 7-compartment, implementation is 16-state                   | Low                        | `pbpk.py:1–13`                                   | Leave alone in P0 (touching live file)                                         |
| GA-9  | ADME pipeline is ML-first, blueprint is ODE-first                          | **Critical philosophical** | `adme_pipeline.py`                               | Demote to property predictor; do not remove from API in P0                     |
| GA-10 | 8-way tight coupling in `derive_pbpk_params`                               | Med                        | `pbpk.py:260–440`                                | New engine avoids this coupling; old `pbpk.py` keeps serving until migration   |
| GA-11 | No mechanism-based inhibition enzyme-turnover ODE                          | **Critical**               | everywhere                                       | Lives in new `engine/enzyme_state.py` (P3)                                     |
| GA-12 | No induction (Emax + enzyme turnover)                                      | **Critical**               | everywhere                                       | Same as GA-11, P3                                                              |
| GA-13 | Protein binding fu is static, not unbound-[I] propagated through DDI math  | Med                        | `protein_binding.py` (exists)                    | Wire into new DDI resolver in P3                                               |
| GA-14 | No `SimulationOrchestrator` for multi-drug simulation                      | **Critical**               | —                                                | New `engine/orchestrator.py` skeleton in P0, wired in P3                       |
| GA-15 | No validation cohort in-repo                                               | High                       | `reports/benchmarks/` (exists but scope unknown) | Expand in P1/P2/P3                                                             |
| GA-16 | No coexistence story — new engine must live alongside old                  | High                       | architecture                                     | `neogen/engine/` new subpackage; `simulation/` untouched until migration       |

---

## 7. Coexistence Strategy (non-destructive migration)

**Principle:** the new mechanistic engine must ship behind `/v2/simulate` while `/predict` (ADME pipeline + `PBPKModel`) keeps serving production. No destructive changes until parity is proven on the validation cohort.

```
neogen/
  core/            NEW — Drug, PatientProfile, DoseEvent, SimulationConfig (Pydantic)
    entities.py
    physiology.py  (canonical 14-compartment ICRP 89)
    enzymes.py     (enzyme DB loader)
  engine/          NEW — the V1 ODE engine
    state.py       (SimulationState, multi-drug × multi-compartment × multi-enzyme)
    pbpk.py        (canonical 14-compartment right-hand side)
    enzyme_state.py(enzyme pools + turnover + inhibition + induction)
    bioavailability.py (runtime F computer)
    ddi_resolver.py(emergent fm-rule resolver)
    orchestrator.py(top-level Simulation)
    integrator.py  (scipy BDF wrapper with event handling)
  physics/         NEW — deterministic sub-models
    dissolution.py (Noyes-Whitney)
    qgut.py        (gut wall metabolism)
    well_stirred.py(hepatic first-pass)
    rodgers_rowland.py (Kp prediction)
  ml/              NEW — property predictors only
    fu.py          (wraps existing protein_binding logic)
    peff.py        (wraps Caco-2 Papp predictor)
    cyp_profile.py (wraps CYP substrate/inhibitor classifiers)
  validate/        NEW — validation harness
    cohort.py
    tier1.py … tier4.py
    invariants.py  (mass balance, IV F=1, etc.)
  simulation/      EXISTING — left untouched in P0; keeps serving /predict
  api/
    predict.py     EXISTING — unchanged in P0
    simulate.py    NEW — /v2/simulate endpoint, added in P7
```

Migration happens only after validation parity: **neither** `simulation/` nor `adme_pipeline.py` is modified in P0. All P0 work is additive.

---

## 8. Immediate P0 Decisions Needed From You

| #   | Question                                                                                                                                                       | My recommendation                                                                                                                             |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | Canonical 14-compartment list — go with blueprint (arterial/venous/heart, no pancreas/RBC/bile) or keep current (pancreas/RBC/bile, no arterial/venous/heart)? | **Blueprint.** Match FDA-PBPK conventions. Pancreas/RBC/bile can return as a V2 extension.                                                    |
| D2  | `ddi_lookup.py` — retain as labeling fallback only?                                                                                                            | **Yes**, retain read-only, never call from prediction path.                                                                                   |
| D3  | Multi-drug state vector — per-drug stacked (`y = [drug1_state, drug2_state, ...]`) or interleaved (`y[(compartment, drug)]`)?                                  | **Stacked** — simpler, easier to slice, cleaner Jacobian sparsity.                                                                            |
| D4  | Enzyme DB primary source — SimCYP textbook, DDInter 2.0, UW Metabolism DB, other?                                                                              | **DDInter 2.0 + FDA DDI guidance + Goodman & Gilman** for Ki/Km/Vmax; **PharmGKB** for fm (already integrated). Seed with top 40 drugs in P0. |
| D5  | fm values — who curates, and at what fidelity?                                                                                                                 | Manually curated CSV in `data/fm_values.csv` with DOI citation column. Start with 40 drugs × 8 enzymes.                                       |
| D6  | Multi-dose regimen support in V1, or V2?                                                                                                                       | **V1** — MBI validation requires multi-dose. Override the PDF "MVP: single dose" and add to P3 scope. Flag explicitly.                        |
| D7  | Integrator — SciPy BDF default with RK45 fallback, or LSODA?                                                                                                   | **BDF default.** Stiff by construction.                                                                                                       |
| D8  | New engine should be named `neogen.engine` or `neogen.sim2`?                                                                                                   | **`neogen.engine`** — forward-looking, clean separation.                                                                                      |
| D9  | Validation cohort storage — JSON, CSV, or Pydantic-modeled?                                                                                                    | **Pydantic-modeled TOML** for editability + type safety.                                                                                      |
| D10 | PD scope in V1 — Emax only, or Emax + receptor occupancy + cardiac wire-up?                                                                                    | **Emax + CiPA wire-up.** Receptor kinetics → V2.                                                                                              |

---

## 9. Ready-to-start P0 task breakdown (only when you greenlight)

All additive. No deletions. No git actions.

1. `docs/PACKAGE_TOPOLOGY.md` — formalize §7 above as a design doc.
2. `neogen/core/__init__.py` + `entities.py` — Pydantic `Drug`, `PatientProfile`, `DoseEvent`, `DoseSchedule`, `SimulationConfig`.
3. `neogen/core/physiology.py` — canonical 14-compartment ICRP 89 reference (Q, V per organ) — blueprint-compliant.
4. `neogen/core/enzymes.py` + `data/enzyme_db.json` seed — CYP3A4 + 2D6 top 20 drugs (Km, Vmax, Ki catalog).
5. `neogen/engine/state.py` — `SimulationState` dataclass: drugs × compartments × enzyme pools.
6. `neogen/engine/orchestrator.py` — signature-only skeleton.
7. `tests/invariants/` — 8 property tests: IV F=1, mass balance, dose conservation, non-negative concentrations, monotone decay post-dose, CL↔AUC relation, MM boundaries, fu propagation.
8. Update `CLAUDE.md` or `AGENTS.md` with phase-tagging convention.

**P0 total:** ~600 LOC new code, 1 doc, 1 seed JSON, 8 tests. Zero changes to existing files. Existing tests continue to pass. Production API untouched.

---

## 10. Execution Gate

Per operating rules, Claude does **not** proceed to P0 implementation without:

1. Your answers to D1–D10 above (or "take your recommendation on all" if you trust the defaults).
2. Explicit permission to begin writing new files in `neogen/core/`, `neogen/engine/`, `docs/`, and `tests/invariants/`.
3. Explicit permission for any git action — none needed for P0 scaffolding but will be asked before first commit.
