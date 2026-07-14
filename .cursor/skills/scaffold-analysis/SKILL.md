---
name: scaffold-analysis
description: >-
  Interviews users about intracranial EEG / LFP datasets and scaffolds hybrid
  analysis notebooks under notebooks/<slug>/ using LFPAnalysis APIs, with path
  placeholders plus sample-data dry-run defaults. Use when a user describes
  their dataset, asks to scaffold notebooks, set up TFR / band-power /
  regression / PSD / connectivity analyses, bring their own data, or wants
  analysis notebooks with only paths left to fill.
---

# Scaffold Analysis Notebooks

Guide a user from a dataset description to runnable hybrid notebooks that follow LFPAnalysis conventions.

## When this skill applies

- User describes a dataset, task, or analysis goals and wants notebooks
- User asks to scaffold / template / bootstrap analyses for their data
- User has epochs (or earlier stage data) and wants TFR, band power, regressions, PSD/FOOOF, connectivity, or dataframe assembly

## Workflow checklist

Copy and track:

```
Scaffold Progress:
- [ ] Step 1: Read repo context (if needed)
- [ ] Step 2: Intake interview (one question at a time)
- [ ] Step 3: Write Dataset Profile
- [ ] Step 4: Select recipes
- [ ] Step 5: Create notebooks under notebooks/<slug>/
- [ ] Step 6: Handoff summary
```

### Step 1: Read repo context

Before asking questions the docs already answer, skim:

- [docs/AI_CONTEXT.md](../../../docs/AI_CONTEXT.md) — stable vs utility layers
- [intake.md](intake.md) — question bank
- [recipes.md](recipes.md) — recipe menu and canonical sources
- [notebook-conventions.md](notebook-conventions.md) — PATHS / sample hybrid pattern

### Step 2: Intake interview

Follow [intake.md](intake.md). Discipline:

- Ask **exactly one** focused question per turn
- Pair every question with a **recommended answer** the user can accept or correct
- Skip branches already answered in the user's initial description
- Stop when these are known: preprocessing stage, modalities, subject structure, events/baseline (if relevant), regressors/conditions, analysis goals, slug

### Step 3: Dataset Profile

Emit a short profile (markdown in chat is enough; optionally save as `notebooks/<slug>/DATASET_PROFILE.md`):

```markdown
# Dataset Profile
- slug: ...
- stage: raw | referenced_raw | epochs | tfr
- subjects: single | multi
- modalities: neural, behavior, electrodes, ...
- events: locking event, tmin/tmax, baseline
- regressors / conditions: ...
- goals → recipes: ...
- path keys needed: epochs, behavior, electrodes, ...
```

### Step 4: Select recipes

Use [recipes.md](recipes.md). Choose the **minimal** set that covers stated goals.

Default bundle for preprocessed re-referenced MNE epochs + behavioral dataframes aiming at baselined TFR, band power, and regressions:

`tfr_baselined` + `band_power` + `regression_avg` + `regression_timeresolved`

Add `dataframe_assembly` for multi-ROI / multi-subject tidy tables. Add `preprocess_to_epochs` only when stage is before epochs.

### Step 5: Create notebooks

1. Create `notebooks/<slug>/` if missing
2. For each selected recipe, write the notebook named in [recipes.md](recipes.md)
3. Follow [notebook-conventions.md](notebook-conventions.md) for structure and PATHS
4. Read canonical book/worked-example sources and adapt code — do not invent APIs
5. Never overwrite `LFPAnalysisBook/worked-examples/` or book chapters

### Step 6: Handoff

Tell the user:

1. Folder path and notebook list
2. Which `PATHS` keys to fill, then set `USE_SAMPLE_DATA = False`
3. How to dry-run with sample data first (`USE_SAMPLE_DATA = True`)
4. Links to the book chapters for each recipe

## Hard rules

1. **Stable API for preparation:** `load_lfp`, config builders, `run_pipeline` for load / reference / artifact / epoch / baseline.
2. **Utilities / MNE for advanced steps:** TFR (`epochs.compute_tfr` or `lfp_preprocess_utils.compute_and_baseline_tfr`), band power, connectivity, `statistics_utils.permutation_regression_zscore`, `time_resolved_mlm`. Do **not** invent stable-API TFR coverage.
3. **Output only under** `notebooks/<slug>/`.
4. **Every notebook structure:** Goal → PATHS + sample dry-run → analysis cells → Next steps (book links).
5. **Stats dataframes:** use chapter 13 columns — `participant`, `unique_label`, `trial`, `ts`, `tfr` (or `power`), plus regressors.
6. **Prefer** book and worked-example patterns over legacy scripts under `scripts/`.

## Additional resources

- [intake.md](intake.md) — interview question bank
- [recipes.md](recipes.md) — recipe selection matrix
- [notebook-conventions.md](notebook-conventions.md) — hybrid notebook template
