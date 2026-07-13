# LFPAnalysis Book

This book now has two explicit tracks.

## Beginner Track

Choose this path if you are new to the repository, new to MNE, or want the shortest route from a sample file to an interpretable result.

This book walks one gambling-task iEEG dataset end-to-end: load, reference, synchronize, artifact QC, epoch with real behavioral times, then reward contrasts in PSD/TFR/connectivity and time-resolved regression against reward prediction error.

## Coming From The Old Repo

Choose this path if you previously worked from `scripts/Condensed Notebook.ipynb` or the older step-by-step notebooks.

You will learn:

- how the refactored repo is organized
- which old entry points now have compatibility shims
- how to translate the condensed notebook, TFR workflow, and connectivity workflow
- which workflows are still advanced or legacy-only

## Reusable Workflows (after metrics)

Once you can compute per-electrode, per-pair, or per-patient measures, chapters 12–16 cover the workflows you will repeat across projects:

- ROI / anatomy assignment
- assembling tidy analysis dataframes
- group-level and permutation statistics (`statistics_utils`)
- saving and organizing results
- plotting recipes

Start that track at {doc}`12_anatomy_and_roi_assignment` after finishing the beginner case study (through {doc}`11_advanced_utility_interoperability`).

## Interface taxonomy

- `stable beginner-facing API`: typed configs, convenience builders, and `run_pipeline`
- `compatibility/legacy shims`: `LFPAnalysis.legacy`
- `advanced legacy utility modules`: `lfp_preprocess_utils`, `analysis_utils`, `oscillation_utils`, and related helpers

## Where to start

- New user: begin with {doc}`00_interface_guide` and {doc}`03_first_load`
- After metrics / group analyses: begin with {doc}`12_anatomy_and_roi_assignment`
- Returning old user: begin with {doc}`20_old_repo_mental_model`

```{tableofcontents}
```
