# Which Interface Should I Use?

## stable beginner-facing API (prep spine → Epochs handoff → analysis spine)

### What this step is for

This interface is the recommended starting point for almost everyone. It gives you typed configuration objects, convenience builders, and a single end-to-end pipeline entry point.

The beginner track walks one gambling-task dataset from raw load through reward contrasts and time-resolved statistics. Each chapter adds one preprocessing or analysis decision on the same files in `../data/`.

### When you should use it

Use it when you want a guided path, clear defaults, and documentation that matches the public API.

### Required inputs

- the bundled sample files (or your own analog: neural FIF, behavior CSV, electrode table)
- optional photodiode FIF for synchronization
- optional electrode metadata when referencing requires it

### Minimal example

```python
from pathlib import Path
from LFPAnalysis import build_basic_pipeline_config, run_pipeline
from LFPAnalysis import build_prep_config, run_prep, build_analysis_config, run_analysis

# Tutorial convenience (prep then analysis):
config = build_basic_pipeline_config(Path("../data/sample_ieeg.fif"), file_format="mne")
result = run_pipeline(config)

# Explicit dual spine (preferred when prep may change later):
prep = run_prep(build_prep_config(Path("../data/sample_ieeg.fif"), file_format="mne"))
# analysis = run_analysis(prep.epochs, build_analysis_config(...))  # when you have Epochs
```

### How to inspect the result

For `run_pipeline`, start with `result.raw`, `result.referenced`, `result.epochs`, and `result.metadata`.
For `run_prep`, the handoff is `prep.epochs` plus `prep.sync` / `prep.electrode_df` / `prep.artifact_tables`.

### Common mistakes

- importing advanced utilities before trying the stable API
- putting sync or electrode localization into analysis — those belong in prep
- using the config dataclasses directly when a builder would do
- assuming the stable API already covers every old notebook workflow

### Old-to-new translation note

If you used `make_mne`, `ref_mne`, or `make_epochs` directly in the old repo, read the migration chapters before choosing between the stable API and the compatibility shims.

## compatibility/legacy shims

Use `LFPAnalysis.legacy` when you need a gentle bridge from old notebook code. These wrappers emit deprecation warnings and point to the new equivalent.

## advanced legacy utilities

Use `LFPAnalysis.advanced` (or the utility modules) only when the stable prep/analysis spines do not yet cover your workflow, especially for connectivity and time-resolved statistics after you already have Epochs.

Read {doc}`11_advanced_utility_interoperability` before chaining utility modules directly. It documents the shared naming, warning, baseline, and surrogate conventions that now span the utility layer.

## Decision guide

- I want the full case study: follow chapters 02 → 10b in order
- I just want to load sample data: `build_basic_pipeline_config`
- I want to epoch task events with behavior metadata: `build_event_locked_pipeline_config` + `sample_beh.csv`
- I want PSD or FOOOF: `build_spectral_pipeline_config` / `run_pipeline`, or `build_analysis_config` + `run_analysis` on existing Epochs
- I want baseline-only on existing Epochs: `build_analysis_config` + `run_analysis`
- I want beginner Morlet TFR: `build_analysis_config(..., tfr_method="morlet")` + `run_analysis`
- I used the old notebooks: `LFPAnalysis.legacy` plus the migration chapters

Next step: {doc}`01_installation`
