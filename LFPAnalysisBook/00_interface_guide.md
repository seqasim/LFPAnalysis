# Which Interface Should I Use?

## stable beginner-facing API

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

config = build_basic_pipeline_config(Path("../data/sample_ieeg.fif"), file_format="mne")
result = run_pipeline(config)
```

### How to inspect the result

Start by checking `result.raw`, `result.referenced`, and `result.metadata`.

### Common mistakes

- importing advanced utilities before trying the stable API
- using the config dataclasses directly when a builder would do
- assuming the stable API already covers every old notebook workflow

### Old-to-new translation note

If you used `make_mne`, `ref_mne`, or `make_epochs` directly in the old repo, read the migration chapters before choosing between the stable API and the compatibility shims.

## compatibility/legacy shims

Use `LFPAnalysis.legacy` when you need a gentle bridge from old notebook code. These wrappers emit deprecation warnings and point to the new equivalent.

## advanced legacy utilities

Use the advanced modules only when the stable API or compatibility layer does not yet cover your workflow, especially for time-frequency, connectivity, and time-resolved statistics.

Read {doc}`11_advanced_utility_interoperability` before chaining utility modules directly. It documents the shared naming, warning, baseline, and surrogate conventions that now span the utility layer.

## Decision guide

- I want the full case study: follow chapters 02 → 10b in order
- I just want to load sample data: `build_basic_pipeline_config`
- I want to epoch task events with behavior metadata: `build_event_locked_pipeline_config` + `sample_beh.csv`
- I want PSD or FOOOF: `build_spectral_pipeline_config`
- I used the old notebooks: `LFPAnalysis.legacy` plus the migration chapters

Next step: {doc}`01_installation`
