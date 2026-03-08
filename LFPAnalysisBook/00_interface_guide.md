# Which Interface Should I Use?

## stable beginner-facing API

### What this step is for

This interface is the recommended starting point for almost everyone. It gives you typed configuration objects, convenience builders, and a single end-to-end pipeline entry point.

### When you should use it

Use it when you want a guided path, clear defaults, and documentation that matches the public API.

### Required inputs

- one supported electrophysiology input
- optional electrode metadata when referencing requires it
- optional event timestamps if you are making epochs

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

Use the advanced modules only when the stable API or compatibility layer does not yet cover your workflow, especially for time-frequency and connectivity analyses.

## Decision guide

- I just want to load sample data: `build_basic_pipeline_config`
- I want to epoch task events: `build_event_locked_pipeline_config`
- I want PSD or FOOOF: `build_spectral_pipeline_config`
- I used the old notebooks: `LFPAnalysis.legacy` plus the migration chapters

Next step: {doc}`01_installation`
