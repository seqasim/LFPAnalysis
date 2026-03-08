# Your First PSD and FOOOF Workflow

## What this step is for

This chapter shows the first spectral workflow that remains mostly inside the stable API.

## When you should use it

Use this after load, reference, and epoching are already believable.

## Required inputs

- continuous or epoched data
- `analysis` dependencies if you want FOOOF

## Minimal example

```python
from pathlib import Path
from LFPAnalysis import build_spectral_pipeline_config, run_pipeline

config = build_spectral_pipeline_config(
    Path("../data/sample_feedback_start-epo.fif"),
    file_format="mne",
    spectral_method="psd",
)
result = run_pipeline(config)
```

## How to inspect the result

Confirm that `result.spectral` includes the chosen method and inspect the returned spectrum or FOOOF table.

## Common mistakes

- trying FOOOF before confirming the PSD is sensible
- assuming TFR and connectivity are covered by the same stable spectral wrapper
- forgetting to install `analysis` extras

## Old-to-new translation note

The old notebooks often jumped directly into PSD and FOOOF functions. The refactored path encourages getting a valid `PipelineResult` first and then using the advanced utilities only where needed.

Next step: {doc}`09_first_time_frequency`
