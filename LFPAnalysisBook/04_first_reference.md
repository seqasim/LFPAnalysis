# Your First Reference Choice

## What this step is for

This chapter helps you make your first explicit referencing decision instead of inheriting one from a notebook.

## When you should use it

Use this after loading data and before interpreting amplitudes, PSDs, or event-locked averages.

## Required inputs

- loaded electrophysiology data
- electrode metadata if you want `wm` or `bipolar`

## Minimal example

```python
from pathlib import Path
from LFPAnalysis import build_basic_pipeline_config, run_pipeline

config = build_basic_pipeline_config(
    Path("../data/sample_ieeg.fif"),
    file_format="mne",
    reference_method="none",
)
result = run_pipeline(config)
```

## How to inspect the result

Compare `result.raw` and `result.referenced`. If you used `none`, they should have the same channels and overall shape.

## Common mistakes

- choosing `wm` or `bipolar` without a valid electrode table
- assuming `laplacian` is ready in the stable path
- forgetting to write down which reference was used

## Old-to-new translation note

The old repo encouraged direct calls to `ref_mne`. The new stable path wraps the same decision in `ReferenceConfig` or the convenience builders.

## Not yet supported in the stable path

`laplacian` remains reserved and should be treated as unavailable in the stable beginner-facing API.

Next step: {doc}`05_first_artifact_pass`
