# Your First Artifact Pass

## What this step is for

This chapter teaches you how to run a first-pass artifact screen and interpret the returned event tables.

## When you should use it

Use this before event-locked averaging or spectral interpretation.

## Required inputs

- referenced or unreferenced continuous data
- a detector choice such as `misc`, `ied`, or `custom`

## Minimal example

```python
from pathlib import Path
from LFPAnalysis import build_basic_pipeline_config, run_pipeline

config = build_basic_pipeline_config(
    Path("../data/sample_ieeg_continuous_rest.fif"),
    file_format="mne",
    artifact_methods=["misc"],
)
result = run_pipeline(config)
artifact_table = result.artifact_tables["misc"]
```

## How to inspect the result

Look at the number of rows, the affected channels, and whether the timestamps cluster suspiciously in one recording segment.

## Common mistakes

- treating detector output as ground truth instead of a quality-control signal
- adding `ied` by default even when the task does not require it
- forgetting that `custom` detectors must return either a dataframe or a channel-event mapping

## Old-to-new translation note

The old notebooks often called `detect_misc_artifacts` or `detect_IEDs` directly. The stable path wraps those into named artifact tables so later steps have a common output shape.

Next step: {doc}`06_first_baseline`
