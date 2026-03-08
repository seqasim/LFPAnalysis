# Your First Baseline

## What this step is for

This chapter helps you choose and inspect a baseline instead of inheriting one from a specific notebook.

## When you should use it

Use this after you understand your event timing and before comparing power or event-locked responses.

## Required inputs

- continuous or epoched data
- a baseline mode
- a baseline window if baselining is enabled

## Minimal example

```python
from pathlib import Path
from LFPAnalysis import build_event_locked_pipeline_config, run_pipeline

config = build_event_locked_pipeline_config(
    Path("../data/sample_ieeg_continuous_rest.fif"),
    file_format="mne",
    event_name="demo",
    event_times=[5.0, 10.0, 15.0],
    baseline_mode="zscore",
    baseline_window=(-0.5, 0.0),
)
result = run_pipeline(config)
```

## How to inspect the result

Review `result.baseline_summary` and confirm that the chosen window overlaps the data you think it does.

## Common mistakes

- enabling baselining without setting a window
- using a baseline mode by habit instead of because it fits the design
- assuming the baseline summary is optional bookkeeping instead of QA

## Old-to-new translation note

Older notebooks often mixed epoch creation and baselining in one flow. The refactored API makes the baseline choice explicit and returns a dedicated summary table.

Next step: {doc}`07_first_event_locked_workflow`
