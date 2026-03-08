# Your First Event-Locked Workflow

## What this step is for

This is the first full beginner workflow: load, optional artifact pass, epoch around behavior, and baseline the result.

## When you should use it

Use this when you have event timestamps and want a clean event-locked analysis object before moving to PSD, TFR, or statistics.

## Required inputs

- continuous electrophysiology data
- event timestamps in seconds
- a named event family

## Minimal example

```python
from pathlib import Path
from LFPAnalysis import build_event_locked_pipeline_config, run_pipeline

config = build_event_locked_pipeline_config(
    Path("../data/sample_ieeg_continuous_rest.fif"),
    file_format="mne",
    event_name="demo",
    event_times=[5.0, 10.0, 15.0],
)
result = run_pipeline(config)
```

## How to inspect the result

Check `len(result.epochs)`, `result.epochs.times[[0, -1]]`, and `result.baseline_summary.head()`.

## Common mistakes

- passing milliseconds instead of seconds
- forgetting that `tmin` is usually negative for pre-event windows
- expecting the stable API to recreate every legacy side effect such as writing artifact CSVs to disk

## Old-to-new translation note

This chapter is the stable replacement for the most common old `make_epochs(...)` pattern when you only need epochs themselves, not every legacy side effect.

Next step: {doc}`08_first_psd_and_fooof`
