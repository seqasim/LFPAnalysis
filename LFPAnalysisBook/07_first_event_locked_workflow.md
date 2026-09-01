# Your First Event-Locked Workflow

## What this step is for

Build **feedback-locked and baseline-locked raw epochs** from real trial times in `sample_beh.csv`, attach `reward`, `rpe`, and `gamble_rt` metadata, and save two buffered epoch files for TFR in chapter 09.

## When you should use it

Use this when you have behavioral timestamps and want a clean event-locked object before PSD, TFR, connectivity, or statistics.

## Required inputs

- `../data/sample_ieeg_bp.fif` — bipolar continuous (times already in neural seconds)
- `../data/sample_beh.csv` — `feedback_start`, `baseline_start`, `reward`, `rpe`, `gamble_rt`

## Minimal example (raw epochs, no voltage baseline)

```python
import pandas as pd
from pathlib import Path
from LFPAnalysis import build_event_locked_pipeline_config, run_pipeline

beh = pd.read_csv(Path("../data/sample_beh.csv"))

config = build_event_locked_pipeline_config(
    Path("../data/sample_ieeg_bp.fif"),
    file_format="mne",
    event_name="feedback_start",
    event_times=beh["feedback_start"].tolist(),
    baseline_mode="none",
    tmin=-0.5,
    tmax=1.5,
    buffer_s=1.0,
    metadata={
        "reward": beh["reward"].tolist(),
        "rpe": beh["rpe"].tolist(),
        "gamble_rt": beh["gamble_rt"].tolist(),
    },
)
result = run_pipeline(config)
print(f"epochs: {len(result.epochs)}")
print("epoch time span:", result.epochs.times[0], result.epochs.times[-1])
```

Epochs remain raw by design. Baseline normalization belongs to the TFR stage (chapter 09), not voltage epochs.

## Two epoch files for cross-event TFR (recommended)

Save **separate** task and baseline epoch sets (each with `buffer_s=1.0`). Chapter 09 loads both and crops buffers on the **TFR** axis after Morlet.

```python
# Task: feedback_start, core window [-0.5, 1.5] + 1 s buffer
task = run_pipeline(
    build_event_locked_pipeline_config(
        Path("../data/sample_ieeg_bp.fif"),
        file_format="mne",
        event_name="feedback_start",
        event_times=beh["feedback_start"].tolist(),
        baseline_mode="none",
        tmin=-0.5,
        tmax=1.5,
        buffer_s=1.0,
        metadata={"reward": beh["reward"].tolist(), "rpe": beh["rpe"].tolist()},
    )
)

# Baseline: baseline_start, core window [-0.5, 0.0] + 1 s buffer
baseline = run_pipeline(
    build_event_locked_pipeline_config(
        Path("../data/sample_ieeg_bp.fif"),
        file_format="mne",
        event_name="baseline_start",
        event_times=beh["baseline_start"].tolist(),
        baseline_mode="none",
        tmin=-0.5,
        tmax=0.0,
        buffer_s=1.0,
        metadata={"reward": beh["reward"].tolist(), "rpe": beh["rpe"].tolist()},
    )
)

task.epochs.save(Path("../data/sample_feedback_start-epo.fif"), overwrite=True)
baseline.epochs.save(Path("../data/sample_baseline_start-epo.fif"), overwrite=True)
```

Alternative in-memory path: pass `baseline_event_times` in one `run_pipeline` call to carry baseline epochs without saving a second file. The book prefers two saved `-epo.fif` files so TFR never re-epochs from raw.

## How to inspect the result

- `len(result.epochs)` — expect 80
- `result.epochs.times[[0, -1]]` — includes 1 s buffer beyond core window (e.g. ~-1.5 to 2.5 for feedback)
- `result.epochs.metadata[["reward", "rpe"]].describe()`

## Common mistakes

- Passing milliseconds instead of seconds
- Using hardcoded demo times `[5.0, 10.0, 15.0]` instead of real `feedback_start` values
- Forgetting `metadata` keys must match `event_times` length
- Cropping voltage epochs before TFR (drop buffers on the TFR power axis in chapter 09 instead)
- Expecting voltage-domain baselining in this chapter (removed by design)

## Old-to-new translation note

This replaces the old `make_epochs(...)` pattern for creating event-locked epochs with metadata. Legacy-style normalization now occurs in chapter 09 at the TFR stage.

Next step: {doc}`08_first_psd_and_fooof`
