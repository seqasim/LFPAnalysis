# Your First Event-Locked Workflow

## What this step is for

Build **feedback-locked epochs** from real trial times in `sample_beh.csv` and attach `reward`, `rpe`, and `gamble_rt` as epoch metadata. This enables condition contrasts (`reward == 1` vs `reward == 0`) for every downstream chapter.

## When you should use it

Use this when you have behavioral timestamps and want a clean event-locked object before PSD, TFR, connectivity, or statistics.

## Required inputs

- `../data/sample_ieeg_bp.fif` — bipolar continuous (times already in neural seconds)
- `../data/sample_beh.csv` — `feedback_start`, `reward`, `rpe`, `gamble_rt`

## Minimal example

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
    baseline_mode="zscore",
    baseline_window=(-0.5, 0.0),
    tmin=-0.5,
    tmax=1.5,
    metadata={
        "reward": beh["reward"].tolist(),
        "rpe": beh["rpe"].tolist(),
        "gamble_rt": beh["gamble_rt"].tolist(),
    },
)
result = run_pipeline(config)

reward_epochs = result.epochs[result.epochs.metadata["reward"] == 1]
loss_epochs = result.epochs[result.epochs.metadata["reward"] == 0]
print(f"Reward trials: {len(reward_epochs)}, Loss trials: {len(loss_epochs)}")
```

The worked notebook ({doc}`worked-examples/07_first_epoching_run`) plots the evoked average for reward vs no-reward on one ACC channel.

## How to inspect the result

- `len(result.epochs)` — expect 80
- `result.epochs.times[[0, -1]]` — pre/post window
- `result.epochs.metadata[["reward", "rpe"]].describe()`
- `result.baseline_summary.head()`

## Common mistakes

- Passing milliseconds instead of seconds
- Using hardcoded demo times `[5.0, 10.0, 15.0]` instead of real `feedback_start` values
- Forgetting `metadata` dict keys must have the same length as `event_times`
- Expecting the stable API to write artifact CSV sidecars to disk

## Old-to-new translation note

This is the stable replacement for the old `make_epochs(...)` pattern when you need epochs with behavioral metadata, not every legacy side effect.

Next step: {doc}`08_first_psd_and_fooof`
