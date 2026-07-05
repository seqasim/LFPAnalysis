# Your First Synchronization

## What this step is for

Align the behavioral task clock to the neural recording clock using the photodiode and sync-pulse log. This step generalizes to any lab where Psychopy (or similar) timestamps must be mapped onto Neuralynx/MNE time.

## When you should use it

Use this after referencing (or loading) and **before** epoching behavioral events from a log file. In this bundled dataset, `sample_beh.csv` times are already in neural seconds, but the sync workflow is still worth learning because your own data will require it.

## Required inputs

- `../data/sample_photodiode.fif` — photodiode Raw, 1024 Hz
- `../data/sample_ts.csv` — behavioral sync pulses (`beh_ts` column, seconds)

## Minimal example

```python
import pandas as pd
import mne
from pathlib import Path
from LFPAnalysis import sync_utils

beh_ts = pd.read_csv(Path("../data/sample_ts.csv"))["beh_ts"].values
photodiode = mne.io.read_raw_fif(Path("../data/sample_photodiode.fif"), preload=True)

slope, offset = sync_utils.synchronize_data(
    beh_ts=beh_ts,
    mne_sync=photodiode,
    sync_source="photodiode",
)

# Transform a behavioral timestamp to neural time:
neural_time = slope * beh_ts[0] + offset
print(f"slope={slope:.4f}, offset={offset:.2f}")
print(f"First pulse: beh {beh_ts[0]:.2f} s → neural {neural_time:.2f} s")
```

The worked notebook plots matched pulse trains to visualize alignment quality.

## How to inspect the result

- `slope` should be close to 1.0 (same clock rate)
- `offset` captures the start-time difference between behavioral and neural recordings
- Apply `neural_time = beh_time * slope + offset` before epoching from a behavioral log

For this sample, `sample_beh.csv` event times already index correctly into the ~788 s neural recording (all trial times < recording duration).

## Common mistakes

- Using `sample_beh.csv` for sync instead of `sample_ts.csv` (behavior table is per-trial, not the full pulse train)
- Assuming sync pulses beyond the neural recording end are valid epoch anchors
- Skipping sync when your own behavioral log uses a different clock than the amplifier

## Old-to-new translation note

The Condensed Notebook called `sync_utils.synchronize_data` after loading the photodiode. The stable API does not wrap sync yet—use `sync_utils` directly, then pass transformed times to `build_event_locked_pipeline_config`.

Next step: {doc}`05_first_artifact_pass`
