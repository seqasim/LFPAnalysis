# Your First Time-Frequency Workflow

## What this step is for

Compute Morlet time-frequency representations on real feedback epochs and visualize the **reward vs no-reward** difference using legacy-parity cross-event trialwise baseline normalization.

## When you should use it

Use this only after load, reference, artifact QC, and event-locked epoch setup are believable.

## Required inputs

- `../data/sample_ieeg_bp.fif`
- `../data/sample_beh.csv` with `feedback_start`, `baseline_start`, and `reward`
- Frequency grid and wavelet parameters

## Minimal example

```python
import numpy as np
import pandas as pd
from pathlib import Path
from LFPAnalysis import build_event_locked_pipeline_config, run_pipeline

beh = pd.read_csv(Path("../data/sample_beh.csv"))
chan = "racas1-racas2"
freqs = np.arange(4, 30, 4).tolist()

cfg = build_event_locked_pipeline_config(
    Path("../data/sample_ieeg_bp.fif"),
    file_format="mne",
    event_name="feedback_start",
    event_times=beh["feedback_start"].tolist(),
    baseline_mode="trialwise",
    baseline_event_times=beh["baseline_start"].tolist(),
    baseline_window=(-0.5, 0.0),
    tmin=-0.5,
    tmax=1.5,
    metadata={"reward": beh["reward"].tolist()},
    tfr_method="morlet",
    tfr_freqs=freqs,
    tfr_n_cycles=3.0,
)
result = run_pipeline(cfg)
power = result.tfr["power"].copy().pick([chan])
reward_ix = np.where(power.metadata["reward"].to_numpy() == 1)[0]
loss_ix = np.where(power.metadata["reward"].to_numpy() == 0)[0]
reward_map = np.nanmean(power.data[reward_ix, 0], axis=0)
loss_map = np.nanmean(power.data[loss_ix, 0], axis=0)
diff = reward_map - loss_map
print("TFR shape:", power.data.shape)
```

The worked notebook ({doc}`worked-examples/09_first_tfr_run`) plots reward, no-reward, and difference heatmaps from this output.

## How to inspect the result

- Output shape before averaging: `(n_trials, n_channels, n_freqs, n_times)`
- Frequency axis matches your band of interest
- Time axis aligns with the unbuffered analysis window
- Difference map shows where reward and loss diverge

## Common mistakes

- Forgetting to pass `baseline_event_times` and `baseline_window` for cross-event normalization
- Treating TFR as a first-step visualization instead of a later-stage summary
- Running TFR on all channels/trials without subsetting when iterating quickly
- Assuming baselining should happen on voltage epochs first (it should not)

## Old-to-new translation note

Legacy parity in the stable path is now: raw epochs in chapter 07, then cross-event trialwise TFR baseline in this chapter.

Next step: {doc}`10_first_connectivity_and_surrogates`
