# Your First Time-Frequency Workflow

## What this step is for

Compute Morlet time-frequency representations on real feedback epochs and visualize the **reward vs no-reward** difference. This is where most users move beyond the stable API into MNE and advanced utilities.

## When you should use it

Use this only after load, reference, artifact QC, epoching, and baseline choices are believable.

## Required inputs

- `../data/sample_feedback_start-epo.fif` with `reward` metadata attached
- Frequency grid and wavelet parameters

## Minimal example

```python
import pandas as pd
import mne
import numpy as np
from pathlib import Path
from LFPAnalysis import load_lfp
from LFPAnalysis.config import LoadConfig

beh = pd.read_csv(Path("../../data/sample_beh.csv"))
epochs = load_lfp(LoadConfig(path=Path("../../data/sample_feedback_start-epo.fif"), file_format="mne"))
epochs.metadata = beh[["reward", "rpe"]]

# Subset for speed: one ACC channel, beta band
chan = "racas1-racas2"
freqs = np.arange(4, 30, 4)
reward_tfr = epochs["reward == 1"].copy().pick([chan]).compute_tfr(
    method="morlet", freqs=freqs, n_cycles=freqs / 2.0, average=True
)
loss_tfr = epochs["reward == 0"].copy().pick([chan]).compute_tfr(
    method="morlet", freqs=freqs, n_cycles=freqs / 2.0, average=True
)
diff = reward_tfr.data - loss_tfr.data
```

The worked notebook plots TFR heatmaps and the reward-minus-loss difference.

## How to inspect the result

- Output shape: `(n_freqs, n_times)` after averaging
- Frequency axis matches your band of interest
- Time axis aligns with `tmin`/`tmax` and event onset at 0 s
- Difference map shows where reward and loss diverge

## Common mistakes

- Treating TFR as a first-step visualization instead of a later-stage summary
- Running TFR on all 15 channels × 80 trials without subsetting (slow in CI)
- Assuming the stable API currently wraps the whole TFR pipeline

## Old-to-new translation note

TFR remains an advanced workflow. Use the stable API for preparation, then `epochs.compute_tfr` or `lfp_preprocess_utils.compute_and_baseline_tfr` for full legacy pipelines.

Next step: {doc}`10_first_connectivity_and_surrogates`
