# Your First Time-Frequency Workflow

## What this step is for

Compute Morlet time-frequency representations on real feedback epochs and visualize the **reward vs no-reward** difference. Beginner Morlet TFR is available on the stable analysis spine via `TfrConfig` / `run_analysis`.

## When you should use it

Use this only after load, reference, artifact QC, epoching, and baseline choices are believable.

## Required inputs

- `../data/sample_feedback_start-epo.fif` with `reward` metadata attached
- Frequency grid and wavelet parameters

## Minimal example

```python
import pandas as pd
import numpy as np
from pathlib import Path
from LFPAnalysis import build_analysis_config, load_lfp, run_analysis
from LFPAnalysis.config import LoadConfig

beh = pd.read_csv(Path("../data/sample_beh.csv"))
epochs = load_lfp(
    LoadConfig(path=Path("../data/sample_feedback_start-epo.fif"), file_format="mne", preload=True)
)
epochs.metadata = beh[["reward", "rpe"]]

chan = "racas1-racas2"
freqs = np.arange(4, 30, 4).tolist()
cfg = build_analysis_config(tfr_method="morlet", tfr_freqs=freqs, tfr_n_cycles=3.0)
reward = run_analysis(epochs["reward == 1"].copy().pick([chan]), cfg)
loss = run_analysis(epochs["reward == 0"].copy().pick([chan]), cfg)
diff = reward.tfr["power"].average().data - loss.tfr["power"].average().data
print(reward.tfr["power"].data.shape)
```

The worked notebook ({doc}`worked-examples/09_first_tfr_run`) plots TFR heatmaps and the reward-minus-loss difference.

## How to inspect the result

- Output shape after averaging: `(n_channels, n_freqs, n_times)`
- Frequency axis matches your band of interest
- Time axis aligns with the epoch window and event onset at 0 s
- Difference map shows where reward and loss diverge

## Common mistakes

- Loading with default `preload=False` then calling `.pick()` / TFR helpers that need in-memory data — pass `preload=True` (or call `epochs.load_data()`)
- Treating TFR as a first-step visualization instead of a later-stage summary
- Running TFR on all 15 channels × 80 trials without subsetting (slow in CI)
- Assuming Morlet TFR is legacy-only — beginner Morlet is on `run_analysis`

## Old-to-new translation note

Beginner Morlet TFR is available via `build_analysis_config(..., tfr_method="morlet")` / `run_analysis`. For full legacy orchestration, use `epochs.compute_tfr` or `lfp_preprocess_utils.compute_and_baseline_tfr`.

Next step: {doc}`10_first_connectivity_and_surrogates`
