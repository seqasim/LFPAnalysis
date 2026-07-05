# Your First Connectivity and Surrogate Workflow

## What this step is for

Estimate spectral connectivity between real ACC and frontal channels during feedback and compare against surrogate null distributions. Surrogates are part of the analysis design—not an afterthought.

## When you should use it

Use this after preprocessing and epoching are stable and you know which regions and frequency band to compare.

## Required inputs

- Feedback-locked epochs with metadata (`sample_feedback_start-epo.fif` + `sample_beh.csv`)
- `mne-connectivity` (included in `.[analysis]` / `.[dev]`)
- Seed/target channel pairs from `sample_labels_bp` (`racas` = ACC, `rmolf` = mid-frontal)

## Minimal example

```python
import pandas as pd
import numpy as np
from pathlib import Path
from mne_connectivity import spectral_connectivity_epochs
from LFPAnalysis import load_lfp
from LFPAnalysis.config import LoadConfig
from LFPAnalysis import oscillation_utils

beh = pd.read_csv(Path("../data/sample_beh.csv"))
epochs = load_lfp(LoadConfig(path=Path("../data/sample_feedback_start-epo.fif"), file_format="mne"))
epochs.metadata = beh[["reward", "rpe"]]

# ACC–frontal pair, beta band
epochs_sub = epochs.copy().pick(["racas1-racas2", "rmolf5-rmolf6"])
con = spectral_connectivity_epochs(
    epochs_sub, method="coh", mode="multitaper", fmin=13, fmax=30, faverage=True
)
print(con.get_data().shape)

# Surrogate null for one frequency slice
data_2d = epochs_sub.get_data()[:, 0, :]  # seed channel
surr = oscillation_utils.make_surrogate_arrays(
    data_2d, method="swap_epochs", n_shuffles=50, rng_seed=42, return_generator=False
)
```

The worked notebook plots connectivity and surrogate distributions.

## How to inspect the result

- Output shape and metric name (`coh`, `wpli`, etc.)
- Frequency window used
- Surrogate method and `rng_seed` recorded in your notes
- Compare real connectivity to surrogate distribution

## Common mistakes

- Using connectivity as a first-pass QA tool
- Omitting the surrogate method from analysis notes
- Assuming a stable API wrapper exists for every connectivity metric
- Running on all channel pairs without subsetting (expensive)

## Old-to-new translation note

Connectivity remains an advanced workflow. See {doc}`11_advanced_utility_interoperability` for how `oscillation_utils.compute_connectivity` wraps metrics and surrogates.

Next step: {doc}`10b_first_time_resolved_stats`
