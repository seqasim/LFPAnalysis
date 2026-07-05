# Your First Time-Resolved Statistics

## What this step is for

Culmination of the case study: regress feedback-locked band power against **reward prediction error (RPE)** across time using permutation-based inference. This connects preprocessing choices to a scientific result.

## When you should use it

Use this after TFR or evoked extraction when you have trial-level power estimates and continuous behavioral regressors in epoch metadata.

## Required inputs

- Feedback epochs with `rpe` in metadata
- Band-power estimates per trial and time point
- `statsmodels` (included in `.[analysis]` / `.[dev]`)

## Minimal example

```python
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from LFPAnalysis import statistics_utils

np.random.seed(42)  # statistics_utils permutations are not seeded internally

beh = pd.read_csv(Path("../data/sample_beh.csv"))
epochs = mne.read_epochs(Path("../data/sample_feedback_start-epo.fif"), preload=True)
epochs.metadata = beh[["reward", "rpe", "gamble_rt"]]

# Extract beta-band power per trial at feedback (simplified single timepoint):
chan = "racas1-racas2"
ep = epochs.copy().pick([chan]).filter(13, 30)
power = np.mean(ep.get_data()[:, 0, :] ** 2, axis=1)

model_df = pd.DataFrame({"power": power, "rpe": epochs.metadata["rpe"].values})
results = statistics_utils.permutation_regression_zscore(
    model_df, "power ~ rpe", n_permutations=200
)
print(results)
```

The worked notebook extends this to multiple time points and plots the z-scored beta trace over time.

## How to inspect the result

- `raw_beta` and `z_beta` for the `rpe` predictor
- `z_p` from the permutation null
- Time-resolved plot: does the RPE effect cluster after feedback onset?

**Reproducibility note:** `statistics_utils` uses unseeded `np.random.permutation`. Set `np.random.seed(...)` before calling for reproducible documentation builds.

## Common mistakes

- Regressing without z-scoring or checking collinearity when adding multiple predictors
- Using too few permutations for stable z-scores (200 is a CI budget; use 1000+ for publication)
- Forgetting that `rpe` is continuous while `reward` is binary—choose the regressor that matches your hypothesis

## Old-to-new translation note

The old `TimeResolvedRegression.ipynb` chained utility calls directly. This chapter uses the same `statistics_utils` surface with the stable-prepared epochs from the case study.

Next step: {doc}`11_advanced_utility_interoperability`
