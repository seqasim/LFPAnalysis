# Plotting Recipes

## What this step is for

The library ships few dedicated plotting helpers. Most figures are built with matplotlib, seaborn, and MNE on top of arrays or tidy dataframes. This chapter collects **recipes** you will reuse after computing metrics: single-channel TFR, time-resolved regression traces, connectivity heatmaps, and per-ROI / per-patient summaries.

## When you should use it

Use this after you have either (a) arrays from spectral / connectivity utilities or (b) tidy / stats tables from chapters 13–14. Pair with chapter 15 if you want to save figures under `results/.../figures/`.

## Required inputs

- For `plot_TFR`: a 2D array (frequencies × times), frequency vector, sampling rate, and pre/post window lengths in seconds
- For regression traces: columns `ts` and `z_beta` (or `raw_beta`) from chapter 14
- For connectivity: a square (or seed×target) matrix
- For ROI summaries: a long dataframe with a metric plus `roi` / `participant`

## Recipes

### 1. Single-channel TFR (`analysis_utils.plot_TFR`)

```python
import numpy as np
from LFPAnalysis import analysis_utils

# Demo array: freqs x times, z-scored-like values in [-3, 3]
freqs = np.logspace(np.log10(2), np.log10(100), 40)
n_times = 500
sr = 500.0
pre_win, post_win = 0.5, 1.0
data = np.random.randn(len(freqs), n_times) * 0.5
data[10:15, 200:350] += 2.0  # fake burst after onset

fig = analysis_utils.plot_TFR(
    data, freqs, pre_win=pre_win, post_win=post_win, sr=sr, title="demo TFR"
)
fig.show()
```

`plot_TFR` uses a fixed `RdBu_r` scale of ±3 (typical for baseline z-scored power). Adjust your data scaling to match, or copy the function and change `vmin`/`vmax` for raw power.

### 2. Time-resolved z-beta trace (stats output)

```python
import matplotlib.pyplot as plt
import pandas as pd

# mlm_res from chapter 14; filter to the predictor of interest
trace = mlm_res.loc[mlm_res.parameter == "rpe", ["ts", "z_beta", "z_p"]].sort_values("ts")

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(trace["ts"], trace["z_beta"], "o-")
ax.axvline(0, color="k", ls="--", lw=0.8)
ax.axhline(0, color="0.5", lw=0.5)
sig = trace["z_p"] < 0.05
ax.scatter(trace.loc[sig, "ts"], trace.loc[sig, "z_beta"], color="C3", zorder=3, label="z_p < 0.05")
ax.set(xlabel="Time (s)", ylabel="z-beta", title="RPE effect over time")
ax.legend()
fig.tight_layout()
```

Worked-example {doc}`worked-examples/10b_first_stats_run` plots the same idea for a single channel via repeated `permutation_regression_zscore` calls.

### 3. Connectivity matrix heatmap

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Replace with your metric matrix (channels x channels or seeds x targets)
rng = np.random.default_rng(1)
mat = rng.normal(size=(6, 6))
mat = (mat + mat.T) / 2
labels = [f"ch{i}" for i in range(6)]

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(mat, xticklabels=labels, yticklabels=labels, cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Connectivity (demo)")
fig.tight_layout()
```

For ROI-averaged connectivity, average the relevant seed/target index sets from `make_seed_target_df` (chapter 12) before plotting.

### 4. Per-ROI / per-patient strip + box summary

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

rng = np.random.default_rng(2)
summary = pd.DataFrame(
    {
        "participant": np.repeat(["P01", "P02", "P03"], 8),
        "roi": np.tile(np.repeat(["ACC", "OFC"], 4), 3),
        "metric": rng.normal(size=24),
    }
)

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=summary, x="roi", y="metric", color="lightgray", ax=ax)
sns.stripplot(data=summary, x="roi", y="metric", hue="participant", dodge=True, ax=ax)
ax.axhline(0, color="k", lw=0.8)
ax.set_title("Electrode-level metric by ROI")
fig.tight_layout()
```

This is the visual companion to the “region effect across patients” discussion in chapter 14.

## How to inspect the result

- TFR: onset line at `t=0`, frequency labels readable, color scale matched to baseline units
- Regression traces: zero lines for time and effect size; mark significance transparently (`z_p` vs `count_p`)
- Heatmaps: symmetric color limits when the metric is signed; labeled axes
- ROI strips: points colored by participant so one patient cannot silently dominate the plot

## Common mistakes

- Expecting a rich plotting API — outside `plot_TFR` and FOOOF’s `do_plot`, you own the figure code
- Plotting raw power with `plot_TFR`’s ±3 clim without z-scoring
- Averaging electrodes into an ROI mean **before** checking per-patient spread
- Saving figures without the tidy table that produced them (chapter 15)

## Old-to-new translation note

Legacy notebooks mixed custom plotting with analysis cells. Keep recipes in a shared notebook or script, and keep statistics tables tidy so the same plot code works for power, FOOOF peaks, or connectivity summaries.

Next step: {doc}`20_old_repo_mental_model`
