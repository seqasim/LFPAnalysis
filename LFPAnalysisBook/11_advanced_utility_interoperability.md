# Advanced Utility Interoperability

## What this step is for

This chapter explains how the advanced utility modules fit together once you leave the stable pipeline surface. The goal is not to replace `run_pipeline`. The goal is to make the lower-level utility stack predictable when you need direct control.

## When you should use it

Use this chapter when the stable API is not yet deep enough for your workflow and you need to combine file parsing, synchronization, baselining, PSD/TFR, connectivity, or custom statistics directly.

## Required inputs

- channel metadata or connection tables if you are loading Neuralynx data
- neural signals already loaded or ready to load
- behavioral timestamps if you need synchronization or event-locked analyses
- analysis dependencies if you are using FOOOF or connectivity helpers

## Minimal example

```python
from LFPAnalysis import iowa_utils, nlx_utils, sync_utils, statistics_utils
from LFPAnalysis import lfp_preprocess_utils, analysis_utils, oscillation_utils

eeg_names, resp_names, ekg_names, seeg_names, drop_names = iowa_utils.extract_names_connect_table(
    "subject_connect_table.csv"
)

signals, srs, ch_names, ch_types = nlx_utils.parse_subject_nlx_data(
    ncs_files,
    eeg_names=eeg_names,
    resp_names=resp_names,
    ekg_names=ekg_names,
    seeg_names=seeg_names,
    drop_names=drop_names,
)

slope, offset = sync_utils.synchronize_data(beh_ts=beh_ts, mne_sync=sync_source)

tfr_z = lfp_preprocess_utils.baseline_avg_TFR(tfr_data, baseline_tfr, mode="zscore")
surrogates = oscillation_utils.make_surrogate_arrays(
    tfr_z[0],
    method="swap_time_blocks",
    n_shuffles=100,
    rng_seed=42,
    return_generator=False,
)

fooof_group, fooof_table = analysis_utils.FOOOF_compute_epochs(epochs, tmin=0.0, tmax=1.0, **fooof_kwargs)
stats = statistics_utils.permutation_regression_zscore(model_df, "y ~ condition", n_permutations=1000)
```

## How to inspect the result

Inspect each handoff, not just the final output:

- after `iowa_utils` and `nlx_utils`: verify `ch_names` and `ch_types`
- after `sync_utils`: verify slope, offset, and timestamp alignment on a few known events
- after `lfp_preprocess_utils`: verify array shape and baseline mode before downstream analysis
- after `analysis_utils` or `oscillation_utils`: verify frequency ranges, surrogate method, and random seed
- after `statistics_utils`: verify the model formula and surrogate count before reading z-scores

## Shared conventions after cleanup

- Channel names coming out of `iowa_utils` and `nlx_utils` are intended to match directly through lowercase `lfpx...` naming.
- `nlx_utils` uses warnings for skipped channels so library calls stay scriptable and log-friendly.
- `lfp_preprocess_utils` baseline helpers rely on broadcasting instead of materializing repeated arrays, so you can pass their outputs directly to downstream NumPy-based utilities.
- `oscillation_utils` surrogate helpers keep deterministic behavior through `rng_seed`.
- `analysis_utils` ROI lookups cache the packaged atlas table instead of re-reading it on every call.

## Common mistakes

- mixing the stable API and utility modules without checking which object type each helper expects
- assuming channel labels are already normalized before Iowa or Neuralynx parsing
- reshaping baseline outputs manually before trying surrogate helpers
- treating connectivity utilities as if they were part of the stable public API
- forgetting to record the surrogate method and random seed in analysis notes

## Old-to-new translation note

If your old notebooks chained utility calls directly, keep doing that only where the stable API still does not cover the workflow. For load, reference, and epoch setup, prefer the stable API or `LFPAnalysis.legacy`. For PSD, TFR, connectivity, and custom statistics, the advanced utilities are still the correct surface, but they now share stricter conventions and tests.

Next step: {doc}`20_old_repo_mental_model`
