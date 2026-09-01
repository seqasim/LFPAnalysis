# Your First Time-Frequency Workflow

## What this step is for

Compute Morlet time-frequency representations on **saved** feedback and baseline epoch files from chapter 07, and visualize the **reward vs no-reward** difference using cross-event trialwise baseline normalization. Buffers are cropped on the TFR axis after Morlet, not on voltage epochs.

## When you should use it

Use this only after load, reference, artifact QC, and event-locked epoch setup are believable.

## Required inputs

- `../data/sample_feedback_start-epo.fif` — 80 feedback-locked raw epochs (buffered)
- `../data/sample_baseline_start-epo.fif` — 80 baseline-locked raw epochs (buffered)
- Frequency grid and wavelet parameters

## Minimal example

```python
import numpy as np
from pathlib import Path
from LFPAnalysis import load_lfp, run_analysis
from LFPAnalysis.config import AnalysisConfig, LoadConfig, TfrConfig

chan = "racas1-racas2"
freqs = np.arange(4, 30, 4).tolist()

task_epochs = load_lfp(
    LoadConfig(path=Path("../data/sample_feedback_start-epo.fif"), file_format="mne", preload=True)
)
baseline_epochs = load_lfp(
    LoadConfig(path=Path("../data/sample_baseline_start-epo.fif"), file_format="mne", preload=True)
)

result = run_analysis(
    task_epochs,
    AnalysisConfig(
        tfr=TfrConfig(
            enabled=True,
            method="morlet",
            freqs=freqs,
            n_cycles=3.0,
            baseline_mode="trialwise",
            apply_baseline=True,
            crop_tmin=-0.5,
            crop_tmax=1.5,
            baseline_crop_tmin=-0.5,
            baseline_crop_tmax=0.0,
        )
    ),
    baseline_epochs=baseline_epochs,
)

power = result.tfr["power"].copy().pick([chan])
reward_ix = np.where(power.metadata["reward"].to_numpy() == 1)[0]
loss_ix = np.where(power.metadata["reward"].to_numpy() == 0)[0]
reward_map = np.nanmean(power.data[reward_ix, 0], axis=0)
loss_map = np.nanmean(power.data[loss_ix, 0], axis=0)
diff = reward_map - loss_map
print("TFR shape:", power.data.shape)
print("TFR times:", power.times[0], power.times[-1])
```

The worked notebook ({doc}`worked-examples/09_first_tfr_run`) plots reward, no-reward, and difference heatmaps from this output.

## How to inspect the result

- Output shape before averaging: `(n_trials, n_channels, n_freqs, n_times)`
- Frequency axis matches your band of interest
- TFR time axis matches core analysis windows (~-0.5 to 1.5 s for feedback), not the buffered voltage span
- Difference map shows where reward and loss diverge

## Common mistakes

- Re-epoching from raw with `run_pipeline` on every TFR run instead of loading saved `-epo.fif` files
- Cropping voltage epochs with `Epochs.crop()` before `run_analysis` (drop buffers on TFR via `TfrConfig.crop_tmin` / `crop_tmax` instead)
- Forgetting the second baseline epoch file for cross-event normalization
- Using mismatched trial counts between task and baseline epoch files
- Assuming baselining should happen on voltage epochs first (it should not)

## Old-to-new translation note

Legacy parity in the stable path: raw buffered epochs saved in chapter 07, then cross-event trialwise TFR baseline in this chapter via `run_analysis(..., baseline_epochs=...)`.

Next step: {doc}`10_first_connectivity_and_surrogates`
