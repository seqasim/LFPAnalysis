# Your First PSD and FOOOF Workflow

## What this step is for

Compute power spectral density and optional FOOOF fits on real feedback-locked epochs, then compare **reward vs no-reward** spectra. This is the first spectral contrast in the case study.

## When you should use it

Use this after epoching with behavioral metadata (chapter 07) or load the pre-built `sample_feedback_start-epo.fif` and attach metadata manually.

## Required inputs

- `../data/sample_feedback_start-epo.fif` — 80 feedback epochs, 15 bipolar channels
- `../data/sample_beh.csv` — for reward labels if attaching metadata
- `analysis` dependencies for FOOOF (`pip install -e .[analysis]`)

## Minimal example

```python
import pandas as pd
from pathlib import Path
from LFPAnalysis import build_analysis_config, load_lfp, run_analysis
from LFPAnalysis.config import LoadConfig

beh = pd.read_csv(Path("../data/sample_beh.csv"))
epochs = load_lfp(
    LoadConfig(path=Path("../data/sample_feedback_start-epo.fif"), file_format="mne", preload=True)
)
epochs.metadata = beh[["reward", "rpe"]]
chan = "racas1-racas2"
reward = run_analysis(
    epochs["reward == 1"].copy().pick([chan]),
    build_analysis_config(spectral_method="psd", fmin=1.0, fmax=80.0),
)
loss = run_analysis(
    epochs["reward == 0"].copy().pick([chan]),
    build_analysis_config(spectral_method="psd", fmin=1.0, fmax=80.0),
)
print(reward.spectral["spectrum"].get_data().shape)
```

For a single end-to-end spectral pipeline on a file path (without a reward contrast), use `build_spectral_pipeline_config` + `run_pipeline`. The worked notebook ({doc}`worked-examples/08_first_psd_and_fooof_run`) plots reward vs loss PSD and runs FOOOF through the same analysis spine.

## How to inspect the result

- PSD frequency range and channel count via `result.spectral["spectrum"]`
- Visual comparison of reward vs no-reward spectra (not just shape prints)
- FOOOF table (`result.spectral["table"]`) columns include `frequency`, `PSD_raw`, `PSD_corrected`, `PSD_exp`, plus peak/channel helpers

## Common mistakes

- Loading with default `preload=False` then calling `.pick()` / spectral helpers that need in-memory data — pass `preload=True` (or call `epochs.load_data()`)
- Trying FOOOF before confirming the PSD looks sensible
- Assuming TFR and connectivity are covered by `build_spectral_pipeline_config`
- Forgetting to install `analysis` extras for FOOOF

## Old-to-new translation note

Old notebooks jumped directly into PSD/FOOOF functions. The refactored path encourages `build_analysis_config` / `run_analysis` (or `build_spectral_pipeline_config` / `run_pipeline`) so results live in a typed container first.

Next step: {doc}`09_first_time_frequency`
