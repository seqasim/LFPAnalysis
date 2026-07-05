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
from LFPAnalysis import load_lfp
from LFPAnalysis.config import LoadConfig

beh = pd.read_csv(Path("../data/sample_beh.csv"))
epochs = load_lfp(LoadConfig(path=Path("../data/sample_feedback_start-epo.fif"), file_format="mne"))
epochs.metadata = beh[["reward", "rpe"]]
chan = "racas1-racas2"
reward_psd = epochs["reward == 1"].copy().pick([chan]).compute_psd(fmin=1, fmax=80)
loss_psd = epochs["reward == 0"].copy().pick([chan]).compute_psd(fmin=1, fmax=80)
print(reward_psd.get_data().shape)
```

The worked notebook plots reward vs loss PSD for one channel and runs FOOOF via `analysis_utils`.

## How to inspect the result

- PSD frequency range and channel count
- Visual comparison of reward vs no-reward spectra (not just shape prints)
- FOOOF table columns: `frequency`, `PSD_raw`, `PSD_corrected`, `PSD_exp`

## Common mistakes

- Trying FOOOF before confirming the PSD looks sensible
- Assuming TFR and connectivity are covered by `build_spectral_pipeline_config`
- Forgetting to install `analysis` extras for FOOOF

## Old-to-new translation note

Old notebooks jumped directly into PSD/FOOOF functions. The refactored path encourages a valid `PipelineResult` first, then advanced utilities where needed.

Next step: {doc}`09_first_time_frequency`
