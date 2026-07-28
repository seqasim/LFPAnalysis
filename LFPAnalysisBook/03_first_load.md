# Your First Successful Load

## What this step is for

Load the real 22-channel gambling-task recording and confirm the object contract before any preprocessing. This is the first end-to-end success checkpoint on the case-study path.

## When you should use it

Use this before referencing, synchronization, or epoching. If loading is unclear, every downstream choice becomes harder.

## Required inputs

- `../data/sample_ieeg.fif` — monopolar sEEG, 500 Hz, channels `rmolf1`–`rmolf12` and `racas1`–`racas10`

## Minimal example

```python
from pathlib import Path
from LFPAnalysis import WORKING_DTYPE, build_basic_pipeline_config, run_pipeline

config = build_basic_pipeline_config(Path("../data/sample_ieeg.fif"), file_format="mne")
# Defaults favor local machines: preload=False (lazy/disk-backed) and float32 arrays.
# Pass preload=True to the builder (or LoadConfig(..., preload=True)) when helpers need RAM.
result = run_pipeline(config)

print(f"Sampling rate: {result.raw.info['sfreq']} Hz")
print(f"Channels ({len(result.raw.ch_names)}): {result.raw.ch_names[:5]} …")
print(f"Duration: {result.raw.n_times / result.raw.info['sfreq']:.1f} s")
print(f"Working dtype: {WORKING_DTYPE}")
```

The worked notebook plots a short raw trace and a sanity-check PSD for one channel.

## How to inspect the result

Check:

- `result.raw.info["sfreq"]` — expect 500 Hz
- `result.raw.ch_names` — 22 lowercase sEEG labels
- `result.metadata` — records `input_format`, `reference_method`, `working_dtype`, `preload`, etc.
- Recording duration — ~788 s
- When epoching is enabled, `result.raw` / `result.referenced` may be `None` so peak RAM stays lower; use `result.epochs` as the working object.

## Common mistakes

- Using `sample_ieeg_continuous_rest.fif` when you want the full task recording (that 4-channel file is a lightweight smoke-test clip, not the gambling task)
- Choosing the wrong `file_format`
- Trying to reference channels before confirming the loaded channel list
- Expecting `float64` arrays — the stable workflow now defaults to `float32`

## Old-to-new translation note

If you previously started by calling `make_mne`, the new recommended first step is `build_basic_pipeline_config` plus `run_pipeline`.

## Worked example and smoke checks

Read the worked notebook for plots, then use the smoke notebooks when you only need to verify the environment.

Next step: {doc}`04_first_reference`
