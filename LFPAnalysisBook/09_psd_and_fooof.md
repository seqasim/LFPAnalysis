# PSD and FOOOF

Spectral summaries are often the first analysis users want after preprocessing.

## Stable API support

- `SpectralConfig(method="psd")` computes PSDs through the workflow layer
- FOOOF remains available through the legacy analysis utilities and is wrapped for epoched data in the stable layer

## When to use PSD first

Start with PSD when you want to answer simple questions about line noise, broad-band power, or whether the preprocessing pipeline is behaving sensibly.

## When to add FOOOF

Add FOOOF only after the PSD looks plausible and you have installed the `analysis` extras.
