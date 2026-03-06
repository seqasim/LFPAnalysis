# Connectivity and Surrogate Analysis

Connectivity analyses are powerful but easy to misuse. Treat them as a later-stage analysis after your preprocessing contract is stable.

## Repository support

- connectivity utilities in `LFPAnalysis.oscillation_utils`
- surrogate generation helpers such as `make_surrogate_arrays`
- optional `mne-connectivity`-based workflows through the analysis extras

## Good practice

- keep frequency bands explicit
- record surrogate method and random seed
- inspect intermediate shapes before trusting a metric
- start with a synthetic or toy example before using real patient data
