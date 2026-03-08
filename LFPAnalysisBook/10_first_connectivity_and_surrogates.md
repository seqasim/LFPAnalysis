# Your First Connectivity and Surrogate Workflow

## What this step is for

This chapter introduces the advanced connectivity surface and shows where surrogates fit into interpretation.

## When you should use it

Use this only after preprocessing and epoching are stable and you know exactly which signals and frequency bands you want to compare.

## Required inputs

- epoched or synthetic data
- analysis dependencies including `mne-connectivity`
- an explicit frequency-band choice

## Minimal example

```python
# Prepare data with the stable API, then continue with oscillation_utils or mne-connectivity.
```

## How to inspect the result

Inspect the output array shape, metric name, frequency window, and surrogate method before looking at values.

## Common mistakes

- using connectivity as a first-pass QA tool
- omitting the surrogate method from your notes
- assuming a stable API wrapper exists for every connectivity metric in the old notebooks

## Old-to-new translation note

Connectivity remains an advanced workflow. The migration chapters show how to translate old notebook code, but the stable API is intentionally not pretending to wrap the whole space yet.

Next step: {doc}`20_old_repo_mental_model`
