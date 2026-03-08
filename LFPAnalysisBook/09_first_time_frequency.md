# Your First Time-Frequency Workflow

## What this step is for

This chapter covers the first point where most users move beyond the stable beginner-facing API into advanced utilities.

## When you should use it

Use this only after you trust your load, reference, artifact, epoch, and baseline choices.

## Required inputs

- epoched data
- a frequency grid and transform choice

## Minimal example

```python
# Start from the stable API for preparation, then continue with MNE or advanced utilities.
```

## How to inspect the result

Check the resulting shape, frequency axis, time axis, and whether the transform aligns with your event window.

## Common mistakes

- treating TFR as a first-step visualization instead of a later-stage summary
- forgetting that time-frequency baselines depend on the same event design logic as time-domain baselines
- assuming the stable API currently wraps the whole TFR story

## Old-to-new translation note

The refactored repo still documents TFR as an advanced workflow. The migration chapters show how to translate the old TFR notebook patterns without pretending the stable API fully replaces them yet.

Next step: {doc}`10_first_connectivity_and_surrogates`
