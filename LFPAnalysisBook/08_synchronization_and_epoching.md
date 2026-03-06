# Synchronization and Epoching

Behavioral alignment is usually the most fragile step in a new dataset.

## Workflow support

The stable API accepts behavioral timestamps through `EpochConfig` and converts them into `mne.Epochs` objects.

## Current assumptions

- event timestamps are in seconds
- timestamps can be linearly transformed with `slope` and `offset`
- events represent a single event family at a time

## Common mistakes checklist

- using milliseconds in one file and seconds in another
- applying slope/offset twice
- epoching before confirming the sync channel and file provenance
- forgetting to exclude trials with missing behavior timestamps
