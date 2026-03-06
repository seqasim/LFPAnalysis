# Referencing Strategies

Referencing is one of the highest-impact preprocessing choices in the repository.

## Supported workflow-level options

- `none`: keep the original signal
- `wm`: white-matter-style re-referencing using legacy utilities
- `bipolar`: bipolar re-referencing using adjacent contacts
- `laplacian`: reserved in the workflow registry, but not yet implemented in the legacy code path

## Choosing a reference scheme

### Start with `none` when

- you are validating file loading
- you are checking synchronization or metadata only

### Start with `bipolar` when

- you want a pragmatic default for depth electrodes
- you have ordered contacts and a reliable localization table

### Start with `wm` when

- your lab already uses white-matter reference contacts consistently

## Practical guidance

Make the reference choice explicit in the pipeline config and document it in analysis outputs. Hidden reference decisions are a common reproducibility failure.
