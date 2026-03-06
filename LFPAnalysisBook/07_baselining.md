# Baselining Strategies

The stable workflow API treats baselining as a configurable stage rather than an implicit analysis detail.

## Supported baseline modes

- `none`
- `mean`
- `ratio`
- `percent`
- `zscore`
- `logratio`
- `zlogratio`
- `trialwise`
- `continuous`

## Choosing a baseline scheme

### `zscore`

Good default for many quick comparisons when a stable baseline window exists.

### `trialwise`

Best when each epoch has its own appropriate baseline period.

### `continuous`

Useful when working with long recordings outside an epoch-first workflow.

## Beginner rule of thumb

Do not baseline by habit. Pick a baseline mode because it matches the experimental design and the quantity you plan to interpret.

## Standard diagnostics

The workflow returns a per-channel baseline summary table so you can inspect baseline mean and spread before trusting downstream plots.
