# What Remains Legacy-Only or Advanced

## Still advanced or legacy-oriented

- full time-frequency orchestration
- connectivity orchestration across the many legacy options
- reserved-but-incomplete stable entries such as `laplacian`
- notebook-specific side effects such as writing intermediate artifact CSVs to disk

## How to choose responsibly

- use the stable API first for preparation
- use `LFPAnalysis.legacy` if you are actively translating old code
- drop to advanced utilities only after you know why you need them

## Side-by-side example

#### Old workflow

```python
lfp_preprocess_utils.compute_and_baseline_tfr(...)
```

#### New workflow

```python
from LFPAnalysis import legacy
legacy.compute_and_baseline_tfr(...)
```

## Why this is explicit now

The refactored repo aims to reduce ambiguity. It is better to label a workflow as advanced or compatibility-only than to imply that the stable API already covers it.

Next step: {doc}`30_troubleshooting`
