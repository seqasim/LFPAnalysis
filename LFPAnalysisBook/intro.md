# LFPAnalysis Book

This book now has two explicit tracks.

## Beginner Track

Choose this path if you are new to the repository, new to MNE, or want the shortest route from a sample file to an interpretable result.

You will learn:

- which public interface to choose
- which files you need before starting
- how to load, reference, inspect, epoch, baseline, and summarize data
- where the stable workflow API ends and the advanced utilities begin

## Coming From The Old Repo

Choose this path if you previously worked from `scripts/Condensed Notebook.ipynb` or the older step-by-step notebooks.

You will learn:

- how the refactored repo is organized
- which old entry points now have compatibility shims
- how to translate the condensed notebook, TFR workflow, and connectivity workflow
- which workflows are still advanced or legacy-only

## Interface taxonomy

- `stable beginner-facing API`: typed configs, convenience builders, and `run_pipeline`
- `compatibility/legacy shims`: `LFPAnalysis.legacy`
- `advanced legacy utility modules`: `lfp_preprocess_utils`, `analysis_utils`, `oscillation_utils`, and related helpers

## Where to start

- New user: begin with {doc}`00_interface_guide` and {doc}`03_first_load`
- Returning old user: begin with {doc}`20_old_repo_mental_model`

```{tableofcontents}
```
