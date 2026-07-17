# Troubleshooting and FAQ

## ImportError: No module named 'LFPAnalysis'

This usually means the package was installed into one interpreter while Jupyter (or VS Code) is running another.

1. Check which interpreter is active:

```bash
python -c "import sys; print(sys.executable)"
```

Confirm the path is under the `LFPAnalysis` conda env (for example `.../envs/LFPAnalysis/bin/python`), not base Anaconda.

2. In notebooks, confirm the selected kernel is **LFPAnalysis**. If that kernel is missing, register it and restart Jupyter:

```bash
conda activate LFPAnalysis
python -m ipykernel install --user --name LFPAnalysis --display-name "LFPAnalysis"
```

3. If `import LFPAnalysis` works from the repo folder but fails from elsewhere (for example `cd ~`), the editable install is broken or masked by a stale `LFPAnalysis.egg-info/`. Fix it in the active env:

```bash
cd /path/to/LFPAnalysis
rm -rf LFPAnalysis.egg-info
python -m pip install -e .
cd ~
python -c "import sys, LFPAnalysis; print(sys.executable); print(LFPAnalysis.__file__)"
```

## Missing dependencies

If you see missing-module errors for `mne`, `fooof`, or `mne_connectivity`, install the relevant extras instead of guessing.

- beginner path: `pip install -e .[dev]`
- analysis-heavy path: `pip install -e .[analysis]`

## Mismatched channel labels

If referencing fails, validate the electrode table first. Most label problems are metadata problems, not signal problems.

`match_elec_names` no longer hangs in notebooks/CI by default: ambiguous Levenshtein ties raise `ValueError` listing candidates. Only pass `interactive=True` when you are at a human terminal and want a prompt.

## Reference method confusion

Use `none` for first-load inspection, `bipolar` when you have ordered contacts, and `wm` when your lab already has a white-matter convention.

## Baseline window confusion

A baseline window must overlap the data you are baselining. If it does not, the stable API now fails clearly instead of silently guessing.

## Why did this return empty tables?

An empty artifact table usually means either the detector was `none` or the chosen detector did not find any events under the configured thresholds.

## Why is there both a workflow API and legacy utilities?

Because the repo is in a staged transition. The stable API exists for approachability and reproducibility. The legacy layer exists so existing users can translate workflows without rewriting everything at once.

## Common mistakes

- skipping the interface guide
- reading advanced notebooks before succeeding with the first-load workflow
- assuming a compatibility shim means the stable API already has feature parity
- selecting the wrong Jupyter kernel after a successful conda install

Next step: revisit {doc}`00_interface_guide` if you still are not sure which surface to use.
