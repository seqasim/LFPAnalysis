# LFPAnalysis

LFPAnalysis is a Python toolkit for turning human intracranial and local field potential recordings into MNE-native objects, preprocessing them reproducibly, synchronizing them with behavioral data, and running downstream analyses such as referencing, baselining, PSD/FOOOF, time-frequency analysis, connectivity, and statistics.

## Who this is for

- total beginners who want a guided path from a sample file to an interpretable result
- existing users of the old notebooks who need a concrete migration path
- advanced users who still want direct access to the underlying utility modules

## Public interface choices

### Stable beginner-facing API

Use this first.

```python
from pathlib import Path
from LFPAnalysis import build_basic_pipeline_config, run_pipeline

config = build_basic_pipeline_config(Path("data/sample_ieeg.fif"), file_format="mne")
result = run_pipeline(config)
```

### Compatibility shims for old notebook users

Use `LFPAnalysis.legacy` when translating old code incrementally.

```python
from LFPAnalysis import legacy

# Emits a DeprecationWarning with the new equivalent.
epochs = legacy.make_epochs(
    load_path="data/sample_ieeg_continuous_rest.fif",
    behav_name="demo",
    behav_times=[5.0, 10.0, 15.0],
    ev_start_s=0.5,
    ev_end_s=1.0,
)
```

### Advanced utility modules

Use `lfp_preprocess_utils`, `analysis_utils`, and `oscillation_utils` when the stable API does not yet cover your workflow, especially for time-frequency and connectivity analyses.

## Which entry point should I use?

- I just want to load sample data: `build_basic_pipeline_config(...)`
- I want to epoch task events: `build_event_locked_pipeline_config(...)`
- I want PSD or FOOOF: `build_spectral_pipeline_config(...)`
- I used the old notebooks: `LFPAnalysis.legacy` plus the migration chapters in `LFPAnalysisBook/`

## Install

### Contributor install

```bash
git clone https://github.com/seqasim/LFPAnalysis.git
cd LFPAnalysis
pip install -e .[dev]
```

### Analysis-heavy install

```bash
pip install -e .[analysis]
```

### Conda install

```bash
git clone https://github.com/seqasim/LFPAnalysis.git
cd LFPAnalysis
conda env create -f environment.yml
conda activate LFPAnalysis
pip install -e .
```

## Beginner quickstart

```python
from pathlib import Path
from LFPAnalysis import build_event_locked_pipeline_config, run_pipeline

config = build_event_locked_pipeline_config(
    Path("data/sample_ieeg_continuous_rest.fif"),
    file_format="mne",
    event_name="demo",
    event_times=[5.0, 10.0, 15.0],
    baseline_mode="zscore",
    baseline_window=(-0.5, 0.0),
)
result = run_pipeline(config)
print(result.epochs)
print(result.baseline_summary.head())
```

## Documentation tracks

The canonical onboarding path is `LFPAnalysisBook/`.

- Beginner Track: first load, first reference, first artifact pass, first baseline, first event-locked workflow, first PSD/TFR/connectivity workflow
- Coming From The Old Repo: side-by-side translations for the condensed notebook, TFR, and connectivity workflows

## What is stable vs transitional

### Stable

- typed config dataclasses
- convenience builders
- `run_pipeline`
- standardized artifact and baseline outputs

### Transitional

- compatibility wrappers in `LFPAnalysis.legacy`
- migration documentation for old notebook users

### Still advanced

- most time-frequency orchestration
- most connectivity orchestration
- reserved stable entries such as `laplacian`

## Testing

```bash
nox -s tests
nox -s docs
nox -s notebooks
```

## Contributing and citation

- Contribution guide: `CONTRIBUTING.md`
- Security policy: `SECURITY.md`
- Citation metadata: `CITATION.cff`
- Changelog: `CHANGELOG.md`
