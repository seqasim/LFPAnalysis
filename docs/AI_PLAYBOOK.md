# AI Playbook: LFPAnalysis

Practical how-to guide for AI agents and contributors working in this repository.

**Related:** [AI_CONTEXT.md](AI_CONTEXT.md) for architecture and mental model. **User tutorials:** [LFPAnalysisBook/](../LFPAnalysisBook/).

---

## How to add a new feature

### Decide which layer to extend

| Situation | Where to add code |
|-----------|-------------------|
| New preprocessing step that should be beginner-accessible | Stable API: `config.py` → `workflow.py` → `builders.py` → tests → book chapter |
| Site-specific or analysis-heavy logic | Appropriate utility module (`lfp_preprocess_utils`, `oscillation_utils`, etc.) |
| Backward compatibility for old notebook call pattern | `legacy.py` shim with `DeprecationWarning` |
| Output that users will inspect in pandas | Add/extend schema in `schemas.py` |

**Default rule:** extend the stable API only when the feature has clear config, validation, standardized outputs, and tests. Otherwise add to utilities and document in chapter 11.

### Recipe: add a new artifact detection method (stable API)

Example pattern for extending `misc` / `ied` style detectors.

**1. Add to the registry and Literal type**

In [`LFPAnalysis/workflow.py`](../LFPAnalysis/workflow.py):

```python
ARTIFACT_METHODS = {"none", "misc", "ied", "custom", "my_method"}  # add name
```

In [`LFPAnalysis/config.py`](../LFPAnalysis/config.py):

```python
ArtifactMethod = Literal["none", "misc", "ied", "custom", "my_method"]
```

Add any new threshold fields to `ArtifactConfig`.

**2. Implement detector and wire registry**

```python
def _artifact_my_method(data, config: ArtifactConfig) -> pd.DataFrame:
    legacy = _legacy_preprocess_module()
    channel_events = legacy.detect_my_artifacts(data, thresh=config.my_thresh)
    return build_event_table(channel_events, event_kind="my_method", sfreq=float(data.info["sfreq"]))

_ARTIFACT_REGISTRY["my_method"] = _artifact_my_method
```

Use `build_event_table` so output matches `ARTIFACT_EVENT_COLUMNS`.

**3. Implement utility function** (if needed)

Add `detect_my_artifacts` in `lfp_preprocess_utils.py`. Return a `dict[str, sequence[float]]` mapping channel name → event times in **seconds**.

**4. Expose in builder** (optional)

Update default `artifact_methods` in `build_event_locked_pipeline_config` if it should be on by default.

**5. Add tests**

```python
@pytest.mark.unit
def test_artifact_my_method_returns_schema(synthetic_raw):
    config = ArtifactConfig(methods=["my_method"], my_thresh=5.0)
    tables = detect_artifacts(synthetic_raw, config)
    assert list(tables["my_method"].columns) == list(ARTIFACT_EVENT_COLUMNS)
```

Add an integration test in `test_workflow_integration.py` if it affects `run_pipeline`.

**6. Update documentation**

- Add a subsection to `LFPAnalysisBook/05_first_artifact_pass.md`
- Run `nox -s docs` and `nox -s tests`

### Recipe: add a new baseline mode (stable API)

**1.** Add mode string to `BASELINE_METHODS` in `workflow.py` and `BaselineMode` Literal in `config.py`.

**2.** Implement in `lfp_preprocess_utils.mean_baseline_time` (or call it from `_apply_baseline_array`).

**3.** Add unit test in `test_lfp_preprocess_utils.py` with parametrized modes.

**4.** Update `LFPAnalysisBook/07_first_event_locked_workflow.md`.

### Recipe: add utility-only analysis feature

For TFR, connectivity, or statistics features that are not yet in `run_pipeline`:

1. Add function to the appropriate utility module with NumPy-style docstring.
2. Document expected input shapes and column names in docstring.
3. Add test in `tests/test_<module>_assessment.py` using stubbed heavy dependencies if needed.
4. Add interoperability note to `LFPAnalysisBook/11_advanced_utility_interoperability.md`.
5. Optionally add a worked-example notebook under `LFPAnalysisBook/worked-examples/`.

### Recipe: add a legacy shim

In [`LFPAnalysis/legacy.py`](../LFPAnalysis/legacy.py):

```python
def old_function_name(...):
    _warn(
        "lfp_preprocess_utils.old_function_name(...)",
        "new_stable_equivalent(...)",
        note="Optional explanation.",
    )
    legacy = _legacy_preprocess_module()
    return legacy.old_function_name(...)
```

Add test in `test_builders_and_legacy.py` asserting `DeprecationWarning` is emitted.

### Pre-merge checklist

```bash
nox -s lint
nox -s tests
# If book or notebooks changed:
nox -s docs
nox -s notebooks
```

---

## Scaffolding user analysis notebooks

When a user brings their own dataset and wants analysis notebooks (TFR, band power, regressions, PSD/FOOOF, connectivity, etc.), use the project Cursor skill:

**[`.cursor/skills/scaffold-analysis/`](../.cursor/skills/scaffold-analysis/SKILL.md)**

Workflow: interview (one question at a time) → Dataset Profile → select recipes → write hybrid notebooks under `notebooks/<slug>/` with `PATHS` placeholders and sample-data dry-run defaults.

Canonical code sources for recipes are the LFPAnalysisBook chapters and `LFPAnalysisBook/worked-examples/` (see the skill's `recipes.md`). Do not overwrite book notebooks; generated work stays in `notebooks/` (gitignored).

---

## How to debug

### Use exception types as signposts

| Exception | Likely cause | First check |
|-----------|--------------|-------------|
| `ConfigurationError` | Invalid config value, missing path, window mismatch | Config object fields, file paths, baseline window vs `times` |
| `DataContractError` | Electrode table missing columns | `electrodes.columns`, especially `label` |
| `MissingDependencyError` | Optional package not installed | Run `pip install -e .[dev]` or `.[analysis]` |
| `KeyError` in ROI functions | Channel not in electrode metadata | `elec_data.label` vs MNE `ch_names` |

### Stage-by-stage pipeline inspection

```python
from LFPAnalysis import build_basic_pipeline_config, run_pipeline
from LFPAnalysis.workflow import load_lfp, preprocess_lfp, detect_artifacts, make_epochs, baseline_lfp

config = build_basic_pipeline_config("data/sample_ieeg.fif", file_format="mne")

raw = load_lfp(config.load)
print(raw.ch_names, raw.info["sfreq"])

referenced = preprocess_lfp(raw, config.reference)
print(referenced.ch_names)

artifacts = detect_artifacts(referenced, config.artifact)
for method, table in artifacts.items():
    print(method, len(table))

# Continue stage by stage...
result = run_pipeline(config)
print(result.metadata)
print(result.baseline_summary.head())
```

### Common failure modes and fixes

**Referencing fails / channel mismatch**

```python
from LFPAnalysis import load_electrode_metadata
elec = load_electrode_metadata("path/to/electrodes.csv")
print(set(elec["label"]) - set(raw.ch_names))  # labels in table but not in data
print(set(raw.ch_names) - set(elec["label"]))  # channels without metadata
```

For fuzzy matching issues, `match_elec_names` defaults to `interactive=False` and raises `ValueError` on ambiguous ties (safe in CI).

**Baseline window error**

```python
# For epochs:
print(epochs.times[0], epochs.times[-1])
# For raw:
import numpy as np
times = np.arange(raw.n_times) / raw.info["sfreq"]
print(times[0], times[-1])
```

Ensure `baseline_window` tuple overlaps this range.

**Empty artifact tables**

```python
print(config.artifact.methods)  # ["none"] → empty by design
print(config.artifact.ied_peak_thresh)  # try lowering threshold
```

**FOOOF fails on stable API**

- Requires `pip install -e .[analysis]`
- Requires epoched data — use `build_spectral_pipeline_config` with `event_name` and `event_times`, or epoch manually first.

**Neuralynx load fails**

- All `.ncs` files must share sampling rate
- Check `nlx_utils` warnings for skipped/empty channels
- Verify channel name lists (`seeg_names`, `drop_names`) match filenames

### Debugging utility-module workflows

Enable MNE verbose if needed: `mne.set_log_level('INFO')`.

For connectivity/surrogate issues, fix `rng_seed` first to isolate logic bugs from randomness:

```python
from LFPAnalysis.oscillation_utils import make_surrogate_data
surrogates = make_surrogate_data(epochs, n_shuffles=10, rng_seed=42)
```

For statistics permutations, note they are **not** seeded — set `np.random.seed(n)` before calls if you need reproducibility during debugging.

### Interactive hang

If execution blocks with no error, search for `input(` in the call stack — only `match_elec_names(..., interactive=True)` still prompts.

---

## How tests work

### Test layout

```
tests/
├── conftest.py              # Shared fixtures
├── data/electrodes.csv      # Minimal electrode table
├── test_workflow_unit.py    # Fast workflow tests
├── test_workflow_integration.py  # Full pipeline on synthetic data
├── test_builders_and_legacy.py
├── test_schemas.py
├── test_validation.py
├── test_docs_content.py     # Book structure validation
├── test_*_utils.py          # Module-specific unit tests
└── test_*_assessment.py     # Utility tests with stubbed deps
```

### Pytest markers (`pytest.ini`)

| Marker | Meaning | When to use |
|--------|---------|-------------|
| `unit` | Fast, pure or lightly mocked | Validation, schemas, single functions |
| `integration` | Multi-step, real MNE objects | `run_pipeline`, load+epoch flows |
| `notebook` | Notebook execution | Defined but run via separate nox session |
| `slow` | Heavy end-to-end | Excluded from default CI |
| `optional_dep` | Needs analysis extras | fooof, mne_connectivity tests |

### Synthetic fixtures (`conftest.py`)

```python
# synthetic_raw: 2-channel seeg Raw, sfreq=200, 20 s, channels l1/l2
# synthetic_epochs: 3 demo events, tmin=-0.5, tmax=1.0
```

Unit tests should use these — not `data/sample_ieeg.fif`.

### Assessment tests (`*_assessment.py`)

Reload utility modules under `monkeypatch` with lightweight stubs for MNE, Levenshtein, connectivity, etc. No explicit markers; included in default pytest run. Use this pattern when testing logic without installing full analysis stack.

### Commands

```bash
# Quick iteration
pytest -m "unit and not optional_dep"

# CI-equivalent (what nox -s tests runs)
pytest -m "not notebook and not slow" \
  --cov=LFPAnalysis.workflow \
  --cov=LFPAnalysis.builders \
  --cov=LFPAnalysis.legacy \
  --cov-fail-under=80

# Single file
pytest tests/test_workflow_integration.py -v

# With optional deps
pytest -m optional_dep
```

### Nox sessions (`noxfile.py`)

| Session | Command | Purpose |
|---------|---------|---------|
| `lint` | `ruff check .` + `ruff format --check .` | Style |
| `tests` | pytest excluding notebook/slow, 80% cov | Main test gate |
| `docs` | `jupyter-book build --html --ci` in `LFPAnalysisBook/` | Book builds cleanly |
| `notebooks` | pytest-nbmake on all smoke-tests + all mapped worked examples | Notebook execution |

### CI (`.github/workflows/ci.yml`)

Runs on push/PR to `main`/`master`:

- **lint** — Python 3.11
- **tests** — matrix 3.10, 3.11, 3.12
- **macos-smoke** — loads `data/sample_ieeg.fif`, runs `run_pipeline`
- **docs** — jupyter-book build
- **notebooks** — nbmake on smoke-tests

### Writing a good test

```python
import pytest
from LFPAnalysis.config import ArtifactConfig
from LFPAnalysis.workflow import detect_artifacts
from LFPAnalysis.schemas import ARTIFACT_EVENT_COLUMNS

@pytest.mark.unit
def test_detect_artifacts_none_returns_empty_schema(synthetic_raw):
    config = ArtifactConfig(methods=["none"])
    tables = detect_artifacts(synthetic_raw, config)
    assert "none" in tables
    assert list(tables["none"].columns) == list(ARTIFACT_EVENT_COLUMNS)
    assert len(tables["none"]) == 0
```

Rules from [`TESTING.md`](../TESTING.md):

- Canonical fixtures under `tests/data/`
- Prefer synthetic MNE objects over large binaries
- Repo `data/` only for docs/notebook smoke tests

---

## Deployment

LFPAnalysis is a **library**, not a deployed service. "Deployment" means packaging, versioning, and publishing documentation.

### Install variants

```bash
# Contributors
pip install -e .[dev]
pre-commit install

# Analysis-heavy users
pip install -e .[analysis]

# Conda
conda env create -f environment.yml
conda activate LFPAnalysis
pip install -e .
```

### Versioning

- Version in [`pyproject.toml`](../pyproject.toml) (`project.version`, currently `1.1.0`)
- Changelog in [`CHANGELOG.md`](../CHANGELOG.md) — Keep a Changelog format, Semantic Versioning
- Citation metadata in [`CITATION.cff`](../CITATION.cff)

### Release checklist (manual)

1. Update version in `pyproject.toml`
2. Move `[Unreleased]` entries to new version section in `CHANGELOG.md`
3. Run full CI locally: `nox -s lint && nox -s tests && nox -s docs && nox -s notebooks`
4. Tag release on GitHub
5. Build and publish to PyPI if applicable (`python -m build`)

### Documentation deployment

Jupyter Book site built from `LFPAnalysisBook/`:

```bash
nox -s docs
# Output: LFPAnalysisBook/_build/html/
```

Hosted documentation URL is referenced in `pyproject.toml` project URLs (GitHub tree path to `LFPAnalysisBook`).

### What is NOT automated

- No Docker/container deployment
- No cloud service or API server
- No database migrations
- PyPI publish workflow not defined in `.github/workflows/` (only CI)

---

## Common mistakes

### For users / analysis workflows

| Mistake | Fix |
|---------|-----|
| Importing utilities before trying stable API | Start with `build_*_pipeline_config` + `run_pipeline` |
| Assuming shims = stable parity | Check migration chapters; TFR/connectivity still utility-only |
| Behavioral times in milliseconds | Convert to seconds for `EpochConfig.event_times` |
| Skipping electrode validation | Run `load_electrode_metadata` and compare labels to `ch_names` |
| Expecting reproducible permutation stats | Seed `np.random` globally or accept non-determinism in `statistics_utils` |
| Using `laplacian` reference | Not implemented; use `bipolar` or `wm` |
| Running FOOOF on continuous raw via stable API | Epoch first or use utility path |
| Ignoring empty artifact tables | Check `config.artifact.methods` and thresholds |

### For contributors

| Mistake | Fix |
|---------|-----|
| Skipping `nox -s lint` | Required before PR |
| Not running `docs`/`notebooks` after book changes | CI will fail |
| Adding output columns without updating `schemas.py` | Breaks `ARTIFACT_EVENT_COLUMNS` / `BASELINE_SUMMARY_COLUMNS` contract |
| Using `data/sample_ieeg.fif` in unit tests | Use `synthetic_raw` / `synthetic_epochs` fixtures |
| Large binary test fixtures | Keep under `tests/data/`, stay small |
| Calling `match_elec_names(..., interactive=True)` in tests | May block on `input()`; default path raises instead |
| Top-level import of optional deps | Use `ensure_dependency` in workflow layer |
| Forgetting `@pytest.mark.unit` or `integration` | Helps selective test runs |

### For AI agents

| Mistake | Fix |
|---------|-----|
| Editing utility modules when stable API suffices | Prefer `workflow.py` for beginner-facing features |
| Silent stub functions returning `None` | Check [AI_CONTEXT.md stubs table](AI_CONTEXT.md#incomplete--stub-surfaces) |
| Reordering `run_pipeline` stages | Preserve invariant stage order |
| Creating docs in wrong location | User tutorials → `LFPAnalysisBook/`; AI context → `docs/` |

---

## Examples

All examples assume repo root as working directory and `pip install -e .[dev]`.

### Basic load and inspect

```python
from pathlib import Path
from LFPAnalysis import build_basic_pipeline_config, run_pipeline

config = build_basic_pipeline_config(
    Path("data/sample_ieeg.fif"),
    file_format="mne",
)
result = run_pipeline(config)

print(f"Channels: {result.raw.ch_names[:5]}...")
print(f"Sampling rate: {result.raw.info['sfreq']} Hz")
print(f"Reference method: {result.metadata['reference_method']}")
```

### Event-locked pipeline with baseline

```python
from pathlib import Path
from LFPAnalysis import build_event_locked_pipeline_config, run_pipeline

config = build_event_locked_pipeline_config(
    Path("data/sample_ieeg_continuous_rest.fif"),
    file_format="mne",
    event_name="demo",
    event_times=[5.0, 10.0, 15.0],  # seconds
    baseline_mode="zscore",
    baseline_window=(-0.5, 0.0),
    tmin=-0.5,
    tmax=1.0,
)
result = run_pipeline(config)

print(result.epochs)  # mne.Epochs
print(result.baseline_summary)
print(result.artifact_tables.keys())
```

### Spectral analysis (PSD)

```python
from pathlib import Path
from LFPAnalysis import build_spectral_pipeline_config, run_pipeline

config = build_spectral_pipeline_config(
    Path("data/sample_feedback_start-epo.fif"),
    file_format="mne",
    spectral_method="psd",
    fmin=1.0,
    fmax=150.0,
)
result = run_pipeline(config)

spectrum = result.spectral["spectrum"]
print(spectrum.freqs.shape, spectrum.get_data().shape)
```

### Spectral analysis (FOOOF, requires `.[analysis]`)

```python
from pathlib import Path
from LFPAnalysis import build_spectral_pipeline_config, run_pipeline

config = build_spectral_pipeline_config(
    Path("data/sample_feedback_start-epo.fif"),
    file_format="mne",
    spectral_method="fooof",
    fooof_range=(1.0, 40.0),
)
result = run_pipeline(config)

fooof_table = result.spectral["table"]
print(fooof_table.columns)
print(fooof_table.head())
```

### Load with referencing and artifacts

```python
from pathlib import Path
from LFPAnalysis import build_basic_pipeline_config, run_pipeline

config = build_basic_pipeline_config(
    Path("data/sample_ieeg.fif"),
    file_format="mne",
    reference_method="bipolar",
    electrode_path=Path("tests/data/electrodes.csv"),
    artifact_methods=["misc"],
)
result = run_pipeline(config)
print(result.artifact_tables["misc"].head())
```

### Validate electrode metadata

```python
from LFPAnalysis import load_electrode_metadata

electrodes = load_electrode_metadata("tests/data/electrodes.csv")
print(electrodes.columns.tolist())
assert "label" in electrodes.columns
```

### Legacy shim (migration)

```python
import warnings
from LFPAnalysis import legacy

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    epochs = legacy.make_epochs(
        load_path="data/sample_ieeg_continuous_rest.fif",
        behav_name="demo",
        behav_times=[5.0, 10.0, 15.0],
        ev_start_s=0.5,
        ev_end_s=1.0,
    )
    assert any(issubclass(x.category, DeprecationWarning) for x in w)

print(epochs)
```

### Advanced: TFR after stable preprocessing (utility layer)

The stable API does not yet include TFR. After loading and epoching via stable API or legacy:

```python
# Conceptual — see LFPAnalysisBook/09_first_time_frequency.md and worked-examples/09_first_tfr_run.ipynb
from LFPAnalysis import legacy

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    tfr = legacy.compute_and_baseline_tfr(
        # legacy-specific arguments — see 23_translate_tfr_workflow.md
    )
```

### Advanced: connectivity with deterministic surrogates

```python
import warnings
from LFPAnalysis.oscillation_utils import make_surrogate_data

# epochs: mne.Epochs object with shape (n_epochs, n_channels, n_times)
surrogates = make_surrogate_data(
    epochs,
    n_shuffles=50,
    swap_method="epochs",  # or "time_blocks"
    rng_seed=42,
)
```

### Inspect pipeline config without running

```python
from pathlib import Path
from LFPAnalysis import build_event_locked_pipeline_config

config = build_event_locked_pipeline_config(
    Path("data/sample_ieeg_continuous_rest.fif"),
    event_name="demo",
    event_times=[5.0, 10.0, 15.0],
)
print(config)  # PipelineConfig with nested dataclasses
```

---

## Quick command reference

```bash
# Setup
pip install -e .[dev]
pre-commit install

# Before every PR
nox -s lint
nox -s tests

# Book/notebook changes
nox -s docs
nox -s notebooks

# Fast test loop
pytest -m "unit and not optional_dep" -q
```
