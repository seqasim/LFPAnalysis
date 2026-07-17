# LFPAnalysis — Comprehensive Repository Overview

> **Purpose of this document.** This is a single-file, ground-truth description of the
> LFPAnalysis repository: what it is, how it is organized, why it is organized that way,
> and where the load-bearing constraints and hazards live. It is written to be handed to
> another agent (or human) as the *only* context they need before attempting a large-scale
> refactor, without re-scoping the whole repo from scratch.
>
> It is grounded in the current source, tests, CI, and docs (package version `1.1.0`).
> For task-oriented how-to recipes see [`AI_PLAYBOOK.md`](AI_PLAYBOOK.md); for the
> architectural mental model in condensed form see [`AI_CONTEXT.md`](AI_CONTEXT.md); the
> canonical *user* documentation is the Jupyter Book under [`../LFPAnalysisBook/`](../LFPAnalysisBook/).

---

## 1. What this repository is

LFPAnalysis is a **Python scientific library** for analyzing human intracranial /
local field potential (iEEG / LFP) recordings. It is a research toolkit built by and for
a neuroscience lab, not a service or application. It wraps **MNE-Python** with:

- a small, typed, beginner-facing **workflow API** (load → reference → artifact-detect →
  epoch → baseline → spectral), returning standardized objects and pandas tables;
- **legacy compatibility shims** that bridge old notebook call patterns to the new API
  while emitting `DeprecationWarning`;
- large, analysis-heavy **utility modules** for site-specific I/O, time-frequency,
  connectivity / information theory, oscillation detection, behavioral synchronization,
  ROI assignment, and statistics that are *not yet* covered by the stable API.

It ships as an installable package (`pip install -e .`, or conda via `environment.yml`),
is documented as a Jupyter Book, and is tested with pytest + nox + GitHub Actions.

### Core facts

| Fact | Value |
|------|-------|
| Package name / import | `LFPAnalysis` |
| Version | `1.1.0` (`pyproject.toml`), Development Status: 4 - Beta |
| License | BSD-3-Clause |
| Python support | 3.10 – 3.12 |
| Author | Salman E. Qasim |
| Build backend | setuptools (`pyproject.toml`, `[build-system]`) |
| Primary dependency | MNE-Python (`mne>=1.6`) |
| Repo home | https://github.com/seqasim/LFPAnalysis |
| Package Python source | ~10.7k lines across 16 modules (see §4) |

### Intended audiences (explicitly designed for three)

1. **Total beginners** who want a guided path from a sample file to an interpretable result.
2. **Existing users of the old notebooks** who need a concrete migration path.
3. **Advanced users** who still want direct access to the underlying utility modules.

This three-audience framing is the single most important thing to understand about the
repo — it explains the three-layer architecture, the legacy shims, and the heavy
documentation investment.

---

## 2. Design philosophy

These are the recurring principles encoded throughout the code and docs. A refactor that
violates them will fight the grain of the repository.

1. **Three layers with a deliberate stability gradient.** Stable workflow API → legacy
   shims → advanced utilities. New beginner-facing behavior belongs in the stable layer;
   analysis-heavy or site-specific behavior belongs in utilities; nothing old is deleted,
   it is wrapped and deprecated. (See §3.)

2. **Typed configuration over positional arguments.** The stable API is driven by frozen-ish
   `@dataclass(slots=True)` config objects (`config.py`). Users compose a `PipelineConfig`
   (usually via a `build_*` helper) and hand it to a single orchestrator, `run_pipeline`.

3. **Standardized, schema-stable outputs.** Workflow outputs are pandas DataFrames with
   *fixed column tuples* declared in `schemas.py`. Empty results still carry the full schema
   (never omit columns). Downstream code and doc-tests depend on these contracts.

4. **Fail loud at boundaries with project-specific exceptions.** Config/path/column problems
   raise `ConfigurationError`, `DataContractError`, or `MissingDependencyError` (all subclass
   `LFPAnalysisError`) instead of bare `ValueError`/`KeyError`. Validation is centralized in
   `validation.py`.

5. **Heavy dependencies are optional and lazily imported.** The base install is intentionally
   light (MNE, numpy, pandas, scipy, …). Analysis-heavy packages (fooof, mne-connectivity,
   tensorpac, statsmodels, numba, …) live in the `analysis`/`dev` extras and are imported only
   through `ensure_dependency()` at call time, never at module top level in the stable layer.

6. **Memory-consciousness for local machines.** Recent changes (`1.1.0` Unreleased) push a
   `float32` working dtype (`WORKING_DTYPE`), lazy/disk-backed loading (`LoadConfig.preload=False`,
   `memmap=True`), generator-based surrogates, and dropping superseded continuous stages from
   `PipelineResult` once epoching supersedes them. Refactors should preserve these RAM guardrails.

7. **Documentation is a first-class deliverable, and it is tested.** User tutorials live in the
   Jupyter Book, are executed in CI (nbmake), and their *structure and content* are asserted by
   `tests/test_docs_content.py`. Docs are not optional prose.

8. **Determinism where it is claimed, honesty where it is not.** Oscillation/surrogate code takes
   an explicit `rng_seed` and is reproducible; `statistics_utils` permutations are *not* seeded and
   are documented as non-reproducible. Do not silently "fix" this without updating the contract.

9. **The library consumes electrode/anatomy tables; it does not localize electrodes.** ROI/anatomy
   logic assumes pre-existing metadata tables with specific columns.

---

## 3. Architecture: dual spine + three layers

Public exports are defined in [`LFPAnalysis/__init__.py`](../LFPAnalysis/__init__.py). Everything
in `__all__` there is the stable surface; utility modules are imported by path, not re-exported.

**Prep vs analysis:** Sync, electrode-table consumption, referencing, and continuous artifact
detection belong in prep (`run_prep` → `PrepResult.epochs`). Analysis starts from MNE Epochs
(`run_analysis`) so prep backends can change without rewriting science code.
`run_pipeline` is a tutorial convenience that runs prep then analysis.

```
                ┌─────────────────────────────────────────────┐
   user  ─────► │  Stable dual spine (typed, tested)           │
                │  Prep: run_prep / PrepConfig → Epochs        │
                │  Analysis: run_analysis / AnalysisConfig     │
                │  Tutorial: run_pipeline = prep then analysis │
                │  __init__ · config · builders · prep ·       │
                │  workflow · results · schemas · validation   │
                └───────────────┬─────────────────────────────┘
                                │ delegates to
                ┌───────────────▼─────────────────────────────┐
   old code ──► │  Legacy Shims  (legacy.py)                   │
                │  DeprecationWarning + route to stable/utils │
                └───────────────┬─────────────────────────────┘
                                │ calls
                ┌───────────────▼─────────────────────────────┐
  advanced ───► │  Advanced escape hatch                       │
                │  LFPAnalysis.advanced (+ *_utils modules)    │
                │  connectivity · stats · ROI · site I/O       │
                └───────────────┬─────────────────────────────┘
                                │ built on
                    MNE-Python, fooof, mne_connectivity, tensorpac, statsmodels, …
```

### Layer A — Stable workflow API

Small, typed, and CI-gated (the coverage gate targets `workflow`, `builders`, `legacy`).

| Module | Role |
|--------|------|
| `__init__.py` | Declares the stable public surface (`__all__`). |
| `config.py` | Config dataclasses + `Literal` aliases + `WORKING_DTYPE`. |
| `builders.py` | Prep/analysis/pipeline convenience constructors. |
| `prep.py` | Prep spine: load → ref → artifacts → sync → electrodes → Epochs. |
| `workflow.py` | Analysis stages + `run_analysis` + tutorial `run_pipeline`. |
| `results.py` | `PrepResult`, `AnalysisResult`, `PipelineResult`. |
| `schemas.py` | Column contracts + DataFrame builders. |
| `validation.py` | `ensure_supported`, path/column helpers. |
| `exceptions.py` | `LFPAnalysisError` hierarchy. |

**Config dataclasses** (`config.py`): `LoadConfig`, `ReferenceConfig`, `ArtifactConfig`,
`SyncConfig`, `ElectrodeConfig`, `BaselineConfig`, `EpochConfig`, `SpectralConfig`,
`TfrConfig`, composing `PrepConfig` / `AnalysisConfig`, and flat `PipelineConfig` for tutorials.
String-valued fields are constrained by `Literal` aliases (`ArtifactMethod`, `BaselineMode`,
`ReferenceMethod`, `SpectralMethod`, `InputFormat`).

**Method registries** (`workflow.py`) — string values must belong to these sets, enforced by
`ensure_supported`:

```python
REFERENCE_METHODS = {"none", "bipolar", "wm"}
ARTIFACT_METHODS  = {"none", "misc", "ied", "custom"}
BASELINE_METHODS  = {"none", "mean", "ratio", "percent", "zscore",
                     "logratio", "zlogratio", "trialwise", "continuous"}
SPECTRAL_METHODS  = {"none", "psd", "fooof"}
```

`laplacian` is not a registered reference method (unimplemented; was a false-support trap).
Artifact detection uses an internal dispatch dict `_ARTIFACT_REGISTRY` (`none`/`misc`/`ied`),
plus a `custom` branch that accepts a caller-supplied detector callable.

**`run_pipeline` stages** (tutorial wrapper = prep then analysis):

1. `load_lfp` → MNE `Raw` or `Epochs`
2. `preprocess_lfp` → re-referenced `Raw`
3. `detect_artifacts` → `dict[str, DataFrame]` (on referenced *continuous* data, before epoching)
4. optional sync + electrode handoff (`run_prep`)
5. `make_epochs` → optional `Epochs` (from referenced raw, not yet baselined)
6. `baseline_lfp` → applied to epochs if present, else to referenced raw
7. `compute_spectral_features` / optional TFR → analysis dicts
8. returns `PipelineResult`

`PipelineResult` fields: `raw`, `referenced`, `epochs`, `artifact_tables`, `baseline_summary`,
`spectral`, `tfr`, `sync`, `electrode_df`, `metadata`. To save RAM, when epoching is enabled the
superseded `raw`/`referenced` are set to `None`.

### Layer B — Legacy shims (`legacy.py`)

Bridges the four most common old entry points — `make_mne`, `ref_mne`, `make_epochs`,
`compute_and_baseline_tfr`. Each emits a `DeprecationWarning` via `_warn(old_call, new_call, note)`
and then either routes to the stable path or delegates to `lfp_preprocess_utils`. Prefer
`run_analysis(..., tfr=...)` for new Morlet TFR; the full legacy TFR orchestrator remains available
via shims / utilities until 2.0 parity is complete.

### Layer C — Advanced utilities

Large, older, analysis-focused modules. They predate the stable API, use looser conventions
(module-level heavy imports, no `__all__`, NumPy-style docstrings, some interactive prompts and
stub functions), and are where the bulk of the scientific logic lives.

| Module | Lines | Domain |
|--------|------:|--------|
| `oscillation_utils.py` | 4226 | Connectivity + Gaussian-copula mutual information / transfer entropy, PAC, surrogates, eBOSC-style oscillation detection. Largest, densest module. |
| `lfp_preprocess_utils.py` | 2722 | Load, re-reference (bipolar/wm), artifact/IED detection, baselining, TFR compute+baseline, channel matching. The legacy workhorse. |
| `analysis_utils.py` | 1003 | ROI selection (`select_rois_picks`, `select_picks_rois`), FOOOF over epochs, spike-triggered averages, burst detection, plotting, catch22 features. |
| `statistics_utils.py` | 639 | Permutation regression (z-scored), mixed-effects models, time-resolved MLM (parallel via joblib). |
| `sync_utils.py` | 321 | Photodiode / TTL behavioral synchronization → `(slope, offset)`. |
| `nlx_utils.py` | 469 | Neuralynx `.ncs`/`.nev` binary I/O (adapted from NeuralynxIO). |
| `iowa_utils.py` | 140 | Iowa-site channel table parsing → `lfpx{N}` names. |

Also packaged: `YBA_ROI_labelled.xlsx` (ROI lookup table, declared as `package-data`).

---

## 4. Full package module map

```
LFPAnalysis/
├── __init__.py             # stable public exports (__all__)
├── config.py               # typed config dataclasses + Literals + WORKING_DTYPE
├── builders.py             # build_*_pipeline_config helpers
├── workflow.py             # staged functions, registries, run_pipeline
├── results.py              # PipelineResult
├── schemas.py              # column contracts + table builders
├── validation.py           # ensure_supported / ensure_dependency / path & column checks
├── exceptions.py           # LFPAnalysisError, ConfigurationError, DataContractError, MissingDependencyError
├── legacy.py               # deprecation shims
├── lfp_preprocess_utils.py # (advanced) load, ref, artifacts, TFR
├── oscillation_utils.py    # (advanced) connectivity, GCMI/TE, PAC, surrogates, eBOSC
├── analysis_utils.py       # (advanced) ROI, FOOOF, STA, bursts, plotting
├── statistics_utils.py     # (advanced) permutation regression, time-resolved MLM
├── sync_utils.py           # (advanced) photodiode/TTL sync
├── nlx_utils.py            # (advanced) Neuralynx I/O
├── iowa_utils.py           # (advanced) Iowa channel tables
├── _scratch_utils.py       # soft-archived stubs / unused helpers (private)
└── YBA_ROI_labelled.xlsx   # packaged ROI lookup table
```

Utility modules export every non-underscore name implicitly (no `__all__`). Treat
underscore-prefixed helpers as private. Soft-archived names remain importable from their original
modules via deprecation shims for one release. There is no enforced "public utility API" boundary —
any rename in a utility module is potentially a breaking change for notebooks and downstream lab
code that are not in this repository.

---

## 5. Data flow and delegation

### Stable pipeline flow

```
EDF / NLX / FIF ──load_lfp──► raw ──preprocess_lfp──► referenced
                                                        │
                              detect_artifacts ◄────────┤ (continuous, pre-epoch)
                                                        │
                              make_epochs ◄─────────────┘
                                   │
                          epochs? ─yes─► baseline_lfp(epochs)
                                   └─no──► baseline_lfp(referenced)
                                                        │
                                       compute_spectral_features
                                                        │
                                                 PipelineResult
```

### Stable → utility delegation

| Stable function | Delegates to |
|-----------------|--------------|
| `load_lfp` (neuralynx) | `nlx_utils.parse_subject_nlx_data` |
| `preprocess_lfp` (wm/bipolar) | `lfp_preprocess_utils.ref_mne` |
| `detect_artifacts` (misc) | `lfp_preprocess_utils.detect_misc_artifacts` |
| `detect_artifacts` (ied) | `lfp_preprocess_utils.detect_IEDs` |
| `baseline_lfp` | `lfp_preprocess_utils.mean_baseline_time` |
| `compute_spectral_features` (fooof) | `analysis_utils.FOOOF_compute_epochs` |

### Advanced lab workflow (beyond `run_pipeline`)

`make_mne`/`load_lfp` → `ref_mne` → `synchronize_data` (photodiode/TTL → slope/offset) →
`make_epochs` (+ IED/artifact CSV sidecars) → `compute_and_baseline_tfr` → `oscillation_utils`
connectivity/surrogates → `statistics_utils.time_resolved_mlm`.

### File sidecar conventions (legacy utilities write these)

| Pattern | Content |
|---------|---------|
| `{event}-epo.fif` | Epoched MNE data |
| `{event}_IED_df.csv` | IED detection results |
| `{event}_artifact_df.csv` | Misc artifact results |
| `{event}-tfr.h5` | Baseline-corrected TFR |
| `lfp_data.fif` | Default continuous save from `make_mne` |

Sample versions of these live in `data/`.

---

## 6. Repository layout (top level)

```
LFPAnalysis/                    # repo root
├── LFPAnalysis/                # the Python package (§4)
├── LFPAnalysisBook/            # Jupyter Book — canonical user docs (§7)
│   ├── 00–16_*.md              # beginner + intermediate track
│   ├── 20–26_*.md              # migration-from-old-repo track + resources
│   ├── 30_troubleshooting.md
│   ├── smoke-tests/            # 8 deterministic CI notebooks
│   ├── worked-examples/        # 11 tutorial notebooks
│   ├── _config.yml, _toc.yml   # Jupyter Book config + table of contents
│   ├── references.bib, logo.png, requirements.txt, intro.md
├── tests/                      # pytest suite + tests/data/ (§8)
├── data/                       # sample FIF/CSV/XLSX for docs & smoke tests (§9)
├── scripts/                    # non-canonical exploratory notebooks + unmaintained/
│   └── _build_book_notebooks.py# generator for book notebooks
├── docs/                       # AI/contributor context: AI_CONTEXT, AI_PLAYBOOK, this file
├── .cursor/skills/scaffold-analysis/  # versioned Cursor skill (§10)
├── .github/                    # CI workflow, issue/PR templates
├── noxfile.py                  # lint/tests/docs/notebooks sessions
├── pyproject.toml              # metadata, deps, Ruff config
├── pytest.ini                  # markers + test discovery
├── environment.yml / environment-dev.yml   # conda envs
├── requirements.txt
├── conftest.py                 # root conftest (numpy preload for coverage)
├── CHANGELOG.md CITATION.cff CODE_OF_CONDUCT.md CONTRIBUTING.md
├── LICENSE README.md SECURITY.md TESTING.md
├── .editorconfig .pre-commit-config.yaml .gitignore .vscode/
```

---

## 7. Documentation (`LFPAnalysisBook/`)

The Jupyter Book is the canonical onboarding surface (README and `scripts/README.md` both defer to
it). It is organized as an ordered set of Markdown chapters, with `.ipynb` notebooks attached as
sections in `_toc.yml`.

- **Beginner / intermediate track (00–16):** interface guide, installation, data model, first load,
  first reference, first synchronization, first artifact pass, first baseline, first event-locked
  workflow, first PSD/FOOOF, first time-frequency, first connectivity + surrogates, first
  time-resolved stats, advanced utility interoperability, anatomy & ROI assignment, assembling
  analysis dataframes, group-level statistics, saving/organizing results, plotting recipes.
- **Migration track (20–26):** old-repo mental model, legacy function mapping, condensed-notebook
  translation, TFR-workflow translation, connectivity-workflow translation, legacy-only surfaces,
  resources & references.
- **30_troubleshooting.md.**

Two notebook collections back the prose:

- **`smoke-tests/`** (8 notebooks): deterministic, bounded, run fully in CI via nbmake. They validate
  imports, sample-data loading, and one small example per workflow area.
- **`worked-examples/`** (11 notebooks): richer tutorials. Only a subset is executed in CI (see
  `noxfile.py notebooks`): `01`, `07`, `08`, `09`.

**Important for refactors:** `tests/test_docs_content.py` asserts chapter *structure* — e.g. beginner
chapters share a teaching structure, migration chapters contain old+new code examples, the interface
guide names all public surfaces, and the advanced-utility chapter names the shared module stack. If
you rename a public symbol, add a chapter, or restructure headings, this test will likely fail until
the book is updated. `scripts/_build_book_notebooks.py` programmatically generates book notebooks.

---

## 8. Tests

Layout under `tests/` (18 test files + `conftest.py` + `data/electrodes.csv`):

- `test_workflow_unit.py`, `test_workflow_integration.py` — stable API (fast vs full-pipeline).
- `test_builders_and_legacy.py` — builders + deprecation-warning assertions.
- `test_schemas.py`, `test_validation.py` — output contracts and validation gate.
- `test_docs_content.py` — book structure/content assertions (see §7).
- `test_<module>.py` — module unit tests (`nlx_utils`, `sync_utils`, `oscillation_utils`).
- `test_<module>_assessment.py` — utility tests that reload modules under `monkeypatch` with
  lightweight stubs for MNE / Levenshtein / connectivity, so logic can be tested without the full
  analysis stack installed.
- `test_utils_interoperability.py` — cross-module contracts (channel-name sharing, sync→analysis
  shape compatibility, surrogate/statistics shapes).

**Fixtures** (`tests/conftest.py`): `synthetic_raw` (2-channel sEEG Raw, sfreq=200, 20 s, channels
`l1`/`l2`), `synthetic_epochs` (3 demo events, tmin=-0.5, tmax=1.0), `electrode_csv_path`, `mne_module`.
Unit tests must use synthetic fixtures — **not** `data/sample_ieeg.fif`.

**Markers** (`pytest.ini`, `--strict-markers`): `unit`, `integration`, `notebook`, `slow`,
`optional_dep`. Config is `--strict-config`; discovery is limited to `testpaths = tests`.

**Root `conftest.py`** imports NumPy before coverage begins tracing, to avoid a NumPy/pandas reload
breakage under pytest-cov.

---

## 9. Sample data (`data/`)

Used by book notebooks and the macOS CI smoke test — **not** by unit tests. Contents:
`sample_ieeg.fif`, `sample_ieeg_continuous_rest.fif`, `sample_ieeg_bp.fif`, `sample_*-epo.fif`
(baseline/feedback epochs), `sample_beh.csv`, `sample_ts.csv`, `sample_photodiode.fif`,
`sample_labels.xlsx`, `sample_labels_bp`, IED/artifact sidecar CSVs, and ROI tables
(`YBA_1.0_Full_Parcels_List.xlsx`, `YBA_ROI_labelled.xlsx`). `tests/data/electrodes.csv` is the
minimal electrode table for validation tests.

---

## 10. Tooling, environments, and CI

### Build & install

- `pyproject.toml` (setuptools). Base deps: `h5io`, `joblib`, `mne>=1.6`, `numpy>=1.24`, `openpyxl`,
  `pandas>=2.0`, `python-Levenshtein`, `PyYAML`, `scipy>=1.11`.
- Optional extras: `analysis` (fooof==1.0.0, mne-connectivity, tensorpac, neurodsp==2.2.0, numba,
  pycatch22, statsmodels, dcor, sparse, seaborn, tabulate, ipywidgets/ipyevents), `docs`
  (jupyter-book, myst-nb, matplotlib, pooch, …), `test` (pytest, pytest-cov, nbmake, coverage),
  and `dev` (superset: analysis + docs tools + test + ruff + nox + pre-commit).
- `fooof` is pinned to `==1.0.0` (upstream renamed to `specparam`); `neurodsp==2.2.0` is also pinned.
- Conda: `environment.yml` (`LFPAnalysis`) and `environment-dev.yml` (`LFPAnalysis-dev`); both run
  `pip install -e .`.

### Nox sessions (`noxfile.py`; default sessions = `lint`, `tests`)

| Session | What it runs |
|---------|--------------|
| `lint` | `ruff check .` + `ruff format --check .` |
| `tests` | installs `.[dev]`, runs pytest excluding `notebook`/`slow`, coverage on `workflow`/`builders`/`legacy`, `--cov-fail-under=80` |
| `docs` | `jupyter-book build LFPAnalysisBook` |
| `notebooks` | nbmake on all `smoke-tests/` + worked examples `01`, `07`, `08`, `09` (timeout 1200s) |

### CI (`.github/workflows/ci.yml`, on push/PR to main/master)

Jobs: **lint** (3.11), **tests** (matrix 3.10/3.11/3.12), **macos-smoke** (loads
`data/sample_ieeg.fif` and runs `run_pipeline`), **docs** (jupyter-book build), **notebooks** (nbmake
on smoke-tests). No PyPI publish workflow exists; there is no Docker/service/DB deployment — this is a
library.

### Style / conventions

- Ruff: line length 100, rules `E,F,I,W`, ignore `E501` (`pyproject.toml`).
- `.editorconfig`: UTF-8, LF, 4-space indent, final newline; `[*.md]` preserves trailing whitespace.
- Pre-commit (`.pre-commit-config.yaml`): merge-conflict check, EOF, trailing-whitespace, YAML,
  large-file guard, ruff lint + format.
- Python patterns: `from __future__ import annotations`, `@dataclass(slots=True)`, NumPy-style
  docstrings, project exceptions at boundaries, lazy optional imports via `ensure_dependency`.

### Cursor tooling

- `.cursor/skills/scaffold-analysis/` is a **versioned** Cursor skill (SKILL.md + intake.md +
  recipes.md + notebook-conventions.md) that interviews a user about their dataset and scaffolds
  hybrid analysis notebooks into `notebooks/<slug>/` (gitignored). It references this `docs/` folder
  as its repo-context source.
- `.cursor/rules/` contains local rule drafts (`handoff`, `grill-me`, `llm-council`).

---

## 11. File management conventions

- **`.gitignore` intent:** ignores build artifacts, tooling caches (`.pytest_cache`, `.ruff_cache`,
  `.nox`, `.venv`), `LFPAnalysisBook/_build/`, `.DS_Store`, and `notebooks/` (user-scaffolded output).
  `.cursor/*` is ignored **except** `.cursor/skills/**`, which is intentionally versioned.
- **Where things go:** user tutorials → `LFPAnalysisBook/`; AI/contributor context → `docs/`;
  exploratory/legacy notebooks → `scripts/` and `scripts/unmaintained/` (explicitly *not* the
  supported path); generated per-user analyses → `notebooks/` (never committed).
- **Test data policy** (`TESTING.md`): canonical fixtures under `tests/data/`, prefer synthetic MNE
  objects over binaries, repo `data/` reserved for docs and notebook smoke tests only.
- **Repo hygiene note:** a few tracked-but-noise artifacts exist (`.DS_Store`, `.coverage`,
  `_build/logs/myst.build.json`, and an untracked `LFPAnalysis.egg-info/`). These are not part of the
  source of truth; a refactor may clean them but should not depend on them.

---

## 12. Hidden assumptions and invariants (read before refactoring)

These are the constraints most likely to be violated by a naive large-scale change.

### Ordering / structural invariants
1. **Pipeline stage order** is fixed: load → reference → artifacts → epoch → baseline → spectral.
   Reordering requires updating tests and docs.
2. **Artifacts are detected on referenced *continuous* data, before epoching.**
3. **Baselining target** is epochs if epoching is enabled, else the referenced raw.
4. **Registry membership**: any config method string must be added to the corresponding `*_METHODS`
   set in `workflow.py` *and* the matching `Literal` in `config.py`.
5. **Schema columns**: artifact and baseline DataFrames always carry the full column tuple from
   `schemas.py`, even when empty. Adding an output column means editing `schemas.py`.
6. **Epoch metadata length** must equal `len(event_times)` when metadata is supplied.

### Data / domain assumptions
7. **Channel naming is lowercased** after load (spaces stripped). MSSM sEEG uses hemisphere prefixes
   `l`/`r` (e.g. `la1`, `rh3`); micros prefix `u` (dropped unless `include_micros=True`). Iowa uses
   `lfpx{N}`. Bipolar labels are `la1-la2` style. NLX names derive from filename stems (`_0000` stripped).
8. **Site branching**: the Iowa site string is `"UI"` (not `"Iowa"`), MSSM is the default. Electrode
   coordinate columns differ (`x/y/z` for MSSM vs `mni_x/y/z` for Iowa); WM reference needs a `manual`
   column (+ `gm`) at MSSM.
9. **Electrode contract**: `load_electrode_metadata` requires only a `label` column; MSSM white-matter
   reference additionally requires `label,x,y,z,manual` (raises if `manual` missing). ROI analysis expects
   `NMM`, `BN246`, `YBA_1`, `collapsed_manual`, and a derived `salman_region`.
10. **Time units**: `event_times` are in **seconds**; `neural_time = beh_time * slope + offset`. Sync
    tools may return ms → seconds. Mixing ms and seconds is the single most common user error.
11. **Sampling / filtering defaults**: resample 500 Hz; line noise 60 Hz with notches at 60/120/180/240.
    Neuralynx: all loaded channels must share one sampling rate (stable API enforces this).
12. **MNE type expectations** per stage: `preprocess_lfp`/`detect_artifacts` want continuous `Raw`;
    `make_epochs` takes `Raw` → `Epochs`; stable FOOOF works on `Epochs` only; TFR utilities expect
    `(n_epochs, n_channels, n_freqs, n_times)`; connectivity expects `(n_epochs, n_channels, n_times)`.

### Determinism
13. Oscillation/surrogate functions are reproducible with explicit `rng_seed` (default 42).
    `statistics_utils` permutations are **not** seeded and are documented as non-reproducible unless
    the caller sets a global NumPy seed.

---

## 13. Incomplete / stub / hazardous surfaces

Stubs and unused helpers are soft-archived in [`LFPAnalysis/_scratch_utils.py`](../LFPAnalysis/_scratch_utils.py).
Original module names still import (with `DeprecationWarning`) and stubs raise `NotImplementedError`
with a pointer to the supported API.

| Symbol | Status |
|--------|--------|
| `laplacian` reference | Omitted from `REFERENCE_METHODS` / `ReferenceMethod` (false-support trap removed). Utility `laplacian_ref` is archived. |
| `analysis_utils.FOOOF_continuous`, `sliding_FOOOF` | Archived stubs → `NotImplementedError`. Prefer `FOOOF_compute_epochs` / spectral workflow. |
| `sync_utils.get_behav_ts` | Archived stub → `NotImplementedError`. |
| `nlx_utils.merge_multiple_ncs_files` | Archived stub → `NotImplementedError`. |
| `iowa_utils.rename_mne_channels` | Archived incomplete helper → `NotImplementedError`. |
| `lfp_preprocess_utils.match_elec_names` | Default `interactive=False` raises `ValueError` on ambiguous Levenshtein ties (CI/agent safe). Pass `interactive=True` only in a human terminal. |

### Dual-spine promotions (landed on `feature/dual-spine-replatform`; more planned for 2.0)

1. **Sync** — typed via `SyncConfig` / `run_prep` (prep spine only).
2. **TFR** — typed via `TfrConfig` / `run_analysis` (Morlet beginner path); full legacy orchestrator parity still planned.
3. **Advanced package** — `LFPAnalysis.advanced` lazy exports; mega-module split still planned for 2.0.

---

## 14. Common pitfalls (condensed)

- Importing utility modules before trying the stable API — leads to wrong assumptions about defaults.
- Assuming legacy shims imply full stable-API parity — connectivity and full TFR orchestration remain advanced.
- Using `laplacian` reference (not registered), or running FOOOF on continuous raw via the stable API.
- Passing `build_spectral_pipeline_config` a TFR/connectivity method (raises `ConfigurationError`).
- Channel-label mismatches: referencing failures are usually electrode-table problems, not signal
  problems — validate `label` against MNE `ch_names`.
- Baseline window outside the data time axis (now raises `ConfigurationError`).
- Empty artifact tables are valid (method was `none`, or thresholds found nothing) — not an error.
- Installing base only and expecting fooof / mne-connectivity — use `.[analysis]` or `.[dev]`.

---

## 15. Refactor-readiness checklist

When planning a large-scale change, work through these:

1. **Which layer?** Beginner-facing behavior → stable API (`config`→`workflow`→`builders`→tests→book).
   Analysis/site-specific → utility module + chapter 11 doc. Old-call compatibility → `legacy.py` shim.
2. **Touching a config string?** Update both the `*_METHODS` set (`workflow.py`) and the `Literal`
   (`config.py`), and add validation/tests.
3. **Changing an output table?** Update `schemas.py` column tuples and every consumer + `test_schemas.py`.
4. **Renaming a public symbol?** Grep the book (`test_docs_content.py` asserts public-surface naming),
   the migration chapters, worked examples, and `__init__.__all__`.
5. **Renaming a utility symbol?** There is no `__all__` guard — assume external notebooks depend on it;
   prefer a deprecation shim over a hard rename.
6. **Preserve memory guardrails:** `WORKING_DTYPE=float32`, lazy `preload`, generator surrogates, and
   dropping superseded stages in `PipelineResult`.
7. **Preserve determinism contracts:** seedable oscillation/surrogate code; explicitly-unseeded stats.
8. **Keep heavy deps lazy:** import optional packages only through `ensure_dependency`, never at
   stable-layer module top level.
9. **Run the gates:** `nox -s lint`, `nox -s tests` (80% coverage on workflow/builders/legacy); if the
   book or notebooks changed, `nox -s docs` and `nox -s notebooks`.
10. **Do not treat `scripts/` as canonical**, do not commit `notebooks/`, and do not use `data/` in unit
    tests.

---

## 16. Quick entry-point reference

| Goal | Entry point |
|------|-------------|
| Load sample data | `build_basic_pipeline_config` + `run_pipeline` |
| Event-locked analysis | `build_event_locked_pipeline_config` + `run_pipeline` |
| PSD / FOOOF | `build_spectral_pipeline_config` + `run_pipeline` |
| Old notebook code | `LFPAnalysis.legacy` + migration chapters 20–26 |
| TFR / connectivity | utility modules after stable preprocessing; book chapters 09–11 |
| ROI-based channel selection | `analysis_utils.select_picks_rois`, `select_rois_picks` |
| Behavioral synchronization | `sync_utils.synchronize_data` / `synchronize_data_robust` |
| Time-resolved statistics | `statistics_utils.time_resolved_mlm` |
```
