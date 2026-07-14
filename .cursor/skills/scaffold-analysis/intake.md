# Intake Question Bank

Ask **one** question per turn. Prefer multiple choice. Always give a **recommended** answer.

Skip any item already answered in the user's description. Pick the highest-uncertainty unanswered branch next.

## 1. Preprocessing stage

**Ask:** What stage is the neural data at?

| Choice | Meaning |
|--------|---------|
| A. Raw (EDF / Neuralynx / continuous FIF) | Needs load → reference → artifacts → epochs |
| B. Re-referenced continuous | Needs epoching (and maybe artifacts) |
| C. Event-locked MNE Epochs (re-referenced) | Ready for TFR / band power / spectral |
| D. TFR already computed | Skip TFR; go to band-power averages / stats |

**Recommended default:** C if the user mentions preprocessed / re-referenced epochs; A if they only mention raw recordings.

**Maps to:** include `preprocess_to_epochs` only for A or B.

## 2. Modalities / files present

**Ask:** Which of these do you have (or will provide paths for)?

- Neural (epochs or continuous)
- Behavioral table (trial metadata / regressors)
- Electrode / bipolar labels + ROI columns
- Anatomy (localization) beyond electrode table

**Recommended default:** neural + behavior if they describe a task; add electrodes if they mention ROI or anatomy.

**Maps to:** `PATHS` keys in [notebook-conventions.md](notebook-conventions.md).

## 3. Subject structure

**Ask:** Single participant or multi-participant group analysis?

- Single subject (explore / electrode-wise permutation OLS)
- Multiple subjects (group MLM / hierarchical designs)

**Recommended default:** single unless they mention cohort / patients / group.

**Maps to:** `regression_avg` vs also `regression_timeresolved` + `dataframe_assembly`; warn that `time_resolved_mlm` needs multiple higher-level units.

## 4. Event structure (if stage ≤ epochs or analysis is event-locked)

**Ask:** What is the locking event, epoch window (`tmin`/`tmax`), and baseline?

**Recommended default for reward / feedback-style tasks:** lock to feedback onset, window ≈ `-0.5` to `1.5` s (or user's known window), baseline from pre-event period used in their preprocessing.

If epochs already exist, ask only for the event name in their filename/metadata and whether baseline was already applied.

## 5. Behavioral regressors / conditions

**Ask:** Which conditions or continuous regressors matter?

Examples: binary `reward`, continuous `rpe`, RT, condition labels.

**Recommended default:** use the variables they named in the task description; for bandit/feedback demos use `reward` + `rpe` like the sample dataset.

**Maps to:** metadata columns attached to epochs; formula strings in regression notebooks.

## 6. Target analyses

**Ask:** Which analyses do you want notebooks for?

| Goal | Recipe IDs |
|------|------------|
| Baselined TFRs / condition TFR contrast | `tfr_baselined` |
| Time-averaged band power extraction | `band_power` |
| Power ~ behavior (single window / averaged) | `regression_avg` |
| Time-resolved regression / MLM | `regression_timeresolved` |
| PSD / FOOOF | `psd_fooof` |
| Tidy long dataframe for stats | `dataframe_assembly` |
| Connectivity + surrogates | `connectivity` |
| Build epochs from earlier stage | `preprocess_to_epochs` |

**Recommended default** when they want TFR → band power → regressions on preprocessed epochs:

`tfr_baselined`, `band_power`, `regression_avg`, `regression_timeresolved`

## 7. Slug / output folder

**Ask:** What short folder name under `notebooks/` should hold these notebooks?

**Recommended default:** kebab-case from task + analysis, e.g. `feedback-tfr-regression` or `sub-all-reward-power`.

## Stop conditions

End intake when stage, modalities, subject structure, events (if needed), regressors, recipe IDs, and slug are known. Then write the Dataset Profile and proceed to recipe selection in [recipes.md](recipes.md).
