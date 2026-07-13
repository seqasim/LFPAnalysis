# Group-Level Statistics

## What this step is for

Once you have a tidy analysis dataframe (chapter 13), the same statistical patterns recur across measures: **permutation OLS** for single-electrode / single-time tests, and **parallelized time-resolved mixed-effects models** when electrodes nest inside patients. This chapter is the reusable stats playbook built on `statistics_utils`.

## When you should use it

- Per-electrode continuous regressors (e.g. power ~ RPE) → `permutation_regression_zscore`
- Time-resolved power across many electrodes and patients → `time_resolved_mlm`
- Asking whether a regional effect is consistent across patients (not driven by one patient contributing many electrodes) → mixed-effects intercept model (API currently commented out; pattern documented below)

## Required inputs

- A long dataframe with columns matching `statistics_utils` defaults: dependent variable (`tfr` or rename), `ts`, `unique_label`, `participant`, `trial`, plus regressors
- `statsmodels` and `joblib` (via `.[analysis]` / `.[dev]`)
- For the multi-patient demo below: **no extra data files** — we synthesize a small dataframe inline because the packaged sample is single-patient

## Minimal example

### 1. Permutation OLS (single unit / single time)

```python
import numpy as np
import pandas as pd
from LFPAnalysis import statistics_utils

np.random.seed(42)  # permutations are not seeded inside statistics_utils

n = 80
df = pd.DataFrame(
    {
        "power": np.random.randn(n) + 0.3 * np.linspace(-1, 1, n),
        "rpe": np.linspace(-1, 1, n) + 0.1 * np.random.randn(n),
    }
)
results = statistics_utils.permutation_regression_zscore(
    df, "power ~ rpe", n_permutations=200
)
print(results)
```

Returns `raw_beta`, `raw_p`, `z_beta`, and `z_p` per predictor. Use ≥1000 permutations for publication; 100–200 is fine for smoke tests and CI.

### 2. Time-resolved mixed-effects with hierarchical shuffles

`time_resolved_mlm` fits `smf.mixedlm` at each `ts`, then builds a null by shuffling trial labels **within participant** (`shuffle_data_for_mlm`) while preserving electrode structure. Surrogate fits yield `z_beta`, `z_p`, and empirical `count_p`. Timepoints are parallelized with joblib (`n_jobs=-1` by default).

```python
import numpy as np
import pandas as pd
from LFPAnalysis import statistics_utils

np.random.seed(0)
rng = np.random.default_rng(0)

participants = ["P01", "P02", "P03"]
electrodes = {"P01": ["e1", "e2"], "P02": ["e1"], "P03": ["e1", "e2", "e3"]}
times = np.array([-0.2, 0.0, 0.2, 0.4])
rows = []
for p in participants:
    n_trials = 30
    rpe = rng.normal(size=n_trials)
    for elec in electrodes[p]:
        for trial in range(n_trials):
            for ts in times:
                # Plant a small RPE effect after t=0
                signal = 0.4 * rpe[trial] * (ts > 0) + rng.normal(scale=1.0)
                rows.append(
                    {
                        "participant": p,
                        "unique_label": f"{p}_{elec}",
                        "trial": trial,
                        "ts": ts,
                        "tfr": signal,
                        "rpe": rpe[trial],
                    }
                )
smoothed_df = pd.DataFrame(rows)

# Keep permutations small for a book/CI-friendly run
mlm_res = statistics_utils.time_resolved_mlm(
    smoothed_df,
    y="tfr",
    formula="tfr ~ 1 + rpe",
    lower_group="unique_label",
    higher_group="participant",
    trial_key="trial",
    n_permutations=20,
    n_jobs=1,
)
print(mlm_res.head())
print(mlm_res.loc[mlm_res.parameter == "rpe", ["ts", "raw_beta", "z_beta", "z_p", "count_p"]])
```

Helpers you will also see in the call stack: `process_single_timepoint`, `generate_surrogate_results`, `shuffle_data_for_mlm`.

### 3. Region effects across patients (documented pattern; helper commented out)

A common question: given one summary metric per electrode (e.g. peak beta), is the regional mean different from zero **after accounting for patient**? Naively t-testing all electrodes against zero overweights patients with denser coverage.

The intended helper `mixed_effects_electrodes` in `statistics_utils` is currently **commented out**. The intended model is an intercept-only mixed model with patient as the grouping factor:

```python
import statsmodels.formula.api as smf

# model_df: one row per electrode, columns include metric + participant
# results = smf.mixedlm("metric ~ 1", data=model_df, groups=model_df["participant"]).fit()
# print(results.summary())
```

Until the helper is restored, use `smf.mixedlm` directly as above, or stick to `time_resolved_mlm` when your data are already in long trial×time form.

## How to inspect the result

- For OLS: `z_beta` / `z_p` for the predictor of interest (not only `raw_p`)
- For MLM: traces of `z_beta` vs `ts`; compare `z_p` (normal approximation of the surrogate distribution) with `count_p` (proportion of |surrogate| ≥ |observed|)
- Confirm hierarchical shuffle: null should destroy predictor–neural coupling within patient without inventing impossible electrode×trial combinations

## Common mistakes

- Calling `time_resolved_mlm` on single-patient data with many electrodes — random effects need multiple higher-level units; use permutation OLS per electrode instead, or the synthetic multi-patient pattern above
- Too few permutations for stable tails (especially `count_p`)
- Forgetting `np.random.seed` before demos (surrogates use unseeded `np.random.permutation`)
- Formula / `y` mismatch (`formula` must include the dependent variable name for mixedlm)
- Setting `n_jobs=-1` on a shared CI machine without need — prefer `n_jobs=1` for documentation builds

## Old-to-new translation note

Legacy `TimeResolvedRegression.ipynb` chains the same conceptual steps. Prefer `permutation_regression_zscore` for single-series tests and `time_resolved_mlm` for the parallel multi-electrode path; assemble the dataframe as in chapter 13 first.

Next step: {doc}`15_saving_and_organizing_results`
