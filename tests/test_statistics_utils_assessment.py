import numpy as np
import pandas as pd

from LFPAnalysis import statistics_utils


def test_permutation_regression_zscore_matches_manual_surrogate_stats(monkeypatch):
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 4.0, 8.0],
            "x": [0.0, 1.0, 0.0, 1.0],
        }
    )
    surrogate_ys = iter(
        [
            np.array([8.0, 4.0, 2.0, 1.0]),
            np.array([2.0, 1.0, 8.0, 4.0]),
        ]
    )
    monkeypatch.setattr(
        statistics_utils.np.random,
        "permutation",
        lambda values: next(surrogate_ys),
    )

    results = statistics_utils.permutation_regression_zscore(
        data,
        "y ~ x",
        n_permutations=2,
    ).set_index("predictor")

    y, X = statistics_utils.patsy.dmatrices("y ~ x", data, return_type="dataframe")
    original_params = statistics_utils.OLS(y, X).fit().params.to_numpy()
    surrogate_params = np.vstack(
        [
            statistics_utils.OLS(np.array([8.0, 4.0, 2.0, 1.0]), X).fit().params.to_numpy(),
            statistics_utils.OLS(np.array([2.0, 1.0, 8.0, 4.0]), X).fit().params.to_numpy(),
        ]
    )
    expected_z = (original_params - surrogate_params.mean(axis=0)) / surrogate_params.std(
        axis=0,
        ddof=1,
    )

    np.testing.assert_allclose(results["z_beta"].to_numpy(), expected_z)


def test_shuffle_data_for_mlm_reassigns_trials_within_subject_and_label(monkeypatch):
    df = pd.DataFrame(
        {
            "participant": ["p1", "p1", "p1", "p1", "p2", "p2", "p2", "p2"],
            "unique_label": ["a", "a", "b", "b", "a", "a", "b", "b"],
            "trial": [1, 2, 1, 2, 1, 2, 1, 2],
            "tfr": [10, 20, 100, 200, 30, 40, 300, 400],
        }
    )
    monkeypatch.setattr(
        statistics_utils.np.random,
        "permutation",
        lambda values: values[::-1],
    )

    shuffled = statistics_utils.shuffle_data_for_mlm(df)

    expected = pd.Series([20, 10, 200, 100, 40, 30, 400, 300], name="tfr")
    pd.testing.assert_series_equal(shuffled["tfr"].reset_index(drop=True), expected)
