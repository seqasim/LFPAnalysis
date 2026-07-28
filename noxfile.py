"""Local automation entry points for linting, tests, docs, and notebook smoke runs."""

from __future__ import annotations

import nox

nox.options.sessions = ["lint", "tests"]


@nox.session
def lint(session: nox.Session) -> None:
    session.install("ruff>=0.7")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session
def tests(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    # Preload NumPy before pytest-cov traces package imports (avoids NumPy/pandas reload breakage).
    session.run(
        "python",
        "-c",
        "import numpy; import pytest; raise SystemExit(pytest.main(["
        "'-m', 'not notebook and not slow', "
        "'--cov=LFPAnalysis.workflow', "
        "'--cov=LFPAnalysis.builders', "
        "'--cov=LFPAnalysis.legacy', "
        "'--cov-fail-under=80'"
        "]))",
    )


@nox.session
def docs(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.chdir("LFPAnalysisBook")
    session.run("jupyter-book", "build", "--html", "--ci")


@nox.session
def notebooks(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run(
        "pytest",
        "--nbmake",
        "--nbmake-timeout=1200",
        "LFPAnalysisBook/smoke-tests",
        "LFPAnalysisBook/worked-examples/03_first_import_and_load.ipynb",
        "LFPAnalysisBook/worked-examples/04_first_preprocessing_run.ipynb",
        "LFPAnalysisBook/worked-examples/04b_first_synchronization_run.ipynb",
        "LFPAnalysisBook/worked-examples/05_first_artifact_pass.ipynb",
        "LFPAnalysisBook/worked-examples/06_first_baseline_run.ipynb",
        "LFPAnalysisBook/worked-examples/07_first_epoching_run.ipynb",
        "LFPAnalysisBook/worked-examples/08_first_psd_and_fooof_run.ipynb",
        "LFPAnalysisBook/worked-examples/09_first_tfr_run.ipynb",
        "LFPAnalysisBook/worked-examples/10_first_connectivity_run.ipynb",
        "LFPAnalysisBook/worked-examples/10b_first_stats_run.ipynb",
        "LFPAnalysisBook/worked-examples/13_assembling_dataframes.ipynb",
        "LFPAnalysisBook/worked-examples/14_group_statistics.ipynb",
        "LFPAnalysisBook/worked-examples/22_migrating_condensed_notebook.ipynb",
    )
