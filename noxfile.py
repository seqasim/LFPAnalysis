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
    session.run(
        "pytest",
        "-m",
        "not notebook and not slow",
        "--cov=LFPAnalysis.workflow",
        "--cov-fail-under=80",
    )


@nox.session
def docs(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("jupyter-book", "build", "LFPAnalysisBook")


@nox.session
def notebooks(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run(
        "pytest",
        "--nbmake",
        "--nbmake-timeout=1200",
        "LFPAnalysisBook/smoke-tests",
    )
