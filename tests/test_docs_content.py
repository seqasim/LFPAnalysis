"""Content checks for the beginner and migration documentation tracks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "LFPAnalysisBook"
ADVANCED_UTILITY_CHAPTER = BOOK / "11_advanced_utility_interoperability.md"

BEGINNER_CHAPTERS = [
    BOOK / "03_first_load.md",
    BOOK / "04_first_reference.md",
    BOOK / "05_first_artifact_pass.md",
    BOOK / "06_first_baseline.md",
    BOOK / "07_first_event_locked_workflow.md",
    BOOK / "08_first_psd_and_fooof.md",
    BOOK / "09_first_time_frequency.md",
    BOOK / "10_first_connectivity_and_surrogates.md",
]
MIGRATION_CHAPTERS = [
    BOOK / "20_old_repo_mental_model.md",
    BOOK / "21_legacy_function_mapping.md",
    BOOK / "22_translate_condensed_notebook.md",
    BOOK / "23_translate_tfr_workflow.md",
    BOOK / "24_translate_connectivity_workflow.md",
    BOOK / "25_legacy_only_surfaces.md",
]


@pytest.mark.unit
def test_beginner_chapters_share_teaching_structure():
    required_headings = [
        "## What this step is for",
        "## When you should use it",
        "## Required inputs",
        "## Minimal example",
        "## How to inspect the result",
        "## Common mistakes",
        "## Old-to-new translation note",
    ]
    for path in BEGINNER_CHAPTERS:
        text = path.read_text()
        for heading in required_headings:
            assert heading in text, f"{path.name} is missing {heading}"
        assert "Next step:" in text


@pytest.mark.unit
def test_migration_chapters_include_old_and_new_code_examples():
    for path in MIGRATION_CHAPTERS:
        text = path.read_text()
        assert "### Old workflow" in text
        assert "### New workflow" in text
        assert text.count("```python") >= 2


@pytest.mark.unit
def test_interface_guide_names_all_public_surfaces():
    text = (BOOK / "00_interface_guide.md").read_text()
    assert "stable beginner-facing API" in text
    assert "compatibility/legacy shims" in text
    assert "advanced legacy utilities" in text
    assert "11_advanced_utility_interoperability" in text


@pytest.mark.unit
def test_advanced_utility_chapter_names_the_shared_module_stack():
    text = ADVANCED_UTILITY_CHAPTER.read_text()
    for module_name in [
        "iowa_utils",
        "nlx_utils",
        "sync_utils",
        "lfp_preprocess_utils",
        "analysis_utils",
        "oscillation_utils",
        "statistics_utils",
    ]:
        assert module_name in text
    assert "## Shared conventions after cleanup" in text


@pytest.mark.unit
def test_worked_notebooks_reference_advanced_utility_guidance():
    notebook_paths = [
        BOOK / "worked-examples" / "04_first_psd_and_fooof_run.ipynb",
        BOOK / "worked-examples" / "06_first_connectivity_run.ipynb",
        BOOK / "worked-examples" / "07_migrating_condensed_notebook.ipynb",
    ]
    for path in notebook_paths:
        notebook = json.loads(path.read_text())
        markdown = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        )
        assert "11_advanced_utility_interoperability" in markdown


@pytest.mark.unit
def test_worked_notebooks_exist_for_beginner_and_migration_paths():
    beginner = BOOK / "worked-examples" / "01_first_import_and_load.ipynb"
    migration = BOOK / "worked-examples" / "07_migrating_condensed_notebook.ipynb"
    for path in [beginner, migration]:
        notebook = json.loads(path.read_text())
        assert notebook["cells"]
        markdown = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        )
        assert "## Goal" in markdown
        assert "## Next step" in markdown or "## Next Step" in markdown
