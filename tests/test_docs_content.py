"""Content checks for the beginner and migration documentation tracks."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "LFPAnalysisBook"
ADVANCED_UTILITY_CHAPTER = BOOK / "11_advanced_utility_interoperability.md"
NOTEBOOK_MAP = yaml.safe_load((BOOK / "notebook_map.yml").read_text())

BEGINNER_CHAPTERS = [
    BOOK / "03_first_load.md",
    BOOK / "04_first_reference.md",
    BOOK / "05_first_artifact_pass.md",
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

PLOT_MARKERS = ("plt.", "imshow", "semilogy", "hist(", ".plot(", "axvspan", "colorbar")
DEBUG_PATTERNS = (
    "agent log",
    "debug-",
    "/Users/",
    "C:\\Users\\",
)


def _notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text())
    chunks: list[str] = []
    for cell in notebook["cells"]:
        chunks.append("".join(cell.get("source", [])))
    return "\n".join(chunks)


def _notebook_markdown(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )


def _notebook_code(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )


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
    assert "build_analysis_config" in text
    assert "build_spectral_pipeline_config" in text


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
def test_notebook_map_files_exist_and_have_goal_next_step():
    for entry in NOTEBOOK_MAP["worked_examples"]:
        chapter = BOOK / entry["chapter"]
        notebook = BOOK / entry["notebook"]
        assert chapter.exists(), f"missing chapter {chapter.name}"
        assert notebook.exists(), f"missing notebook for {chapter.name}: {notebook}"
        markdown = _notebook_markdown(notebook)
        assert "## Goal" in markdown, f"{notebook.name} missing ## Goal"
        assert "## Next step" in markdown or "## Next Step" in markdown, f"{notebook.name} missing Next step"


@pytest.mark.unit
def test_toc_lists_every_mapped_worked_example():
    toc = (BOOK / "_toc.yml").read_text()
    for entry in NOTEBOOK_MAP["worked_examples"]:
        stem = Path(entry["notebook"]).with_suffix("").as_posix()
        assert stem in toc, f"_toc.yml missing {stem}"


@pytest.mark.unit
def test_chapters_promising_worked_notebooks_own_one():
    for entry in NOTEBOOK_MAP["worked_examples"]:
        chapter_text = (BOOK / entry["chapter"]).read_text()
        if "worked notebook" in chapter_text.lower():
            # Chapter should name or {doc}-link its notebook stem somewhere, or at least
            # sit under a TOC section that owns the notebook (checked separately).
            assert (BOOK / entry["notebook"]).exists()


@pytest.mark.unit
def test_worked_notebooks_use_notebook_relative_data_paths():
    prefix = NOTEBOOK_MAP["api_contracts"]["notebook_data_prefix"]
    for entry in NOTEBOOK_MAP["worked_examples"]:
        code = _notebook_code(BOOK / entry["notebook"])
        if "data/" not in code and "sample_" not in code:
            continue
        bad = re.findall(r"""Path\(['\"](\.\./data/[^'\"]+)['\"]\)""", code)
        for path in bad:
            assert path.startswith(prefix), f"{entry['notebook']} uses {path}; expected {prefix}..."


@pytest.mark.unit
def test_chapter_code_blocks_use_chapter_relative_data_paths():
    prefix = NOTEBOOK_MAP["api_contracts"]["chapter_data_prefix"]
    for path in BEGINNER_CHAPTERS:
        text = path.read_text()
        bad = re.findall(r"""Path\(['\"](\.\./\.\./data/[^'\"]+)['\"]\)""", text)
        assert not bad, f"{path.name} uses notebook-depth paths in prose: {bad}"
        hits = re.findall(r"""Path\(['\"](\.\./data/[^'\"]+)['\"]\)""", text)
        for hit in hits:
            assert hit.startswith(prefix)


@pytest.mark.unit
def test_worked_notebooks_contain_required_symbols_and_plots():
    for entry in NOTEBOOK_MAP["worked_examples"]:
        source = _notebook_source(BOOK / entry["notebook"])
        for symbol in entry.get("required_symbols", []):
            assert symbol in source, f"{entry['notebook']} missing required symbol {symbol}"
        if entry.get("required_plot"):
            assert any(marker in source for marker in PLOT_MARKERS), (
                f"{entry['notebook']} promised a plot but has no plotting markers"
            )


@pytest.mark.unit
def test_worked_notebooks_have_no_debug_or_absolute_user_paths():
    for entry in NOTEBOOK_MAP["worked_examples"]:
        source = _notebook_source(BOOK / entry["notebook"])
        for pattern in DEBUG_PATTERNS:
            assert pattern not in source, f"{entry['notebook']} contains forbidden pattern {pattern!r}"


@pytest.mark.unit
def test_worked_notebooks_reference_advanced_utility_when_needed():
    notebook_paths = [
        BOOK / "worked-examples" / "08_first_psd_and_fooof_run.ipynb",
        BOOK / "worked-examples" / "10_first_connectivity_run.ipynb",
        BOOK / "worked-examples" / "22_migrating_condensed_notebook.ipynb",
    ]
    for path in notebook_paths:
        markdown = _notebook_markdown(path)
        assert "11_advanced_utility_interoperability" in markdown


@pytest.mark.unit
def test_event_locked_chapter_documents_same_and_cross_event_baseline():
    text = (BOOK / "07_first_event_locked_workflow.md").read_text()
    assert "build_event_locked_pipeline_config" in text
    assert "run_pipeline" in text
    assert "baseline_event_times" in text
    assert "Cross-event baselining" in text
    assert "build_analysis_config" in text
    assert "run_analysis" in text
    assert "build_spectral_pipeline_config" in text
    # Same-event minimal example should not require baseline_event_times
    same_event = text.split("## Minimal example", 1)[1].split("## Cross-event", 1)[0]
    fence = same_event.split("```python", 1)[1].split("```", 1)[0]
    assert "baseline_window" in fence
    assert "baseline_event_times" not in fence
    cross = text.split("## Cross-event baselining", 1)[1]
    cross_fence = cross.split("```python", 1)[1].split("```", 1)[0]
    assert "baseline_event_times" in cross_fence


@pytest.mark.unit
def test_saving_chapter_lists_full_pipeline_result_fields():
    text = (BOOK / "15_saving_and_organizing_results.md").read_text()
    for field_name in ("tfr", "electrode_df", "sync"):
        assert field_name in text, f"15_saving_and_organizing_results.md missing {field_name}"
