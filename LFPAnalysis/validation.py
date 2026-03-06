"""Validation helpers used by the stable workflow layer."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from random import Random
from typing import Iterable, Sequence

import pandas as pd

from .exceptions import ConfigurationError, DataContractError, MissingDependencyError


PATHLIKE_TYPES = (str, Path)


def ensure_supported(value: str, *, field_name: str, supported: Sequence[str]) -> str:
    """Validate that a string config value belongs to a supported set."""
    if value not in supported:
        supported_text = ", ".join(sorted(supported))
        raise ConfigurationError(
            f"Unsupported {field_name} '{value}'. Supported values: {supported_text}."
        )
    return value


def ensure_dependency(module_name: str, *, install_hint: str | None = None):
    """Import a dependency or raise a workflow-specific error."""
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        hint = f" Install with `{install_hint}`." if install_hint else ""
        raise MissingDependencyError(
            f"The '{module_name}' dependency is required for this operation.{hint}"
        ) from exc


def resolve_existing_path(path_value: str | Path, *, field_name: str) -> Path:
    """Resolve a path-like value and ensure it exists."""
    if not isinstance(path_value, PATHLIKE_TYPES):
        raise ConfigurationError(f"{field_name} must be a filesystem path.")
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise ConfigurationError(f"{field_name} does not exist: {path}")
    return path


def normalize_name_list(values: Iterable[str] | None) -> list[str]:
    """Normalize optional channel-name lists."""
    if not values:
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def validate_required_columns(
    dataframe: pd.DataFrame,
    *,
    required_columns: Sequence[str],
    schema_name: str,
) -> pd.DataFrame:
    """Validate that a dataframe satisfies a required column contract."""
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise DataContractError(f"{schema_name} is missing required columns: {missing_text}.")
    return dataframe


def stable_random(seed: int | None) -> Random:
    """Return a deterministic Python RNG wrapper."""
    return Random(seed if seed is not None else 0)
