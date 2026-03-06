"""Stable beginner-facing workflow API built on top of legacy utility modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import nlx_utils
from .config import (
    ArtifactConfig,
    BaselineConfig,
    EpochConfig,
    LoadConfig,
    PipelineConfig,
    ReferenceConfig,
    SpectralConfig,
)
from .exceptions import ConfigurationError, MissingDependencyError
from .results import PipelineResult
from .schemas import (
    ELECTRODE_REQUIRED_COLUMNS,
    build_baseline_summary,
    build_event_table,
    empty_baseline_summary,
    empty_event_table,
)
from .validation import (
    ensure_dependency,
    ensure_supported,
    normalize_name_list,
    resolve_existing_path,
    validate_required_columns,
)


REFERENCE_METHODS = {"none", "bipolar", "wm", "laplacian"}
ARTIFACT_METHODS = {"none", "misc", "ied", "custom"}
BASELINE_METHODS = {
    "none",
    "mean",
    "ratio",
    "percent",
    "zscore",
    "logratio",
    "zlogratio",
    "trialwise",
    "continuous",
}
SPECTRAL_METHODS = {"none", "psd", "fooof"}


def _require_mne():
    return ensure_dependency("mne", install_hint="pip install -e .[dev]")


def _legacy_preprocess_module():
    return ensure_dependency("LFPAnalysis.lfp_preprocess_utils", install_hint="pip install -e .[dev]")


def load_electrode_metadata(path: str | Path) -> pd.DataFrame:
    """Load and validate electrode metadata from CSV or XLSX files."""
    path_obj = resolve_existing_path(path, field_name="electrode_path")
    if path_obj.suffix.lower() == ".csv":
        dataframe = pd.read_csv(path_obj)
    elif path_obj.suffix.lower() in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(path_obj)
    else:
        raise ConfigurationError("electrode_path must point to a CSV or Excel file.")
    return validate_required_columns(
        dataframe,
        required_columns=ELECTRODE_REQUIRED_COLUMNS,
        schema_name="Electrode metadata",
    )


def load_lfp(config: LoadConfig):
    """Load LFP data into an MNE Raw or Epochs object."""
    ensure_supported(config.file_format, field_name="file_format", supported=("edf", "neuralynx", "mne"))
    mne = _require_mne()

    if hasattr(config.path, "info") and hasattr(config.path, "copy"):
        data = config.path.copy()
    elif config.file_format == "mne":
        path = resolve_existing_path(config.path, field_name="path")
        if path.suffix.lower() == ".fif" and ("epo" in path.name or path.name.endswith("-epo.fif")):
            data = mne.read_epochs(path, preload=config.preload)
        else:
            data = mne.io.read_raw_fif(path, preload=config.preload)
    elif config.file_format == "edf":
        path = resolve_existing_path(config.path, field_name="path")
        data = mne.io.read_raw_edf(path, preload=config.preload)
    else:
        if isinstance(config.path, (str, Path)):
            base_path = resolve_existing_path(config.path, field_name="path")
            ncs_files = sorted(base_path.glob("*.ncs")) if base_path.is_dir() else [base_path]
        else:
            ncs_files = [resolve_existing_path(path, field_name="path") for path in config.path]
        if not ncs_files:
            raise ConfigurationError("No Neuralynx .ncs files were found for loading.")
        signals, srs, ch_names, ch_types = nlx_utils.parse_subject_nlx_data(
            [str(path) for path in ncs_files],
            eeg_names=normalize_name_list(config.eeg_names),
            resp_names=normalize_name_list(config.resp_names),
            ekg_names=normalize_name_list(config.ekg_names),
            seeg_names=normalize_name_list(config.seeg_names),
            drop_names=normalize_name_list(config.drop_names),
            include_micros=config.include_micros,
        )
        if not signals:
            raise ConfigurationError("No valid Neuralynx channels were loaded from the provided files.")
        if len(set(srs)) != 1:
            raise ConfigurationError("Neuralynx channels must share a common sampling rate.")
        info = mne.create_info(ch_names=ch_names, sfreq=float(srs[0]), ch_types=ch_types)
        data = mne.io.RawArray(np.vstack(signals), info)

    if config.pick_channels and hasattr(data, "pick"):
        data.pick(config.pick_channels)
    if config.resample_sfreq and hasattr(data, "resample"):
        data.resample(config.resample_sfreq)
    return data


def preprocess_lfp(data, config: ReferenceConfig):
    """Apply an optional re-referencing step to continuous data."""
    ensure_supported(config.method, field_name="reference.method", supported=REFERENCE_METHODS)
    if config.method == "none":
        return data.copy() if hasattr(data, "copy") else data
    if not hasattr(data, "copy"):
        raise ConfigurationError("Reference operations require an MNE Raw-like object.")
    if not config.electrode_path:
        raise ConfigurationError("Reference methods other than 'none' require electrode_path.")

    electrode_path = str(resolve_existing_path(config.electrode_path, field_name="electrode_path"))
    legacy = _legacy_preprocess_module()
    if config.method in {"wm", "bipolar"}:
        return legacy.ref_mne(
            mne_data=data.copy(),
            elec_path=electrode_path,
            method=config.method,
            site=config.site,
        )
    raise ConfigurationError(
        "The 'laplacian' registry entry is reserved, but the legacy laplacian implementation is not yet complete."
    )


def _artifact_none(data, config: ArtifactConfig) -> pd.DataFrame:
    return empty_event_table()


def _artifact_misc(data, config: ArtifactConfig) -> pd.DataFrame:
    legacy = _legacy_preprocess_module()
    channel_events = legacy.detect_misc_artifacts(data, peak_thresh=config.misc_peak_thresh)
    return build_event_table(channel_events, event_kind="misc", sfreq=float(data.info["sfreq"]))


def _artifact_ied(data, config: ArtifactConfig) -> pd.DataFrame:
    legacy = _legacy_preprocess_module()
    channel_events = legacy.detect_IEDs(
        data,
        peak_thresh=config.ied_peak_thresh,
        closeness_thresh=config.ied_closeness_thresh,
        width_thresh=config.ied_width_thresh,
    )
    return build_event_table(channel_events, event_kind="ied", sfreq=float(data.info["sfreq"]))


_ARTIFACT_REGISTRY = {
    "none": _artifact_none,
    "misc": _artifact_misc,
    "ied": _artifact_ied,
}


def detect_artifacts(data, config: ArtifactConfig) -> dict[str, pd.DataFrame]:
    """Run one or more artifact detectors and return standardized tables."""
    results: dict[str, pd.DataFrame] = {}
    for method in config.methods:
        ensure_supported(method, field_name="artifact.methods", supported=ARTIFACT_METHODS)
        if method == "custom":
            if config.custom_detector is None:
                raise ConfigurationError("artifact.custom_detector is required when using method='custom'.")
            custom_result = config.custom_detector(data)
            if isinstance(custom_result, pd.DataFrame):
                results[method] = custom_result
            elif isinstance(custom_result, dict):
                results[method] = build_event_table(
                    custom_result,
                    event_kind="custom",
                    sfreq=float(data.info["sfreq"]),
                )
            else:
                raise ConfigurationError("custom_detector must return a DataFrame or channel-event mapping.")
            continue
        results[method] = _ARTIFACT_REGISTRY[method](data, config)
    return results


def _get_baseline_indices(times: np.ndarray, window: tuple[float, float] | None) -> np.ndarray:
    if window is None:
        raise ConfigurationError("baseline_window is required when baselining is enabled.")
    start, stop = window
    indices = np.where((times >= start) & (times <= stop))[0]
    if len(indices) == 0:
        raise ConfigurationError("baseline_window does not overlap the data time axis.")
    return indices


def _apply_baseline_array(data: np.ndarray, baseline: np.ndarray, mode: str) -> np.ndarray:
    legacy = _legacy_preprocess_module()
    if mode == "continuous":
        mode = "zscore"
    if mode == "trialwise":
        mode = "zscore"
    return legacy.mean_baseline_time(data, baseline, mode=mode)


def baseline_lfp(data, config: BaselineConfig):
    """Baseline continuous or epoched data using the shared summary schema."""
    ensure_supported(config.mode, field_name="baseline.mode", supported=BASELINE_METHODS)
    if not config.enabled or config.mode == "none":
        return data, empty_baseline_summary()

    mne = _require_mne()
    if isinstance(data, mne.BaseEpochs):
        indices = _get_baseline_indices(data.times, config.baseline_window)
        data_copy = data.copy()
        all_data = data_copy.get_data(copy=True)
        baseline = all_data[:, :, indices]
        corrected = np.empty_like(all_data)
        for epoch_index in range(all_data.shape[0]):
            corrected[epoch_index] = _apply_baseline_array(
                all_data[epoch_index],
                baseline[epoch_index],
                config.mode,
            )
        data_copy._data = corrected
        summary = build_baseline_summary(
            target="epochs",
            channel_names=data_copy.ch_names,
            mode=config.mode,
            baseline_start=config.baseline_window[0],
            baseline_stop=config.baseline_window[1],
            baseline_mean=baseline.mean(axis=(0, 2)),
            baseline_std=baseline.std(axis=(0, 2)),
        )
        return data_copy, summary

    if not hasattr(data, "get_data"):
        raise ConfigurationError("Baselining requires an MNE Raw or Epochs object.")

    data_copy = data.copy()
    time_axis = np.arange(data_copy.n_times) / float(data_copy.info["sfreq"])
    indices = _get_baseline_indices(time_axis, config.baseline_window)
    all_data = data_copy.get_data()
    baseline = all_data[:, indices]
    data_copy._data = _apply_baseline_array(all_data, baseline, config.mode)
    summary = build_baseline_summary(
        target="raw",
        channel_names=data_copy.ch_names,
        mode=config.mode,
        baseline_start=config.baseline_window[0],
        baseline_stop=config.baseline_window[1],
        baseline_mean=baseline.mean(axis=1),
        baseline_std=baseline.std(axis=1),
    )
    return data_copy, summary


def make_epochs(data, config: EpochConfig):
    """Create epochs around event timestamps on an MNE Raw object."""
    if not config.enabled:
        return None
    mne = _require_mne()
    if not hasattr(data, "info") or not hasattr(data, "copy"):
        raise ConfigurationError("Epoch extraction requires an MNE Raw object.")
    if not config.event_times:
        raise ConfigurationError("epoch.event_times must be provided when epoching is enabled.")

    transformed_times = [(time_value * config.slope) + config.offset for time_value in config.event_times]
    events = np.column_stack(
        [
            np.asarray(transformed_times, dtype=float) * float(data.info["sfreq"]),
            np.zeros(len(transformed_times), dtype=int),
            np.ones(len(transformed_times), dtype=int),
        ]
    ).astype(int)
    metadata = None
    if config.metadata:
        metadata = pd.DataFrame(config.metadata)
        if len(metadata) != len(events):
            raise ConfigurationError("epoch.metadata must have the same number of rows as event_times.")
    return mne.Epochs(
        data.copy(),
        events=events,
        event_id={config.event_name: 1},
        tmin=config.tmin,
        tmax=config.tmax,
        baseline=None,
        preload=True,
        metadata=metadata,
        verbose=False,
    )


def compute_spectral_features(data, config: SpectralConfig) -> dict[str, Any]:
    """Compute optional spectral features using the stable registry interface."""
    ensure_supported(config.method, field_name="spectral.method", supported=SPECTRAL_METHODS)
    if not config.enabled or config.method == "none":
        return {}

    if config.method == "psd":
        spectrum = data.compute_psd(fmin=config.fmin, fmax=config.fmax, n_fft=config.n_fft)
        return {"method": "psd", "spectrum": spectrum}

    if not hasattr(data, "compute_psd"):
        raise ConfigurationError("FOOOF features require an MNE Raw or Epochs object.")
    ensure_dependency("fooof", install_hint="pip install -e .[analysis]")
    from . import analysis_utils

    if hasattr(data, "events"):
        fooof_group, fooof_table = analysis_utils.FOOOF_compute_epochs(
            data,
            tmin=float(data.times[0]),
            tmax=float(data.times[-1]),
            **{
                "peak_width_limits": config.fooof_kwargs.get("peak_width_limits", (1, 12)),
                "min_peak_height": config.fooof_kwargs.get("min_peak_height", 0.0),
                "peak_threshold": config.fooof_kwargs.get("peak_threshold", 2.0),
                "max_n_peaks": config.fooof_kwargs.get("max_n_peaks", 6),
                "freq_range": config.fooof_range,
            },
        )
        return {"method": "fooof", "group": fooof_group, "table": fooof_table}
    raise MissingDependencyError(
        "FOOOF computation is currently implemented for epoched data only in the stable workflow layer."
    )


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the stable end-to-end workflow and return structured outputs."""
    raw = load_lfp(config.load)
    referenced = preprocess_lfp(raw, config.reference)
    artifact_tables = detect_artifacts(referenced, config.artifact)
    epochs = make_epochs(referenced, config.epoch)

    target = epochs if epochs is not None else referenced
    baselined_target, baseline_summary = baseline_lfp(target, config.baseline)
    if epochs is not None:
        epochs = baselined_target
    else:
        referenced = baselined_target

    spectral = compute_spectral_features(baselined_target, config.spectral)
    metadata = {
        "input_format": config.load.file_format,
        "reference_method": config.reference.method,
        "artifact_methods": list(config.artifact.methods),
        "baseline_mode": config.baseline.mode,
        "spectral_method": config.spectral.method,
    }
    return PipelineResult(
        raw=raw,
        referenced=referenced,
        epochs=epochs,
        artifact_tables=artifact_tables,
        baseline_summary=baseline_summary,
        spectral=spectral,
        metadata=metadata,
    )
