"""Stable beginner-facing workflow API built on top of legacy utility modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from . import nlx_utils
from .config import (
    WORKING_DTYPE,
    AnalysisConfig,
    ArtifactConfig,
    BaselineConfig,
    EpochConfig,
    LoadConfig,
    PipelineConfig,
    ReferenceConfig,
    SpectralConfig,
    TfrConfig,
    analysis_config_from_pipeline,
    prep_config_from_pipeline,
)
from .exceptions import ConfigurationError, MissingDependencyError
from .results import AnalysisResult, PipelineResult
from .schemas import (
    ELECTRODE_REQUIRED_COLUMNS,
    build_baseline_summary,
    build_event_table,
    build_tfr_metadata,
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


REFERENCE_METHODS = {"none", "bipolar", "wm", "car", "car_trimmed"}
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
TFR_METHODS = {"none", "morlet"}
_WORKING_DTYPE = np.dtype(WORKING_DTYPE).type


def _require_mne():
    return ensure_dependency("mne", install_hint="pip install -e .[dev]")


def _legacy_preprocess_module():
    return ensure_dependency("LFPAnalysis.lfp_preprocess_utils", install_hint="pip install -e .[dev]")


def _ensure_preloaded(data):
    """Load data into memory when an in-RAM array is required."""
    if hasattr(data, "preload") and not data.preload and hasattr(data, "load_data"):
        data.load_data()
    return data


def _get_data_array(data, *, copy: bool = False, dtype=None):
    """Return signal data as a NumPy array, compatible with Raw and Epochs.

    MNE Raw.get_data does not accept ``copy``; Epochs.get_data does. Prefer the
    preloaded ``_data`` buffer when available to avoid an extra materialization.
    """
    if dtype is None:
        dtype = _WORKING_DTYPE
    _ensure_preloaded(data)
    if hasattr(data, "_data") and data._data is not None and getattr(data, "preload", True):
        arr = data._data
        if copy:
            arr = np.array(arr, copy=True)
    else:
        try:
            arr = data.get_data(copy=copy)
        except TypeError:
            arr = data.get_data()
            if copy:
                arr = np.array(arr, copy=True)
    return np.asarray(arr, dtype=dtype)


def _downcast_mne_data(data, dtype=None):
    """Cast in-memory MNE signal arrays to the working dtype when present.

    Keys off ``_data`` rather than ``preload`` so float64 upcasts are always
    restored even if ``preload`` is False on a nonstandard object.
    """
    if dtype is None:
        dtype = _WORKING_DTYPE
    if not hasattr(data, "_data") or data._data is None:
        return data
    skipped = False
    try:
        mne = _require_mne()
        if isinstance(data, mne.BaseEpochs):
            # Keep Epochs in float64 for MNE FIF save compatibility.
            skipped = True
    except Exception:
        pass
    if (not skipped) and data._data.dtype != dtype:
        data._data = np.asarray(data._data, dtype=dtype)
    return data


def load_electrode_metadata(path: str | Path) -> pd.DataFrame:
    """Load and validate electrode metadata from CSV or XLSX files."""
    path_obj = resolve_existing_path(path, field_name="electrode_path")
    if path_obj.suffix.lower() == ".csv":
        dataframe = pd.read_csv(path_obj)
    elif path_obj.suffix.lower() in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(path_obj)
    else:
        raise ConfigurationError("electrode_path must point to a CSV or Excel file.")
    if "label" not in dataframe.columns and "NMMlabel" in dataframe.columns:
        dataframe = dataframe.rename(columns={"NMMlabel": "label"})
    return validate_required_columns(
        dataframe,
        required_columns=ELECTRODE_REQUIRED_COLUMNS,
        schema_name="Electrode metadata",
    )


def derive_referenced_electrode_df(
    electrode_df: pd.DataFrame,
    ch_names: list[str],
    method: str,
) -> pd.DataFrame:
    """Derive electrode metadata aligned to re-referenced channel names.

    For bipolar channels (``anode-cathode``), the derived row inherits
    categorical metadata from the anode and averages coordinate columns from
    the anode/cathode pair. For white-matter reference pairs, metadata is
    inherited from the anode contact (no coordinate averaging).
    """
    if method not in {"bipolar", "wm"}:
        return electrode_df.copy()
    if "label" not in electrode_df.columns:
        raise ConfigurationError("electrode_df must include a 'label' column.")

    base = electrode_df.copy()
    base["label"] = base["label"].astype(str)
    by_label = {
        str(row["label"]).lower(): row.copy()
        for _, row in base.iterrows()
    }
    coord_cols = [
        name
        for name in ("x", "y", "z", "mni_x", "mni_y", "mni_z")
        if name in base.columns
    ]
    derived_rows: list[pd.Series] = []

    for ch_name in ch_names:
        if "-" not in ch_name:
            src = by_label.get(ch_name.lower())
            if src is not None:
                derived_rows.append(src.copy())
            continue

        anode, cathode = ch_name.split("-", 1)
        anode_row = by_label.get(anode.lower())
        cathode_row = by_label.get(cathode.lower())
        if anode_row is None or cathode_row is None:
            warnings.warn(
                f"Could not derive electrode row for '{ch_name}' because "
                "one or both contacts are missing from the electrode table.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        out_row = anode_row.copy()
        out_row["label"] = ch_name
        out_row["anode"] = anode
        out_row["cathode"] = cathode
        if method == "bipolar":
            for coord in coord_cols:
                anode_v = pd.to_numeric(pd.Series([anode_row[coord]]), errors="coerce").iloc[0]
                cathode_v = pd.to_numeric(pd.Series([cathode_row[coord]]), errors="coerce").iloc[0]
                if pd.notna(anode_v) and pd.notna(cathode_v):
                    out_row[coord] = (float(anode_v) + float(cathode_v)) / 2.0
        derived_rows.append(out_row)

    if not derived_rows:
        return base.iloc[0:0].copy()
    derived = pd.DataFrame(derived_rows).reset_index(drop=True)
    return derived


def load_lfp(config: LoadConfig):
    """Load LFP data into an MNE Raw or Epochs object."""
    ensure_supported(config.file_format, field_name="file_format", supported=("edf", "neuralynx", "mne"))
    mne = _require_mne()

    if hasattr(config.path, "info") and hasattr(config.path, "copy"):
        # Caller-owned object: return as-is (no defensive copy) and downcast if preloaded.
        data = config.path
    elif config.file_format == "mne":
        path = resolve_existing_path(config.path, field_name="path")
        if path.suffix.lower() == ".fif" and ("epo" in path.name or path.name.endswith("-epo.fif")):
            data = mne.read_epochs(path, preload=config.preload)
        else:
            # memmap=True uses MNE's disk-backed path when preload is False.
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
        stacked = np.asarray(np.vstack(signals), dtype=_WORKING_DTYPE)
        data = mne.io.RawArray(stacked, info)

    if config.pick_channels and hasattr(data, "pick"):
        data.pick(config.pick_channels)

    clinical_format = config.file_format in {"edf", "neuralynx"}
    if clinical_format and config.notch_freqs:
        _ensure_preloaded(data)
        data.info["line_freq"] = 60.0
        data.notch_filter(freqs=tuple(float(freq) for freq in config.notch_freqs))
    target_sfreq = config.resample_sfreq
    if target_sfreq == "auto":
        target_sfreq = 500.0 if clinical_format else None
    if target_sfreq is not None and hasattr(data, "resample"):
        _ensure_preloaded(data)
        data.resample(float(target_sfreq))
    if clinical_format and config.check_bad_channels:
        legacy = _legacy_preprocess_module()
        channel_types = data.get_channel_types()
        seeg_names = [
            name
            for name, ch_type in zip(data.ch_names, channel_types)
            if ch_type in {"seeg", "eeg"}
        ]
        if not seeg_names:
            seeg_names = list(data.ch_names)
        bads = legacy.detect_bad_elecs(
            data.copy(),
            {name: "seeg" for name in seeg_names},
        )
        data.info["bads"] = list(bads)
    return _downcast_mne_data(data)


def preprocess_lfp(data, config: ReferenceConfig):
    """Apply an optional re-referencing step to continuous data."""
    ensure_supported(config.method, field_name="reference.method", supported=REFERENCE_METHODS)
    if config.method == "none":
        return data
    if not hasattr(data, "copy"):
        raise ConfigurationError("Reference operations require an MNE Raw-like object.")
    _ensure_preloaded(data)
    if config.method in {"car", "car_trimmed"}:
        trim_proportion = 0.0 if config.method == "car" else float(config.car_trim_proportion)
        return _common_average_reference(data, trim_proportion=trim_proportion)

    if not config.electrode_path:
        raise ConfigurationError("Reference methods 'wm' and 'bipolar' require electrode_path.")

    electrode_path = str(resolve_existing_path(config.electrode_path, field_name="electrode_path"))
    legacy = _legacy_preprocess_module()
    if config.method in {"wm", "bipolar"}:
        # Single copy into the legacy rereferencer (which may also copy internally).
        referenced = legacy.ref_mne(
            mne_data=data,
            elec_path=electrode_path,
            method=config.method,
            site=config.site,
        )
        return _downcast_mne_data(referenced)
    raise ConfigurationError(f"Unsupported reference method '{config.method}'.")


def _common_average_reference(data, trim_proportion: float = 0.0):
    """Apply common-average reference with optional trimmed-mean robustness."""
    if trim_proportion < 0.0 or trim_proportion >= 0.5:
        raise ConfigurationError("reference.car_trim_proportion must be in [0.0, 0.5).")

    referenced = data.copy()
    _ensure_preloaded(referenced)
    channel_types = referenced.get_channel_types()
    picked_names = [
        name
        for name, ch_type in zip(referenced.ch_names, channel_types)
        if ch_type in {"seeg", "eeg"}
    ]
    if not picked_names:
        picked_names = list(referenced.ch_names)

    name_to_ix = {name: ix for ix, name in enumerate(referenced.ch_names)}
    picked_ix = [name_to_ix[name] for name in picked_names]
    bad_names = set(referenced.info.get("bads", []))
    good_ref_ix = [name_to_ix[name] for name in picked_names if name not in bad_names]
    if not good_ref_ix:
        raise ConfigurationError("No non-bad channels are available for common-average reference.")

    arr = _get_data_array(referenced, copy=False, dtype=_WORKING_DTYPE)
    reference_input = np.asarray(arr[good_ref_ix, :], dtype=np.float64)
    if trim_proportion > 0.0:
        scipy_stats = ensure_dependency("scipy.stats", install_hint="pip install -e .[dev]")
        reference_signal = scipy_stats.trim_mean(
            reference_input,
            proportiontocut=trim_proportion,
            axis=0,
        )
    else:
        reference_signal = reference_input.mean(axis=0)
    arr[picked_ix, :] = arr[picked_ix, :] - reference_signal
    referenced._data = np.asarray(arr, dtype=_WORKING_DTYPE)
    return _downcast_mne_data(referenced)


def _artifact_none(data, config: ArtifactConfig) -> pd.DataFrame:
    return empty_event_table()


def _artifact_misc(data, config: ArtifactConfig) -> pd.DataFrame:
    legacy = _legacy_preprocess_module()
    _ensure_preloaded(data)
    channel_events = legacy.detect_misc_artifacts(data, peak_thresh=config.misc_peak_thresh)
    return build_event_table(channel_events, event_kind="misc", sfreq=float(data.info["sfreq"]))


def _artifact_ied(data, config: ArtifactConfig) -> pd.DataFrame:
    legacy = _legacy_preprocess_module()
    _ensure_preloaded(data)
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


def baseline_lfp(data, config: BaselineConfig, baseline_epochs=None):
    """Baseline continuous or epoched data using the shared summary schema.

    Parameters
    ----------
    data
        MNE Raw or Epochs to correct.
    config
        Baseline mode / window configuration.
    baseline_epochs
        Optional MNE Epochs locked to a different per-trial event. When
        provided, per-trial baseline statistics are taken from these epochs
        instead of ``config.baseline_window`` on ``data.times``.
    """
    ensure_supported(config.mode, field_name="baseline.mode", supported=BASELINE_METHODS)
    if not config.enabled or config.mode == "none":
        return data, empty_baseline_summary()

    mne = _require_mne()
    _ensure_preloaded(data)

    if baseline_epochs is not None:
        if not isinstance(data, mne.BaseEpochs):
            raise ConfigurationError(
                "Cross-event baselining requires task data to be MNE Epochs."
            )
        if not isinstance(baseline_epochs, mne.BaseEpochs):
            raise ConfigurationError("baseline_epochs must be an MNE Epochs object.")
        _ensure_preloaded(baseline_epochs)
        if len(baseline_epochs) != len(data):
            raise ConfigurationError(
                "baseline_epochs must have the same number of trials as the task epochs."
            )
        if list(baseline_epochs.ch_names) != list(data.ch_names):
            raise ConfigurationError(
                "baseline_epochs channel names must match the task epochs."
            )
        all_data = _get_data_array(data, copy=False)
        baseline = _get_data_array(baseline_epochs, copy=False)
        # mean_baseline_time broadcasts over leading dims (n_trials, n_channels).
        corrected = _apply_baseline_array(all_data, baseline, config.mode)
        data._data = np.asarray(corrected, dtype=_WORKING_DTYPE)
        bl_start = (
            float(config.baseline_window[0])
            if config.baseline_window is not None
            else float(baseline_epochs.times[0])
        )
        bl_stop = (
            float(config.baseline_window[1])
            if config.baseline_window is not None
            else float(baseline_epochs.times[-1])
        )
        summary = build_baseline_summary(
            target="epochs",
            channel_names=data.ch_names,
            mode=config.mode,
            baseline_start=bl_start,
            baseline_stop=bl_stop,
            baseline_mean=baseline.mean(axis=(0, 2)),
            baseline_std=baseline.std(axis=(0, 2)),
        )
        return data, summary

    if isinstance(data, mne.BaseEpochs):
        indices = _get_baseline_indices(data.times, config.baseline_window)
        # Single materialization; mutate in place on a float32 working buffer.
        all_data = _get_data_array(data, copy=False)
        baseline = all_data[:, :, indices]
        # Vectorized across epochs via the shared baseline helper (operates on full array).
        corrected = _apply_baseline_array(all_data, baseline, config.mode)
        data._data = np.asarray(corrected, dtype=_WORKING_DTYPE)
        summary = build_baseline_summary(
            target="epochs",
            channel_names=data.ch_names,
            mode=config.mode,
            baseline_start=config.baseline_window[0],
            baseline_stop=config.baseline_window[1],
            baseline_mean=baseline.mean(axis=(0, 2)),
            baseline_std=baseline.std(axis=(0, 2)),
        )
        return data, summary

    if not hasattr(data, "get_data"):
        raise ConfigurationError("Baselining requires an MNE Raw or Epochs object.")

    time_axis = np.arange(data.n_times) / float(data.info["sfreq"])
    indices = _get_baseline_indices(time_axis, config.baseline_window)
    all_data = _get_data_array(data, copy=False)
    baseline = all_data[:, indices]
    data._data = np.asarray(_apply_baseline_array(all_data, baseline, config.mode), dtype=_WORKING_DTYPE)
    summary = build_baseline_summary(
        target="raw",
        channel_names=data.ch_names,
        mode=config.mode,
        baseline_start=config.baseline_window[0],
        baseline_stop=config.baseline_window[1],
        baseline_mean=baseline.mean(axis=1),
        baseline_std=baseline.std(axis=1),
    )
    return data, summary


def make_epochs(data, config: EpochConfig):
    """Create epochs around event timestamps on an MNE Raw object."""
    if not config.enabled:
        return None
    mne = _require_mne()
    if not hasattr(data, "info") or not hasattr(data, "copy"):
        raise ConfigurationError("Epoch extraction requires an MNE Raw object.")
    if not config.event_times:
        raise ConfigurationError("epoch.event_times must be provided when epoching is enabled.")

    _ensure_preloaded(data)
    keep_indices: list[int] = []
    filtered_times: list[float] = []
    for ix, raw_time in enumerate(config.event_times):
        if str(raw_time) == "None":
            continue
        time_value = float(raw_time)
        if np.isnan(time_value):
            continue
        keep_indices.append(ix)
        filtered_times.append(time_value)
    if not filtered_times:
        raise ConfigurationError("epoch.event_times contained no valid timestamps after dropping NaNs.")

    transformed_times = [(time_value * config.slope) + config.offset for time_value in filtered_times]
    events = np.column_stack(
        [
            np.asarray(transformed_times, dtype=_WORKING_DTYPE) * float(data.info["sfreq"]),
            np.zeros(len(transformed_times), dtype=int),
            np.ones(len(transformed_times), dtype=int),
        ]
    ).astype(int)
    metadata = None
    if config.metadata:
        metadata = pd.DataFrame(config.metadata)
        if len(metadata) != len(config.event_times):
            raise ConfigurationError("epoch.metadata must have the same number of rows as event_times.")
        metadata = metadata.iloc[keep_indices].reset_index(drop=True)
    # Avoid an extra full Raw copy: Epochs will materialize its own preloaded array.
    epochs = mne.Epochs(
        data,
        events=events,
        event_id={config.event_name: 1},
        tmin=config.tmin,
        tmax=config.tmax,
        baseline=None,
        preload=True,
        metadata=metadata,
        verbose=False,
    )
    return _downcast_mne_data(epochs)


def compute_spectral_features(data, config: SpectralConfig) -> dict[str, Any]:
    """Compute optional spectral features using the stable registry interface."""
    ensure_supported(config.method, field_name="spectral.method", supported=SPECTRAL_METHODS)
    if not config.enabled or config.method == "none":
        return {}

    _ensure_preloaded(data)
    if config.method == "psd":
        psd_kwargs: dict[str, Any] = {"fmin": config.fmin, "fmax": config.fmax}
        # Epochs default to multitaper (no n_fft); only pass n_fft for welch-style calls.
        if config.n_fft is not None:
            psd_kwargs["method"] = "welch"
            psd_kwargs["n_fft"] = config.n_fft
        spectrum = data.compute_psd(**psd_kwargs)
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


def _crop_tfr_power(
    power,
    config: TfrConfig,
    *,
    crop_tmin: float | None = None,
    crop_tmax: float | None = None,
):
    tmin_bound = config.crop_tmin if crop_tmin is None else crop_tmin
    tmax_bound = config.crop_tmax if crop_tmax is None else crop_tmax
    if tmin_bound is None or tmax_bound is None:
        return power
    tmin = max(float(tmin_bound), float(power.times[0]))
    tmax = min(float(tmax_bound), float(power.times[-1]))
    if tmin < tmax:
        power.crop(tmin=tmin, tmax=tmax)
    return power


def _baseline_tfr_crop_bounds(config: TfrConfig) -> tuple[float | None, float | None]:
    if config.baseline_crop_tmin is not None or config.baseline_crop_tmax is not None:
        return config.baseline_crop_tmin, config.baseline_crop_tmax
    return config.crop_tmin, config.crop_tmax


def _nan_mask_tfr_tables(power, tables: list[pd.DataFrame]):
    legacy = _legacy_preprocess_module()
    for table in tables:
        if table is None or table.empty:
            continue
        power.data = np.asarray(power.data, dtype=_WORKING_DTYPE)
        legacy._nan_mask_tfr_events(
            power.data,
            table,
            power.ch_names,
            float(power.info["sfreq"]),
            int(power.data.shape[-1]),
        )
    return power


def _cross_event_trialwise_zscore(task_power, baseline_power, config: TfrConfig):
    legacy = _legacy_preprocess_module()
    working = np.asarray(task_power.data, dtype=_WORKING_DTYPE)
    baseline_data = np.asarray(baseline_power.data, dtype=_WORKING_DTYPE)
    z_data = legacy.baseline_trialwise_TFR(
        data=working,
        include_epoch_in_baseline=False,
        baseline_mne=baseline_data,
        mode="zscore",
        ev_axis=0,
        elec_axis=1,
        freq_axis=2,
        time_axis=3,
    )
    if config.uncaptured_z_thresh:
        threshold = float(config.z_thresh)
        max_iter = 10
        iteration = 0
        while iteration < max_iter:
            large_mask = np.where(np.abs(z_data) > threshold)
            if large_mask[0].shape[0] == 0:
                break
            working[large_mask] = np.nan
            z_data = legacy.baseline_trialwise_TFR(
                data=working,
                include_epoch_in_baseline=False,
                baseline_mne=baseline_data,
                mode="zscore",
                ev_axis=0,
                elec_axis=1,
                freq_axis=2,
                time_axis=3,
            )
            iteration += 1
    return z_data


def compute_tfr_features(data, config: TfrConfig, baseline_epochs=None, artifact_tables: dict[str, pd.DataFrame] | None = None) -> dict[str, Any]:
    """Compute optional TFR features on epoched data (analysis spine)."""
    ensure_supported(config.method, field_name="tfr.method", supported=TFR_METHODS)
    if not config.enabled or config.method == "none":
        return {}

    mne = _require_mne()
    if not isinstance(data, mne.BaseEpochs):
        raise ConfigurationError("TFR in the analysis spine currently requires MNE Epochs.")
    if config.freqs is None:
        raise ConfigurationError("tfr.freqs is required when TFR is enabled.")

    _ensure_preloaded(data)
    freqs = np.asarray(config.freqs, dtype=float)
    power = data.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=config.n_cycles,
        decim=config.decim,
        n_jobs=config.n_jobs,
        average=False,
        return_itc=False,
        verbose=False,
    )

    if baseline_epochs is not None:
        baseline_power = baseline_epochs.compute_tfr(
            method="morlet",
            freqs=freqs,
            n_cycles=config.n_cycles,
            decim=config.decim,
            n_jobs=config.n_jobs,
            average=False,
            return_itc=False,
            verbose=False,
        )
        power = _crop_tfr_power(power, config)
        bl_tmin, bl_tmax = _baseline_tfr_crop_bounds(config)
        baseline_power = _crop_tfr_power(
            baseline_power,
            config,
            crop_tmin=bl_tmin,
            crop_tmax=bl_tmax,
        )
        if config.mask_artifacts and artifact_tables:
            power = _nan_mask_tfr_tables(
                power,
                [
                    artifact_tables.get("ied_task_epoched"),
                    artifact_tables.get("misc_task_epoched"),
                ],
            )
            baseline_power = _nan_mask_tfr_tables(
                baseline_power,
                [
                    artifact_tables.get("ied_baseline_epoched"),
                    artifact_tables.get("misc_baseline_epoched"),
                ],
            )
        z_data = _cross_event_trialwise_zscore(power, baseline_power, config)
        z_power = mne.time_frequency.EpochsTFRArray(
            data.info,
            np.asarray(z_data, dtype=_WORKING_DTYPE),
            power.times,
            freqs,
        )
        z_power.metadata = data.metadata
        meta = build_tfr_metadata(
            method="morlet",
            baseline_mode="trialwise",
            freqs=freqs,
            n_cycles=config.n_cycles,
            decim=config.decim,
        )
        return {"method": "morlet", "power": z_power, "metadata": meta}

    if config.apply_baseline and config.baseline_mode not in {"none", None}:
        # EpochsTFR baseline expects a (tmin, tmax) window; use full epoch pre-zero when present.
        times = np.asarray(power.times, dtype=float)
        if times.size and times[0] < 0:
            baseline_window = (float(times[0]), 0.0)
        else:
            baseline_window = None
        if baseline_window is not None:
            mode = config.baseline_mode
            if mode in {"trialwise", "continuous"}:
                mode = "zscore"
            power.apply_baseline(baseline=baseline_window, mode=mode, verbose=False)

    meta = build_tfr_metadata(
        method="morlet",
        baseline_mode=config.baseline_mode if config.apply_baseline else "none",
        freqs=freqs,
        n_cycles=config.n_cycles,
        decim=config.decim,
    )
    return {"method": "morlet", "power": power, "metadata": meta}


def run_analysis(epochs, config: AnalysisConfig, baseline_epochs=None, artifact_tables: dict[str, pd.DataFrame] | None = None) -> AnalysisResult:
    """Run the analysis spine starting from MNE Epochs (or continuous fallback).

    Does not perform sync or electrode localization — those belong in :func:`run_prep`.

    Parameters
    ----------
    epochs
        Task epochs (or continuous data) to analyze.
    config
        Analysis configuration.
    baseline_epochs
        Optional per-trial baseline epochs locked to a different event stream.
    """
    if epochs is None:
        raise ConfigurationError(
            "run_analysis requires Epochs (or continuous data). Run prep first, "
            "or pass your own MNE Epochs."
        )

    if config.baseline.enabled:
        warnings.warn(
            "analysis.baseline is deprecated in the analysis spine; epochs remain raw "
            "and baselining should happen in the TFR stage.",
            DeprecationWarning,
            stacklevel=2,
        )
    baseline_summary = empty_baseline_summary()
    spectral = compute_spectral_features(epochs, config.spectral)
    tfr = compute_tfr_features(
        epochs,
        config.tfr,
        baseline_epochs=baseline_epochs,
        artifact_tables=artifact_tables,
    )
    metadata = {
        "spine": "analysis",
        "baseline_mode": "none",
        "spectral_method": config.spectral.method,
        "tfr_method": config.tfr.method,
        "working_dtype": str(WORKING_DTYPE),
        "cross_event_baseline": baseline_epochs is not None,
    }
    return AnalysisResult(
        epochs=epochs,
        baseline_summary=baseline_summary,
        spectral=spectral,
        tfr=tfr,
        metadata=metadata,
    )


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Tutorial convenience: run prep then analysis and compose :class:`PipelineResult`.

    Prefer :func:`LFPAnalysis.prep.run_prep` + :func:`run_analysis` when you want
    a swappable prep backend. Superseded continuous stages are dropped when
    epoching is enabled (``raw`` / ``referenced`` become ``None``).
    """
    from .prep import run_prep

    prep = run_prep(prep_config_from_pipeline(config))
    analysis_cfg = analysis_config_from_pipeline(config)
    if (
        config.epoch.enabled
        and analysis_cfg.tfr.enabled
        and analysis_cfg.tfr.method != "none"
        and (analysis_cfg.tfr.crop_tmin is None or analysis_cfg.tfr.crop_tmax is None)
    ):
        analysis_cfg.tfr.crop_tmin = float(config.epoch.tmin)
        analysis_cfg.tfr.crop_tmax = float(config.epoch.tmax)
    analysis_input = prep.epochs if prep.epochs is not None else prep.referenced
    needs_analysis = (
        analysis_cfg.baseline.enabled
        or analysis_cfg.spectral.enabled
        or analysis_cfg.tfr.enabled
    )

    if needs_analysis:
        if analysis_input is None:
            raise ConfigurationError(
                "Analysis stages are enabled but prep produced neither Epochs nor continuous data."
            )
        analysis = run_analysis(
            analysis_input,
            analysis_cfg,
            baseline_epochs=prep.baseline_epochs,
            artifact_tables=prep.artifact_tables,
        )
        if prep.epochs is not None:
            raw_out = None
            referenced_out = None
            epochs_out = analysis.epochs
        else:
            raw_out = prep.raw
            referenced_out = analysis.epochs
            epochs_out = None
        baseline_summary = analysis.baseline_summary
        spectral = analysis.spectral
        tfr = analysis.tfr
        analysis_meta = analysis.metadata
    else:
        raw_out = prep.raw
        referenced_out = prep.referenced
        epochs_out = prep.epochs
        baseline_summary = empty_baseline_summary()
        spectral = {}
        tfr = {}
        analysis_meta = {"spine": "analysis", "skipped": True}

    metadata = {
        **prep.metadata,
        **analysis_meta,
        "input_format": config.load.file_format,
        "reference_method": config.reference.method,
        "artifact_methods": list(config.artifact.methods),
        "baseline_mode": "none",
        "spectral_method": config.spectral.method,
        "tfr_method": config.tfr.method,
        "working_dtype": str(WORKING_DTYPE),
        "preload": bool(config.load.preload),
        "notch_freqs": list(config.load.notch_freqs) if config.load.notch_freqs else None,
        "buffer_s": float(config.epoch.buffer_s),
        "tfr_baseline": (
            "cross_event_trialwise_zscore" if prep.baseline_epochs is not None else config.tfr.baseline_mode
        ),
        "tfr_mask_artifacts": bool(config.tfr.mask_artifacts),
        "tfr_uncaptured_z_thresh": bool(config.tfr.uncaptured_z_thresh),
        "cross_event_baseline": prep.baseline_epochs is not None,
    }
    return PipelineResult(
        raw=raw_out,
        referenced=referenced_out,
        epochs=epochs_out,
        artifact_tables=prep.artifact_tables,
        baseline_summary=baseline_summary,
        spectral=spectral,
        tfr=tfr,
        electrode_df=prep.electrode_df,
        sync=prep.sync,
        metadata=metadata,
    )
