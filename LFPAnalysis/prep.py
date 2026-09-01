"""Prep spine: raw clinical files → event-locked MNE Epochs (+ metadata).

This spine is intentionally separate from analysis so sync, electrode tables,
and site I/O can evolve without rewriting science code.
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import pandas as pd

from .config import (
    WORKING_DTYPE,
    ElectrodeConfig,
    EpochConfig,
    PrepConfig,
    SyncConfig,
)
from .exceptions import ConfigurationError
from .results import PrepResult
from .schemas import build_sync_summary, empty_sync_summary
from .validation import ensure_supported, resolve_existing_path
from .workflow import (
    _legacy_preprocess_module,
    derive_referenced_electrode_df,
    detect_artifacts,
    load_electrode_metadata,
    load_lfp,
    make_epochs,
    preprocess_lfp,
)


SYNC_SOURCES = {"none", "photodiode", "ttl", "precomputed"}


def synchronize_lfp(data, config: SyncConfig) -> tuple[float, float, dict[str, Any], pd.DataFrame]:
    """Run behavioral–neural sync and return slope, offset, details, summary table.

    Parameters
    ----------
    data
        MNE Raw (or sync-channel subset) used when ``source`` is photodiode/ttl.
    config
        Sync configuration (prep spine only).
    """
    ensure_supported(config.source, field_name="sync.source", supported=SYNC_SOURCES)
    if not config.enabled or config.source == "none":
        slope = 1.0 if config.slope is None else float(config.slope)
        offset = 0.0 if config.offset is None else float(config.offset)
        summary = build_sync_summary(
            source=config.source if config.enabled else "none",
            slope=slope,
            offset=offset,
        )
        return slope, offset, {"source": "none", "slope": slope, "offset": offset}, summary

    if config.source == "precomputed":
        if config.slope is None or config.offset is None:
            raise ConfigurationError(
                "sync.source='precomputed' requires sync.slope and sync.offset."
            )
        slope = float(config.slope)
        offset = float(config.offset)
        details = {"source": "precomputed", "slope": slope, "offset": offset}
        summary = build_sync_summary(source="precomputed", slope=slope, offset=offset)
        return slope, offset, details, summary

    if not config.behav_times:
        raise ConfigurationError("sync.behav_times is required when sync is enabled.")

    from . import sync_utils

    if config.source == "ttl":
        if config.nev_data is None:
            raise ConfigurationError("sync.source='ttl' requires sync.nev_data.")
        if config.use_robust:
            slope, offset, r_value = sync_utils.synchronize_data_robust(
                beh_ts=config.behav_times,
                neural_ts=sync_utils.get_neural_ts_ttl(config.nev_data),
                window_size=config.wind_size,
            )
            details = {
                "source": "ttl",
                "slope": float(slope),
                "offset": float(offset),
                "r_value": float(r_value),
                "n_behav_pulses": len(config.behav_times),
            }
            summary = build_sync_summary(
                source="ttl",
                slope=float(slope),
                offset=float(offset),
                n_behav_pulses=len(config.behav_times),
                r_value=float(r_value),
            )
            return float(slope), float(offset), details, summary
        slope, offset = sync_utils.synchronize_data(
            beh_ts=config.behav_times,
            mne_sync=config.nev_data,
            smoothSize=config.smooth_size,
            windSize=config.wind_size,
            height=config.height,
            sync_source="ttl",
        )
        details = {
            "source": "ttl",
            "slope": float(slope),
            "offset": float(offset),
            "n_behav_pulses": len(config.behav_times),
        }
        summary = build_sync_summary(
            source="ttl",
            slope=float(slope),
            offset=float(offset),
            n_behav_pulses=len(config.behav_times),
        )
        return float(slope), float(offset), details, summary

    # photodiode
    if config.sync_channel is None:
        raise ConfigurationError("sync.source='photodiode' requires sync.sync_channel.")
    if not hasattr(data, "copy"):
        raise ConfigurationError("Photodiode sync requires an MNE Raw-like object.")
    mne_sync = data.copy().pick([config.sync_channel])
    if config.use_robust:
        neural_ts = sync_utils.get_neural_ts_photodiode(
            mne_sync, smoothSize=config.smooth_size, height=config.height
        )
        slope, offset, r_value = sync_utils.synchronize_data_robust(
            beh_ts=config.behav_times,
            neural_ts=neural_ts,
            window_size=config.wind_size,
        )
        details = {
            "source": "photodiode",
            "slope": float(slope),
            "offset": float(offset),
            "r_value": float(r_value),
            "sync_channel": config.sync_channel,
            "n_behav_pulses": len(config.behav_times),
            "n_neural_pulses": len(neural_ts),
        }
        summary = build_sync_summary(
            source="photodiode",
            slope=float(slope),
            offset=float(offset),
            n_behav_pulses=len(config.behav_times),
            n_neural_pulses=len(neural_ts),
            r_value=float(r_value),
        )
        return float(slope), float(offset), details, summary

    slope, offset = sync_utils.synchronize_data(
        beh_ts=config.behav_times,
        mne_sync=mne_sync,
        smoothSize=config.smooth_size,
        windSize=config.wind_size,
        height=config.height,
        sync_source="photodiode",
    )
    details = {
        "source": "photodiode",
        "slope": float(slope),
        "offset": float(offset),
        "sync_channel": config.sync_channel,
        "n_behav_pulses": len(config.behav_times),
    }
    summary = build_sync_summary(
        source="photodiode",
        slope=float(slope),
        offset=float(offset),
        n_behav_pulses=len(config.behav_times),
    )
    return float(slope), float(offset), details, summary


def _load_electrodes(config: ElectrodeConfig) -> pd.DataFrame | None:
    if not config.load_into_result or config.path is None:
        return None
    path = resolve_existing_path(config.path, field_name="electrode.path")
    if config.site == "UI":
        legacy = _legacy_preprocess_module()
        return legacy.load_elec(str(path), site="UI")
    return load_electrode_metadata(path)


def _epoch_config_with_sync(epoch: EpochConfig, slope: float, offset: float) -> EpochConfig:
    """Return an EpochConfig using sync-derived slope/offset when sync ran."""
    keep_indices: list[int] = []
    clean_event_times: list[float] = []
    for ix, raw_time in enumerate(epoch.event_times):
        if str(raw_time) == "None":
            continue
        time_value = float(raw_time)
        if np.isnan(time_value):
            continue
        keep_indices.append(ix)
        clean_event_times.append(time_value)

    metadata = epoch.metadata
    if metadata is not None:
        frame = pd.DataFrame(metadata)
        if len(frame) == len(epoch.event_times):
            frame = frame.iloc[keep_indices].reset_index(drop=True)
            metadata = frame.to_dict(orient="list")

    baseline_event_times = None
    if epoch.baseline_event_times is not None:
        if len(epoch.baseline_event_times) != len(epoch.event_times):
            raise ConfigurationError(
                "epoch.baseline_event_times must have the same length as epoch.event_times."
            )
        baseline_event_times = [epoch.baseline_event_times[ix] for ix in keep_indices]
        valid_pairs: list[int] = []
        for ix, raw_time in enumerate(baseline_event_times):
            if str(raw_time) == "None":
                continue
            time_value = float(raw_time)
            if np.isnan(time_value):
                continue
            valid_pairs.append(ix)
        if len(valid_pairs) != len(baseline_event_times):
            clean_event_times = [clean_event_times[ix] for ix in valid_pairs]
            baseline_event_times = [baseline_event_times[ix] for ix in valid_pairs]
            if metadata is not None:
                frame = pd.DataFrame(metadata).iloc[valid_pairs].reset_index(drop=True)
                metadata = frame.to_dict(orient="list")

    if len(clean_event_times) != len(epoch.event_times):
        warnings.warn(
            "Dropped invalid event timestamps (None/NaN) while building epochs.",
            RuntimeWarning,
            stacklevel=2,
        )

    return EpochConfig(
        enabled=epoch.enabled,
        event_name=epoch.event_name,
        event_times=clean_event_times,
        slope=slope,
        offset=offset,
        tmin=epoch.tmin - float(epoch.buffer_s),
        tmax=epoch.tmax + float(epoch.buffer_s),
        buffer_s=epoch.buffer_s,
        metadata=metadata,
        baseline_event_times=baseline_event_times,
        baseline_tmin=epoch.baseline_tmin,
        baseline_tmax=epoch.baseline_tmax,
    )


def _make_baseline_epochs(data, epoch: EpochConfig):
    """Extract baseline-event epochs when cross-event baselining is configured."""
    if not epoch.baseline_event_times:
        return None
    if epoch.baseline_tmin is None or epoch.baseline_tmax is None:
        raise ConfigurationError(
            "epoch.baseline_tmin and epoch.baseline_tmax are required when "
            "epoch.baseline_event_times is set."
        )
    if len(epoch.baseline_event_times) != len(epoch.event_times):
        raise ConfigurationError(
            "epoch.baseline_event_times must have the same length as epoch.event_times."
        )
    baseline_cfg = EpochConfig(
        enabled=True,
        event_name="baseline_event",
        event_times=list(epoch.baseline_event_times),
        slope=epoch.slope,
        offset=epoch.offset,
        tmin=float(epoch.baseline_tmin) - float(epoch.buffer_s),
        tmax=float(epoch.baseline_tmax) + float(epoch.buffer_s),
        buffer_s=float(epoch.buffer_s),
        metadata=None,
    )
    return make_epochs(data, baseline_cfg)


def _bin_artifact_tables_for_epochs(
    artifact_tables: dict[str, pd.DataFrame],
    epoch: EpochConfig,
) -> dict[str, pd.DataFrame]:
    if not epoch.enabled or not epoch.event_times:
        return artifact_tables
    legacy = _legacy_preprocess_module()
    out = dict(artifact_tables)
    event_times = np.asarray(epoch.event_times, dtype=float)
    event_task_starts = (event_times * float(epoch.slope)) + float(epoch.offset) + float(epoch.tmin)
    event_task_ends = (event_times * float(epoch.slope)) + float(epoch.offset) + float(epoch.tmax)

    for method in ("ied", "misc"):
        table = artifact_tables.get(method)
        if table is None or table.empty:
            continue
        chan_map = (
            table.groupby("channel")["time_seconds"]
            .apply(lambda series: [float(value) for value in series.to_list()])
            .to_dict()
        )
        out[f"{method}_task_epoched"] = legacy._bin_channelwise_times_into_behav_evs(
            chan_map,
            event_task_starts.tolist(),
            event_task_ends.tolist(),
        )

        if epoch.baseline_event_times and epoch.baseline_tmin is not None and epoch.baseline_tmax is not None:
            baseline_times = np.asarray(epoch.baseline_event_times, dtype=float)
            bl_starts = (
                (baseline_times * float(epoch.slope))
                + float(epoch.offset)
                + float(epoch.baseline_tmin)
                - float(epoch.buffer_s)
            )
            bl_ends = (
                (baseline_times * float(epoch.slope))
                + float(epoch.offset)
                + float(epoch.baseline_tmax)
                + float(epoch.buffer_s)
            )
            out[f"{method}_baseline_epoched"] = legacy._bin_channelwise_times_into_behav_evs(
                chan_map,
                bl_starts.tolist(),
                bl_ends.tolist(),
            )
    return out


def run_prep(config: PrepConfig) -> PrepResult:
    """Run the prep spine and return Epochs plus handoff metadata."""
    raw = load_lfp(config.load)
    referenced = preprocess_lfp(raw, config.reference)
    referenced_ch_names = list(getattr(referenced, "ch_names", []))
    keep_raw = referenced is raw

    artifact_tables = detect_artifacts(referenced, config.artifact)
    electrode_df = _load_electrodes(config.electrode)
    electrode_df_referenced = False
    if (
        electrode_df is not None
        and config.reference.method in {"bipolar", "wm"}
        and referenced_ch_names
    ):
        electrode_df = derive_referenced_electrode_df(
            electrode_df=electrode_df,
            ch_names=referenced_ch_names,
            method=config.reference.method,
        )
        electrode_df_referenced = True

    if config.sync.enabled:
        slope, offset, sync_details, sync_summary = synchronize_lfp(referenced, config.sync)
    else:
        slope = float(config.epoch.slope)
        offset = float(config.epoch.offset)
        sync_details = {}
        sync_summary = empty_sync_summary()

    epoch_cfg = _epoch_config_with_sync(config.epoch, slope, offset)
    artifact_tables = _bin_artifact_tables_for_epochs(artifact_tables, epoch_cfg)
    epochs = make_epochs(referenced, epoch_cfg)
    baseline_epochs = _make_baseline_epochs(referenced, epoch_cfg)

    if epochs is not None:
        raw_out = None
        referenced_out = None
        del raw, referenced
    else:
        raw_out = raw if keep_raw else None
        referenced_out = referenced
        if not keep_raw:
            del raw

    metadata = {
        "spine": "prep",
        "input_format": config.load.file_format,
        "reference_method": config.reference.method,
        "artifact_methods": list(config.artifact.methods),
        "sync_source": config.sync.source if config.sync.enabled else "none",
        "electrode_path": str(config.electrode.path) if config.electrode.path else None,
        "working_dtype": str(WORKING_DTYPE),
        "preload": bool(config.load.preload),
        "has_baseline_epochs": baseline_epochs is not None,
        "electrode_df_referenced": electrode_df_referenced,
    }
    return PrepResult(
        epochs=epochs,
        raw=raw_out,
        referenced=referenced_out,
        baseline_epochs=baseline_epochs,
        artifact_tables=artifact_tables,
        electrode_df=electrode_df,
        sync=sync_details,
        sync_summary=sync_summary if config.sync.enabled else empty_sync_summary(),
        metadata=metadata,
    )
