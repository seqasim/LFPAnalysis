"""Compatibility wrappers for the most common legacy notebook entry points."""

from __future__ import annotations

import warnings
from typing import Any

from .config import EpochConfig, LoadConfig, ReferenceConfig
from .exceptions import ConfigurationError
from .validation import ensure_dependency
from .workflow import load_lfp, make_epochs as workflow_make_epochs, preprocess_lfp



def _warn(old_call: str, new_call: str, note: str | None = None) -> None:
    message = f"`{old_call}` is deprecated. Use `{new_call}` instead."
    if note:
        message = f"{message} {note}"
    warnings.warn(message, DeprecationWarning, stacklevel=2)



def _legacy_preprocess_module():
    return ensure_dependency("LFPAnalysis.lfp_preprocess_utils", install_hint="pip install -e .[dev]")



def make_mne(
    load_path=None,
    elec_path=None,
    format: str = "edf",
    site: str = "MSSM",
    resample_sr: int = 500,
    overwrite: bool = True,
    return_data: bool = False,
    include_micros: bool = False,
    eeg_names=None,
    resp_names=None,
    ekg_names=None,
    sync_name=None,
    sync_type: str = "photodiode",
    seeg_names=None,
    drop_names=None,
    seeg_only: bool = True,
    check_bad: bool = False,
):
    """Compatibility wrapper for the legacy `make_mne` workflow."""
    _warn(
        "lfp_preprocess_utils.make_mne(...)",
        "load_lfp(LoadConfig(...)) or run_pipeline(build_basic_pipeline_config(...))",
        note="The compatibility shim preserves the old folder-oriented behavior where needed.",
    )
    if format == "mne":
        if load_path is None:
            raise ConfigurationError("load_path is required for legacy.make_mne(format='mne').")
        data = load_lfp(
            LoadConfig(
                path=load_path,
                file_format="mne",
                resample_sfreq=resample_sr,
                include_micros=include_micros,
                eeg_names=eeg_names or [],
                resp_names=resp_names or [],
                ekg_names=ekg_names or [],
                seeg_names=seeg_names or [],
                drop_names=drop_names or [],
            )
        )
        return data

    legacy = _legacy_preprocess_module()
    return legacy.make_mne(
        load_path=load_path,
        elec_path=elec_path,
        format=format,
        site=site,
        resample_sr=resample_sr,
        overwrite=overwrite,
        return_data=return_data,
        include_micros=include_micros,
        eeg_names=eeg_names,
        resp_names=resp_names,
        ekg_names=ekg_names,
        sync_name=sync_name,
        sync_type=sync_type,
        seeg_names=seeg_names,
        drop_names=drop_names,
        seeg_only=seeg_only,
        check_bad=check_bad,
    )



def ref_mne(mne_data=None, elec_path=None, method: str = "wm", site: str = "MSSM"):
    """Compatibility wrapper for the legacy `ref_mne` helper."""
    _warn(
        "lfp_preprocess_utils.ref_mne(...)",
        "preprocess_lfp(raw, ReferenceConfig(...))",
    )
    return preprocess_lfp(mne_data, ReferenceConfig(method=method, electrode_path=elec_path, site=site))



def make_epochs(
    load_path=None,
    slope=None,
    offset=None,
    behav_name=None,
    behav_times=None,
    ev_start_s: float = 0,
    ev_end_s: float = 1.5,
    buf_s: float = 1,
    downsamp_factor=None,
    IED_args=None,
    baseline=None,
    detrend=None,
):
    """Compatibility wrapper for the legacy `make_epochs` helper."""
    _warn(
        "lfp_preprocess_utils.make_epochs(...)",
        "raw = load_lfp(LoadConfig(...)); make_epochs(raw, EpochConfig(...))",
        note="The compatibility shim uses the stable path only for the no-side-effects case and falls back to the legacy implementation otherwise.",
    )
    if load_path is None or behav_times is None or behav_name is None:
        raise ConfigurationError("load_path, behav_name, and behav_times are required for legacy.make_epochs.")

    stable_only = IED_args is None and baseline is None and detrend is None and downsamp_factor is None and buf_s == 1
    if stable_only:
        raw = load_lfp(LoadConfig(path=load_path, file_format="mne"))
        return workflow_make_epochs(
            raw,
            EpochConfig(
                enabled=True,
                event_name=behav_name,
                event_times=[float(x) for x in behav_times if str(x) != "None"],
                slope=1.0 if slope is None else float(slope),
                offset=0.0 if offset is None else float(offset),
                tmin=-float(ev_start_s),
                tmax=float(ev_end_s),
            ),
        )

    legacy = _legacy_preprocess_module()
    return legacy.make_epochs(
        load_path=load_path,
        slope=slope,
        offset=offset,
        behav_name=behav_name,
        behav_times=behav_times,
        ev_start_s=ev_start_s,
        ev_end_s=ev_end_s,
        buf_s=buf_s,
        downsamp_factor=downsamp_factor,
        IED_args=IED_args,
        baseline=baseline,
        detrend=detrend,
    )



def compute_and_baseline_tfr(*args: Any, **kwargs: Any):
    """Compatibility wrapper for the legacy TFR workflow."""
    _warn(
        "lfp_preprocess_utils.compute_and_baseline_tfr(...)",
        "baseline_lfp(...) plus the advanced time-frequency utilities documented in the migration guide",
        note="A one-to-one stable workflow replacement does not exist yet, so this shim intentionally delegates to the legacy implementation.",
    )
    legacy = _legacy_preprocess_module()
    return legacy.compute_and_baseline_tfr(*args, **kwargs)
