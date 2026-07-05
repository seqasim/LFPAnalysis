"""One-off script to regenerate LFPAnalysisBook notebooks for the real-data case study."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "LFPAnalysisBook"


def nb(cells: list[dict], metadata: dict | None = None) -> dict:
    return {
        "cells": cells,
        "metadata": metadata
        or {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str, tags: list[str] | None = None) -> dict:
    meta = {"tags": tags} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": meta,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def write_nb(path: Path, notebook: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=2))


NOTEBOOKS: dict[str, dict] = {}

# --- Worked 01 ---
NOTEBOOKS["worked-examples/01_first_import_and_load.ipynb"] = nb(
    [
        md(
            "# Worked Example: Load the Gambling-Task Recording\n\n"
            "## Goal\n"
            "Load the real 22-channel `sample_ieeg.fif` and inspect sampling rate, channels, and duration.\n\n"
            "## Expected input files\n"
            "- `../../data/sample_ieeg.fif`\n"
        ),
        code(
            "from pathlib import Path\n"
            "from LFPAnalysis import build_basic_pipeline_config, run_pipeline\n\n"
            "config = build_basic_pipeline_config(Path('../../data/sample_ieeg.fif'), file_format='mne')\n"
            "result = run_pipeline(config)\n"
            "raw = result.raw\n"
            "print(f'sfreq={raw.info[\"sfreq\"]} Hz, n_ch={len(raw.ch_names)}, duration={raw.n_times/raw.info[\"sfreq\"]:.1f} s')\n"
            "print('Channels:', raw.ch_names[:8], '…')",
            tags=["worked"],
        ),
        md("## Plot a raw trace and sanity-check PSD"),
        code(
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n\n"
            "sfreq = raw.info['sfreq']\n"
            "start = int(240 * sfreq)\n"
            "stop = start + int(2 * sfreq)\n"
            "data = raw.get_data(picks=['racas1'])[0, start:stop]\n"
            "times = np.arange(len(data)) / sfreq\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(10, 3))\n"
            "axes[0].plot(times, data, 'k', lw=0.8)\n"
            "axes[0].set(xlabel='Time (s)', ylabel='Amplitude', title='racas1 around first feedback')\n"
            "psd = raw.compute_psd(fmin=1, fmax=80, picks=['racas1'])\n"
            "axes[1].semilogy(psd.freqs, psd.get_data()[0])\n"
            "axes[1].set(xlabel='Frequency (Hz)', ylabel='PSD', title='Welch PSD')\n"
            "fig.tight_layout()\n"
            "plt.show()",
            tags=["worked"],
        ),
        md("## Next step\n\nContinue to chapter 04 for bipolar referencing."),
    ]
)

# --- Worked 02 ---
NOTEBOOKS["worked-examples/02_first_preprocessing_run.ipynb"] = nb(
    [
        md(
            "# Worked Example: Bipolar Reference and Artifact QC\n\n"
            "## Goal\n"
            "Apply bipolar referencing to the real recording and run misc-artifact detection.\n"
        ),
        code(
            "from pathlib import Path\n"
            "from LFPAnalysis import build_basic_pipeline_config, run_pipeline\n\n"
            "config = build_basic_pipeline_config(\n"
            "    Path('../../data/sample_ieeg.fif'),\n"
            "    file_format='mne',\n"
            "    reference_method='bipolar',\n"
            "    electrode_path=Path('../../data/sample_labels.xlsx'),\n"
            "    artifact_methods=['misc'],\n"
            ")\n"
            "result = run_pipeline(config)\n"
            "print(f'Monopolar {len(result.raw.ch_names)} ch → Bipolar {len(result.referenced.ch_names)} ch')\n"
            "artifact_table = result.artifact_tables['misc']\n"
            "print(f'Misc artifacts: {len(artifact_table)} events')\n"
            "print(artifact_table.head())",
            tags=["worked"],
        ),
        md("## Plot artifact density by channel"),
        code(
            "import matplotlib.pyplot as plt\n\n"
            "if len(artifact_table):\n"
            "    counts = artifact_table.groupby('channel').size().sort_values(ascending=False)\n"
            "    counts.head(8).plot(kind='bar', figsize=(8, 3), title='Misc artifacts per channel')\n"
            "    plt.ylabel('Count')\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
            "else:\n"
            "    print('No misc artifacts detected at default threshold.')",
            tags=["worked"],
        ),
        md("## Next step\n\nChapter 04b covers photodiode synchronization."),
    ]
)

# --- Worked 08 sync (NEW) ---
NOTEBOOKS["worked-examples/08_first_synchronization_run.ipynb"] = nb(
    [
        md(
            "# Worked Example: Photodiode Synchronization\n\n"
            "## Goal\n"
            "Align behavioral sync pulses to the neural photodiode recording.\n"
        ),
        code(
            "import pandas as pd\n"
            "import mne\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "from pathlib import Path\n"
            "from LFPAnalysis import sync_utils\n\n"
            "beh_ts = pd.read_csv(Path('../../data/sample_ts.csv'))['beh_ts'].values\n"
            "photodiode = mne.io.read_raw_fif(Path('../../data/sample_photodiode.fif'), preload=True, verbose=False)\n"
            "slope, offset = sync_utils.synchronize_data(beh_ts=beh_ts, mne_sync=photodiode, sync_source='photodiode')\n"
            "print(f'slope={slope:.4f}, offset={offset:.2f}')",
            tags=["worked"],
        ),
        md("## Plot first matched pulses"),
        code(
            "neural_ts = sync_utils.get_neural_ts_photodiode(photodiode)\n"
            "n_show = min(30, len(beh_ts), len(neural_ts))\n"
            "fig, ax = plt.subplots(figsize=(8, 3))\n"
            "ax.plot(np.arange(n_show), beh_ts[:n_show], 'o-', label='behavioral')\n"
            "ax.plot(np.arange(n_show), neural_ts[:n_show], 's-', label='neural (photodiode)')\n"
            "ax.set(xlabel='Pulse index', ylabel='Time (s)', title='Sync pulse alignment')\n"
            "ax.legend()\n"
            "fig.tight_layout()\n"
            "plt.show()",
            tags=["worked"],
        ),
        md("## Next step\n\nContinue to artifact QC (chapter 05)."),
    ]
)

# --- Worked 03 epoching ---
NOTEBOOKS["worked-examples/03_first_epoching_run.ipynb"] = nb(
    [
        md(
            "# Worked Example: Feedback-Locked Epochs with Behavior Metadata\n\n"
            "## Goal\n"
            "Epoch around real `feedback_start` times and attach reward/RPE for condition contrasts.\n"
        ),
        code(
            "import pandas as pd\n"
            "from pathlib import Path\n"
            "from LFPAnalysis import build_event_locked_pipeline_config, run_pipeline\n\n"
            "beh = pd.read_csv(Path('../../data/sample_beh.csv'))\n"
            "config = build_event_locked_pipeline_config(\n"
            "    Path('../../data/sample_ieeg_bp.fif'),\n"
            "    file_format='mne',\n"
            "    event_name='feedback_start',\n"
            "    event_times=beh['feedback_start'].tolist(),\n"
            "    baseline_mode='zscore',\n"
            "    baseline_window=(-0.5, 0.0),\n"
            "    tmin=-0.5,\n"
            "    tmax=1.5,\n"
            "    metadata={'reward': beh['reward'].tolist(), 'rpe': beh['rpe'].tolist()},\n"
            ")\n"
            "result = run_pipeline(config)\n"
            "epochs = result.epochs\n"
            "print(f'{len(epochs)} epochs, reward={epochs.metadata.reward.sum():.0f} win / {(epochs.metadata.reward==0).sum():.0f} loss')",
            tags=["worked"],
        ),
        md("## Plot evoked reward vs no-reward (ACC channel)"),
        code(
            "import matplotlib.pyplot as plt\n\n"
            "chan = 'racas1-racas2'\n"
            "reward_evoked = epochs[epochs.metadata['reward'] == 1].copy().pick([chan]).average()\n"
            "loss_evoked = epochs[epochs.metadata['reward'] == 0].copy().pick([chan]).average()\n"
            "fig, ax = plt.subplots(figsize=(7, 3))\n"
            "ax.plot(reward_evoked.times, reward_evoked.data[0], label='reward')\n"
            "ax.plot(loss_evoked.times, loss_evoked.data[0], label='no reward')\n"
            "ax.axvline(0, color='k', ls='--', lw=0.8)\n"
            "ax.set(xlabel='Time (s)', ylabel='Amplitude (a.u.)', title=f'Evoked {chan}')\n"
            "ax.legend()\n"
            "fig.tight_layout()\n"
            "plt.show()",
            tags=["worked"],
        ),
        md("## Next step\n\nChapter 08 covers PSD and FOOOF."),
    ]
)

# --- Worked 04 PSD/FOOOF ---
NOTEBOOKS["worked-examples/04_first_psd_and_fooof_run.ipynb"] = nb(
    [
        md(
            "# Worked Example: PSD and FOOOF Reward Contrast\n\n"
            "## Goal\n"
            "Compare reward vs no-reward spectra on feedback epochs. FOOOF uses `analysis_utils` — see {doc}`11_advanced_utility_interoperability`.\n"
        ),
        code(
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from pathlib import Path\n"
            "from LFPAnalysis import load_lfp\n"
            "from LFPAnalysis.config import LoadConfig\n\n"
            "beh = pd.read_csv(Path('../../data/sample_beh.csv'))\n"
            "epochs = load_lfp(LoadConfig(path=Path('../../data/sample_feedback_start-epo.fif'), file_format='mne'))\n"
            "epochs.metadata = beh[['reward', 'rpe']]\n"
            "chan = 'racas1-racas2'\n"
            "reward_psd = epochs['reward == 1'].copy().pick([chan]).compute_psd(fmin=1, fmax=80, verbose=False)\n"
            "loss_psd = epochs['reward == 0'].copy().pick([chan]).compute_psd(fmin=1, fmax=80, verbose=False)",
            tags=["worked"],
        ),
        md("## Plot PSD contrast"),
        code(
            "fig, ax = plt.subplots(figsize=(7, 3))\n"
            "ax.semilogy(reward_psd.freqs, reward_psd.get_data()[0, 0], label='reward')\n"
            "ax.semilogy(loss_psd.freqs, loss_psd.get_data()[0, 0], label='no reward')\n"
            "ax.set(xlabel='Frequency (Hz)', ylabel='PSD', title=f'{chan} feedback-locked')\n"
            "ax.legend()\n"
            "fig.tight_layout()\n"
            "plt.show()",
            tags=["worked"],
        ),
        md("## FOOOF via advanced utility (subset for speed)"),
        code(
            "from LFPAnalysis import analysis_utils\n\n"
            "epochs_sub = epochs.copy().pick([chan])[:10]\n"
            "fooof_kwargs = {\n"
            "    'peak_width_limits': (1, 12),\n"
            "    'min_peak_height': 0.0,\n"
            "    'peak_threshold': 2.0,\n"
            "    'max_n_peaks': 6,\n"
            "    'freq_range': (1, 40),\n"
            "}\n"
            "_, fooof_table = analysis_utils.FOOOF_compute_epochs(\n"
            "    epochs_sub, tmin=float(epochs_sub.times[0]), tmax=float(epochs_sub.times[-1]), **fooof_kwargs\n"
            ")\n"
            "print(fooof_table.head())",
            tags=["worked"],
        ),
        md(
            "## Next step\n\n"
            "Advanced utility interoperability: 11_advanced_utility_interoperability. "
            "Next chapter: time-frequency."
        ),
    ]
)

# --- Worked 05 TFR ---
NOTEBOOKS["worked-examples/05_first_tfr_run.ipynb"] = nb(
    [
        md("# Worked Example: Time-Frequency Reward Contrast\n\n## Goal\nMorlet TFR on feedback epochs with reward vs no-reward difference map."),
        code(
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from pathlib import Path\n"
            "from LFPAnalysis import load_lfp\n"
            "from LFPAnalysis.config import LoadConfig\n\n"
            "beh = pd.read_csv(Path('../../data/sample_beh.csv'))\n"
            "epochs = load_lfp(LoadConfig(path=Path('../../data/sample_feedback_start-epo.fif'), file_format='mne'))\n"
            "epochs.metadata = beh[['reward', 'rpe']]\n"
            "chan = 'racas1-racas2'\n"
            "freqs = np.arange(4, 30, 4)\n"
            "reward_tfr = epochs['reward == 1'].copy().pick([chan]).compute_tfr(\n"
            "    method='morlet', freqs=freqs, n_cycles=freqs / 2.0, average=True, verbose=False\n"
            ")\n"
            "loss_tfr = epochs['reward == 0'].copy().pick([chan]).compute_tfr(\n"
            "    method='morlet', freqs=freqs, n_cycles=freqs / 2.0, average=True, verbose=False\n"
            ")\n"
            "diff = reward_tfr.data[0] - loss_tfr.data[0]",
            tags=["worked"],
        ),
        code(
            "fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)\n"
            "for ax, data, title in zip(\n"
            "    axes,\n"
            "    [reward_tfr.data[0], loss_tfr.data[0], diff],\n"
            "    ['reward', 'no reward', 'difference'],\n"
            "):\n"
            "    im = ax.imshow(data, aspect='auto', origin='lower',\n"
            "                   extent=[reward_tfr.times[0], reward_tfr.times[-1], freqs[0], freqs[-1]],\n"
            "                   cmap='RdBu_r')\n"
            "    ax.axvline(0, color='k', ls='--', lw=0.8)\n"
            "    ax.set(xlabel='Time (s)', title=title)\n"
            "axes[0].set_ylabel('Frequency (Hz)')\n"
            "fig.colorbar(im, ax=axes, shrink=0.8, label='Power')\n"
            "fig.tight_layout()\n"
            "plt.show()",
            tags=["worked"],
        ),
        md("## Next step\n\nChapter 10 covers connectivity."),
    ]
)

# --- Worked 06 connectivity ---
NOTEBOOKS["worked-examples/06_first_connectivity_run.ipynb"] = nb(
    [
        md(
            "# Worked Example: ACC–Frontal Connectivity\n\n"
            "## Goal\n"
            "Spectral connectivity on real feedback epochs with surrogate null. See 11_advanced_utility_interoperability.\n"
        ),
        code(
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from pathlib import Path\n"
            "from mne_connectivity import spectral_connectivity_epochs\n"
            "from LFPAnalysis import load_lfp, oscillation_utils\n"
            "from LFPAnalysis.config import LoadConfig\n\n"
            "beh = pd.read_csv(Path('../../data/sample_beh.csv'))\n"
            "epochs = load_lfp(LoadConfig(path=Path('../../data/sample_feedback_start-epo.fif'), file_format='mne'))\n"
            "epochs.metadata = beh[['reward', 'rpe']]\n"
            "epochs_sub = epochs.copy().pick(['racas1-racas2', 'rmolf5-rmolf6'])\n"
            "con = spectral_connectivity_epochs(\n"
            "    epochs_sub, method='coh', mode='multitaper', fmin=13, fmax=30, faverage=True, verbose=False\n"
            ")\n"
            "coh_value = float(con.get_data()[0, 0])\n"
            "print(f'Beta coherence ACC–frontal: {coh_value:.3f}')",
            tags=["worked"],
        ),
        code(
            "seed_data = epochs_sub.get_data()[:, 0, :]\n"
            "surr = oscillation_utils.make_surrogate_arrays(\n"
            "    seed_data, method='swap_epochs', n_shuffles=50, rng_seed=42, return_generator=False\n"
            ")\n"
            "surr_coh = [float(np.corrcoef(surr[i], epochs_sub.get_data()[:, 1, :].mean(axis=0))[0, 1]) for i in range(min(10, len(surr)))]\n"
            "fig, ax = plt.subplots(figsize=(5, 3))\n"
            "ax.hist(surr_coh, bins=15, color='0.7', label='surrogate (approx)')\n"
            "ax.axvline(coh_value, color='r', lw=2, label='observed')\n"
            "ax.set(xlabel='Coupling proxy', title='Surrogate null (illustrative)')\n"
            "ax.legend()\n"
            "fig.tight_layout()\n"
            "plt.show()",
            tags=["worked"],
        ),
        md("## Next step\n\n11_advanced_utility_interoperability for full connectivity API. Chapter 10b for statistics."),
    ]
)

# --- Worked 07 migration ---
NOTEBOOKS["worked-examples/07_migrating_condensed_notebook.ipynb"] = nb(
    [
        md(
            "# Worked Example: Migrating the Condensed Notebook\n\n"
            "## Goal\n"
            "Compare stable API epoching with real `feedback_start` times against the legacy shim.\n"
        ),
        code(
            "import warnings\n"
            "import pandas as pd\n"
            "from pathlib import Path\n"
            "from LFPAnalysis import build_basic_pipeline_config, run_pipeline, legacy\n\n"
            "beh = pd.read_csv(Path('../../data/sample_beh.csv'))\n"
            "config = build_basic_pipeline_config(Path('../../data/sample_ieeg.fif'), file_format='mne')\n"
            "result = run_pipeline(config)\n"
            "print('Stable load:', type(result.raw).__name__)",
            tags=["worked"],
        ),
        code(
            "with warnings.catch_warnings(record=True) as caught:\n"
            "    warnings.simplefilter('always')\n"
            "    epochs = legacy.make_epochs(\n"
            "        load_path='../../data/sample_ieeg_bp.fif',\n"
            "        behav_name='feedback_start',\n"
            "        behav_times=beh['feedback_start'].tolist()[:5],\n"
            "        ev_start_s=0.5,\n"
            "        ev_end_s=1.5,\n"
            "    )\n"
            "    print('Legacy epochs:', len(epochs), 'trials (first 5 for speed)')\n"
            "    if caught:\n"
            "        print('Deprecation:', caught[0].message)",
            tags=["worked"],
        ),
        md(
            "## Next step\n\n"
            "For advanced utility interoperability see 11_advanced_utility_interoperability. "
            "Migration chapters 20–25 cover full old-notebook translation."
        ),
    ]
)

# --- Worked 09 stats (NEW) ---
NOTEBOOKS["worked-examples/09_first_stats_run.ipynb"] = nb(
    [
        md("# Worked Example: Time-Resolved Regression vs RPE\n\n## Goal\nRegress feedback-locked beta power against reward prediction error."),
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import mne\n"
            "from pathlib import Path\n"
            "from LFPAnalysis import statistics_utils\n\n"
            "np.random.seed(42)\n"
            "beh = pd.read_csv(Path('../../data/sample_beh.csv'))\n"
            "epochs = mne.read_epochs(Path('../../data/sample_feedback_start-epo.fif'), preload=True, verbose=False)\n"
            "epochs.metadata = beh[['reward', 'rpe']]\n"
            "chan = 'racas1-racas2'\n"
            "ep = epochs.copy().pick([chan]).filter(13, 30, verbose=False)\n"
            "times = ep.times\n"
            "betas, z_betas = [], []\n"
            "for t_idx in range(0, len(times), max(1, len(times) // 20)):\n"
            "    power = ep.get_data()[:, 0, t_idx] ** 2\n"
            "    df = pd.DataFrame({'power': power, 'rpe': epochs.metadata['rpe'].values})\n"
            "    res = statistics_utils.permutation_regression_zscore(df, 'power ~ rpe', n_permutations=100)\n"
            "    betas.append(res.loc[res.predictor == 'rpe', 'raw_beta'].values[0])\n"
            "    z_betas.append(res.loc[res.predictor == 'rpe', 'z_beta'].values[0])\n"
            "time_sub = times[::max(1, len(times) // 20)]",
            tags=["worked"],
        ),
        code(
            "fig, ax = plt.subplots(figsize=(7, 3))\n"
            "ax.plot(time_sub, z_betas, 'o-')\n"
            "ax.axvline(0, color='k', ls='--', lw=0.8)\n"
            "ax.axhline(0, color='0.5', ls='-', lw=0.5)\n"
            "ax.set(xlabel='Time (s)', ylabel='z-beta (RPE)', title=f'{chan} feedback-locked')\n"
            "fig.tight_layout()\n"
            "plt.show()",
            tags=["worked"],
        ),
        md("## Next step\n\nChapter 11 covers advanced utility interoperability."),
    ]
)

# --- Smoke tests ---
NOTEBOOKS["smoke-tests/01_install_and_import.ipynb"] = nb(
    [
        md("# Smoke Test: Install and Import"),
        code(
            "from LFPAnalysis import LoadConfig, PipelineConfig, run_pipeline\n"
            "import LFPAnalysis\n"
            "print('ok', sorted(LFPAnalysis.__all__)[:5], '…')",
        ),
    ]
)

NOTEBOOKS["smoke-tests/02_load_sample_data.ipynb"] = nb(
    [
        md("# Smoke Test: Load Sample Data"),
        code(
            "from pathlib import Path\n"
            "from LFPAnalysis import load_lfp\n"
            "from LFPAnalysis.config import LoadConfig\n\n"
            "raw = load_lfp(LoadConfig(path=Path('../../data/sample_ieeg.fif'), file_format='mne'))\n"
            "print(raw, raw.info['sfreq'], len(raw.ch_names))",
        ),
    ]
)

NOTEBOOKS["smoke-tests/03_reference_and_artifacts.ipynb"] = nb(
    [
        md("# Smoke Test: Reference and Artifacts"),
        code(
            "from pathlib import Path\n"
            "from LFPAnalysis import load_lfp, preprocess_lfp, detect_artifacts\n"
            "from LFPAnalysis.config import LoadConfig, ReferenceConfig, ArtifactConfig\n\n"
            "raw = load_lfp(LoadConfig(path=Path('../../data/sample_ieeg_bp.fif'), file_format='mne'))\n"
            "raw.crop(tmax=60)\n"
            "ref = preprocess_lfp(raw, ReferenceConfig(method='none'))\n"
            "tables = detect_artifacts(ref, ArtifactConfig(methods=['misc'], misc_peak_thresh=8.0))\n"
            "print(tables['misc'].head(), len(tables['misc']))",
        ),
    ]
)

NOTEBOOKS["smoke-tests/04_baselining_and_epoching.ipynb"] = nb(
    [
        md("# Smoke Test: Baselining and Epoching"),
        code(
            "import pandas as pd\n"
            "from pathlib import Path\n"
            "from LFPAnalysis import load_lfp, make_epochs, baseline_lfp\n"
            "from LFPAnalysis.config import LoadConfig, EpochConfig, BaselineConfig\n\n"
            "beh = pd.read_csv(Path('../../data/sample_beh.csv'))\n"
            "raw = load_lfp(LoadConfig(path=Path('../../data/sample_ieeg_bp.fif'), file_format='mne'))\n"
            "raw.crop(tmax=300)\n"
            "epochs = make_epochs(\n"
            "    raw,\n"
            "    EpochConfig(\n"
            "        enabled=True,\n"
            "        event_name='feedback_start',\n"
            "        event_times=beh['feedback_start'].tolist()[:3],\n"
            "        tmin=-0.5,\n"
            "        tmax=1.0,\n"
            "    ),\n"
            ")\n"
            "ep_base, summary = baseline_lfp(epochs, BaselineConfig(mode='zscore', enabled=True, baseline_window=(-0.5, 0.0)))\n"
            "print(ep_base, summary.head())",
        ),
    ]
)

NOTEBOOKS["smoke-tests/05_psd_and_fooof.ipynb"] = nb(
    [
        md("# Smoke Test: PSD and FOOOF"),
        code(
            "from pathlib import Path\n"
            "from LFPAnalysis import load_lfp\n"
            "from LFPAnalysis.config import LoadConfig\n\n"
            "epochs = load_lfp(LoadConfig(path=Path('../../data/sample_feedback_start-epo.fif'), file_format='mne'))\n"
            "epochs = epochs[:5].pick(['racas1-racas2'])\n"
            "spectrum = epochs.compute_psd(fmin=1, fmax=40, verbose=False)\n"
            "print('psd', spectrum.get_data().shape)",
        ),
    ]
)

NOTEBOOKS["smoke-tests/06_time_frequency.ipynb"] = nb(
    [
        md("# Smoke Test: Time-Frequency"),
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "from pathlib import Path\n"
            "from LFPAnalysis import load_lfp\n"
            "from LFPAnalysis.config import LoadConfig\n\n"
            "epochs = load_lfp(LoadConfig(path=Path('../../data/sample_feedback_start-epo.fif'), file_format='mne'))\n"
            "epochs = epochs[:5].pick(['racas1-racas2'])\n"
            "freqs = np.arange(4, 20, 4)\n"
            "tfr = epochs.compute_tfr(method='morlet', freqs=freqs, n_cycles=3, average=True, verbose=False)\n"
            "print(tfr.data.shape)",
        ),
    ]
)

NOTEBOOKS["smoke-tests/07_connectivity_and_surrogates.ipynb"] = nb(
    [
        md("# Smoke Test: Connectivity and Surrogates"),
        code(
            "import pandas as pd\n"
            "from pathlib import Path\n"
            "from mne_connectivity import spectral_connectivity_epochs\n"
            "from LFPAnalysis import load_lfp, oscillation_utils\n"
            "from LFPAnalysis.config import LoadConfig\n\n"
            "epochs = load_lfp(LoadConfig(path=Path('../../data/sample_feedback_start-epo.fif'), file_format='mne'))\n"
            "epochs = epochs[:8].pick(['racas1-racas2', 'rmolf5-rmolf6'])\n"
            "con = spectral_connectivity_epochs(epochs, method='coh', fmin=13, fmax=30, faverage=True, verbose=False)\n"
            "surr = oscillation_utils.make_surrogate_arrays(epochs.get_data()[:, 0, :], n_shuffles=3, rng_seed=42, return_generator=False)\n"
            "print(con.get_data().shape, len(surr))",
        ),
    ]
)

NOTEBOOKS["smoke-tests/08_synchronization.ipynb"] = nb(
    [
        md("# Smoke Test: Synchronization"),
        code(
            "import pandas as pd\n"
            "import mne\n"
            "from pathlib import Path\n"
            "from LFPAnalysis import sync_utils\n\n"
            "beh_ts = pd.read_csv(Path('../../data/sample_ts.csv'))['beh_ts'].values\n"
            "pd_raw = mne.io.read_raw_fif(Path('../../data/sample_photodiode.fif'), preload=True, verbose=False)\n"
            "slope, offset = sync_utils.synchronize_data(beh_ts=beh_ts, mne_sync=pd_raw)\n"
            "print(round(slope, 4), round(offset, 2))",
        ),
    ]
)


def main() -> None:
    for rel, notebook in NOTEBOOKS.items():
        write_nb(BOOK / rel, notebook)
        print(f"Wrote {rel}")


if __name__ == "__main__":
    main()
