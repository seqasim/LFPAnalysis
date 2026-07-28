"""Regenerate LFPAnalysisBook notebooks from the canonical chapter-aligned map."""

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
    path.write_text(json.dumps(notebook, indent=2) + "\n")


NOTEBOOKS: dict[str, dict] = {}

# --- Worked 03 load ---
NOTEBOOKS["worked-examples/03_first_import_and_load.ipynb"] = nb(
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
            "print('Channels:', raw.ch_names[:8], '…')\n"
            "print('preload in metadata:', result.metadata.get('preload'))",
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
        md("## Next step\n\nContinue to chapter 04 (`04_first_reference`) for bipolar referencing."),
    ]
)

# --- Worked 04 reference ---
NOTEBOOKS["worked-examples/04_first_preprocessing_run.ipynb"] = nb(
    [
        md(
            "# Worked Example: Bipolar Reference\n\n"
            "## Goal\n"
            "Apply bipolar referencing to the real recording. Artifact QC is covered in chapter 05.\n"
        ),
        code(
            "from pathlib import Path\n"
            "from LFPAnalysis import build_basic_pipeline_config, run_pipeline\n\n"
            "config = build_basic_pipeline_config(\n"
            "    Path('../../data/sample_ieeg.fif'),\n"
            "    file_format='mne',\n"
            "    reference_method='bipolar',\n"
            "    electrode_path=Path('../../data/sample_labels.xlsx'),\n"
            ")\n"
            "result = run_pipeline(config)\n"
            "# After bipolar re-reference, prep drops the superseded monopolar Raw to save RAM.\n"
            "print(f'Bipolar channels: {len(result.referenced.ch_names)}')\n"
            "print('First bipolar channels:', result.referenced.ch_names[:5])\n"
            "print('result.raw is None after re-reference:', result.raw is None)",
            tags=["worked"],
        ),
        md("## Plot one bipolar channel"),
        code(
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n\n"
            "chan = result.referenced.ch_names[0]\n"
            "sfreq = result.referenced.info['sfreq']\n"
            "start = int(240 * sfreq)\n"
            "stop = start + int(2 * sfreq)\n"
            "data = result.referenced.get_data(picks=[chan])[0, start:stop]\n"
            "times = np.arange(len(data)) / sfreq\n"
            "fig, ax = plt.subplots(figsize=(8, 3))\n"
            "ax.plot(times, data, 'k', lw=0.8)\n"
            "ax.set(xlabel='Time (s)', ylabel='Amplitude', title=f'Bipolar {chan}')\n"
            "fig.tight_layout()\n"
            "plt.show()",
            tags=["worked"],
        ),
        md("## Next step\n\nChapter 04b (`04b_first_synchronization`) covers photodiode synchronization."),
    ]
)

# --- Worked 04b sync ---
NOTEBOOKS["worked-examples/04b_first_synchronization_run.ipynb"] = nb(
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
        md("## Next step\n\nContinue to artifact QC (chapter 05, `05_first_artifact_pass`)."),
    ]
)

# --- Worked 05 artifacts ---
NOTEBOOKS["worked-examples/05_first_artifact_pass.ipynb"] = nb(
    [
        md(
            "# Worked Example: Artifact Pass on Bipolar Continuous Data\n\n"
            "## Goal\n"
            "Run misc + IED detectors on `sample_ieeg_bp.fif` and plot flagged time ranges.\n"
        ),
        code(
            "from pathlib import Path\n"
            "from LFPAnalysis import build_basic_pipeline_config, run_pipeline\n\n"
            "config = build_basic_pipeline_config(\n"
            "    Path('../../data/sample_ieeg_bp.fif'),\n"
            "    file_format='mne',\n"
            "    artifact_methods=['misc', 'ied'],\n"
            "    preload=True,\n"
            ")\n"
            "result = run_pipeline(config)\n"
            "misc_table = result.artifact_tables['misc']\n"
            "ied_table = result.artifact_tables['ied']\n"
            "print(f'Misc events: {len(misc_table)}, IED events: {len(ied_table)}')\n"
            "print(misc_table.head())",
            tags=["worked"],
        ),
        md("## Plot flagged time ranges on a short raw segment"),
        code(
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n\n"
            "raw = result.referenced if result.referenced is not None else result.raw\n"
            "chan = 'racas1-racas2' if 'racas1-racas2' in raw.ch_names else raw.ch_names[0]\n"
            "sfreq = float(raw.info['sfreq'])\n"
            "t0, t1 = 240.0, 250.0\n"
            "start, stop = int(t0 * sfreq), int(t1 * sfreq)\n"
            "segment = raw.get_data(picks=[chan])[0, start:stop]\n"
            "times = np.arange(len(segment)) / sfreq + t0\n"
            "fig, ax = plt.subplots(figsize=(10, 3))\n"
            "ax.plot(times, segment, 'k', lw=0.7)\n"
            "chan_events = misc_table[misc_table['channel'] == chan] if len(misc_table) else misc_table\n"
            "for _, row in chan_events.iterrows():\n"
            "    t = float(row['time_seconds'])\n"
            "    if t0 <= t <= t1:\n"
            "        ax.axvspan(t - 0.05, t + 0.05, color='C1', alpha=0.35)\n"
            "ax.set(xlabel='Time (s)', ylabel='Amplitude', title=f'{chan} with misc flags')\n"
            "fig.tight_layout()\n"
            "plt.show()",
            tags=["worked"],
        ),
        md("## Next step\n\nChapter 06 (`06_first_baseline`) covers baseline correction on epochs."),
    ]
)

# --- Worked 06 baseline ---
NOTEBOOKS["worked-examples/06_first_baseline_run.ipynb"] = nb(
    [
        md(
            "# Worked Example: Baseline Correction on Feedback Epochs\n\n"
            "## Goal\n"
            "Apply z-score baselining to pre-built feedback epochs via the analysis spine "
            "(`build_analysis_config` + `run_analysis`) without running PSD.\n"
        ),
        code(
            "from pathlib import Path\n"
            "from LFPAnalysis import build_analysis_config, load_lfp, run_analysis\n"
            "from LFPAnalysis.config import LoadConfig\n\n"
            "epochs = load_lfp(\n"
            "    LoadConfig(path=Path('../../data/sample_feedback_start-epo.fif'), file_format='mne', preload=True)\n"
            ")\n"
            "analysis = run_analysis(\n"
            "    epochs,\n"
            "    build_analysis_config(baseline_mode='zscore', baseline_window=(-0.5, 0.0)),\n"
            ")\n"
            "print(analysis.baseline_summary.head())\n"
            "print('spectral (should be empty):', analysis.spectral)\n"
            "print('epoch window:', analysis.epochs.times[[0, -1]])",
            tags=["worked"],
        ),
        md("## Plot baseline-corrected evoked activity for one channel"),
        code(
            "import matplotlib.pyplot as plt\n\n"
            "chan = 'racas1-racas2'\n"
            "evoked = analysis.epochs.copy().pick([chan]).average()\n"
            "fig, ax = plt.subplots(figsize=(7, 3))\n"
            "ax.plot(evoked.times, evoked.data[0], label='zscore-baselined')\n"
            "ax.axvline(0, color='k', ls='--', lw=0.8)\n"
            "ax.axvspan(-0.5, 0.0, color='0.85', label='baseline window')\n"
            "ax.set(xlabel='Time (s)', ylabel='Amplitude (a.u.)', title=f'Baselined evoked {chan}')\n"
            "ax.legend()\n"
            "fig.tight_layout()\n"
            "plt.show()",
            tags=["worked"],
        ),
        md("## Next step\n\nChapter 07 (`07_first_event_locked_workflow`) epochs continuous data with behavior metadata."),
    ]
)

# --- Worked 07 epoching ---
NOTEBOOKS["worked-examples/07_first_epoching_run.ipynb"] = nb(
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
            "print(f'{len(epochs)} epochs, reward={epochs.metadata.reward.sum():.0f} win / {(epochs.metadata.reward==0).sum():.0f} loss')\n"
            "print(result.baseline_summary.head())",
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
        md("## Next step\n\nChapter 08 (`08_first_psd_and_fooof`) covers PSD and FOOOF."),
    ]
)

# --- Worked 08 PSD/FOOOF ---
NOTEBOOKS["worked-examples/08_first_psd_and_fooof_run.ipynb"] = nb(
    [
        md(
            "# Worked Example: PSD and FOOOF Reward Contrast\n\n"
            "## Goal\n"
            "Compare reward vs no-reward spectra on feedback epochs using the stable analysis spine "
            "(`build_analysis_config` + `run_analysis`). FOOOF details: 11_advanced_utility_interoperability.\n"
        ),
        code(
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from pathlib import Path\n"
            "from LFPAnalysis import build_analysis_config, load_lfp, run_analysis\n"
            "from LFPAnalysis.config import LoadConfig\n\n"
            "beh = pd.read_csv(Path('../../data/sample_beh.csv'))\n"
            "epochs = load_lfp(\n"
            "    LoadConfig(path=Path('../../data/sample_feedback_start-epo.fif'), file_format='mne', preload=True)\n"
            ")\n"
            "epochs.metadata = beh[['reward', 'rpe']]\n"
            "chan = 'racas1-racas2'\n"
            "reward_epochs = epochs['reward == 1'].copy().pick([chan])\n"
            "loss_epochs = epochs['reward == 0'].copy().pick([chan])\n"
            "reward_result = run_analysis(reward_epochs, build_analysis_config(spectral_method='psd', fmin=1.0, fmax=80.0))\n"
            "loss_result = run_analysis(loss_epochs, build_analysis_config(spectral_method='psd', fmin=1.0, fmax=80.0))\n"
            "reward_psd = reward_result.spectral['spectrum']\n"
            "loss_psd = loss_result.spectral['spectrum']\n"
            "print('reward PSD shape:', reward_psd.get_data().shape)",
            tags=["worked"],
        ),
        md("## Plot PSD contrast"),
        code(
            "fig, ax = plt.subplots(figsize=(7, 3))\n"
            "ax.semilogy(reward_psd.freqs, reward_psd.get_data().mean(axis=0)[0], label='reward')\n"
            "ax.semilogy(loss_psd.freqs, loss_psd.get_data().mean(axis=0)[0], label='no reward')\n"
            "ax.set(xlabel='Frequency (Hz)', ylabel='PSD', title=f'{chan} feedback-locked')\n"
            "ax.legend()\n"
            "fig.tight_layout()\n"
            "plt.show()",
            tags=["worked"],
        ),
        md("## FOOOF via stable analysis spine (subset for speed)"),
        code(
            "epochs_sub = epochs.copy().pick([chan])[:10]\n"
            "fooof_result = run_analysis(\n"
            "    epochs_sub,\n"
            "    build_analysis_config(spectral_method='fooof', fooof_range=(1.0, 40.0)),\n"
            ")\n"
            "fooof_table = fooof_result.spectral['table']\n"
            "print(fooof_table.head())",
            tags=["worked"],
        ),
        md(
            "## Next step\n\n"
            "Advanced utility interoperability: 11_advanced_utility_interoperability. "
            "Next chapter: time-frequency (`09_first_time_frequency`)."
        ),
    ]
)

# --- Worked 09 TFR ---
NOTEBOOKS["worked-examples/09_first_tfr_run.ipynb"] = nb(
    [
        md(
            "# Worked Example: Time-Frequency Reward Contrast\n\n"
            "## Goal\n"
            "Morlet TFR on feedback epochs via `build_analysis_config` / `run_analysis`, "
            "with reward vs no-reward difference map."
        ),
        code(
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from pathlib import Path\n"
            "from LFPAnalysis import build_analysis_config, load_lfp, run_analysis\n"
            "from LFPAnalysis.config import LoadConfig\n\n"
            "beh = pd.read_csv(Path('../../data/sample_beh.csv'))\n"
            "epochs = load_lfp(\n"
            "    LoadConfig(path=Path('../../data/sample_feedback_start-epo.fif'), file_format='mne', preload=True)\n"
            ")\n"
            "epochs.metadata = beh[['reward', 'rpe']]\n"
            "chan = 'racas1-racas2'\n"
            "freqs = np.arange(4, 30, 4).tolist()\n"
            "tfr_cfg = build_analysis_config(tfr_method='morlet', tfr_freqs=freqs, tfr_n_cycles=3.0)\n"
            "reward_result = run_analysis(epochs['reward == 1'].copy().pick([chan]), tfr_cfg)\n"
            "loss_result = run_analysis(epochs['reward == 0'].copy().pick([chan]), tfr_cfg)\n"
            "reward_power = reward_result.tfr['power'].average()\n"
            "loss_power = loss_result.tfr['power'].average()\n"
            "diff = reward_power.data[0] - loss_power.data[0]\n"
            "print('TFR shape:', reward_power.data.shape)",
            tags=["worked"],
        ),
        code(
            "fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)\n"
            "freq_arr = np.asarray(freqs)\n"
            "for ax, data, title in zip(\n"
            "    axes,\n"
            "    [reward_power.data[0], loss_power.data[0], diff],\n"
            "    ['reward', 'no reward', 'difference'],\n"
            "):\n"
            "    im = ax.imshow(\n"
            "        data,\n"
            "        aspect='auto',\n"
            "        origin='lower',\n"
            "        extent=[reward_power.times[0], reward_power.times[-1], freq_arr[0], freq_arr[-1]],\n"
            "        cmap='RdBu_r',\n"
            "    )\n"
            "    ax.axvline(0, color='k', ls='--', lw=0.8)\n"
            "    ax.set(xlabel='Time (s)', title=title)\n"
            "axes[0].set_ylabel('Frequency (Hz)')\n"
            "fig.colorbar(im, ax=axes, shrink=0.8, label='Power')\n"
            "fig.tight_layout()\n"
            "plt.show()",
            tags=["worked"],
        ),
        md("## Next step\n\nChapter 10 (`10_first_connectivity_and_surrogates`) covers connectivity."),
    ]
)

# --- Worked 10 connectivity ---
NOTEBOOKS["worked-examples/10_first_connectivity_run.ipynb"] = nb(
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
            "epochs = load_lfp(\n"
            "    LoadConfig(path=Path('../../data/sample_feedback_start-epo.fif'), file_format='mne', preload=True)\n"
            ")\n"
            "epochs.metadata = beh[['reward', 'rpe']]\n"
            "epochs_sub = epochs.copy().pick(['racas1-racas2', 'rmolf5-rmolf6'])\n"
            "con = spectral_connectivity_epochs(\n"
            "    epochs_sub, method='coh', mode='multitaper', fmin=13, fmax=30, faverage=True, verbose=False\n"
            ")\n"
            "coh_mat = con.get_data(output='dense')[:, :, 0]\n"
            "coh_value = float(coh_mat[1, 0])\n"
            "print(f'Beta coherence ACC–frontal: {coh_value:.3f}')",
            tags=["worked"],
        ),
        md("## Plot connectivity value and surrogate distribution"),
        code(
            "seed_data = epochs_sub.get_data()[:, 0, :]\n"
            "surr = oscillation_utils.make_surrogate_arrays(\n"
            "    seed_data, method='swap_epochs', n_shuffles=50, rng_seed=42, return_generator=False\n"
            ")\n"
            "target_mean = epochs_sub.get_data()[:, 1, :].mean(axis=0)\n"
            "surr_coh = [\n"
            "    float(np.corrcoef(surr[i].mean(axis=0), target_mean)[0, 1]) for i in range(min(20, len(surr)))\n"
            "]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(10, 3))\n"
            "im = axes[0].imshow(coh_mat, vmin=0, vmax=1, cmap='viridis')\n"
            "axes[0].set_xticks([0, 1])\n"
            "axes[0].set_yticks([0, 1])\n"
            "axes[0].set_xticklabels(['ACC', 'frontal'])\n"
            "axes[0].set_yticklabels(['ACC', 'frontal'])\n"
            "axes[0].set_title('Beta coherence')\n"
            "fig.colorbar(im, ax=axes[0], shrink=0.8)\n"
            "axes[1].hist(surr_coh, bins=15, color='0.7', label='surrogate (approx)')\n"
            "axes[1].axvline(coh_value, color='r', lw=2, label='observed')\n"
            "axes[1].set(xlabel='Coupling proxy', title='Surrogate null (illustrative)')\n"
            "axes[1].legend()\n"
            "fig.tight_layout()\n"
            "plt.show()",
            tags=["worked"],
        ),
        md(
            "## Next step\n\n"
            "11_advanced_utility_interoperability for full connectivity API. "
            "Chapter 10b (`10b_first_time_resolved_stats`) for statistics."
        ),
    ]
)

# --- Worked 10b stats ---
NOTEBOOKS["worked-examples/10b_first_stats_run.ipynb"] = nb(
    [
        md(
            "# Worked Example: Time-Resolved Regression vs RPE\n\n"
            "## Goal\n"
            "Regress feedback-locked beta power against reward prediction error."
        ),
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
        md("## Next step\n\nChapter 11 (`11_advanced_utility_interoperability`) covers advanced utility interoperability."),
    ]
)

# --- Worked 13 dataframes ---
NOTEBOOKS["worked-examples/13_assembling_dataframes.ipynb"] = nb(
    [
        md(
            "# Worked Example: Assembling Analysis DataFrames\n\n"
            "## Goal\n"
            "Build a tidy long dataframe (participant, electrode, trial, time, power, regressors)\n"
            "from sample feedback epochs — the input shape expected by `statistics_utils`.\n"
        ),
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import mne\n"
            "from pathlib import Path\n\n"
            "beh = pd.read_csv(Path('../../data/sample_beh.csv'))\n"
            "epochs = mne.read_epochs(Path('../../data/sample_feedback_start-epo.fif'), preload=True, verbose=False)\n"
            "epochs.metadata = beh[['reward', 'rpe']].copy()\n"
            "epochs.metadata['trial'] = np.arange(len(epochs))\n\n"
            "elec_df = pd.read_csv(Path('../../data/sample_labels_bp'))\n"
            "chan = 'racas1-racas2'\n"
            "roi = elec_df.loc[elec_df.label == chan, 'salman_region'].iloc[0]\n\n"
            "ep = epochs.copy().pick([chan]).filter(13, 30, verbose=False)\n"
            "times = ep.times\n"
            "step = max(1, len(times) // 20)\n"
            "rows = []\n"
            "for t_idx in range(0, len(times), step):\n"
            "    power = ep.get_data()[:, 0, t_idx] ** 2\n"
            "    for trial_i, p in enumerate(power):\n"
            "        rows.append({\n"
            "            'participant': 'sample',\n"
            "            'unique_label': chan,\n"
            "            'roi': roi,\n"
            "            'trial': trial_i,\n"
            "            'ts': times[t_idx],\n"
            "            'tfr': p,\n"
            "            'rpe': epochs.metadata['rpe'].iloc[trial_i],\n"
            "            'reward': epochs.metadata['reward'].iloc[trial_i],\n"
            "        })\n"
            "smoothed_df = pd.DataFrame(rows)\n"
            "smoothed_df.head()",
            tags=["worked"],
        ),
        code(
            "print('shape:', smoothed_df.shape)\n"
            "print('n timepoints:', smoothed_df['ts'].nunique())\n"
            "print('n trials:', smoothed_df['trial'].nunique())\n"
            "print(smoothed_df.groupby('ts')['tfr'].mean().head())",
            tags=["worked"],
        ),
        md("## Next step\n\nChapter 14 (`14_group_level_statistics`) covers group-level statistics."),
    ]
)

# --- Worked 14 group stats ---
NOTEBOOKS["worked-examples/14_group_statistics.ipynb"] = nb(
    [
        md(
            "# Worked Example: Group-Level Statistics\n\n"
            "## Goal\n"
            "Run permutation OLS and a small synthetic multi-patient `time_resolved_mlm`\n"
            "demo (sample data is single-patient).\n"
        ),
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "from LFPAnalysis import statistics_utils\n\n"
            "np.random.seed(42)\n"
            "n = 80\n"
            "df = pd.DataFrame({\n"
            "    'power': np.random.randn(n) + 0.3 * np.linspace(-1, 1, n),\n"
            "    'rpe': np.linspace(-1, 1, n) + 0.1 * np.random.randn(n),\n"
            "})\n"
            "ols_res = statistics_utils.permutation_regression_zscore(\n"
            "    df, 'power ~ rpe', n_permutations=100\n"
            ")\n"
            "ols_res",
            tags=["worked"],
        ),
        code(
            "np.random.seed(0)\n"
            "rng = np.random.default_rng(0)\n"
            "participants = ['P01', 'P02', 'P03']\n"
            "electrodes = {'P01': ['e1', 'e2'], 'P02': ['e1'], 'P03': ['e1', 'e2', 'e3']}\n"
            "times = np.array([-0.2, 0.0, 0.2, 0.4])\n"
            "rows = []\n"
            "for p in participants:\n"
            "    n_trials = 30\n"
            "    rpe = rng.normal(size=n_trials)\n"
            "    for elec in electrodes[p]:\n"
            "        for trial in range(n_trials):\n"
            "            for ts in times:\n"
            "                signal = 0.4 * rpe[trial] * (ts > 0) + rng.normal(scale=1.0)\n"
            "                rows.append({\n"
            "                    'participant': p,\n"
            "                    'unique_label': f'{p}_{elec}',\n"
            "                    'trial': trial,\n"
            "                    'ts': ts,\n"
            "                    'tfr': signal,\n"
            "                    'rpe': rpe[trial],\n"
            "                })\n"
            "smoothed_df = pd.DataFrame(rows)\n\n"
            "mlm_res = statistics_utils.time_resolved_mlm(\n"
            "    smoothed_df,\n"
            "    y='tfr',\n"
            "    formula='tfr ~ 1 + rpe',\n"
            "    lower_group='unique_label',\n"
            "    higher_group='participant',\n"
            "    trial_key='trial',\n"
            "    n_permutations=20,\n"
            ")\n"
            "print(mlm_res.head())",
            tags=["worked"],
        ),
        md("## Next step\n\nChapter 15 (`15_saving_and_organizing_results`) covers saving results."),
    ]
)

# --- Worked 22 migration ---
NOTEBOOKS["worked-examples/22_migrating_condensed_notebook.ipynb"] = nb(
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
            "        tmax=1.5,\n"
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
            "from LFPAnalysis import build_analysis_config, load_lfp, run_analysis\n"
            "from LFPAnalysis.config import LoadConfig\n\n"
            "epochs = load_lfp(\n"
            "    LoadConfig(path=Path('../../data/sample_feedback_start-epo.fif'), file_format='mne', preload=True)\n"
            ")\n"
            "epochs = epochs[:5].pick(['racas1-racas2'])\n"
            "psd_result = run_analysis(epochs, build_analysis_config(spectral_method='psd', fmin=1.0, fmax=40.0))\n"
            "print('psd', psd_result.spectral['spectrum'].get_data().shape)\n"
            "fooof_result = run_analysis(epochs, build_analysis_config(spectral_method='fooof', fooof_range=(1.0, 40.0)))\n"
            "print('fooof rows', len(fooof_result.spectral['table']))",
        ),
    ]
)

NOTEBOOKS["smoke-tests/06_time_frequency.ipynb"] = nb(
    [
        md("# Smoke Test: Time-Frequency"),
        code(
            "import numpy as np\n"
            "from pathlib import Path\n"
            "from LFPAnalysis import build_analysis_config, load_lfp, run_analysis\n"
            "from LFPAnalysis.config import LoadConfig\n\n"
            "epochs = load_lfp(\n"
            "    LoadConfig(path=Path('../../data/sample_feedback_start-epo.fif'), file_format='mne', preload=True)\n"
            ")\n"
            "epochs = epochs[:5].pick(['racas1-racas2'])\n"
            "freqs = np.arange(4, 20, 4).tolist()\n"
            "result = run_analysis(epochs, build_analysis_config(tfr_method='morlet', tfr_freqs=freqs, tfr_n_cycles=3.0))\n"
            "print(result.tfr['power'].data.shape)",
        ),
    ]
)

NOTEBOOKS["smoke-tests/07_connectivity_and_surrogates.ipynb"] = nb(
    [
        md("# Smoke Test: Connectivity and Surrogates"),
        code(
            "from pathlib import Path\n"
            "from mne_connectivity import spectral_connectivity_epochs\n"
            "from LFPAnalysis import load_lfp, oscillation_utils\n"
            "from LFPAnalysis.config import LoadConfig\n\n"
            "epochs = load_lfp(\n"
            "    LoadConfig(path=Path('../../data/sample_feedback_start-epo.fif'), file_format='mne', preload=True)\n"
            ")\n"
            "epochs = epochs[:8].pick(['racas1-racas2', 'rmolf5-rmolf6'])\n"
            "con = spectral_connectivity_epochs(epochs, method='coh', fmin=13, fmax=30, faverage=True, verbose=False)\n"
            "surr = oscillation_utils.make_surrogate_arrays(\n"
            "    epochs.get_data()[:, 0, :], n_shuffles=3, rng_seed=42, return_generator=False\n"
            ")\n"
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
    # Remove superseded numeric prefixes that no longer match chapter numbers.
    worked = BOOK / "worked-examples"
    keep = {Path(rel).name for rel in NOTEBOOKS if rel.startswith("worked-examples/")}
    for path in worked.glob("*.ipynb"):
        if path.name not in keep:
            path.unlink()
            print(f"Removed stale {path.relative_to(BOOK)}")


if __name__ == "__main__":
    main()
