# Resources and References

Educational resources and methodological references behind the workflows in this
book. Entries marked **(paper)** or **(video/course)** are conceptual reading;
entries marked **(code/provenance)** point to source repositories, license
notices, or implementation details that a given utility was ported from or that
explain a specific engineering decision.

Most of these previously lived only as inline comments in `LFPAnalysis/*.py`.
They are collated here so readers have a single starting point for the literature
and code behind each methodology. The file path and line noted after each entry
is where the reference is used in the source, so you can jump to the exact
function that relies on it.

```{note}
This list started from references embedded in the codebase. It is not a complete
bibliography of the field — add the canonical texts and papers you cite verbally
(see *Suggested additions* at the bottom) as you have time.
```

## Referencing and preprocessing

- **(paper)** Re-referencing and per-trial normalization rationale —
  <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5309795/>
  (`LFPAnalysis/lfp_preprocess_utils.py:237`, and the baseline z-scoring note at
  `:375`). Motivates subtracting the baseline-period mean when z-scoring across
  the whole trial.
- **(paper)** Variance-normalization caveat — high-power artifacts/outliers can
  contaminate normalization —
  <https://www.sciencedirect.com/science/article/abs/pii/S1053811913009919>
  (`LFPAnalysis/lfp_preprocess_utils.py:306`).
- **(paper)** Referencing and scalp/depth topography (Brain Topography) —
  <https://link.springer.com/article/10.1007/s10548-014-0379-1#Sec2>
  (`LFPAnalysis/lfp_preprocess_utils.py:1247`).

## Time-frequency, oscillations, and spectral analysis

- **(paper)** Oscillations review used as the source for the surrogate/permutation
  approach —
  <https://www.sciencedirect.com/science/article/pii/S0959438814001640>
  (`LFPAnalysis/oscillation_utils.py:103`).
- **(paper)** High-gamma amplitude via squaring and log of the analytic amplitude —
  <https://www.nature.com/articles/nn.3101#Sec15>
  (`LFPAnalysis/oscillation_utils.py:1975`).
- **(code/provenance)** eBOSC (extended Better OSCillation detection) usage example
  that the eBOSC workflow follows —
  <https://github.com/jkosciessa/eBOSC_py/blob/main/examples/eBOSC_example_empirical.ipynb>
  (`LFPAnalysis/oscillation_utils.py:4173`).
- **(code/provenance)** BOSC library GPL license notice embedded with the ported
  BOSC code (Caplan, Hughes, Whitten, Dickson, 2010) —
  <http://www.gnu.org/licenses/>
  (`LFPAnalysis/oscillation_utils.py:2982`).

## Connectivity and mutual information (GCMI)

- **(code/provenance)** Gaussian-Copula Mutual Information — the connectivity/MI
  helpers are ported from the `gcmi` reference implementation —
  <https://github.com/robince/gcmi/blob/master/python/gcmi.py>
  (`LFPAnalysis/oscillation_utils.py:737`).
- **(paper)** Mixture-entropy approximation via the unscented transform (Huber,
  Bailey, Durrant-Whyte & Hanebeck, "On entropy approximation for Gaussian
  mixture random vectors") — <http://dx.doi.org/10.1109/MFI.2008.4648062>
  (`LFPAnalysis/oscillation_utils.py:1508`).
- **(paper)** KL-divergence approximation between two Gaussian mixtures
  (Goldberger, Gordon & Greenspan, "An efficient image similarity measure based
  on approximations of KL-divergence between two Gaussian mixtures") —
  <http://dx.doi.org/10.1109/ICCV.2003.1238387>
  (`LFPAnalysis/oscillation_utils.py:1513`).

## I/O and synchronization

- **(code/provenance)** Neuralynx I/O — the `.ncs` reading code is adapted from
  NeuralynxIO —
  <https://github.com/alafuzof/NeuralynxIO/blob/master/neuralynx_io/neuralynx_io.py>
  (`LFPAnalysis/nlx_utils.py:1`).
- **(code/provenance)** Photodiode-based synchronization — potential improvements
  by synergizing with `pd-parser` —
  <https://github.com/alexrockhill/pd-parser>
  (`LFPAnalysis/sync_utils.py:7`).

## Engineering notes and code utilities

These are implementation-detail links (library issues, gists, Q&A) rather than
methodology references, kept here for completeness so nothing in the source is
undocumented.

- **(code/utility)** MNE-Python: all EDF channels must share the same sample rate —
  <https://github.com/mne-tools/mne-python/issues/10635>
  (`LFPAnalysis/lfp_preprocess_utils.py:1665`).
- **(code/utility)** Resampling helper gist (larsoner) —
  <https://gist.github.com/larsoner/01642cb3789992fbca59>
  (`LFPAnalysis/lfp_preprocess_utils.py:1807`).
- **(code/utility)** MNE-Python: post-resampling end timings may not match
  perfectly — <https://github.com/mne-tools/mne-python/issues/8257>
  (`LFPAnalysis/lfp_preprocess_utils.py:1823`).
- **(code/utility)** Deleting multiple indices from a list at once (Stack Overflow) —
  <https://stackoverflow.com/questions/21032034/deleting-multiple-indexes-from-a-list-at-once-python>
  (`LFPAnalysis/oscillation_utils.py:3629`).

## Software this book builds on

- **MNE-Python** — <https://mne.tools/> — core object model (`Raw`, `Epochs`,
  `compute_tfr`) used throughout the book.
- **FOOOF / specparam** — <https://fooof-tools.github.io/fooof/> — aperiodic +
  periodic spectral parameterization used in {doc}`08_first_psd_and_fooof`.
- **mne-connectivity** — <https://mne.tools/mne-connectivity/> — connectivity
  spectra used in {doc}`10_first_connectivity_and_surrogates`.

## Suggested additions (populate as needed)

Papers/texts commonly cited for these methods that are not yet referenced in the
code — add links and annotations when you get a chance:

- Cohen, M. X. *Analyzing Neural Time Series Data: Theory and Practice* (MIT
  Press) — general TFR / phase / connectivity reference.
- The FOOOF/specparam paper (Donoghue et al., 2020, *Nature Neuroscience*).
- MNE-Python methods paper (Gramfort et al., 2013).
- Ince et al. (2017) GCMI paper (the theory behind the `gcmi` port above).
- Any lab-specific tutorials, lecture recordings, or course videos you want
  users to watch first.
