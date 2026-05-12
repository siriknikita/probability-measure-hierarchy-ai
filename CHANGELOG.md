# Changelog

Each release is one iteration of the experiment. Tags (`vX.Y.Z`) mark the state of
the repo after that iteration's notebook was run and its result logged; the full
write-up for each run lives in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## [0.1.0] — 2026-05-12

### Added

- `01_dynamic_hierarchical_random_measure.ipynb` — hierarchy of probability measures over greed/fear, soft-membership features, baseline vs hierarchy MLP.
- `02_markov_kernel_hierarchy.ipynb` — empirical and neural Markov kernels over hierarchy states.
- `03_neural_measure_valued_sequence_model.ipynb` — measure-valued sequence model over latent atoms.
- uv environment (Python 3.12) with numpy, pandas, matplotlib, torch, jupyter.

### Findings

- Synthetic data only — the three formalizations run end-to-end; no real-data claims yet.
