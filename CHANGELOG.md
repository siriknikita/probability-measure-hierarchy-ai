# Changelog

Each release is one iteration of the experiment. Tags (`vX.Y.Z`) mark the state of
the repo after that iteration's notebook was run and its result logged; the full
write-up for each run lives in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## [0.9.0] — 2026-05-12

### Added

- `11_two_stage_horizon_sweep.ipynb` — NB11 two-stage horizon sweep (BTC-USD, horizons=[1, 4, 24]h, K=4).

### Changed

- `EXPERIMENTS.md`: NB11 entry; running synthesis updated.

### Findings

- **Two-stage horizon sweep** (hourly BTC, horizons {1, 4, 24}h): **4h is the clean positive: IC lift +0.022, bal_acc +0.28 pp, both with 4/5 folds positive.** 1h within noise; 24h underpowered

## [0.8.0] — 2026-05-12

### Added

- `10_two_stage_transition_return.ipynb` — NB10 two-stage transition-aware return prediction (BTC-USD, 1h, K=4).

### Changed

- `EXPERIMENTS.md`: NB10 entry; running synthesis updated.

### Findings

- **Two-stage transition-aware (50/50 split, 1h)** (hourly BTC, walk-forward): hypothesis test null at 1h: transition_rf IC ≈ hierarchy_rf IC, +0.0005 mean lift, 2/5 folds positive

## [0.7.0] — 2026-05-12

### Added

- `09_hourly_multi_crypto_walkforward.ipynb` — NB09 multi-crypto hourly robustness (1h horizon, K=4, 5-fold WF).

### Changed

- `EXPERIMENTS.md`: NB09 entry; running synthesis updated.

### Findings

- **Multi-crypto hourly robustness** (hourly BTC/ETH/SOL/XRP): baseline RF beats trivial baselines on 4/4 tickers; hierarchy IC lift positive on 4/4 (range +0.000 to +0.010); balanced-acc lift split 2/2 (BTC/SOL positive, ETH/XRP negative)

## [0.6.0] — 2026-05-12

### Added

- `08_hourly_btc_walkforward.ipynb` — NB08 hourly BTC-USD walk-forward (horizons=[1, 4]h, K=4).

### Changed

- `EXPERIMENTS.md`: NB08 entry; running synthesis updated.

### Findings

- **Next-hour log return (single ticker)** (hourly BTC-USD, walk-forward): hierarchy adds small but real signal: IC +0.034, balanced acc 51.9%

## [0.5.0] — 2026-05-12

### Added

- `07_markov_kernel_depth_sweep.ipynb` — NB07 Markov-kernel depth sweep (depths=[1, 2, 3, 4], horizon=5).

### Changed

- `EXPERIMENTS.md`: NB07 entry; running synthesis updated.

### Findings

- **Next-state — depth sweep** (daily, 3 tickers, K∈{2,4,8,16}): robust at every depth; soft-membership features help at K=4–8, hurt at K=16

## [0.4.0] — 2026-05-12

### Added

- `06_real_data_markov_kernel.ipynb` — NB06 Markov-kernel on real data (K=4, horizon=5).

### Changed

- `EXPERIMENTS.md`: NB06 entry; running synthesis updated.

### Findings

- **Next-state (which leaf in K leaves)** (daily, 3 tickers, K=4): hierarchy framing works — RF beats persistence by 7–13 pp everywhere

## [0.3.0] — 2026-05-12

### Added

- `05_multi_ticker_validation_tuned.ipynb` — NB05 multi-ticker ablation (val-tuned RF, hierarchy max_depth=2).
- `EXPERIMENTS.md` — append-only experiment journal with running synthesis and metric glossary.
- Dependency: `tabulate`.

### Findings

- **Direction (up/down next H bars)** (daily, 3 tickers): hierarchy does NOT help — baseline wins on balanced accuracy on all tickers

## [0.2.0] — 2026-05-12

### Added

- `04_real_data_price_prediction_yfinance.ipynb` — first real-data run: baseline vs hierarchy-augmented sklearn models on close / return / direction.
- Dependencies: `scikit-learn`, `yfinance`.

### Changed

- README documents NB04.

### Findings

- Single untuned run on SPY: hierarchy-augmented direction classifier +0.9 pp over baseline. Needs per-feature-set tuning and more tickers before it means anything.

## [0.1.0] — 2026-05-12

### Added

- `01_dynamic_hierarchical_random_measure.ipynb` — hierarchy of probability measures over greed/fear, soft-membership features, baseline vs hierarchy MLP.
- `02_markov_kernel_hierarchy.ipynb` — empirical and neural Markov kernels over hierarchy states.
- `03_neural_measure_valued_sequence_model.ipynb` — measure-valued sequence model over latent atoms.
- uv environment (Python 3.12) with numpy, pandas, matplotlib, torch, jupyter.

### Findings

- Synthetic data only — the three formalizations run end-to-end; no real-data claims yet.
