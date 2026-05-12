# Experiments journal

Append-only log of empirical tests of the **hierarchy-of-probability-measures** idea on real market data. Each entry is one notebook execution; newest entries are at the bottom.

## What this project is testing

A hierarchy is constructed over the 2D `(greed_index, fear_index)` feature space, where greed/fear are sigmoid-compressed combinations of returns, volatility, drawdown, MA-distance, and volume z-score. The hierarchy has K leaves, each corresponding to a region of behavioral feature space. For each row, soft membership across the K leaves gives a K-dimensional probability vector summing to 1.

The recurring question across every entry below: **does this hierarchy representation add predictive value beyond plain price/volume features?**

Three feature sets are compared (with naming that drifts slightly across notebooks but means the same thing):

- `baseline` — price/volume features only (returns at multiple lags, volatility, drawdown, price-to-MA, volume z-score). ~17 columns.
- `indices` — `baseline` + scalar `greed_index` + scalar `fear_index`.
- `hierarchy` / `rf_joint` — `indices` + K soft-membership columns from the leaf decomposition.

A `baseline_rf` vs `hierarchy_rf` win establishes whether the soft-membership representation carries additional signal; a `baseline` vs `indices` comparison isolates the two scalar indices' contribution.

## Findings to date (running synthesis)

| target | resolution | result | notebook |
|---|---|---|---|
| **Direction (up/down next H bars)** | daily, 3 tickers | hierarchy does NOT help — baseline wins on balanced accuracy on all tickers | NB05 |

Net read (updated after NB05):

- **The hierarchy does not help direction prediction on daily data** (NB05). Baseline wins on balanced accuracy on SPY, NVDA and BTC-USD once each feature set tunes its own RF; the earlier NB04 "+0.9 pp on SPY" was a hyperparameter artifact.

## Metric glossary

**Classification metrics (direction or multi-class state):**

- **Accuracy** — fraction of correct predictions. Sensitive to class imbalance.
- **Balanced accuracy** — mean of per-class recall. 0.5 = random regardless of imbalance. The preferred single-number metric when classes are skewed.

**Regression metrics (log return prediction):**

- **MAE / RMSE** — error of predicted return. *Only informative compared to a `zero` baseline.* On low-signal targets the model correctly predicts near-zero and MAE will match `zero` exactly — that is success, not failure.

**Trivial baselines (always scored alongside the models):**

- `majority` — predict the more frequent class.

**Validation protocol:**

- **Time-respecting split** (NB05) — single train/val/test split, chronologically ordered, no shuffling. Val used for RF hyperparameter selection; model refit on train+val before scoring on test.

---

## 2026-05-12 15:12:40 — NB05 multi-ticker ablation (val-tuned RF, hierarchy max_depth=2)

**TL;DR — direction prediction, daily.** Three-way ablation (`baseline` / `indices` / `hierarchy`) for 5-day-ahead direction prediction, run with proper validation-set hyperparameter tuning on SPY, NVDA, BTC-USD. **Negative result: baseline wins on balanced accuracy on every ticker** (SPY −1.29 pp for indices, NVDA −0.56 pp, BTC −0.96 pp). The earlier NB04 "+0.9 pp on SPY" finding was a hyperparameter artifact — when each feature set is allowed to tune itself, the indices/hierarchy models pick deeper RFs that look slightly better on validation but don't generalize. The leaf decomposition is the culprit: 16 noisy columns competing with 17 baseline features triggers the same overfit pathology RandomForest is prone to.

**Read on permutation importance:** the top-MDI features in the classifier (`vol_5`, `vol_20`, `vol_10`) have the *worst* held-out permutation importance (vol_20: MDI rank 3, perm rank 33 of 34). The features the model trains on most aren't the ones that generalize. Only `ret_20` and `price_to_ma_50` carry held-out signal consistently across all three tickers. `greed_index` has negative permutation importance on SPY (-0.0114) and zero on NVDA inside the joint model.

**Config**

```json
{
  "notebook": "05_multi_ticker_validation_tuned.ipynb",
  "tickers": [
    "SPY",
    "NVDA",
    "BTC-USD"
  ],
  "start_date": "2010-01-01",
  "horizon": 5,
  "test_fraction": 0.2,
  "validation_fraction": 0.15,
  "max_hierarchy_depth": 2,
  "min_leaf_count": 200,
  "rf_n_estimators": 300,
  "rf_grid": [
    {
      "max_depth": 4,
      "min_samples_leaf": 10
    },
    {
      "max_depth": 4,
      "min_samples_leaf": 20
    },
    {
      "max_depth": 4,
      "min_samples_leaf": 40
    },
    {
      "max_depth": 6,
      "min_samples_leaf": 10
    },
    {
      "max_depth": 6,
      "min_samples_leaf": 20
    },
    {
      "max_depth": 6,
      "min_samples_leaf": 40
    },
    {
      "max_depth": 8,
      "min_samples_leaf": 10
    },
    {
      "max_depth": 8,
      "min_samples_leaf": 20
    },
    {
      "max_depth": 8,
      "min_samples_leaf": 40
    }
  ],
  "seed": 42
}
```

**Per-ticker test-set results**

| ticker   | model     |   n_train |   n_val |   n_test |   n_leaves |   test_return_mae |   test_dir_acc |   test_bal_acc |   test_dir_acc_from_reg |   test_majority_acc | rf_reg_params                            | rf_cls_params                            |
|:---------|:----------|----------:|--------:|---------:|-----------:|------------------:|---------------:|---------------:|------------------------:|--------------------:|:-----------------------------------------|:-----------------------------------------|
| SPY      | baseline  |      2759 |     488 |      812 |          4 |           0.0146  |         0.564  |         0.5138 |                  0.6232 |              0.6281 | {'max_depth': 4, 'min_samples_leaf': 40} | {'max_depth': 6, 'min_samples_leaf': 20} |
| SPY      | indices   |      2759 |     488 |      812 |          4 |           0.0146  |         0.5419 |         0.5009 |                  0.6244 |              0.6281 | {'max_depth': 4, 'min_samples_leaf': 40} | {'max_depth': 8, 'min_samples_leaf': 20} |
| SPY      | hierarchy |      2759 |     488 |      812 |          4 |           0.0146  |         0.5456 |         0.5025 |                  0.6232 |              0.6281 | {'max_depth': 4, 'min_samples_leaf': 40} | {'max_depth': 6, 'min_samples_leaf': 10} |
| NVDA     | baseline  |      2759 |     488 |      812 |          4 |           0.05073 |         0.548  |         0.5034 |                  0.5727 |              0.5998 | {'max_depth': 4, 'min_samples_leaf': 20} | {'max_depth': 6, 'min_samples_leaf': 10} |
| NVDA     | indices   |      2759 |     488 |      812 |          4 |           0.05067 |         0.5431 |         0.4978 |                  0.5764 |              0.5998 | {'max_depth': 4, 'min_samples_leaf': 10} | {'max_depth': 8, 'min_samples_leaf': 10} |
| NVDA     | hierarchy |      2759 |     488 |      812 |          4 |           0.05064 |         0.5394 |         0.4963 |                  0.5788 |              0.5998 | {'max_depth': 4, 'min_samples_leaf': 10} | {'max_depth': 8, 'min_samples_leaf': 10} |
| BTC-USD  | baseline  |      2856 |     505 |      841 |          4 |           0.04234 |         0.5101 |         0.5258 |                  0.5268 |              0.5446 | {'max_depth': 6, 'min_samples_leaf': 20} | {'max_depth': 8, 'min_samples_leaf': 10} |
| BTC-USD  | indices   |      2856 |     505 |      841 |          4 |           0.04239 |         0.5006 |         0.5163 |                  0.5375 |              0.5446 | {'max_depth': 6, 'min_samples_leaf': 20} | {'max_depth': 8, 'min_samples_leaf': 10} |
| BTC-USD  | hierarchy |      2856 |     505 |      841 |          4 |           0.04241 |         0.503  |         0.5178 |                  0.5232 |              0.5446 | {'max_depth': 6, 'min_samples_leaf': 10} | {'max_depth': 8, 'min_samples_leaf': 10} |

**Deltas vs baseline (per ticker)**

| ticker   |   ('indices_minus_baseline', 'test_return_mae') |   ('indices_minus_baseline', 'test_bal_acc') |   ('indices_minus_baseline', 'test_dir_acc') |   ('indices_minus_baseline', 'test_dir_acc_from_reg') |   ('hierarchy_minus_baseline', 'test_return_mae') |   ('hierarchy_minus_baseline', 'test_bal_acc') |   ('hierarchy_minus_baseline', 'test_dir_acc') |   ('hierarchy_minus_baseline', 'test_dir_acc_from_reg') |
|:---------|------------------------------------------------:|---------------------------------------------:|---------------------------------------------:|------------------------------------------------------:|--------------------------------------------------:|-----------------------------------------------:|-----------------------------------------------:|--------------------------------------------------------:|
| SPY      |                                           1e-05 |                                     -0.01292 |                                     -0.02217 |                                               0.00123 |                                             1e-05 |                                       -0.01133 |                                       -0.01847 |                                                 0       |
| NVDA     |                                          -6e-05 |                                     -0.00564 |                                     -0.00493 |                                               0.00369 |                                            -9e-05 |                                       -0.00719 |                                       -0.00862 |                                                 0.00616 |
| BTC-USD  |                                           5e-05 |                                     -0.00959 |                                     -0.00951 |                                               0.0107  |                                             7e-05 |                                       -0.00805 |                                       -0.00713 |                                                -0.00357 |
