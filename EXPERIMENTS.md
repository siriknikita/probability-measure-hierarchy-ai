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
| **Next-state (which leaf in K leaves)** | daily, 3 tickers, K=4 | hierarchy framing works — RF beats persistence by 7–13 pp everywhere | NB06 |
| **Next-state — depth sweep** | daily, 3 tickers, K∈{2,4,8,16} | robust at every depth; soft-membership features help at K=4–8, hurt at K=16 | NB07 |
| **Next-hour log return (single ticker)** | hourly BTC-USD, walk-forward | hierarchy adds small but real signal: IC +0.034, balanced acc 51.9% | NB08 |

Net read (updated after NB08):

- **The hierarchy is useful as a target labeling** (NB06/07 — predicting which state the market moves to). This is the most robust positive result; it survives a depth sweep across K ∈ {2, 4, 8, 16} on three daily tickers. Soft-membership *as inputs* helps at K=4–8 and hurts at K=16.
- **First price-related signal at intraday resolution** (NB08). Hourly BTC-USD, walk-forward: `hierarchy_rf` IC +0.034 (4/5 folds positive), balanced accuracy 51.9%, both ahead of `baseline_rf`. Signal collapses at 4h. Research finding, not a tradeable edge.
- **The hierarchy never helps direction prediction on daily data** (NB05). The disagreement with the intraday results suggests the hierarchy's value, if any, emerges at short-horizon resolution.

## Metric glossary

**Classification metrics (direction or multi-class state):**

- **Accuracy** — fraction of correct predictions. Sensitive to class imbalance.
- **Balanced accuracy** — mean of per-class recall. 0.5 = random regardless of imbalance. The preferred single-number metric when classes are skewed.
- **Macro F1** — unweighted mean of per-class F1. Used for multi-class state prediction (NB06/07).
- **ROC AUC** — quality of the up-probability ranking. 0.5 = random; 0.52 = NB08 result; 0.55+ would be a strong intraday edge.
- **Transition accuracy** (NB06/07) — accuracy computed only over rows where the state actually changed (`state_t_future != state_t`). Persistence scores 0 here by construction. This isolates the model's ability to detect regime *changes*.

**Regression metrics (log return prediction):**

- **MAE / RMSE** — error of predicted return. *Only informative compared to a `zero` baseline.* On low-signal targets the model correctly predicts near-zero and MAE will match `zero` exactly — that is success, not failure.
- **Information Coefficient (IC)** — Spearman rank correlation between predicted and realized returns. The headline metric for intraday quant. +0.03 to +0.05 with sign-consistency across folds is a real edge; ±0.01 is noise.

**Trivial baselines (always scored alongside the models):**

- `zero` — predict 0 return / probability 0.5.
- `persistence` — predict next return = last return (regression); predict last direction (classification).
- `majority` — predict the more frequent class.
- `empirical_markov` (NB06/07) — `argmax_u P(z_{t+h}=u | z_t=s)` from the training transition matrix. Uses `state_t` only, no features.
- `random` (NB06/07) — uniform over K classes.

**Validation protocol:**

- **Time-respecting split** (NB05/06/07) — single train/val/test split, chronologically ordered, no shuffling. Val used for RF hyperparameter selection; model refit on train+val before scoring on test.
- **Walk-forward CV** (NB08) — N expanding-window folds. Each fold tunes on its own val slice, refits on full train, scores on the next test window, then slides. Per-fold metrics aggregated to mean ± std.

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

---

## 2026-05-12 15:19:06 — NB06 Markov-kernel on real data (K=4, horizon=5)

**TL;DR — next-state prediction, daily.** Reframes the prediction task: instead of "is the next-week return positive?" (NB05), this asks "which leaf of the K=4 hierarchy is the market in 5 days from now?". Chance is now 0.25, the hard baseline is **persistence** (`predict z_{t+5} = z_t`). **First positive result for the hierarchy idea on real data:** RF on baseline features beats persistence by +10.5 pp on SPY, +7.1 pp on NVDA, +6.8 pp on BTC-USD. The `rf_joint` model (features + soft-membership of `z_t`) edges `rf_features_only` by 1–2 pp on SPY and BTC, ties on NVDA.

**Transition accuracy (the hard subset)** — rows where state actually changed: persistence is 0 by construction; `rf_joint` scores **0.395 on SPY**, 0.297 on NVDA, 0.291 on BTC-USD vs random ≈ 0.25. The model genuinely detects regime changes, not just predicts "same state again."

**Critical caveat:** the empirical Markov kernel `argmax_u P(z_{t+5}=u | z_t=s)` ties persistence on all three tickers — its argmax is always the diagonal of the transition matrix. **Feature context is required; the state alone is not enough.** This kills the simplest version of ChatGPT's "Markov kernel" formalization (state-only prediction) and salvages a richer version (state + features).

**Transition matrices** (in entry below) show clear off-diagonal mass with 1-D ordering on SPY and stickier diagonals on BTC — both expected from a coherent state space.

**Config**

```json
{
  "notebook": "06_real_data_markov_kernel.ipynb",
  "tickers": [
    "SPY",
    "NVDA",
    "BTC-USD"
  ],
  "start_date": "2010-01-01",
  "horizon": 5,
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

| ticker   | model            |   K |   n_train |   n_val |   n_test |   accuracy |   macro_f1 |   transition_count |   transition_accuracy |
|:---------|:-----------------|----:|----------:|--------:|---------:|-----------:|-----------:|-------------------:|----------------------:|
| SPY      | random           |   4 |      2754 |     483 |      807 |     0.2454 |     0.2454 |                522 |                0.251  |
| SPY      | marginal         |   4 |      2754 |     483 |      807 |     0.2528 |     0.1009 |                522 |                0.2816 |
| SPY      | persistence      |   4 |      2754 |     483 |      807 |     0.3532 |     0.3527 |                522 |                0      |
| SPY      | empirical_markov |   4 |      2754 |     483 |      807 |     0.3606 |     0.3004 |                522 |                0.2146 |
| SPY      | rf_features_only |   4 |      2754 |     483 |      807 |     0.4634 |     0.4452 |                522 |                0.3831 |
| SPY      | rf_joint         |   4 |      2754 |     483 |      807 |     0.487  |     0.4733 |                522 |                0.3946 |
| NVDA     | random           |   4 |      2754 |     483 |      807 |     0.233  |     0.2321 |                491 |                0.2464 |
| NVDA     | marginal         |   4 |      2754 |     483 |      807 |     0.2763 |     0.1083 |                491 |                0.2505 |
| NVDA     | persistence      |   4 |      2754 |     483 |      807 |     0.3916 |     0.3868 |                491 |                0      |
| NVDA     | empirical_markov |   4 |      2754 |     483 |      807 |     0.3916 |     0.3868 |                491 |                0      |
| NVDA     | rf_features_only |   4 |      2754 |     483 |      807 |     0.4622 |     0.4484 |                491 |                0.3238 |
| NVDA     | rf_joint         |   4 |      2754 |     483 |      807 |     0.4622 |     0.4412 |                491 |                0.2974 |
| BTC-USD  | random           |   4 |      2851 |     500 |      836 |     0.2512 |     0.2457 |                484 |                0.2397 |
| BTC-USD  | marginal         |   4 |      2851 |     500 |      836 |     0.2309 |     0.0938 |                484 |                0.2231 |
| BTC-USD  | persistence      |   4 |      2851 |     500 |      836 |     0.4211 |     0.4029 |                484 |                0      |
| BTC-USD  | empirical_markov |   4 |      2851 |     500 |      836 |     0.4211 |     0.4029 |                484 |                0      |
| BTC-USD  | rf_features_only |   4 |      2851 |     500 |      836 |     0.4892 |     0.4557 |                484 |                0.314  |
| BTC-USD  | rf_joint         |   4 |      2851 |     500 |      836 |     0.4952 |     0.4621 |                484 |                0.2913 |

**Empirical transition matrices (training data)**


`SPY`

```
       0      1      2      3
0  0.436  0.245  0.160  0.159
1  0.321  0.276  0.212  0.190
2  0.178  0.296  0.284  0.242
3  0.067  0.186  0.342  0.406
```

`NVDA`

```
       0      1      2      3
0  0.373  0.225  0.222  0.180
1  0.190  0.453  0.185  0.172
2  0.265  0.191  0.281  0.262
3  0.172  0.133  0.310  0.384
```

`BTC-USD`

```
       0      1      2      3
0  0.473  0.185  0.211  0.130
1  0.168  0.516  0.207  0.109
2  0.235  0.190  0.312  0.262
3  0.125  0.109  0.266  0.500
```

**RF hyperparameter choices (val macro-F1)**

| ticker   | rf_features_only                         |   val_f1_features | rf_joint                                 |   val_f1_joint |
|:---------|:-----------------------------------------|------------------:|:-----------------------------------------|---------------:|
| SPY      | {'max_depth': 8, 'min_samples_leaf': 10} |            0.4568 | {'max_depth': 8, 'min_samples_leaf': 10} |         0.4499 |
| NVDA     | {'max_depth': 8, 'min_samples_leaf': 10} |            0.5244 | {'max_depth': 6, 'min_samples_leaf': 40} |         0.5315 |
| BTC-USD  | {'max_depth': 6, 'min_samples_leaf': 40} |            0.5009 | {'max_depth': 8, 'min_samples_leaf': 20} |         0.5013 |

---

## 2026-05-12 15:23:24 — NB07 Markov-kernel depth sweep (depths=[1, 2, 3, 4], horizon=5)

**TL;DR — robustness check of NB06.** Same setup as NB06 but sweeps hierarchy depth across `{1, 2, 3, 4}` → `K ∈ {2, 4, 8, 16}` leaves. **The NB06 finding is robust at every depth on every ticker.** RF beats persistence by a minimum of +3.7 pp (BTC, K=16) and a maximum of +15.0 pp (SPY, K=2).

**Multiplicative gain over chance grows with K:**

| ticker | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|
| SPY rf_joint / chance | 1.50× | 1.96× | 2.24× | 2.38× |
| NVDA | 1.43× | 1.85× | 2.20× | 2.27× |
| BTC | 1.51× | 1.98× | 2.67× | 2.83× |

The absolute gap over persistence shrinks as K grows (because persistence shrinks faster) but the multiplicative information gain *grows*. The RF recovers progressively more structure as the partition gets finer.

**Soft-membership feature contribution (`rf_joint − rf_features_only`):**

| ticker | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|
| SPY | −0.87 | **+3.22** | +0.24 | −0.62 |
| NVDA | +0.13 | +0.13 | −1.11 | −2.47 |
| BTC-USD | +0.36 | +0.60 | +1.55 | −1.19 |

Soft-membership features as inputs help most at K=4–8 and **hurt on every ticker at K=16** (the same overfit pathology NB05 saw with direction prediction). The "sweet spot for hierarchy as features" is moderate depth.

**Empirical Markov kernel collapses to persistence at every depth** — its diagonal is the argmax in essentially every row. State-only prediction has no value beyond persistence regardless of how finely you partition.

**Config**

```json
{
  "notebook": "07_markov_kernel_depth_sweep.ipynb",
  "tickers": [
    "SPY",
    "NVDA",
    "BTC-USD"
  ],
  "depths": [
    1,
    2,
    3,
    4
  ],
  "start_date": "2010-01-01",
  "horizon": 5,
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

**Per (ticker, depth, model) test-set results**

| ticker   |   depth |   K | model            |   n_test |   accuracy |   macro_f1 |   transition_count |   transition_accuracy |
|:---------|--------:|----:|:-----------------|---------:|-----------:|-----------:|-------------------:|----------------------:|
| SPY      |       1 |   2 | random           |      807 |     0.4907 |     0.4907 |                315 |                0.5365 |
| SPY      |       1 |   2 | marginal         |      807 |     0.5093 |     0.3374 |                315 |                0.5016 |
| SPY      |       1 |   2 | persistence      |      807 |     0.6097 |     0.6095 |                315 |                0      |
| SPY      |       1 |   2 | empirical_markov |      807 |     0.6097 |     0.6095 |                315 |                0      |
| SPY      |       1 |   2 | rf_features_only |      807 |     0.7596 |     0.7589 |                315 |                0.5841 |
| SPY      |       1 |   2 | rf_joint         |      807 |     0.7509 |     0.7496 |                315 |                0.5397 |
| SPY      |       2 |   4 | random           |      807 |     0.2342 |     0.2341 |                522 |                0.228  |
| SPY      |       2 |   4 | marginal         |      807 |     0.2528 |     0.1009 |                522 |                0.2816 |
| SPY      |       2 |   4 | persistence      |      807 |     0.3532 |     0.3527 |                522 |                0      |
| SPY      |       2 |   4 | empirical_markov |      807 |     0.3606 |     0.3004 |                522 |                0.2146 |
| SPY      |       2 |   4 | rf_features_only |      807 |     0.4585 |     0.4401 |                522 |                0.3793 |
| SPY      |       2 |   4 | rf_joint         |      807 |     0.4907 |     0.4775 |                522 |                0.3985 |
| SPY      |       3 |   8 | random           |      807 |     0.1264 |     0.1263 |                644 |                0.1289 |
| SPY      |       3 |   8 | marginal         |      807 |     0.1202 |     0.0268 |                644 |                0.1335 |
| SPY      |       3 |   8 | persistence      |      807 |     0.202  |     0.2009 |                644 |                0      |
| SPY      |       3 |   8 | empirical_markov |      807 |     0.2032 |     0.1571 |                644 |                0.1196 |
| SPY      |       3 |   8 | rf_features_only |      807 |     0.2776 |     0.2422 |                644 |                0.2283 |
| SPY      |       3 |   8 | rf_joint         |      807 |     0.28   |     0.2564 |                644 |                0.2096 |
| SPY      |       4 |  16 | random           |      807 |     0.0458 |     0.0465 |                720 |                0.0403 |
| SPY      |       4 |  16 | marginal         |      807 |     0.0533 |     0.0063 |                720 |                0.0556 |
| SPY      |       4 |  16 | persistence      |      807 |     0.1078 |     0.106  |                720 |                0      |
| SPY      |       4 |  16 | empirical_markov |      807 |     0.114  |     0.0894 |                720 |                0.0819 |
| SPY      |       4 |  16 | rf_features_only |      807 |     0.1549 |     0.13   |                720 |                0.1458 |
| SPY      |       4 |  16 | rf_joint         |      807 |     0.1487 |     0.1287 |                720 |                0.1361 |
| NVDA     |       1 |   2 | random           |      807 |     0.4857 |     0.4854 |                288 |                0.5035 |
| NVDA     |       1 |   2 | marginal         |      807 |     0.5254 |     0.3444 |                288 |                0.5035 |
| NVDA     |       1 |   2 | persistence      |      807 |     0.6431 |     0.6423 |                288 |                0      |
| NVDA     |       1 |   2 | empirical_markov |      807 |     0.6431 |     0.6423 |                288 |                0      |
| NVDA     |       1 |   2 | rf_features_only |      807 |     0.7125 |     0.7118 |                288 |                0.4722 |
| NVDA     |       1 |   2 | rf_joint         |      807 |     0.7138 |     0.7121 |                288 |                0.4792 |
| NVDA     |       2 |   4 | random           |      807 |     0.2416 |     0.2405 |                491 |                0.2546 |
| NVDA     |       2 |   4 | marginal         |      807 |     0.2763 |     0.1083 |                491 |                0.2505 |
| NVDA     |       2 |   4 | persistence      |      807 |     0.3916 |     0.3868 |                491 |                0      |
| NVDA     |       2 |   4 | empirical_markov |      807 |     0.3916 |     0.3868 |                491 |                0      |
| NVDA     |       2 |   4 | rf_features_only |      807 |     0.4622 |     0.449  |                491 |                0.3238 |
| NVDA     |       2 |   4 | rf_joint         |      807 |     0.4634 |     0.4424 |                491 |                0.2974 |
| NVDA     |       3 |   8 | random           |      807 |     0.1276 |     0.126  |                649 |                0.1341 |
| NVDA     |       3 |   8 | marginal         |      807 |     0.1314 |     0.029  |                649 |                0.1371 |
| NVDA     |       3 |   8 | persistence      |      807 |     0.1958 |     0.1929 |                649 |                0      |
| NVDA     |       3 |   8 | empirical_markov |      807 |     0.2144 |     0.1599 |                649 |                0.094  |
| NVDA     |       3 |   8 | rf_features_only |      807 |     0.2862 |     0.2595 |                649 |                0.228  |
| NVDA     |       3 |   8 | rf_joint         |      807 |     0.2751 |     0.2447 |                649 |                0.2173 |
| NVDA     |       4 |  16 | random           |      807 |     0.0694 |     0.0703 |                728 |                0.0646 |
| NVDA     |       4 |  16 | marginal         |      807 |     0.0731 |     0.0085 |                728 |                0.0728 |
| NVDA     |       4 |  16 | persistence      |      807 |     0.0979 |     0.0965 |                728 |                0      |
| NVDA     |       4 |  16 | empirical_markov |      807 |     0.1029 |     0.0729 |                728 |                0.0824 |
| NVDA     |       4 |  16 | rf_features_only |      807 |     0.166  |     0.1372 |                728 |                0.1456 |
| NVDA     |       4 |  16 | rf_joint         |      807 |     0.1413 |     0.1166 |                728 |                0.1209 |
| BTC-USD  |       1 |   2 | random           |      836 |     0.4797 |     0.479  |                276 |                0.5109 |
| BTC-USD  |       1 |   2 | marginal         |      836 |     0.5203 |     0.3423 |                276 |                0.5    |
| BTC-USD  |       1 |   2 | persistence      |      836 |     0.6699 |     0.6693 |                276 |                0      |
| BTC-USD  |       1 |   2 | empirical_markov |      836 |     0.6699 |     0.6693 |                276 |                0      |
| BTC-USD  |       1 |   2 | rf_features_only |      836 |     0.7548 |     0.7539 |                276 |                0.5217 |
| BTC-USD  |       1 |   2 | rf_joint         |      836 |     0.7584 |     0.7576 |                276 |                0.4674 |
| BTC-USD  |       2 |   4 | random           |      836 |     0.256  |     0.2533 |                484 |                0.2521 |
| BTC-USD  |       2 |   4 | marginal         |      836 |     0.2309 |     0.0938 |                484 |                0.2231 |
| BTC-USD  |       2 |   4 | persistence      |      836 |     0.4211 |     0.4029 |                484 |                0      |
| BTC-USD  |       2 |   4 | empirical_markov |      836 |     0.4211 |     0.4029 |                484 |                0      |
| BTC-USD  |       2 |   4 | rf_features_only |      836 |     0.4892 |     0.4557 |                484 |                0.314  |
| BTC-USD  |       2 |   4 | rf_joint         |      836 |     0.4952 |     0.4621 |                484 |                0.2913 |
| BTC-USD  |       3 |   8 | random           |      836 |     0.1292 |     0.1236 |                618 |                0.1197 |
| BTC-USD  |       3 |   8 | marginal         |      836 |     0.067  |     0.0157 |                618 |                0.068  |
| BTC-USD  |       3 |   8 | persistence      |      836 |     0.2608 |     0.2389 |                618 |                0      |
| BTC-USD  |       3 |   8 | empirical_markov |      836 |     0.2572 |     0.1962 |                618 |                0.0663 |
| BTC-USD  |       3 |   8 | rf_features_only |      836 |     0.3182 |     0.2739 |                618 |                0.2411 |
| BTC-USD  |       3 |   8 | rf_joint         |      836 |     0.3337 |     0.2808 |                618 |                0.233  |
| BTC-USD  |       4 |  16 | random           |      836 |     0.061  |     0.0567 |                719 |                0.0626 |
| BTC-USD  |       4 |  16 | marginal         |      836 |     0.0215 |     0.0026 |                719 |                0.0195 |
| BTC-USD  |       4 |  16 | persistence      |      836 |     0.14   |     0.1364 |                719 |                0      |
| BTC-USD  |       4 |  16 | empirical_markov |      836 |     0.1316 |     0.0904 |                719 |                0.0682 |
| BTC-USD  |       4 |  16 | rf_features_only |      836 |     0.189  |     0.1627 |                719 |                0.1697 |
| BTC-USD  |       4 |  16 | rf_joint         |      836 |     0.177  |     0.1515 |                719 |                0.1419 |

**Accuracy gap vs persistence**

|                |   empirical_markov |   marginal |   persistence |   random |   rf_features_only |   rf_joint |
|:---------------|-------------------:|-----------:|--------------:|---------:|-------------------:|-----------:|
| ('BTC-USD', 1) |             0      |    -0.1495 |             0 |  -0.1902 |             0.0849 |     0.0885 |
| ('BTC-USD', 2) |             0      |    -0.1902 |             0 |  -0.1651 |             0.0682 |     0.0742 |
| ('BTC-USD', 3) |            -0.0036 |    -0.1938 |             0 |  -0.1316 |             0.0574 |     0.073  |
| ('BTC-USD', 4) |            -0.0084 |    -0.1184 |             0 |  -0.0789 |             0.049  |     0.0371 |
| ('NVDA', 1)    |             0      |    -0.1177 |             0 |  -0.1574 |             0.0694 |     0.0706 |
| ('NVDA', 2)    |             0      |    -0.1152 |             0 |  -0.1499 |             0.0706 |     0.0719 |
| ('NVDA', 3)    |             0.0186 |    -0.0644 |             0 |  -0.0682 |             0.0905 |     0.0793 |
| ('NVDA', 4)    |             0.005  |    -0.0248 |             0 |  -0.0285 |             0.0682 |     0.0434 |
| ('SPY', 1)     |             0      |    -0.1004 |             0 |  -0.119  |             0.1499 |     0.1413 |
| ('SPY', 2)     |             0.0074 |    -0.1004 |             0 |  -0.119  |             0.1053 |     0.1375 |
| ('SPY', 3)     |             0.0012 |    -0.0818 |             0 |  -0.0756 |             0.0756 |     0.0781 |
| ('SPY', 4)     |             0.0062 |    -0.0545 |             0 |  -0.062  |             0.0471 |     0.0409 |

---

## 2026-05-12 15:31:13 — NB08 hourly BTC-USD walk-forward (horizons=[1, 4]h, K=4)

**TL;DR — next-hour and next-4h return prediction, hourly BTC, 5 walk-forward folds.** First test of the hierarchy on a price-related (not state-related) intraday target. **Result at 1h:** `hierarchy_rf` IC = **+0.034 ± 0.030** (positive on 4 of 5 folds), balanced accuracy = **51.87 ± 0.82 %** (above 50% on every fold), ROC AUC = **0.521**. `baseline_rf` is +0.6 pp behind on balanced accuracy and +0.006 behind on IC. Both models beat all trivial baselines (`zero`, `persistence`, `majority`) on every classification metric. **At 4h horizon signal collapses** — IC ≈ +0.01 (indistinguishable from zero given fold std), MAE marginally worse than `zero`.

**`persistence` has NEGATIVE IC at both horizons** (−0.019 at 1h, −0.021 at 4h). Hourly BTC returns mean-revert at this resolution; "predict last return" is actively wrong.

**Calibration (1h, pooled across folds):** monotonic-ish curve, realized up rate spans **0.49 → 0.54** across the 10 predicted-probability deciles. The top decile (avg pred 0.577) realizes 54.1% up — a real conditional edge for the most-confident 10% of predictions.

**MAE matches `zero` baseline exactly** at 1h. This is *correct* behavior: the conditional mean really is near zero, the model has learned that, and IC captures the tiny directional signal that MAE cannot.

**Why this entry is interesting.** Across NB04→NB07 the hierarchy never helped on a price-related target (NB05 negative for direction, NB06/07 positive only for the state-prediction reframe). **NB08 is the first place hierarchy adds detectable signal on a price-related target** — and it does so at the resolution (1h) the original ChatGPT formalization didn't explicitly aim at.

**Honest caveat:** 51.87% balanced accuracy with average top-decile |predicted return| of 12 bps is well below retail tradeability after bid-ask + exchange fees (~5–15 bps round-trip on BTC). This is a research finding, not a strategy.

**Config**

```json
{
  "notebook": "08_hourly_btc_walkforward.ipynb",
  "ticker": "BTC-USD",
  "period": "720d",
  "interval": "1h",
  "horizons_hours": [
    1,
    4
  ],
  "n_folds": 5,
  "min_train_fraction": 0.4,
  "test_fraction_per_fold": 0.1,
  "val_fraction_within_train": 0.15,
  "max_hierarchy_depth": 2,
  "min_leaf_count": 200,
  "rf_n_estimators": 200,
  "rf_grid": [
    {
      "max_depth": 6,
      "min_samples_leaf": 20
    },
    {
      "max_depth": 6,
      "min_samples_leaf": 50
    },
    {
      "max_depth": 10,
      "min_samples_leaf": 20
    },
    {
      "max_depth": 10,
      "min_samples_leaf": 50
    }
  ],
  "seed": 42
}
```

**Regression summary (mean ± std across folds)**

```
                          mae              rmse                ic         
                         mean      std     mean      std     mean      std
horizon model                                                             
1       baseline_rf   0.00296  0.00061  0.00464  0.00122  0.02848  0.02800
        hierarchy_rf  0.00296  0.00061  0.00464  0.00122  0.03367  0.02995
        persistence   0.00429  0.00088  0.00656  0.00167 -0.01871  0.02712
        zero          0.00296  0.00060  0.00464  0.00121  0.00000  0.00000
4       baseline_rf   0.00625  0.00155  0.00943  0.00278  0.01069  0.02047
        hierarchy_rf  0.00624  0.00155  0.00943  0.00278  0.00905  0.02279
        persistence   0.00897  0.00209  0.01299  0.00373 -0.02092  0.02153
        zero          0.00605  0.00134  0.00920  0.00252  0.00000  0.00000
```

**Classification summary (mean ± std across folds)**

```
                     accuracy          balanced_accuracy           roc_auc  \
                         mean      std              mean      std     mean   
horizon model                                                                
1       baseline_rf   0.51075  0.00781           0.51238  0.00603  0.51992   
        hierarchy_rf  0.51746  0.00799           0.51867  0.00823  0.52081   
        majority      0.50428  0.00669           0.50000  0.00000  0.50000   
        persistence   0.49255  0.01225           0.49244  0.01212  0.50000   
4       baseline_rf   0.50672  0.01155           0.51016  0.01148  0.51580   
        hierarchy_rf  0.50880  0.01167           0.51114  0.01256  0.51432   
        majority      0.50537  0.01271           0.50000  0.00000  0.50000   
        persistence   0.48632  0.01481           0.48598  0.01477  0.50000   

                               
                          std  
horizon model                  
1       baseline_rf   0.00790  
        hierarchy_rf  0.01259  
        majority      0.00000  
        persistence   0.00000  
4       baseline_rf   0.02561  
        hierarchy_rf  0.02189  
        majority      0.00000  
        persistence   0.00000  
```
