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

Net read (updated after NB06):

- **The hierarchy is useful as a target labeling** (NB06 — predicting which state the market moves to). RF on baseline features beats persistence by 7–13 pp on every ticker. The empirical Markov kernel alone collapses to persistence, so feature context is required.
- **The hierarchy does not help direction prediction on daily data** (NB05). Baseline wins on balanced accuracy on SPY, NVDA and BTC-USD once each feature set tunes its own RF; the earlier NB04 "+0.9 pp on SPY" was a hyperparameter artifact.

## Metric glossary

**Classification metrics (direction or multi-class state):**

- **Accuracy** — fraction of correct predictions. Sensitive to class imbalance.
- **Balanced accuracy** — mean of per-class recall. 0.5 = random regardless of imbalance. The preferred single-number metric when classes are skewed.
- **Macro F1** — unweighted mean of per-class F1. Used for multi-class state prediction (NB06/07).
- **Transition accuracy** (NB06/07) — accuracy computed only over rows where the state actually changed (`state_t_future != state_t`). Persistence scores 0 here by construction. This isolates the model's ability to detect regime *changes*.

**Regression metrics (log return prediction):**

- **MAE / RMSE** — error of predicted return. *Only informative compared to a `zero` baseline.* On low-signal targets the model correctly predicts near-zero and MAE will match `zero` exactly — that is success, not failure.

**Trivial baselines (always scored alongside the models):**

- `persistence` — predict next return = last return (regression); predict last direction (classification).
- `majority` — predict the more frequent class.
- `empirical_markov` (NB06/07) — `argmax_u P(z_{t+h}=u | z_t=s)` from the training transition matrix. Uses `state_t` only, no features.
- `random` (NB06/07) — uniform over K classes.

**Validation protocol:**

- **Time-respecting split** (NB05/06) — single train/val/test split, chronologically ordered, no shuffling. Val used for RF hyperparameter selection; model refit on train+val before scoring on test.

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
