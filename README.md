# Probability Measure Hierarchy AI Prototype

This package contains three complete Jupyter notebooks for formalizing and testing a hierarchy-of-measures idea on market-like time series.

## Files

1. `01_dynamic_hierarchical_random_measure.ipynb`
   - Builds a dynamic hierarchy of regions over greed/fear indices.
   - Converts hierarchy leaves into learnable features.
   - Trains baseline and hierarchy-augmented MLP classifiers.

2. `02_markov_kernel_hierarchy.ipynb`
   - Treats hierarchy states as a probabilistic transition system.
   - Estimates a Markov kernel between hierarchy states.
   - Trains a neural Markov kernel to predict next hierarchy state.

3. `03_neural_measure_valued_sequence_model.ipynb`
   - Builds a trainable sequence model.
   - Outputs a probability measure over latent hierarchy atoms.
   - Predicts future movement and reconstructs greed/fear indices.

## How to run

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Install uv (e.g. `brew install uv`), then from the repo root:

```bash
uv sync          # creates .venv and installs from pyproject.toml + uv.lock
uv run jupyter notebook
```

To add or change a dependency, edit `pyproject.toml` and re-run `uv sync` (the lockfile updates automatically). To run a one-off command inside the env without activating it, use `uv run <cmd>`; or `source .venv/bin/activate` if you prefer the classic flow.

Then open the notebooks in order.

## Data

The notebooks use synthetic data so they run immediately.

To use real data, replace the function:

```python
make_synthetic_market_data(...)
```

with a loader that returns a DataFrame containing at least:

```python
close
volume
```

The notebooks derive returns, volatility, greed index, fear index, and labels from those columns.

## Scientific caution

The notebooks are prototypes. They do not prove that price charts literally encode psychology.
They test whether hierarchy-derived variables add predictive or representational value under time-respecting validation.

