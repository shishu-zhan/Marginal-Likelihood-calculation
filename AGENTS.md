# AGENTS.md

## Project overview

Research project comparing Nested Sampling (dynesty, UltraNest) vs MCMC-based methods for marginal likelihood estimation in Bayesian logistic regression across three datasets of increasing dimensionality (Pima=9d, CreditCard=30d, TCGA=61d).

Only the `nested Sampling/` directory contains active code. `AIS-HMC/` and `Bridge Sampling/` are placeholders (`.gitkeep` only).

## Commands

All experiments run from the **`nested Sampling/`** directory:

```bash
# Quick test (5 runs per config, ~5-10 min)
python experiments/run_experiments.py --phase test --method dynesty --dataset pima

# Full experiment (30 runs per config)
python experiments/run_experiments.py --phase final

# Single method, single dataset
python experiments/run_experiments.py --method dynesty --dataset creditcard --runs 3
```

**Always** run from inside `nested Sampling/`. The code resolves paths relative to `src/` by walking up two directory levels, so running from any other CWD will fail to find data files.

## Architecture

```
nested Sampling/src/
  model.py          — log_likelihood, log_prior, log_posterior, prior_transform (Gaussian σ²=10)
  data_loader.py    — loads CSVs from ../data preprocessing/; defines DATASETS dict
  dynesty_runner.py — wraps dynesty.NestedSampler, returns metrics dict
  ultranest_runner.py — wraps ultranest.ReactiveNestedSampler
  io_utils.py       — save_run(), save_summary(), load_runs() → results/{method}/{dataset}/{phase}/

nested Sampling/experiments/
  config.py         — hyperparameters per method×dataset, TEST_RUNS=5, FINAL_RUNS=30
  run_experiments.py — CLI entrypoint, orchestrates runs

nested Sampling/results/  — output: {method}/{dataset}/{test_runs,final_runs}/run_*.csv + summary.csv
```

## Key conventions

- **Prior**: Gaussian N(0, 10·I), NOT standard N(0,1).
- **Data files** live in `data preprocessing/` at the repo root. CSVs have column `y` as label, remaining columns as features X. The data loader strips column 0 (the first feature column) as X.
- **Seeds are deterministic**: base seed + run_id*100. Reproducibility depends on `numpy.random.seed()`, not a per-library RNG.
- **Results are CSV-based**, not a database. Single runs saved as `run_NNN.csv`; aggregated stats in `summary.csv`.
- **No test suite**. The file `experiments/_test_dynesty.py` is a manual smoke test with hardcoded Windows paths — it will fail on Linux and should not be relied upon.
- **Notebooks** in `bayes logistic regression/` use R kernel for posterior sampling and have precomputed posterior samples as CSV outputs for later use.
- **dynesty** is listed as a required dependency. **UltraNest** has a try/except ImportError guard at call time.
