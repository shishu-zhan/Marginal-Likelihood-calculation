import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
import time
import os

prior_var = 10.0

def log_prior(beta, d):
    const_term = -0.5 * d * np.log(2 * np.pi * prior_var)
    quad_term = -0.5 * np.sum(beta**2) / prior_var
    return const_term + quad_term

def log_likelihood(beta, X, y):
    z = X @ beta
    return np.dot(y, z) - np.sum(np.logaddexp(0, z))

def log_posterior(beta, X, y, d):
    return log_likelihood(beta, X, y) + log_prior(beta, d)

def grad_log_posterior(beta, X, y):
    z = X @ beta
    p = expit(z)
    grad_ll = X.T @ (y - p)
    grad_prior = -beta / prior_var
    return grad_ll + grad_prior

def hmc_sampler(X, y, num_samples, burn_in, epsilon, L, seed):
    np.random.seed(seed)
    N, d = X.shape
    res = minimize(
        lambda b: -log_posterior(b, X, y, d),
        np.zeros(d),
        jac=lambda b: -grad_log_posterior(b, X, y),
        method='L-BFGS-B'
    )
    q = res.x.copy()
    samples = np.zeros((num_samples, d))
    accepted = 0
    grad_calls = 0
    total_iter = num_samples + burn_in

    for i in range(total_iter):
        q_old = q.copy()
        p = np.random.normal(0, 1, size=d)
        p_old = p.copy()
        p = p + 0.5 * epsilon * grad_log_posterior(q, X, y)
        grad_calls += 1
        for l in range(L - 1):
            q = q + epsilon * p
            p = p + epsilon * grad_log_posterior(q, X, y)
            grad_calls += 1
        q = q + epsilon * p
        p = p + 0.5 * epsilon * grad_log_posterior(q, X, y)
        grad_calls += 1
        current_U = -log_posterior(q_old, X, y, d)
        current_K = 0.5 * np.sum(p_old**2)
        proposed_U = -log_posterior(q, X, y, d)
        proposed_K = 0.5 * np.sum(p**2)
        if np.log(np.random.rand()) < (current_U + current_K - proposed_U - proposed_K):
            if i >= burn_in:
                accepted += 1
        else:
            q = q_old
        if i >= burn_in:
            samples[i - burn_in] = q
    acc_rate = accepted / num_samples
    return samples, acc_rate, grad_calls

DATASETS = {
    'pima': {
        'file': 'pima_preprocessed_低维.csv',
        'num_samples': 10000, 'burn_in': 2000, 'epsilon': 0.10, 'L': 10,
        'col_slice': slice(1, None)
    },
    'creditcard': {
        'file': 'creditcard_preprocessed_中维.csv',
        'num_samples': 10000, 'burn_in': 3000, 'epsilon': 0.10, 'L': 12,
        'col_slice': slice(1, -1)
    },
    'tcga': {
        'file': 'tcga_preprocessed_高维.csv',
        'num_samples': 15000, 'burn_in': 5000, 'epsilon': 0.08, 'L': 12,
        'col_slice': slice(1, None)
    }
}

BASE_SEED = 42
N_RUNS = 30
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'hmc_runs')
os.makedirs(RESULTS_DIR, exist_ok=True)

all_meta = []

for ds_name, ds in DATASETS.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    print(f"{'='*60}")
    df = pd.read_csv(os.path.join(OUTPUT_DIR, ds['file']))
    y = df['y'].values
    X = df.iloc[:, ds['col_slice']].values
    N, d = X.shape
    print(f"  Dimensions: {N} samples x {d} features")
    ds_dir = os.path.join(RESULTS_DIR, ds_name)
    os.makedirs(ds_dir, exist_ok=True)

    for run_id in range(N_RUNS):
        seed = BASE_SEED + run_id * 100
        print(f"  Run {run_id+1}/{N_RUNS} (seed={seed})...", end=' ', flush=True)
        t0 = time.time()
        samples, acc_rate, grad_calls = hmc_sampler(
            X, y,
            num_samples=ds['num_samples'],
            burn_in=ds['burn_in'],
            epsilon=ds['epsilon'],
            L=ds['L'],
            seed=seed
        )
        runtime = time.time() - t0
        feature_names = df.columns[ds['col_slice']]
        samples_df = pd.DataFrame(samples, columns=feature_names)
        out_file = os.path.join(ds_dir, f'run_{run_id:03d}.csv')
        samples_df.to_csv(out_file, index=False)
        meta = {
            'dataset': ds_name,
            'run_id': run_id,
            'seed': seed,
            'runtime': round(runtime, 3),
            'acc_rate': round(acc_rate, 4),
            'grad_calls': grad_calls,
            'd': d,
            'n_samples': ds['num_samples'],
            'burn_in': ds['burn_in'],
            'epsilon': ds['epsilon'],
            'L': ds['L']
        }
        all_meta.append(meta)
        print(f"done ({runtime:.1f}s, acc={acc_rate:.2%}, grads={grad_calls})")

meta_df = pd.DataFrame(all_meta)
meta_df.to_csv(os.path.join(RESULTS_DIR, 'hmc_metadata.csv'), index=False)
print(f"\n{'='*60}")
print(f"All runs complete. Metadata saved to {os.path.join(RESULTS_DIR, 'hmc_metadata.csv')}")
print(f"{'='*60}")
