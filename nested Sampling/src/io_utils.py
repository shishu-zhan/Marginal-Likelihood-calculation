import os
import csv
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

METRIC_KEYS = ['logz', 'logzerr', 'logz_resample_std', 'H', 'runtime', 'ncall', 'n_iter', 'eff_nlive', 'nlive_used', 'seed', 'converged']


def save_run(method, dataset, run_id, metrics, phase='test_runs'):
    out_dir = os.path.join(RESULTS_DIR, method, dataset, phase)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'run_{run_id:03d}.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(METRIC_KEYS)
        writer.writerow([metrics.get(k, np.nan) for k in METRIC_KEYS])


def save_summary(method, dataset, all_metrics, phase='test_runs'):
    out_dir = os.path.join(RESULTS_DIR, method, dataset, phase)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'summary.csv')

    keys = METRIC_KEYS
    rows = [['metric', 'mean', 'std', 'min', 'max']]
    for k in keys:
        vals = [m[k] for m in all_metrics if k in m and not np.isnan(m[k])]
        if vals:
            rows.append([k, np.mean(vals), np.std(vals, ddof=1), np.min(vals), np.max(vals)])
        else:
            rows.append([k, np.nan, np.nan, np.nan, np.nan])

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return out_path


def load_runs(method, dataset, phase='test_runs'):
    out_dir = os.path.join(RESULTS_DIR, method, dataset, phase)
    all_metrics = []
    for fname in sorted(os.listdir(out_dir)):
        if fname.startswith('run_') and fname.endswith('.csv'):
            path = os.path.join(out_dir, fname)
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metrics = {k: float(v) for k, v in row.items()}
                    all_metrics.append(metrics)
    return all_metrics
