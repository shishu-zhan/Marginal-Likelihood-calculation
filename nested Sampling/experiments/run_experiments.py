import sys
import os
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_dataset, get_param_names
from src.dynesty_runner import run_dynesty
from src.ultranest_runner import run_ultranest
from src.io_utils import save_run, save_summary
from experiments.config import (
    DATASET_LIST, METHOD_LIST, METHOD_CONFIG,
    TEST_RUNS, FINAL_RUNS, TOTAL_PHASES
)


def run_single(method, dataset, X, y, ndim, param_names, seed):
    cfg = METHOD_CONFIG[method][dataset]
    if method == 'dynesty':
        metrics = run_dynesty(
            X, y, ndim,
            nlive=cfg['nlive'],
            sample=cfg['sample'],
            walks=cfg['walks'],
            dlogz=cfg['dlogz'],
            maxcall=cfg['maxcall'],
            seed=seed,
        )
    elif method == 'ultranest':
        metrics = run_ultranest(
            X, y, ndim, param_names,
            min_live=cfg['min_live'],
            min_ess=cfg['min_ess'],
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return metrics


def run_experiments(datasets, methods, phase='test', seed=42):
    phase_info = TOTAL_PHASES[phase]
    n_runs = phase_info['runs']
    phase_dir = phase_info['phase_dir']

    print(f"=== Phase: {phase} ({n_runs} runs per config) ===")
    print(f"Datasets: {datasets}")
    print(f"Methods: {methods}")
    print()

    total_tasks = len(datasets) * len(methods) * n_runs
    pbar = tqdm(total=total_tasks, desc="Total progress")

    for dataset in datasets:
        X, y, info = load_dataset(dataset)
        ndim = info['ndim']
        param_names = get_param_names(dataset)

        print(f"\n--- {info['label']} | X: {X.shape} | y: {y.shape} ---")

        for method in methods:
            all_metrics = []
            print(f"  [{method.upper()}] Running {n_runs} iterations...")

            for run_id in range(n_runs):
                run_seed = seed + run_id * 100
                metrics = run_single(method, dataset, X, y, ndim, param_names, run_seed)
                all_metrics.append(metrics)
                save_run(method, dataset, run_id, metrics, phase=phase_dir)
                pbar.set_postfix_str(
                    f"{dataset} {method} run {run_id+1}/{n_runs} "
                    f"logZ={metrics['logz']:.2f}"
                )
                pbar.update(1)

            summary_path = save_summary(method, dataset, all_metrics, phase=phase_dir)
            mean_logz = sum(m['logz'] for m in all_metrics) / len(all_metrics)
            std_logz = (sum((m['logz'] - mean_logz)**2 for m in all_metrics) / (len(all_metrics) - 1))**0.5 if len(all_metrics) > 1 else 0
            mean_runtime = sum(m['runtime'] for m in all_metrics) / len(all_metrics)
            print(f"    logZ = {mean_logz:.4f} +/- {std_logz:.4f} | runtime = {mean_runtime:.1f}s")
            print(f"    Summary saved: {summary_path}")

    pbar.close()
    print("\n=== All experiments completed ===")


def main():
    parser = argparse.ArgumentParser(description='Nested Sampling Marginal Likelihood Experiments')
    parser.add_argument('--method', type=str, default='both',
                        choices=['dynesty', 'ultranest', 'both'],
                        help='Nested sampling method (default: both)')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['pima', 'creditcard', 'tcga', 'all'],
                        help='Dataset to run (default: all)')
    parser.add_argument('--runs', type=int, default=None,
                        help='Number of runs (default: 5 for test, 30 for final)')
    parser.add_argument('--phase', type=str, default='test',
                        choices=['test', 'final'],
                        help='Experiment phase (default: test)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    if args.runs is not None:
        TOTAL_PHASES[args.phase]['runs'] = args.runs

    datasets = DATASET_LIST if args.dataset == 'all' else [args.dataset]
    methods = METHOD_LIST if args.method == 'both' else [args.method]

    run_experiments(datasets, methods, phase=args.phase, seed=args.seed)


if __name__ == '__main__':
    main()
