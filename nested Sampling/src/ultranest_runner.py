import time
import numpy as np
from .model import make_loglike, make_prior_transform


def run_ultranest(X, y, ndim, param_names, min_live=400, min_ess=400, seed=None):
    loglike = make_loglike(X, y)
    prior_transform = make_prior_transform(ndim)

    if seed is not None:
        np.random.seed(seed)

    try:
        import ultranest
        sampler = ultranest.ReactiveNestedSampler(
            param_names, loglike,
            transform=prior_transform,
        )
    except ImportError:
        raise RuntimeError("ultranest not installed. Install with: pip install ultranest")

    t0 = time.perf_counter()
    result = sampler.run(
        min_num_live_points=min_live,
        min_ess=min_ess,
        max_num_improvement_loops=999999,
    )
    runtime = time.perf_counter() - t0

    logz = float(result['logz'])
    logzerr = float(result['logzerr'])
    h_info = float(result.get('H', result.get('information', np.nan)))
    ncall = int(result.get('ncall', 0))
    ess = int(result.get('ess', 0))

    eff_nlive = int(ncall / (h_info * ndim)) if h_info > 0 and h_info != np.nan else 0

    return {
        'logz': logz,
        'logzerr': logzerr,
        'logz_resample_std': np.nan,
        'H': h_info,
        'runtime': runtime,
        'ncall': ncall,
        'ess': ess,
        'eff_nlive': eff_nlive,
        'nlive_used': min_live,
    }
