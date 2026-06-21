import time
import numpy as np
import dynesty
from .model import make_loglike, make_prior_transform


def run_dynesty(X, y, ndim, nlive=500, sample='rslice', walks=None,
                dlogz=0.5, maxcall=None, seed=None):
    loglike = make_loglike(X, y)
    ptform = make_prior_transform(ndim)

    if seed is not None:
        np.random.seed(seed)

    sampler_kwargs = dict(
        nlive=nlive, bound='multi', sample=sample,
    )
    if sample in ('rwalk', 'rstagger') and walks is not None:
        sampler_kwargs['walks'] = walks

    sampler = dynesty.NestedSampler(
        loglike, ptform, ndim,
        **sampler_kwargs,
    )

    t0 = time.perf_counter()
    sampler.run_nested(dlogz=dlogz, maxcall=maxcall, print_progress=False)
    runtime = time.perf_counter() - t0

    results = sampler.results

    logz = float(results.logz[-1])
    logzerr = float(results.logzerr[-1])
    h_info = float(results.information[-1])
    ncall = int(np.sum(results.ncall) if hasattr(results.ncall, '__iter__') else results.ncall)
    n_iter = len(results.samples)
    eff_nlive = int(ncall / (h_info * ndim)) if h_info > 0 else 0
    logz_resample_std = np.sqrt(h_info / nlive) if h_info > 0 else np.nan
    converged = 1 if (maxcall is None or ncall < maxcall) else 0

    return {
        'logz': logz,
        'logzerr': logzerr,
        'logz_resample_std': logz_resample_std,
        'H': h_info,
        'runtime': runtime,
        'ncall': ncall,
        'n_iter': n_iter,
        'eff_nlive': eff_nlive,
        'nlive_used': nlive,
        'seed': seed if seed is not None else 0,
        'converged': converged,
    }
