import numpy as np
from scipy.stats import norm

PRIOR_VAR = 10.0
PRIOR_SD = np.sqrt(PRIOR_VAR)


def log_prior_density(beta):
    d = len(beta)
    const = -0.5 * d * np.log(2 * np.pi * PRIOR_VAR)
    quad = -0.5 * np.sum(beta ** 2) / PRIOR_VAR
    return const + quad


def log_likelihood(beta, X, y):
    z = X @ beta
    return np.dot(y, z) - np.sum(np.logaddexp(0, z))


def log_posterior(beta, X, y):
    return log_likelihood(beta, X, y) + log_prior_density(beta)


def prior_transform(u):
    return PRIOR_SD * norm.ppf(u)


def make_loglike(X, y):
    def _loglike(beta):
        return log_likelihood(beta, X, y)
    return _loglike


def make_prior_transform(ndim):
    def _pt(u):
        return prior_transform(u)
    return _pt
