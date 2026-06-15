log_prior <- function(beta) {
  d <- length(beta)
  const <- -0.5 * d * log(2 * pi * PRIOR_VAR)
  quad  <- -0.5 * sum(beta^2) / PRIOR_VAR
  const + quad
}

log1pexp <- function(x) {
  ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))
}

log_likelihood <- function(beta, X, y) {
  z <- drop(X %*% beta)
  sum(y * z) - sum(log1pexp(z))
}

log_posterior <- function(beta, X, y) {
  log_prior(beta) + log_likelihood(beta, X, y)
}
