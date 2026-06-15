log_sum_exp <- function(x) {
  if (length(x) == 0) return(-Inf)
  m <- max(x)
  m + log(sum(exp(x - m)))
}

log_sum_exp_pair <- function(a, b) {
  m <- pmax(a, b)
  m + log(exp(a - m) + exp(b - m))
}

bridge_sampling <- function(posterior_samples, X, y, n_proposal = N_PROPOSAL,
                            max_iter = BS_MAX_ITER, tol = BS_TOL,
                            lambda = SHRINK_LAMBDA) {

  n1 <- nrow(posterior_samples)
  d  <- ncol(posterior_samples)
  s1 <- n1 / (n1 + n_proposal)
  s2 <- n_proposal / (n1 + n_proposal)

  mu_hat    <- colMeans(posterior_samples)
  S_raw     <- cov(posterior_samples)
  S_shrink  <- (1 - lambda) * S_raw + lambda * diag(diag(S_raw), nrow = d)

  proposal_samples <- MASS::mvrnorm(n_proposal, mu = mu_hat, Sigma = S_shrink)

  log_post_theta <- apply(posterior_samples, 1, function(th) {
    log_prior(th) + log_likelihood(th, X, y)
  })
  log_post_tilde <- apply(proposal_samples, 1, function(th) {
    log_prior(th) + log_likelihood(th, X, y)
  })

  log_q_theta <- mvtnorm::dmvnorm(posterior_samples, mean = mu_hat,
                                   sigma = S_shrink, log = TRUE)
  log_q_tilde <- mvtnorm::dmvnorm(proposal_samples, mean = mu_hat,
                                   sigma = S_shrink, log = TRUE)
  log_l_theta <- log_post_theta - log_q_theta
  log_l_tilde <- log_post_tilde - log_q_tilde
  log_r <- log_sum_exp(log_l_tilde) - log(n_proposal)
  log_r_trace <- numeric(max_iter)
  log_r_trace[1] <- log_r

  for (iter in seq_len(max_iter)) {
    a1 <- log(s1) + log_l_tilde
    b1 <- log(s2) + log_r
    log_denom_num <- log_sum_exp_pair(a1, b1)
    log_num_term  <- log_l_tilde - log_denom_num
    log_num       <- log_sum_exp(log_num_term) - log(n_proposal)

    a2 <- log(s1) + log_l_theta
    b2 <- log(s2) + log_r
    log_denom_den <- log_sum_exp_pair(a2, b2)
    log_den_term  <- -log_denom_den
    log_den       <- log_sum_exp(log_den_term) - log(n1)

    log_r_new <- log_num - log_den
    log_r_trace[iter + 1] <- log_r_new

    if (abs(log_r_new - log_r) < tol) {
      log_r <- log_r_new
      break
    }
    log_r <- log_r_new
  }

  log_ratios <- log_l_tilde - log_r
  log_mcse   <- sd(log_ratios) / sqrt(n_proposal)

  list(
    logz        = log_r,
    logzerr     = log_mcse,
    iterations  = iter,
    converged   = (iter < max_iter),
    mu_hat      = mu_hat,
    log_r_trace = log_r_trace[1:(iter + 1)]
  )
}
