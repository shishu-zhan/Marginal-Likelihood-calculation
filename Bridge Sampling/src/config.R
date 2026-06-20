if (!exists("PROJECT_ROOT")) {
  PROJECT_ROOT <- normalizePath(file.path("..", ".."), winslash = "/")
}

DATA_DIR      <- file.path(PROJECT_ROOT, "data preprocessing")
POSTERIOR_DIR <- file.path(PROJECT_ROOT, "bayes logistic regression")
HMC_RUNS_DIR  <- file.path(POSTERIOR_DIR, "hmc_runs")
RESULTS_DIR   <- file.path(PROJECT_ROOT, "Bridge Sampling", "results")

PRIOR_VAR <- 10.0
PRIOR_SD  <- sqrt(PRIOR_VAR)

DATASETS <- list(
  pima = list(
    file      = "pima_preprocessed.csv",
    posterior = "pima_posterior_samples.csv",
    ndim      = 9L,
    label     = "Pima (d = 9)",
    n_post    = 10000L
  ),
  creditcard = list(
    file      = "creditcard_preprocessed.csv",
    posterior = "creditcard_posterior_samples.csv",
    ndim      = 29L,
    label     = "CreditCard (d = 29)",
    n_post    = 10000L,
    drop_col  = "Amount"
  ),
  tcga = list(
    file      = "tcga_preprocessed.csv",
    posterior = "tcga_posterior_samples.csv",
    ndim      = 61L,
    label     = "TCGA (d = 61)",
    n_post    = 15000L
  )
)

TEST_RUNS  <- 5L
FINAL_RUNS <- 30L
BASE_SEED  <- 42L

N_PROPOSAL    <- 10000L
BS_TOL        <- 1e-10
BS_MAX_ITER   <- 10000L
SHRINK_LAMBDA <- 0.05
