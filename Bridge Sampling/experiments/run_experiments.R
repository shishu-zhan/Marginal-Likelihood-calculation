suppressPackageStartupMessages({
  library(MASS)
  library(mvtnorm)
})

PROJECT_ROOT <- normalizePath(file.path("..", ".."), winslash = "/")
source("../src/config.R")
source("../src/model.R")
source("../src/data_loader.R")
source("../src/bridge.R")
source("../src/io_utils.R")

run_single <- function(dataset, X, y, ndim, seed, n_proposal) {
  set.seed(seed)
  posterior_samples <- load_posterior_samples(dataset)

  t0 <- proc.time()[3]
  result <- bridge_sampling(
    posterior_samples = posterior_samples,
    X = X,
    y = y,
    n_proposal = n_proposal
  )
  runtime <- proc.time()[3] - t0

  n_eff <- DATASETS[[dataset]]$n_post

  list(
    logz       = result$logz,
    logzerr    = result$logzerr,
    H          = NA_real_,
    runtime    = runtime,
    ncall      = n_eff + n_proposal,
    n_iter     = result$iterations,
    eff_nlive  = NA_integer_,
    nlive_used = NA_integer_,
    seed       = seed,
    converged  = as.integer(result$converged)
  )
}

run_experiments <- function(datasets, phase = "test") {
  phase_info <- if (phase == "test") {
    list(runs = TEST_RUNS, phase_dir = "test_runs")
  } else {
    list(runs = FINAL_RUNS, phase_dir = "final_runs")
  }

  cat(sprintf("=== Phase: %s (%d runs per dataset) ===\n", phase, phase_info$runs))
  cat(sprintf("Datasets: %s\n\n", paste(datasets, collapse = ", ")))

  for (dataset in datasets) {
    dat <- load_dataset(dataset)
    ndim <- dat$info$ndim
    cat(sprintf("--- %s | X: %s | n_post: %d ---\n",
                dat$info$label,
                paste(dim(dat$X), collapse = " x "),
                dat$info$n_post))

    all_metrics <- list()

    for (run_id in seq_len(phase_info$runs)) {
      seed <- BASE_SEED + (run_id - 1L) * 100L
      metrics <- run_single(dataset, dat$X, dat$y, ndim, seed, N_PROPOSAL)
      all_metrics[[run_id]] <- metrics
      save_run(dataset, run_id - 1L, metrics, phase_dir = phase_info$phase_dir)
      cat(sprintf("  run %d/%d | logZ = %.4f | iter = %d | %.1fs\n",
                  run_id, phase_info$runs, metrics$logz,
                  metrics$n_iter, metrics$runtime))
    }

    save_summary(dataset, all_metrics, phase_dir = phase_info$phase_dir)

    logz_vals <- sapply(all_metrics, `[[`, "logz")
    runtime_vals <- sapply(all_metrics, `[[`, "runtime")
    cat(sprintf("  >> logZ = %.4f +/- %.4f | runtime = %.1f +/- %.1f s\n\n",
                mean(logz_vals), sd(logz_vals),
                mean(runtime_vals), sd(runtime_vals)))
  }

  cat("=== All experiments completed ===\n")
}

args <- commandArgs(trailingOnly = TRUE)

parse_arg <- function(args, name, default) {
  idx <- grep(paste0("^--", name, "="), args)
  if (length(idx) > 0) return(sub(paste0("^--", name, "="), "", args[idx[1]]))
  idx <- grep(paste0("^--", name, "$"), args)
  if (length(idx) > 0 && (idx[1] + 1) <= length(args) &&
      !grepl("^--", args[idx[1] + 1])) return(args[idx[1] + 1])
  default
}

phase <- parse_arg(args, "phase", "test")
dataset_arg <- parse_arg(args, "dataset", "all")

datasets <- if (dataset_arg == "all") names(DATASETS) else dataset_arg
run_experiments(datasets, phase = phase)
