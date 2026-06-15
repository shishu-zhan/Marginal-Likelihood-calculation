METRIC_KEYS <- c("logz", "logzerr", "H", "runtime", "ncall", "n_iter",
                 "eff_nlive", "nlive_used", "seed", "converged")

save_run <- function(dataset, run_id, metrics, phase_dir) {
  out_dir <- file.path(RESULTS_DIR, "bridge_sampling", dataset, phase_dir)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  out_path <- file.path(out_dir, sprintf("run_%03d.csv", run_id))

  row <- vapply(METRIC_KEYS, function(k) {
    if (is.null(metrics[[k]])) NA_real_ else as.numeric(metrics[[k]])
  }, numeric(1))

  df <- as.data.frame(t(row))
  colnames(df) <- METRIC_KEYS
  write.csv(df, out_path, row.names = FALSE)
  invisible(out_path)
}

save_summary <- function(dataset, all_metrics, phase_dir) {
  out_dir <- file.path(RESULTS_DIR, "bridge_sampling", dataset, phase_dir)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  out_path <- file.path(out_dir, "summary.csv")

  rows <- data.frame(
    metric = METRIC_KEYS,
    mean   = NA_real_,
    std    = NA_real_,
    min    = NA_real_,
    max    = NA_real_,
    stringsAsFactors = FALSE
  )

  for (k in METRIC_KEYS) {
    vals <- na.omit(vapply(all_metrics, function(m) {
      if (is.null(m[[k]])) NA_real_ else as.numeric(m[[k]])
    }, numeric(1)))
    if (length(vals) > 0) {
      rows[rows$metric == k, c("mean", "std", "min", "max")] <-
        c(mean(vals), if (length(vals) > 1) sd(vals) else 0, min(vals), max(vals))
    }
  }

  write.csv(rows, out_path, row.names = FALSE)
  invisible(out_path)
}

load_bridge_runs <- function(dataset, phase_dir) {
  dir_path <- file.path(RESULTS_DIR, "bridge_sampling", dataset, phase_dir)
  if (!dir.exists(dir_path)) return(list())
  files <- list.files(dir_path, pattern = "^run_\\d+\\.csv$", full.names = TRUE)
  lapply(files, function(f) {
    as.list(read.csv(f))
  })
}
