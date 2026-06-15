load_dataset <- function(name) {
  info <- DATASETS[[name]]
  path <- file.path(DATA_DIR, info$file)
  df <- read.csv(path)
  y <- df$y
  X <- as.matrix(df[, setdiff(names(df), "y")])

  if (!is.null(info$drop_col)) {
    drop_idx <- which(colnames(X) == info$drop_col)
    if (length(drop_idx) > 0) X <- X[, -drop_idx, drop = FALSE]
  }

  list(X = X, y = y, info = info)
}

load_posterior_samples <- function(name) {
  info <- DATASETS[[name]]
  path <- file.path(POSTERIOR_DIR, info$posterior)
  as.matrix(read.csv(path))
}

load_ground_truth <- function(method, dataset, phase_dir) {
  path <- file.path(PROJECT_ROOT, "nested Sampling", "results",
                    method, dataset, phase_dir, "summary.csv")
  if (!file.exists(path)) return(NULL)
  df <- read.csv(path, row.names = 1)
  as.list(structure(df$mean, names = df$metric))
}
