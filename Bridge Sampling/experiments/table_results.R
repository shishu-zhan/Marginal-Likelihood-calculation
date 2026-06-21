suppressPackageStartupMessages(library(dplyr))

RESULTS_DIR <- normalizePath(file.path("..", "results"), winslash = "/")

datasets <- c("pima", "creditcard", "tcga")
labels   <- c(pima = "Pima (d = 9)", creditcard = "CreditCard (d = 29)", tcga = "TCGA (d = 61)")

load_runs <- function(dataset) {
  path <- file.path(RESULTS_DIR, "bridge_sampling", dataset, "final_runs")
  files <- list.files(path, pattern = "^run_\\d+\\.csv$", full.names = TRUE)
  do.call(rbind, lapply(seq_along(files), function(i) {
    d <- read.csv(files[i])
    d$Run_index <- i
    d$Dataset <- labels[dataset]
    d
  }))
}

all_runs <- do.call(rbind, lapply(datasets, load_runs))

cat("============================================================\n")
cat("   Updated Bridge Sampling Results (30 independent HMC runs)\n")
cat("============================================================\n\n")

for (ds in names(labels)) {
  sub <- all_runs[all_runs$Dataset == labels[ds], ]
  n <- nrow(sub)
  
  cat(sprintf("--- %s ---\n", labels[ds]))
  cat(sprintf("  logZ:                %.4f +/- %.4f\n", mean(sub$logz), sd(sub$logz)))
  cat(sprintf("  Total runtime (HMC+BS):\n"))
  cat(sprintf("    mean: %.1f +/- %.1f s (incl. HMC)\n", mean(sub$runtime), sd(sub$runtime)))
  
  rt_vals <- sub$runtime
  if (ds == "tcga") {
    rt_ok <- rt_vals[rt_vals < 1000]
    cat(sprintf("    median: %.1f s (excl. %d outlier run(s))\n", median(rt_ok), sum(rt_vals >= 1000)))
    cat(sprintf("    HMC sampling:     median %.1f s\n",
                median(sub$hmc_runtime[sub$hmc_runtime < 1000])))
  } else {
    cat(sprintf("    median: %.1f s\n", median(rt_vals)))
    cat(sprintf("    HMC: %.1f +/- %.1f s | BS: %.1f +/- %.1f s\n",
                mean(sub$hmc_runtime), sd(sub$hmc_runtime),
                mean(sub$bs_runtime), sd(sub$bs_runtime)))
  }
  cat(sprintf("  Function evaluations:\n"))
  cat(sprintf("    HMC grad calls:     %.0f\n", mean(sub$hmc_ncall)))
  cat(sprintf("    Bridge (post+prop): %.0f\n", mean(sub$ncall - sub$hmc_ncall)))
  cat(sprintf("    Total:              %.0f\n", mean(sub$ncall)))
  cat(sprintf("  BS iterations:       %.1f\n", mean(sub$n_iter)))
  cat(sprintf("  Converged:           %d/30\n", sum(sub$converged)))
  cat("\n")
}

cat("--- Comparison with Nested Sampling (dynesty) ---\n\n")
cat(sprintf("%-25s %-25s %-25s\n", "Metric", "Bridge Sampling", "Nested Sampling"))
cat(sprintf("%-25s %-25s %-25s\n", paste0(rep("-", 24), collapse=""),
            paste0(rep("-", 24), collapse=""),
            paste0(rep("-", 24), collapse="")))

ns_results <- list(
  pima       = list(logZ = -387.49, sd = 0.28, rt = 27.5),
  creditcard = list(logZ = -96.95,  sd = 0.68, rt = 147.6),
  tcga       = list(logZ = -57.57,  sd = 2.82, rt = 126.3)
)

for (ds in names(labels)) {
  bs_sub <- all_runs[all_runs$Dataset == labels[ds], ]
  ns <- ns_results[[ds]]
  
  cat(sprintf("\n%s:\n", labels[ds]))
  cat(sprintf("  logZ         BS: %.2f +/- %.4f  |  NS: %.2f +/- %.2f\n",
              mean(bs_sub$logz), sd(bs_sub$logz), ns$logZ, ns$sd))
  cat(sprintf("  Runtime      BS: %.1f s (HMC+BS)  |  NS: %.1f s\n",
              mean(bs_sub$runtime), ns$rt))
  cat(sprintf("  BS runtime breakdown: HMC %.1f s + Bridge %.1f s\n",
              mean(bs_sub$hmc_runtime), mean(bs_sub$bs_runtime)))
}

cat("\n============================================================\n")
cat("NOTE: BS logZ std now reflects full variability across\n")
cat("30 independent HMC + Bridge runs (not conditional variance).\n")
cat("BS ncall and runtime include HMC gradient evaluations.\n")
cat("============================================================\n")
