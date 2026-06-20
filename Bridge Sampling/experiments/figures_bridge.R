suppressPackageStartupMessages({
  library(ggplot2)
  library(  ggdist)
  library(  dplyr)
  library(showtext)
  library(sysfonts)
})

font_add("Times New Roman",
  regular    = "C:/WINDOWS/Fonts/times.ttf",
  italic     = "C:/WINDOWS/Fonts/timesi.ttf",
  bold       = "C:/WINDOWS/Fonts/timesbd.ttf",
  bolditalic = "C:/WINDOWS/Fonts/timesbi.ttf")
showtext_auto()

RESULTS_DIR <- normalizePath(
  file.path("..", "results"), winslash = "/"
)

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
all_runs$Dataset <- factor(all_runs$Dataset, levels = labels)
all_runs_rt <- all_runs
tcga_outliers <- all_runs_rt$Dataset == "TCGA (d = 61)" & all_runs_rt$runtime > 1000
all_runs_rt$runtime[tcga_outliers] <- NA
all_runs_rt$hmc_runtime[tcga_outliers] <- NA

summ <- all_runs %>%
  group_by(Dataset) %>%
  summarise(
    mean_logZ  = mean(logz),
    sd_logZ    = sd(logz),
    .groups = "drop"
  )

summ_rt <- all_runs_rt %>%
  group_by(Dataset) %>%
  summarise(
    median_runtime = median(runtime, na.rm = TRUE),
    mad_runtime    = mad(runtime, na.rm = TRUE),
    mean_hmc_rt    = mean(hmc_runtime, na.rm = TRUE),
    mean_bs_rt     = mean(bs_runtime, na.rm = TRUE),
    .groups = "drop"
  )

FONT <- "Times New Roman"

fill_colors <- c("Pima (d = 9)"          = "#fae9ae",
                 "CreditCard (d = 29)"   = "#b9dafa",
                 "TCGA (d = 61)"         = "#f1c8c8")

p1 <- ggplot(summ, aes(x = Dataset, y = mean_logZ, fill = Dataset)) +
  geom_col(width = 0.45, color = "#333333", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%.1f", mean_logZ)),
            vjust = -0.8, size = 3.5, fontface = "bold",
            family = FONT) +
  scale_fill_manual(values = fill_colors) +
  labs(x = NULL, y = expression(log(italic(Z))),
       title = "Marginal Likelihood via Bridge Sampling") +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.12))) +
  theme_minimal(base_family = FONT) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(linetype = "dotted", colour = "grey75", linewidth = 0.35),
    panel.background   = element_rect(fill = "white", colour = NA),
    plot.background    = element_rect(fill = "white", colour = NA),
    axis.line         = element_blank(),
    axis.text         = element_text(size = 11, colour = "black"),
    axis.title.y      = element_text(size = 12),
    plot.title        = element_text(size = 13, face = "bold", hjust = 0.5,
                                     margin = margin(b = 10)),
    legend.position   = "none"
  )

runtime_long <- all_runs_rt %>%
  select(Dataset, hmc_runtime, bs_runtime) %>%
  tidyr::pivot_longer(cols = c(hmc_runtime, bs_runtime),
                      names_to = "Phase", values_to = "Runtime") %>%
  mutate(Phase = recode(Phase,
                        hmc_runtime = "HMC Sampling",
                        bs_runtime  = "Bridge Iteration"))

ggplot2_colors <- c("HMC Sampling" = "#A8DADC", "Bridge Iteration" = "#E63946")

p2 <- ggplot(runtime_long, aes(x = Dataset, y = Runtime, fill = Phase)) +
  stat_summary(fun = median, geom = "bar", width = 0.55,
               position = position_stack(), color = "#333333", linewidth = 0.3) +
  scale_fill_manual(values = ggplot2_colors) +
  labs(x = NULL, y = "Runtime (seconds)",
       title = "Total Computational Cost of Bridge Sampling",
       fill = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
  theme_minimal(base_family = FONT) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(linetype = "dotted", colour = "grey75", linewidth = 0.35),
    panel.background   = element_rect(fill = "white", colour = NA),
    plot.background    = element_rect(fill = "white", colour = NA),
    axis.line         = element_line(linewidth = 0.6, colour = "black"),
    axis.text         = element_text(size = 11, colour = "black"),
    axis.title.y      = element_text(size = 12),
    plot.title        = element_text(size = 13, face = "bold", hjust = 0.5,
                                     margin = margin(b = 10)),
    legend.position   = "top",
    legend.text       = element_text(size = 10, family = FONT)
  )

stab_col <- c("Pima (d = 9)"        = "#94d8da",
              "CreditCard (d = 29)" = "#fbdd9c",
              "TCGA (d = 61)"       = "#fabb9e")

all_runs_stab <- all_runs %>%
  group_by(Dataset) %>%
  mutate(mean_logZ = mean(logz)) %>%
  ungroup()

p3 <- ggplot(all_runs_stab, aes(x = Run_index, y = logz, colour = Dataset)) +
  geom_hline(aes(yintercept = mean_logZ), linewidth = 1) +
  geom_segment(aes(xend = Run_index, y = mean_logZ, yend = logz),
               alpha = 0.6, linewidth = 0.4) +
  geom_point(size = 2.5, alpha = 0.9) +
  scale_colour_manual(values = stab_col) +
  facet_wrap(~ Dataset, scales = "free_y", ncol = 3) +
  labs(x = "Run index", y = expression(log(italic(Z))),
       title = "Bridge Sampling Stability Across 30 Independent Runs") +
  theme_minimal(base_family = FONT) +
  theme(
    panel.grid        = element_blank(),
    panel.background  = element_rect(fill = "white", colour = NA),
    plot.background   = element_rect(fill = "white", colour = NA),
    axis.line         = element_line(linewidth = 0.4, colour = "black"),
    axis.text         = element_text(size = 10, colour = "black"),
    axis.title        = element_text(size = 11),
    plot.title        = element_text(size = 13, face = "bold", hjust = 0.5,
                                     margin = margin(b = 10)),
    strip.text        = element_text(size = 11, face = "bold",
                                     family = FONT),
    strip.background  = element_blank(),
    legend.position   = "none"
  )

rain_col <- c("Pima (d = 9)"        = "#DBEDC5",
              "CreditCard (d = 29)" = "#F7C4C1",
              "TCGA (d = 61)"       = "#C3E2EC")

p4 <- ggplot(all_runs, aes(x = 1, y = logz, fill = Dataset, colour = Dataset)) +
  stat_halfeye(adjust = 0.5, width = 0.6, .width = 0.95, alpha = 0.8,
               show.legend = FALSE) +
  geom_dots(side = "left", alpha = 0.6, show.legend = FALSE) +
  scale_fill_manual(values = rain_col) +
  scale_colour_manual(values = rain_col) +
  facet_wrap(~ Dataset, scales = "free_y", ncol = 3) +
  labs(x = NULL, y = expression(log(italic(Z))),
       title = "Distribution of Bridge Sampling Estimates") +
  theme_minimal(base_family = FONT) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(linetype = "dotted", colour = "grey75", linewidth = 0.35),
    panel.background   = element_rect(fill = "white", colour = NA),
    plot.background    = element_rect(fill = "white", colour = NA),
    axis.line         = element_line(linewidth = 0.4, colour = "black"),
    axis.text.x       = element_blank(),
    axis.ticks.x      = element_blank(),
    axis.text.y       = element_text(size = 10, colour = "black"),
    axis.title.y      = element_text(size = 11),
    plot.title        = element_text(size = 13, face = "bold", hjust = 0.5,
                                     margin = margin(b = 10)),
    strip.text        = element_text(size = 11, face = "bold",
                                     family = FONT),
    strip.background  = element_blank(),
    panel.spacing     = unit(1.2, "lines"),
    legend.position   = "none"
  )

cat("Writing 4 figures ...\n")

ggsave(file.path(RESULTS_DIR, "Bridge_Fig1_logZ.pdf"),
       p1, width = 5, height = 4.5, dpi = 300, device = "pdf")
ggsave(file.path(RESULTS_DIR, "Bridge_Fig1_logZ.png"),
       p1, width = 5, height = 4.5, dpi = 300, device = "png")
cat("  [OK] Bridge_Fig1_logZ\n")

ggsave(file.path(RESULTS_DIR, "Bridge_Fig2_runtime.pdf"),
       p2, width = 5, height = 4.5, dpi = 300, device = "pdf")
ggsave(file.path(RESULTS_DIR, "Bridge_Fig2_runtime.png"),
       p2, width = 5, height = 4.5, dpi = 300, device = "png")
cat("  [OK] Bridge_Fig2_runtime\n")

ggsave(file.path(RESULTS_DIR, "Bridge_Fig3_stability.pdf"),
       p3, width = 10, height = 3.5, dpi = 300, device = "pdf")
ggsave(file.path(RESULTS_DIR, "Bridge_Fig3_stability.png"),
       p3, width = 10, height = 3.5, dpi = 300, device = "png")
cat("  [OK] Bridge_Fig3_stability\n")

ggsave(file.path(RESULTS_DIR, "Bridge_Fig4_raincloud.pdf"),
       p4, width = 10, height = 4, dpi = 300, device = "pdf")
ggsave(file.path(RESULTS_DIR, "Bridge_Fig4_raincloud.png"),
       p4, width = 10, height = 4, dpi = 300, device = "png")
cat("  [OK] Bridge_Fig4_raincloud\n")

cat("Done. All files in:", RESULTS_DIR, "\n")
