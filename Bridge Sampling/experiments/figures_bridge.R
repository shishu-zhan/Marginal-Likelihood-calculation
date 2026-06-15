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

summ <- all_runs %>%
  group_by(Dataset) %>%
  summarise(
    mean_logZ  = mean(logz),
    sd_logZ    = sd(logz),
    mean_runtime = mean(runtime),
    sd_runtime = sd(runtime),
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

lollipop_col <- c("Pima (d = 9)"        = "#A8DADC",
                  "CreditCard (d = 29)" = "#457B9D",
                  "TCGA (d = 61)"       = "#1D3557")

p2 <- ggplot(summ, aes(x = Dataset, y = mean_runtime, colour = Dataset)) +
  geom_segment(aes(xend = Dataset, y = 1, yend = mean_runtime),
               linewidth = 1.5, alpha = 0.8) +
  geom_point(size = 5.5) +
  geom_point(size = 2.2, colour = "white") +
  geom_text(aes(label = sprintf("%.1fs", mean_runtime)),
            vjust = -1.2, size = 3.5, fontface = "bold",
            family = FONT) +
  scale_colour_manual(values = lollipop_col) +
  scale_y_log10(expand = expansion(mult = c(0.08, 0.25))) +
  labs(x = NULL, y = "Runtime (seconds, log scale)",
       title = "Computational Cost of Bridge Sampling") +
  theme_minimal(base_family = FONT) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(linetype = "dotted", colour = "grey75", linewidth = 0.35),
    panel.background   = element_rect(fill = "white", colour = NA),
    plot.background    = element_rect(fill = "white", colour = NA),
    axis.line         = element_line(linewidth = 0.6, colour = "black"),
    axis.text         = element_text(size = 11, colour = "black", face = "bold"),
    axis.title.y      = element_text(size = 12, face = "bold"),
    plot.title        = element_text(size = 13, face = "bold", hjust = 0.5,
                                     margin = margin(b = 10)),
    legend.position   = "none"
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
