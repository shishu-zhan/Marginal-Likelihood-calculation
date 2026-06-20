suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(showtext)
  library(sysfonts)
})

font_add("Times New Roman",
  regular    = "C:/WINDOWS/Fonts/times.ttf",
  italic     = "C:/WINDOWS/Fonts/timesi.ttf",
  bold       = "C:/WINDOWS/Fonts/timesbd.ttf",
  bolditalic = "C:/WINDOWS/Fonts/timesbi.ttf")
showtext_auto()

RESULTS_DIR <- normalizePath(file.path("..", "results"), winslash = "/")
FONT <- "Times New Roman"

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

caterpillar_data <- all_runs %>%
  group_by(Dataset) %>%
  mutate(
    mean_logZ = mean(logz),
    dev       = logz - mean_logZ,
    rank      = order(order(logz)),
    above     = dev >= 0
  ) %>%
  ungroup()

cat_colors <- c("TRUE" = "#E63946", "FALSE" = "#457B9D")

p_cat <- ggplot(caterpillar_data, aes(x = rank, y = logz)) +
  geom_hline(aes(yintercept = mean_logZ), linewidth = 0.8, linetype = "dashed",
             color = "grey40", alpha = 0.7) +
  geom_linerange(aes(ymin = logz - logzerr, ymax = logz + logzerr, color = above),
                 linewidth = 0.6, alpha = 0.5) +
  geom_point(aes(color = above), size = 2.3, alpha = 0.85) +
  scale_color_manual(values = cat_colors) +
  facet_wrap(~ Dataset, scales = "free", ncol = 3) +
  labs(
    x = "Run (sorted by estimate)",
    y = expression(log(italic(Z))),
    title = "Bridge Sampling: Caterpillar Plot of 30 Independent Runs"
  ) +
  theme_minimal(base_family = FONT) +
  theme(
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(linetype = "dotted", color = "grey75", linewidth = 0.3),
    panel.background   = element_rect(fill = "white", color = NA),
    plot.background    = element_rect(fill = "white", color = NA),
    axis.line          = element_line(linewidth = 0.4, color = "black"),
    axis.text          = element_text(size = 10, color = "black"),
    axis.text.x        = element_blank(),
    axis.ticks.x       = element_blank(),
    axis.title         = element_text(size = 11),
    plot.title         = element_text(size = 13, face = "bold", hjust = 0.5,
                                      margin = margin(b = 10)),
    strip.text         = element_text(size = 11, face = "bold", family = FONT),
    strip.background   = element_rect(fill = "grey95", color = "grey80", linewidth = 0.5),
    legend.position    = "none",
    panel.spacing      = unit(1, "lines")
  ) +
  geom_text(
    data = caterpillar_data %>% group_by(Dataset) %>% summarise(
      mean_logZ = unique(mean_logZ), sd_logZ = unique(sd(.data$logz)), .groups = "drop"),
    aes(x = -Inf, y = -Inf, hjust = -0.05, vjust = -1.5,
        label = sprintf("mean +/- sd: %.4f +/- %.4f", mean_logZ, sd_logZ)),
    size = 3.2, color = "grey30", family = FONT
  )

cat("Writing Caterpillar Plot ...\n")

ggsave(file.path(RESULTS_DIR, "Bridge_Fig3_caterpillar.pdf"),
       p_cat, width = 11, height = 4.5, dpi = 300, device = "pdf")
ggsave(file.path(RESULTS_DIR, "Bridge_Fig3_caterpillar.png"),
       p_cat, width = 11, height = 4.5, dpi = 300, device = "png")

cat("[OK] Bridge_Fig3_caterpillar (PDF + PNG)\n")
