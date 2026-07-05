library(tidyverse)

# ===== 1. Load all IPD files =====
file_paths <- list.files("IPD/", pattern = "*.csv", full.names = TRUE)

# ===== Helper: safe numeric conversion =====
safe_num <- function(x) suppressWarnings(as.numeric(x))

# ===== Helper: compute 95% CI =====
compute_ci <- function(x) {
  x <- x[!is.na(x)]
  n <- length(x)
  if (n == 0) return(c(mean = NA, sd = NA, n = 0, ci_low = NA, ci_high = NA))
  m <- mean(x)
  s <- sd(x)
  se <- s / sqrt(n)
  ci_low <- m - 1.96 * se
  ci_high <- m + 1.96 * se
  c(mean = m, sd = s, n = n, ci_low = ci_low, ci_high = ci_high)
}

# ===== 2. Summarize each file =====
summarize_file <- function(path) {

  dat <- read_csv(path, show_col_types = FALSE)

  # --- normalize not_gamified to logical ---
  if ("not_gamified" %in% names(dat)) {
    dat <- dat %>%
      mutate(not_gamified = case_when(
        not_gamified %in% c(TRUE, "TRUE", "true", "1") ~ TRUE,
        not_gamified %in% c(FALSE, "FALSE", "false", "0") ~ FALSE,
        TRUE ~ FALSE
      ))
  } else {
    dat$not_gamified <- FALSE
  }

  dat <- dat %>%
    mutate(gamified_label = if_else(not_gamified, "not gamified", "gamified"))

  # ensure variant exists
  if (!"variant" %in% names(dat)) dat$variant <- "unknown"

  # convert numeric columns safely
  numeric_cols <- c("coop_prob","coop_streak","relative_payoff")
  for (col in numeric_cols) {
    if (col %in% names(dat)) dat[[col]] <- safe_num(dat[[col]])
  }

  # summarise by subgroup
  dat %>%
    group_by(gamified_label, variant) %>%
    summarise(
      coop_prob_stats       = list(compute_ci(coop_prob)),
      coop_streak_stats     = list(compute_ci(coop_streak)),
      relative_payoff_stats = list(compute_ci(relative_payoff)),
      .groups = "drop"
    ) %>%
    mutate(file = basename(path))
}

# ===== 3. Apply to all files =====
raw_results <- map_dfr(file_paths, summarize_file)

# ===== 4. Unnest into tidy columns =====
summary_results <- raw_results %>%
  unnest_wider(coop_prob_stats, names_sep = "_") %>%
  unnest_wider(coop_streak_stats, names_sep = "_") %>%
  unnest_wider(relative_payoff_stats, names_sep = "_")

# ===== 5. Save results =====
if (!dir.exists("results")) dir.create("results")

write_csv(summary_results, "results/ipd_means_sd_ci_by_group.csv")
print("Saved results to results/ipd_means_sd_ci_by_group.csv")

# ===== 6. View in RStudio =====
View(summary_results)