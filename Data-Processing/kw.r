library(tidyverse)

# ===== 1. Load all Dictator OLMO files =====
file_paths <- list.files("Dictator/", pattern = "*.csv", full.names = TRUE)

# ===== Helper: safe numeric conversion =====
safe_num <- function(x) suppressWarnings(as.numeric(x))

# ===== Helper: safe column accessor =====
safe_col <- function(df, col) {
  if (col %in% names(df)) df[[col]] else NA
}

# ===== 2. Summarize each file by subgroup =====
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

  # handle total_a / total_A naming
  if ("total_a" %in% names(dat)) dat$total_A <- dat$total_a
  if ("total_b" %in% names(dat)) dat$total_B <- dat$total_b

  # convert numeric columns safely
  numeric_cols <- c("choice_prob","given","kept","generosity_streak","total_A","total_B")
  for (col in numeric_cols) {
    if (col %in% names(dat)) dat[[col]] <- safe_num(dat[[col]])
  }

  dat %>%
    group_by(gamified_label, variant) %>%
    summarise(
      mean_choice_prob   = mean(pick(choice_prob)[[1]], na.rm = TRUE),
      sd_choice_prob     = sd(pick(choice_prob)[[1]], na.rm = TRUE),

      mean_given         = mean(pick(given)[[1]], na.rm = TRUE),
      sd_given           = sd(pick(given)[[1]], na.rm = TRUE),

      mean_kept          = mean(pick(kept)[[1]], na.rm = TRUE),
      sd_kept            = sd(pick(kept)[[1]], na.rm = TRUE),

      mean_generosity_streak = mean(pick(generosity_streak)[[1]], na.rm = TRUE),
      sd_generosity_streak   = sd(pick(generosity_streak)[[1]], na.rm = TRUE),

      mean_total_a       = mean(pick(total_A)[[1]], na.rm = TRUE),
      sd_total_a         = sd(pick(total_A)[[1]], na.rm = TRUE),

      mean_total_b       = mean(pick(total_B)[[1]], na.rm = TRUE),
      sd_total_b         = sd(pick(total_B)[[1]], na.rm = TRUE),

      .groups = "drop"
    ) %>%
    mutate(file = basename(path))
}

# ===== 3. Apply to all files =====
summary_results <- map_dfr(file_paths, summarize_file)

# ===== 4. Save results =====
if (!dir.exists("results")) dir.create("results")

write_csv(summary_results, "results/dictator_means_sd_by_group.csv")
print("Saved results to results/dictator_means_sd_by_group.csv")

# ===== 5. View in RStudio =====
View(summary_results)