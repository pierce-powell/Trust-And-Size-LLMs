# ===== Libraries =====
suppressPackageStartupMessages({
  library(dplyr)
  library(vroom)      # FAST CSV loading
  library(stringr)
  library(purrr)
  library(furrr)      # parallel map
  library(FSA)        # Dunn's test
  library(boot)       # bootstrapping
  library(readr)
})
options(future.globals.maxSize = 10 * 1024^3)
future::plan(multisession, workers = 4)   # or fewer

# ===== 1. Define folders =====
dictator_dir <- "Dictator"
ipd_dir      <- "IPD"

# ===== 2. Parse filenames into family + size + experiment =====
parse_file_info <- function(path) {
  fname <- basename(path)
  tibble(
    file = path,
    model_family = case_when(
      str_detect(fname, regex("qwen",  ignore_case = TRUE)) ~ "Qwen",
      str_detect(fname, regex("gemma", ignore_case = TRUE)) ~ "GEMMA",
      str_detect(fname, regex("olmo",  ignore_case = TRUE)) ~ "OLMo",
      str_detect(fname, regex("deepseek",   ignore_case = TRUE)) ~ "DeepSeek",
      TRUE ~ "Other"
    ),
    size_category = case_when(
      # small
      str_detect(fname, regex("(^|[_-])(0?5B|05B|1B|1.3B)($|[_-])", ignore_case = TRUE)) ~ "small",
      # medium
      str_detect(fname, regex("(^|[_-])(4B|6.7B|7B|12B|13B|14|14B)($|[_-])", ignore_case = TRUE)) ~ "medium",
      # large
      str_detect(fname, regex("(^|[_-])(27B|32B|32|33B)($|[_-])",  ignore_case = TRUE)) ~ "large",
      TRUE ~ "unknown"
    ),
    experiment = case_when(
      str_detect(fname, regex("dic", ignore_case = TRUE)) ~ "Dictator",
      str_detect(fname, regex("ipd",      ignore_case = TRUE)) ~ "IPD",
      TRUE ~ "Unknown"
    )
  )
}

# ===== 3. Load and tag files (FAST) =====
load_and_tag <- function(file, model_family, size_category, experiment) {
  dat <- vroom::vroom(file, col_types = cols(.default = "c"))  # all char, fast

  dat %>%
    mutate(
      model_family = model_family,
      size_category = size_category,
      experiment = experiment
    )
}

# Dictator
dictator_files <- list.files(dictator_dir, pattern = "*.csv", full.names = TRUE)
dictator_map   <- bind_rows(lapply(dictator_files, parse_file_info))

# IPD
ipd_files <- list.files(ipd_dir, pattern = "*.csv", full.names = TRUE)
ipd_map   <- bind_rows(lapply(ipd_files, parse_file_info))

dictator_data <- dictator_map %>%
  select(file, model_family, size_category, experiment) %>%
  pmap_dfr(load_and_tag)

ipd_data <- ipd_map %>%
  select(file, model_family, size_category, experiment) %>%
  pmap_dfr(load_and_tag)

# ===== 3b. Safe type conversion AFTER merge =====
convert_types <- function(df) {
  safe_as <- function(df, col, fn) {
    if (col %in% names(df)) df[[col]] <- fn(df[[col]])
    df
  }

  numeric_cols <- c(
    "round","choice_prob","given","kept","generosity_streak",
    "total_A","total_B","coop_prob","coop_streak","relative_payoff"
  )
  logical_cols <- c("not_gamified","serious","game_theorist")
  char_cols    <- c("timestamp","seed","model_family","size_category","experiment")

  for (col in numeric_cols) df <- safe_as(df, col, as.numeric)
  for (col in logical_cols) df <- safe_as(df, col, as.logical)
  for (col in char_cols)    df <- safe_as(df, col, as.character)

  df
}

dictator_data <- convert_types(dictator_data)
ipd_data      <- convert_types(ipd_data)

# Drop unknown sizes
dictator_data <- dictator_data %>% filter(size_category %in% c("small","medium","large"))
ipd_data      <- ipd_data      %>% filter(size_category %in% c("small","medium","large"))

# ===== 4. Helper functions =====
boot_median <- function(x, i) median(x[i], na.rm = TRUE)

compute_ci <- function(x, R = 200) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(c(median = NA, ci_low = NA, ci_high = NA))
  if (length(unique(x)) <= 1) {
    m <- median(x)
    return(c(median = m, ci_low = m, ci_high = m))
  }
  b  <- boot(x, statistic = boot_median, R = R)   # <-- remove parallel args
  ci <- suppressWarnings(boot.ci(b, type = "perc"))
  if (is.null(ci)) return(c(median = median(x), ci_low = NA, ci_high = NA))
  c(median = median(x), ci_low = ci$percent[4], ci_high = ci$percent[5])
}

kw_eta2 <- function(H, k, n) max(0, min(1, (H - (k - 1)) / (n - k)))

cliffs_delta <- function(x, y) {
  x <- x[!is.na(x)]
  y <- y[!is.na(y)]
  nx <- length(x); ny <- length(y)
  if (nx == 0 || ny == 0) return(NA_real_)
  count <- 0
  for (xi in x) {
    count <- count + sum(xi > y) - sum(xi < y)
  }
  count / (nx * ny)
}

# ===== 5. Analysis function (FAST) =====
analyze_outcome <- function(df, outcome) {
  if (!(outcome %in% names(df))) {
    return(list(outcome = outcome, kw = NULL, eta = NA_real_, dunn = NULL, deltas = NULL, ci = NULL))
  }

  d <- df %>%
    select(size_category, value = !!sym(outcome)) %>%
    mutate(value = as.numeric(value)) %>%
    filter(!is.na(size_category))

  groups <- unique(d$size_category)

  # CIs only if <2 groups
  if (length(groups) < 2) {
    ci <- d %>%
      group_by(size_category) %>%
      summarise(stats = list(compute_ci(value)), .groups = "drop") %>%
      mutate(
        median = map_dbl(stats, 1),
        ci_low = map_dbl(stats, 2),
        ci_high= map_dbl(stats, 3)
      ) %>%
      select(size_category, median, ci_low, ci_high)

    return(list(outcome = outcome, kw = NULL, eta = NA_real_, dunn = NULL, deltas = NULL, ci = ci))
  }

  # KW
  kw  <- kruskal.test(value ~ factor(size_category), data = d)
  eta <- kw_eta2(unname(kw$statistic), length(groups), nrow(d))

  # Dunn
  dunn <- tryCatch(
    FSA::dunnTest(d$value, factor(d$size_category), method = "bonferroni")$res,
    error = function(e) NULL
  )

  # Cliff’s delta (FAST)
  split_groups <- split(d$value, d$size_category)
  pairs <- combn(groups, 2, simplify = FALSE)
  deltas <- map_dfr(pairs, function(p) {
    tibble(pair = paste(p, collapse = " vs "),
           delta = cliffs_delta(split_groups[[p[1]]], split_groups[[p[2]]]))
  })

  # CIs
  ci <- d %>%
    group_by(size_category) %>%
    summarise(stats = list(compute_ci(value)), .groups = "drop") %>%
    mutate(
      median = map_dbl(stats, 1),
      ci_low = map_dbl(stats, 2),
      ci_high= map_dbl(stats, 3)
    ) %>%
    select(size_category, median, ci_low, ci_high)

  list(outcome = outcome, kw = kw, eta = eta, dunn = dunn, deltas = deltas, ci = ci)
}

# ===== 6. Run for all outcomes (PARALLEL) =====
dictator_outcomes <- c("choice_prob","given","kept","generosity_streak","total_A","total_B")
ipd_outcomes      <- c("coop_prob","coop_streak","relative_payoff")

dictator_results <- future_map(dictator_outcomes, function(o) {
  analyze_outcome(dictator_data[, c("size_category", o), drop = FALSE], o)
})

ipd_results <- future_map(ipd_outcomes, function(o) {
  analyze_outcome(ipd_data[, c("size_category", o), drop = FALSE], o)
})

# ===== 7. Save summaries =====
flatten_results <- function(results_list) {
  bind_rows(map(results_list, function(res) {
    if (is.null(res$ci)) return(NULL)
    res$ci %>%
      mutate(
        outcome = res$outcome,
        eta2    = res$eta,
        kw_p    = if (!is.null(res$kw)) res$kw$p.value else NA_real_
      )
  }))
}

dictator_summary <- flatten_results(dictator_results)
ipd_summary      <- flatten_results(ipd_results)

write_csv(dictator_summary, "dictator_summary.csv")
write_csv(ipd_summary,      "ipd_summary.csv")