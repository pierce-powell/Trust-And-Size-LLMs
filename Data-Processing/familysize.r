suppressPackageStartupMessages({
  library(dplyr)
  library(vroom)      # fast CSV loading
  library(stringr)
  library(purrr)
  library(furrr)
  library(FSA)        # Dunn's test
  library(boot)       # bootstrapping
  library(tidyr)
  library(readr)
})

# ===== Parallel plan (no nested parallelism) =====
options(future.globals.maxSize = 10 * 1024^3)
future::plan(future::multisession, workers = max(1L, parallel::detectCores() - 1L))

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

# ===== 3. Load and tag files (fast, all char) =====
load_and_tag <- function(file, model_family, size_category, experiment) {
  dat <- vroom::vroom(file, col_types = cols(.default = "c"))
  dat %>%
    mutate(
      model_family = model_family,
      size_category = size_category,
      experiment = experiment
    )
}

# Dictator
dictator_files <- list.files(dictator_dir, pattern = "*.csv", full.names = TRUE)
dictator_map   <- bind_rows(map(dictator_files, parse_file_info))
dictator_data  <- pmap_dfr(dictator_map, load_and_tag)

# IPD
ipd_files <- list.files(ipd_dir, pattern = "*.csv", full.names = TRUE)
ipd_map   <- bind_rows(map(ipd_files, parse_file_info))
ipd_data  <- pmap_dfr(ipd_map, load_and_tag)

# ===== 3b. Safe type conversion AFTER merge =====
convert_types <- function(df) {
  safe_as <- function(df, col, fn) {
    if (col %in% names(df)) df[[col]] <- fn(df[[col]])
    df
  }

  numeric_cols <- c(
    "round",
    "choice_prob","given","kept","generosity_streak",
    "total_A","total_B",
    "coop_prob","coop_streak","relative_payoff"
  )
  logical_cols <- c("not_gamified","serious","game_theorist")
  char_cols    <- c("timestamp","seed","model_family","size_category","experiment")

  for (col in numeric_cols) df <- safe_as(df, col, as.numeric)
  for (col in logical_cols) df <- safe_as(df, col, as.logical)
  for (col in char_cols)    df <- safe_as(df, col, as.character)

  df
}

dictator_data <- pmap_dfr(
  dictator_map %>% select(file, model_family, size_category, experiment),
  load_and_tag
)

ipd_data <- pmap_dfr(
  ipd_map %>% select(file, model_family, size_category, experiment),
  load_and_tag
)

table(dictator_data$model_family)
table(ipd_data$model_family)

# ===== 4. Helper functions =====
boot_median <- function(x, i) median(x[i], na.rm = TRUE)

compute_ci <- function(x, R = 200) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(c(median = NA, ci_low = NA, ci_high = NA))
  if (length(unique(x)) <= 1) {
    m <- median(x)
    return(c(median = m, ci_low = m, ci_high = m))
  }
  b  <- boot(x, statistic = boot_median, R = R)  # serial, no nested parallel
  ci <- suppressWarnings(boot.ci(b, type = "perc"))
  if (is.null(ci)) return(c(median = median(x), ci_low = NA, ci_high = NA))
  c(median = median(x), ci_low = ci$percent[4], ci_high = ci$percent[5])
}

kw_eta2 <- function(H, k, n) {
  eta <- (H - (k - 1)) / (n - k)
  max(0, min(1, eta))
}

cliffs_delta <- function(x, y) {
  x <- x[!is.na(x)]; y <- y[!is.na(y)]
  nx <- length(x); ny <- length(y)
  if (nx == 0 || ny == 0) return(NA_real_)
  count <- 0L
  for (xi in x) {
    count <- count + sum(xi > y) - sum(xi < y)
  }
  count / (nx * ny)
}

# ===== 5. Core analysis function (size effects within a family) =====
analyze_outcome <- function(df, outcome) {
  if (!(outcome %in% names(df))) {
    return(list(outcome = outcome, kw = NULL, eta = NA_real_, dunn = NULL, deltas = NULL, ci = NULL))
  }

  d <- df %>%
    select(size_category, value = !!sym(outcome)) %>%
    mutate(value = as.numeric(value)) %>%
    filter(size_category %in% c("small","medium","large"))

  groups <- d %>% distinct(size_category) %>% pull(size_category)

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

  # Dunn (Bonferroni)
  dunn <- tryCatch(
    dunnTest(d$value, factor(d$size_category), method = "bonferroni")$res,
    error = function(e) NULL
  )

  # Cliff's deltas for all pairs (streaming)
  pairs <- combn(groups, 2, simplify = FALSE)
  deltas <- bind_rows(lapply(pairs, function(p) {
    x <- d %>% filter(size_category == p[1]) %>% pull(value)
    y <- d %>% filter(size_category == p[2]) %>% pull(value)
    tibble(pair = paste(p, collapse = " vs "), delta = cliffs_delta(x, y))
  }))

  # CIs per group
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

# ===== 6. Outcomes =====
dictator_outcomes <- c("choice_prob","given","kept","generosity_streak","total_A","total_B")
ipd_outcomes      <- c("coop_prob","coop_streak","relative_payoff")

# ===== 7. Run analyses per family in parallel =====
dictator_by_family <- dictator_data %>% group_split(model_family)
ipd_by_family      <- ipd_data %>% group_split(model_family)

dictator_results <- future_map(dictator_by_family, function(df) {
  family <- df$model_family[1]
  outs   <- intersect(dictator_outcomes, names(df))
  res    <- map(outs, ~ analyze_outcome(df, .x))
  list(family = family, results = res)
})

ipd_results <- future_map(ipd_by_family, function(df) {
  family <- df$model_family[1]
  outs   <- intersect(ipd_outcomes, names(df))
  res    <- map(outs, ~ analyze_outcome(df, .x))
  list(family = family, results = res)
})

# ===== 8. Flatten outputs =====
flatten_summary <- function(family_block) {
  family <- family_block$family
  bind_rows(lapply(family_block$results, function(res) {
    if (is.null(res$ci)) return(NULL)
    res$ci %>%
      mutate(
        outcome      = res$outcome,
        eta2         = res$eta,
        kw_p         = if (!is.null(res$kw)) res$kw$p.value else NA_real_,
        model_family = family
      )
  }))
}

flatten_dunn <- function(family_block) {
  family <- family_block$family
  bind_rows(lapply(family_block$results, function(res) {
    if (is.null(res$dunn)) return(NULL)
    as_tibble(res$dunn) %>%
      mutate(outcome = res$outcome, model_family = family)
  }))
}

flatten_deltas <- function(family_block) {
  family <- family_block$family
  bind_rows(lapply(family_block$results, function(res) {
    if (is.null(res$deltas)) return(NULL)
    res$deltas %>%
      mutate(outcome = res$outcome, model_family = family)
  }))
}

dictator_summary <- bind_rows(map(dictator_results, flatten_summary))
dictator_dunn    <- bind_rows(map(dictator_results, flatten_dunn))
dictator_deltas  <- bind_rows(map(dictator_results, flatten_deltas))

ipd_summary <- bind_rows(map(ipd_results, flatten_summary))
ipd_dunn    <- bind_rows(map(ipd_results, flatten_dunn))
ipd_deltas  <- bind_rows(map(ipd_results, flatten_deltas))

# ===== 9. Save outputs =====
if (!dir.exists("results")) dir.create("results")

write_csv(dictator_summary, "results/dictator_summary_by_family.csv")
write_csv(dictator_dunn,    "results/dictator_dunn_by_family.csv")
write_csv(dictator_deltas,  "results/dictator_deltas_by_family.csv")

write_csv(ipd_summary, "results/ipd_summary_by_family.csv")
write_csv(ipd_dunn,    "results/ipd_dunn_by_family.csv")
write_csv(ipd_deltas,  "results/ipd_deltas_by_family.csv")

message("Saved: results/*_by_family.csv")