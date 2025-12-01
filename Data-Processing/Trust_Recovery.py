#!/usr/bin/env python3
"""
Compare trust-recovery across CSV results (supports multiple files and grouping).

Usage examples:
  # single file (backwards compatible)
  python compare_results_recovery.py --infile qwen_05B_clean.csv --model_name QWEN05B

  # multiple files, no grouping:
  python compare_results_recovery.py --infiles a.csv b.csv c.csv --model_names A B C

  # group every 3 files:
  python compare_results_recovery.py --infiles f1.csv f2.csv f3.csv f4.csv f5.csv f6.csv --group_size 3 --group_labels "GroupA" "GroupB"

  python Trust_Recovery.py --infiles qwen_05B_ipd_cleaned.csv ipd_gemma_cleaned_1b.csv ipd_OLMo_clean_1B.csv qwen_7B_ipd_cleaned.csv ipd_gemma_cleaned_4b.csv ipd_OLMo_clean_7B.csv qwen_14_ipd_cleaned.csv ipd_gemma_cleaned_12b.csv ipd_OLMo_clean_13B.csv qwen_32_ipd_cleaned.csv ipd_gemma_cleaned_27b.csv ipd_OLMo_clean_32B.csv --group_size 3  --group_labels "Small Models (Qwen 0.5B, OLMo 1B, Gemma 1B)" "Small-Medium Models (Qwen 7B, OLMo 7B, Gemma 4B)" "Large-Medium Models (Qwen 14B, OLMo 13B, Gemma 12B)" "Large Models (Qwen 32B, OLMo 32B, Gemma 27B)" 

"""
import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Trust-recovery helpers
# ----------------------------
def compute_recovery_times(model_hist: str, heur_hist: str):
    """
    Given two strings of history (model & heuristic) compute recovery times:
      distance from each heuristic 'D' to the next model 'C'.
    Returns list of positive integers (recovery distances). If no future 'C', event is skipped.
    """
    recovery_times = []
    # be defensive: ensure strings
    model_hist = "" if model_hist is None else str(model_hist)
    heur_hist = "" if heur_hist is None else str(heur_hist)
    length = min(len(model_hist), len(heur_hist))
    for i, h in enumerate(heur_hist[:length]):
        if h == 'D':
            for j in range(i + 1, length):
                if model_hist[j] == 'C':
                    recovery_times.append(j - i)
                    break
            # if no future cooperation found, skip (treated as infinite/unobserved)
    return recovery_times


def compute_recovery_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-(variant, heuristic, not_gamified) recovery summary from the input DataFrame.
    Filters to heuristic == 'Random' and round == 100 (if 'round' exists).
    Returns DataFrame with columns: variant, heuristic, not_gamified, avg_recovery_time, n_recovery_events
    """
    if "heuristic" not in df.columns or "variant" not in df.columns or "not_gamified" not in df.columns:
        raise ValueError("Input DataFrame is missing required grouping columns.")

    df_work = df.copy()
    # filter for Random heuristic (same as your original script)
    df_work = df_work[df_work["heuristic"] == "Random"]
    # keep only summary round if round column exists
    if "round" in df_work.columns:
        df_work = df_work[df_work["round"] == 100]

    group_cols = ["variant", "heuristic", "not_gamified"]

    results = []
    if df_work.empty:
        return pd.DataFrame(results)

    for (variant, heuristic, ng), group in df_work.groupby(group_cols, as_index=False):
        all_recoveries = []
        for _, row in group.iterrows():
            model_hist = row.get("history_model", "")
            heur_hist = row.get("history_heuristic", "")
            recovs = compute_recovery_times(model_hist, heur_hist)
            all_recoveries.extend(recovs)

        avg_recovery = float(np.mean(all_recoveries)) if all_recoveries else np.nan
        results.append({
            "variant": str(variant),
            "heuristic": str(heuristic),
            "not_gamified": str(ng),
            "avg_recovery_time": avg_recovery,
            "n_recovery_events": len(all_recoveries)
        })
    return pd.DataFrame(results)


# ----------------------------
# Summary & plotting helpers
# ----------------------------
def summarize_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Return summary DataFrame grouped by variant, heuristic, not_gamified with mean, std, count, se, ci95 and label.
    If groups have only a single observation, se/ci95 are set to 0 for plotting convenience.
    """
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(["variant", "heuristic", "not_gamified"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"].replace(0, np.nan))
    summary["ci95"] = 1.96 * summary["se"]
    # set se/ci95 to 0 when only a single observation (avoid NaNs)
    single_mask = summary["count"] <= 1
    summary.loc[single_mask, ["se", "ci95"]] = 0.0
    summary["label_tuple"] = list(zip(summary["variant"], summary["heuristic"], summary["not_gamified"]))
    summary["label"] = (
        summary["variant"].astype(str) + " | "
        + summary["heuristic"].astype(str) + " | "
        + summary["not_gamified"].astype(str)
    )
    summary = summary.set_index("label_tuple")
    return summary


def plot_stacked_ci(dfs: List[pd.DataFrame], metric: str, title: str, ylabel: str, model_names: List[str], out_prefix: str = None):
    """
    dfs: list of pandas DataFrames (one per model/group) containing columns:
         variant, heuristic, not_gamified and metric
    metric: column name to plot
    model_names: list of strings, same length as dfs
    """
    summaries = [summarize_df(df, metric) for df in dfs]

    all_label_tuples = sorted({t for s in summaries for t in s.index.tolist()})
    label_strings = []
    for vt, ht, ng in all_label_tuples:
        label_strings.append(f"{vt} | {ht} | {ng}")

    n_models = len(dfs)
    n_labels = len(all_label_tuples)
    if n_labels == 0:
        print(f"No label groups found for metric '{metric}'. Skipping.")
        return

    means = np.zeros((n_models, n_labels), dtype=float)
    cis = np.zeros_like(means)
    counts = np.zeros_like(means, dtype=int)

    for i, s in enumerate(summaries):
        for j, lt in enumerate(all_label_tuples):
            if lt in s.index:
                means[i, j] = s.at[lt, "mean"] if not pd.isna(s.at[lt, "mean"]) else 0.0
                cis[i, j] = s.at[lt, "ci95"] if not pd.isna(s.at[lt, "ci95"]) else 0.0
                counts[i, j] = int(s.at[lt, "count"]) if not pd.isna(s.at[lt, "count"]) else 0
            else:
                means[i, j] = 0.0
                cis[i, j] = 0.0
                counts[i, j] = 0

    has_data_mask = counts > 0
    if np.any(has_data_mask):
        valid_upper = (means + cis)[has_data_mask]
        valid_lower = (means - cis)[has_data_mask]
        global_max = np.nanmax(valid_upper) if valid_upper.size > 0 else 0.0
        global_min = np.nanmin(valid_lower) if valid_lower.size > 0 else 0.0
        if "prob" in metric.lower() or "probability" in metric.lower():
            global_min = 0.0
            global_max = 1.0
        else:
            if global_min > 0:
                global_min = min(0.0, global_min)
            if global_max <= 0:
                global_max = 1.0
    else:
        global_min, global_max = 0.0, 1.0

    padding_factor = 1.12
    if np.isclose(global_max, global_min):
        global_max = global_min + 1.0
    if global_max < global_min:
        global_min, global_max = min(global_min, global_max), max(global_min, global_max)

    range_span = global_max - global_min
    if range_span <= 0:
        range_span = 1.0
    padded_max = global_min + range_span * padding_factor

    fig_height_per_model = 3.5
    fig, axes = plt.subplots(n_models, 1, figsize=(14, fig_height_per_model * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    x = np.arange(n_labels)

    for i, ax in enumerate(axes):
        mask = counts[i] > 0
        bar_positions = x[mask]
        bar_heights = means[i][mask]
        bar_err = cis[i][mask]

        if bar_positions.size == 0:
            ax.text(0.5, 0.5, "No data for this model / metric", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.bar(bar_positions, bar_heights, yerr=bar_err, capsize=4, edgecolor="black", color = "green")

        ax.set_title(model_names[i], fontsize=10, loc="left", pad=6, style="italic")
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_ylim(global_min - 0.05 * range_span, padded_max)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(label_strings, ha="right", fontsize=10)

    plt.subplots_adjust(bottom=0.3, top=0.94, hspace=0.35)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if out_prefix:
        safe_metric = metric.replace(" ", "_")
        out_file = f"{out_prefix}_{safe_metric}.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {out_file}")

    plt.show()


# ----------------------------
# Loading / grouping helpers
# ----------------------------
def load_and_prepare(path: str) -> pd.DataFrame:
    """
    Load CSV and ensure required columns exist.
    Required columns for recovery: variant, heuristic, not_gamified, history_model, history_heuristic
    """
    df = pd.read_csv(path)
    required = ["variant", "heuristic", "not_gamified", "history_model", "history_heuristic"]
    for col in required:
        if col not in df.columns:
            print(f"ERROR: Required column '{col}' not found in {path}.", file=sys.stderr)
            raise SystemExit(1)

    # Coerce not_gamified to string
    df["not_gamified"] = df["not_gamified"].astype(str)

    # If coop/model payoff columns present, coerce numeric (keeps backward compat)
    for col in ["coop_prob", "model_payoff"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute & plot trust recovery time across multiple CSVs with optional grouping.")
    parser.add_argument("--infile", help="Single input CSV (backwards compatible)")
    parser.add_argument("--model_name", help="Single model name (backwards compatible)")
    parser.add_argument("--infiles", nargs="+", help="List of input CSV files (space separated)")
    parser.add_argument("--model_names", nargs="+", help="List of model names (space separated). Must match the number of infiles (if not grouping).")
    parser.add_argument("--group_size", type=int, default=1, help="Number of input files to average together per plotted model (default 1).")
    parser.add_argument("--group_labels", nargs="+", help="Optional labels for each averaged group. Must match len(infiles)/group_size if provided.")
    parser.add_argument("--out_prefix", help="Optional prefix for saved figures (e.g. 'stacked' -> stacked_avg_recovery_time.png)", default=None)
    args = parser.parse_args()

    if args.infiles:
        input_paths = args.infiles
    elif args.infile:
        input_paths = [args.infile]
    else:
        parser.print_help()
        raise SystemExit(1)

    if args.group_size <= 0:
        print("ERROR: --group_size must be >= 1", file=sys.stderr)
        raise SystemExit(1)

    if len(input_paths) % args.group_size != 0:
        print(f"ERROR: Number of input files ({len(input_paths)}) is not divisible by group_size ({args.group_size}).", file=sys.stderr)
        raise SystemExit(1)

    # Load all raw dataframes
    raw_dfs = []
    for p in input_paths:
        if not os.path.exists(p):
            print(f"ERROR: Input file not found: {p}", file=sys.stderr)
            raise SystemExit(1)
        df = load_and_prepare(p)
        raw_dfs.append(df)
        print(f"Loaded {len(df)} rows from {p}")

    # Build grouped/averaged recovery summary DataFrames
    grouped_dfs = []
    grouped_names = []
    n_groups = len(input_paths) // args.group_size

    for gi in range(n_groups):
        start = gi * args.group_size
        end = start + args.group_size
        dfs_group = raw_dfs[start:end]

        if args.group_size > 1:
            # Pool rows across the group and then compute the recovery summary on the pooled rows
            pooled = pd.concat(dfs_group, ignore_index=True)
            averaged_df = compute_recovery_summary(pooled)
        else:
            averaged_df = compute_recovery_summary(dfs_group[0])

        grouped_dfs.append(averaged_df)

        # populate group label
        if args.group_labels:
            if len(args.group_labels) != n_groups:
                print("ERROR: --group_labels must have exactly len(infiles)/group_size entries.", file=sys.stderr)
                raise SystemExit(1)
            # grouped_names will be set outside the loop in this case
        else:
            group_files = input_paths[start:end]
            nice = ", ".join([os.path.splitext(os.path.basename(x))[0] for x in group_files])
            grouped_names.append(f"avg({nice})")

    if args.group_labels:
        model_names_for_plot = args.group_labels
    else:
        model_names_for_plot = grouped_names

    if len(grouped_dfs) != len(model_names_for_plot):
        print("ERROR: Internal mismatch between grouped_dfs and model names.", file=sys.stderr)
        raise SystemExit(1)

    # Plot average recovery time
    plot_stacked_ci(
        grouped_dfs,
        metric="avg_recovery_time",
        title="Trust Recovery Time: Average (distance from D -> next C)",
        ylabel="Average Recovery Time (rounds)",
        model_names=model_names_for_plot,
        out_prefix=args.out_prefix,
    )

    # Optional: also plot number of recovery events (counts)
    plot_stacked_ci(
        grouped_dfs,
        metric="n_recovery_events",
        title="Trust Recovery: Number of Observed Recovery Events",
        ylabel="Number of Recovery Events",
        model_names=model_names_for_plot,
        out_prefix=args.out_prefix,
    )


if __name__ == "__main__":
    main()
