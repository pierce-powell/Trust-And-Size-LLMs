#!/usr/bin/env python3
"""
Usage examples:
    # Single File
    python compare_results.py --infile cleaned_qwen_0.5B_IPD.csv --model_name "Qwen2.5 0.5B"

    # Group by Model Size
    python compare_results_between.py --infiles cleaned_qwen_0.5B_IPD.csv cleaned_gemma_1B_IPD.csv cleaned_olmo_1B_IPD.csv cleaned_deepseek_1.3B_IPD.csv cleaned_qwen_7_14B_IPD.csv cleaned_gemma_4_12B_IPD.csv cleaned_olmo_7_13B_IPD.csv cleaned_deepseek_6.7B_IPD.csv cleaned_qwen_32B_IPD.csv cleaned_gemma_27B_IPD.csv cleaned_olmo_32B_IPD.csv cleaned_deepseek_33B_IPD.csv --group_size 4 --group_labels "Small Models (Qwen2.5 0.5B, OLMo2 1B, Gemma3 1B, DeepSeek 1.3B)" "Medium Models (Qwen2.5 7B, 14B; OLMo2 7B, 13B; Gemma3 4B, 12B; DeepSeek 6.7B)" "Large Models (Qwen2.5 32B, OLMo2 32B, Gemma3 27B, DeepSeek 33B)"

    # Group by Model Source
    python compare_results_between.py --infiles cleaned_qwen_0.5B_IPD.csv cleaned_olmo_1B_IPD.csv cleaned_qwen_7_14B_IPD.csv cleaned_olmo_7_13B_IPD.csv cleaned_qwen_32B_IPD.csv cleaned_olmo_32B_IPD.csv cleaned_gemma_1B_IPD.csv cleaned_deepseek_1.3B_IPD.csv cleaned_gemma_4_12B_IPD.csv cleaned_deepseek_6.7B_IPD.csv cleaned_gemma_27B_IPD.csv cleaned_deepseek_33B_IPD.csv --group_size 6 --group_labels "Open-Source (Qwen2.5, OLMo2)" "Commercially Used (Gemma3, DeepSeek)"
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from typing import List

# ----------------------------
# Summary & plotting helpers
# ----------------------------
def summarize_df(df, metric):
    """Return summary DataFrame grouped by variant, heuristic, not_gamified with mean, std, count, se, ci95 and label."""
    summary = (
        df.groupby(["variant", "heuristic", "not_gamified"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"].replace(0, np.nan))
    summary["ci95"] = 1.96 * summary["se"]
    summary["label_tuple"] = list(zip(summary["variant"], summary["heuristic"], summary["not_gamified"]))
    summary["label"] = (
        summary["variant"].astype(str) + " | "
        + summary["heuristic"].astype(str) + " | "
        + summary["not_gamified"].astype(str)
    )
    summary = summary.set_index("label_tuple")
    return summary

def plot_stacked_ci(dfs, metric, title, ylabel, model_names, out_prefix=None):
    """
    dfs: list of pandas DataFrames (one per model)
    metric: column name to plot
    model_names: list of strings, same length as dfs
    out_prefix: if provided, save to {out_prefix}_{metric}.png
    """
    # Summarize each df
    summaries = [summarize_df(df, metric) for df in dfs]

    # Build the full set of label tuples (consistent order across models)
    all_label_tuples = sorted({t for s in summaries for t in s.index.tolist()})
    # Readable labels in the same order
    label_strings = []
    for vt, ht, ng in all_label_tuples:
        label_strings.append(f"{vt} | {ht} | {ng}")

    n_models = len(dfs)
    n_labels = len(all_label_tuples)
    if n_labels == 0:
        print(f"No label groups found for metric '{metric}'. Skipping.")
        return

    # Create arrays for each model aligned to all_label_tuples
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

    # Determine a global y-axis range across all models for this metric
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
    if "prob" in metric.lower() or "probability" in metric.lower():
        padded_min, padded_max = 0.0, 1.0
    else:
        lower_padding = 0.05 * range_span
        padded_min = global_min - lower_padding
        if padded_min > 0 and global_min >= 0:
            padded_min = 0.0
        padded_max = padded_max

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
            ax.bar(bar_positions, bar_heights, yerr=bar_err, capsize=4, edgecolor="black")

        ax.set_title(model_names[i], fontsize=10, loc="left", pad=6, style="italic")
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_ylim(padded_min, padded_max)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(label_strings, rotation=65, ha="right", fontsize=10)

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
def load_and_prepare(path):
    """Load CSV and coerce expected columns/numeric types. Returns prepared DataFrame."""
    df = pd.read_csv(path)
    # Ensure the column exists
    for col in ["variant", "heuristic", "not_gamified"]:
        if col not in df.columns:
            print(f"ERROR: Required column '{col}' not found in {path}.", file=sys.stderr)
            raise SystemExit(1)

    # Coerce not_gamified to string (keeps grouping stable)
    df["not_gamified"] = df["not_gamified"].astype(str)

    # Ensure numeric columns exist and coerce to numeric; missing -> NaN
    for col in ["coop_prob", "model_payoff"]:
        if col not in df.columns:
            print(f"ERROR: Required metric column '{col}' not found in {path}.", file=sys.stderr)
            raise SystemExit(1)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # drop rows missing both metrics (we will filter per metric later)
    df = df.dropna(subset=["coop_prob", "model_payoff"], how="all")
    return df

def average_group_dfs(dfs_group: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Pool rows from dfs in dfs_group and return a dataframe that contains the per-(variant,heuristic,not_gamified)
    means for all numeric columns. This effectively averages across the files in the group.
    """
    pooled = pd.concat(dfs_group, ignore_index=True)
    # make sure grouping columns coerced properly
    pooled["not_gamified"] = pooled["not_gamified"].astype(str)
    # identify numeric columns to average (exclude grouping cols)
    group_cols = ["variant", "heuristic", "not_gamified"]
    numeric_cols = [c for c in pooled.columns if c not in group_cols and pd.api.types.is_numeric_dtype(pooled[c])]
    if not numeric_cols:
        # fallback to the two metrics explicitly if types are weird
        numeric_cols = ["coop_prob", "model_payoff"]
    agg_dict = {c: "mean" for c in numeric_cols}
    averaged = pooled.groupby(group_cols, as_index=False).agg(agg_dict)
    # keep grouping columns as strings
    averaged["variant"] = averaged["variant"].astype(str)
    averaged["heuristic"] = averaged["heuristic"].astype(str)
    averaged["not_gamified"] = averaged["not_gamified"].astype(str)
    return averaged

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot stacked cooperation probability and model payoff across multiple models.")
    parser.add_argument("--infile", help="Single input CSV (backwards compatible)")
    parser.add_argument("--model_name", help="Single model name (backwards compatible)")
    parser.add_argument("--infiles", nargs="+", help="List of input CSV files (space separated)")
    parser.add_argument("--model_names", nargs="+", help="List of model names (space separated). Must match the number of infiles (if not grouping).")
    parser.add_argument("--group_size", type=int, default=1, help="Number of input files to average together per plotted model (default 1).")
    parser.add_argument("--group_labels", nargs="+", help="Optional labels for each averaged group (space separated). Must match len(infiles)/group_size if provided.")
    parser.add_argument("--out_prefix", help="Optional prefix for saved figures (e.g. 'stacked' -> stacked_coop_prob.png)", default=None)
    args = parser.parse_args()

    # Resolve inputs
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

    # Load all dataframes
    raw_dfs = []
    for p in input_paths:
        if not os.path.exists(p):
            print(f"ERROR: Input file not found: {p}", file=sys.stderr)
            raise SystemExit(1)
        df = load_and_prepare(p)
        raw_dfs.append(df)
        print(f"Loaded {len(df)} rows from {p}")

    # Build grouped/averaged dataframes
    grouped_dfs = []
    grouped_names = []
    n_groups = len(input_paths) // args.group_size
    for gi in range(n_groups):
        start = gi * args.group_size
        end = start + args.group_size
        dfs_group = raw_dfs[start:end]
        averaged_df = average_group_dfs(dfs_group) if args.group_size > 1 else dfs_group[0]
        grouped_dfs.append(averaged_df)

        # choose a group label
        if args.group_labels:
            if len(args.group_labels) != n_groups:
                print("ERROR: --group_labels must have exactly len(infiles)/group_size entries.", file=sys.stderr)
                raise SystemExit(1)
            grouped_names = args.group_labels
        else:
            # auto-generate: concatenate filenames for the group (shortened)
            group_files = input_paths[start:end]
            nice = ", ".join([os.path.splitext(os.path.basename(x))[0] for x in group_files])
            grouped_names.append(f"avg({nice})")

    # If group_labels provided we've already set grouped_names above; otherwise grouped_names list is filled iteratively
    if args.group_labels:
        model_names_for_plot = args.group_labels
    else:
        model_names_for_plot = grouped_names

    # Final check length matching
    if len(grouped_dfs) != len(model_names_for_plot):
        print("ERROR: Internal mismatch between grouped_dfs and model names.", file=sys.stderr)
        raise SystemExit(1)

    # Plot the two metrics using the grouped/averaged dataframes
    plot_stacked_ci(
        grouped_dfs,
        metric="coop_prob",
        title="IPD: Average Cooperation Probability (grouped averages)",
        ylabel="Cooperation Probability",
        model_names=model_names_for_plot,
        out_prefix=args.out_prefix,
    )

    plot_stacked_ci(
        grouped_dfs,
        metric="model_payoff",
        title="Average Model Payoff (grouped averages)",
        ylabel="Model Payoff",
        model_names=model_names_for_plot,
        out_prefix=args.out_prefix,
    )

if __name__ == "__main__":
    main()
