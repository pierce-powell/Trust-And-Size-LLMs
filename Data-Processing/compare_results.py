#!/usr/bin/env python3
"""
Usage examples:
  python compare_results.py --infile qwen_05B_clean.csv --model_name QWEN05B

  python compare_results.py --infiles 05-clean.csv 7-clean.csv qwen_14B_ipd.csv 32-clean.csv --model_names "QWEN-2.5 0.5B" "QWEN-2.5 7B" "QWEN-2.5 14B" "QWEN-2.5 32B" --out_prefix stacked
  python compare_results.py --infiles olmo_1B_ipd.csv olmo_7B_ipd.csv olmo_13B_ipd.csv olmo_32B_ipd.csv --model_names "OLMo2 1B" "OLMo2 7B" "OLMo2 13B" "OLMo2 32B" --out_prefix stacked
  python compare_results.py --infiles 1-clean.csv 4-clean.csv 12-clean.csv 27-clean.csv --model_names "Gemma 3 1B" "Gemma 3 4B" "Gemma 3 12B" "Gemma 3 27B" --out_prefix stacked

This script plots stacked bar charts (one subplot per model) for the two metrics:
 - coop_prob
 - model_payoff

Each stacked figure has one row per input model to save space on the x-axis labels.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

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
    # Consider only entries where count > 0 (actual data)
    has_data_mask = counts > 0
    if np.any(has_data_mask):
        # Compute candidate min/max using mean +/- ci
        valid_upper = (means + cis)[has_data_mask]
        valid_lower = (means - cis)[has_data_mask]
        global_max = np.nanmax(valid_upper) if valid_upper.size > 0 else 0.0
        global_min = np.nanmin(valid_lower) if valid_lower.size > 0 else 0.0
        # If metric looks like a probability, clamp to [0,1]
        if "prob" in metric.lower() or "probability" in metric.lower():
            global_min = 0.0
            global_max = 1.0
        else:
            # ensure sensible bounds: include 0 if all positive and close to zero, and avoid negative bottoms unless actual
            if global_min > 0:
                global_min = min(0.0, global_min)
            # If max is zero or negative (odd), set to 1 as a fallback
            if global_max <= 0:
                global_max = 1.0
    else:
        # No data at all: fallback to 0-1
        global_min, global_max = 0.0, 1.0

    # Add a small top padding (same factor used previously)
    padding_factor = 1.12
    # If global_max equals global_min (flat), expand a little for visibility
    if np.isclose(global_max, global_min):
        global_max = global_min + 1.0

    # Ensure we don't invert the axis
    if global_max < global_min:
        global_min, global_max = min(global_min, global_max), max(global_min, global_max)

    # Apply padding but keep lower bound as-is if it's 0 (probabilities)
    # For lower bound, give a small fraction of the range if it's not zero
    range_span = global_max - global_min
    if range_span <= 0:
        range_span = 1.0
    padded_max = global_min + range_span * padding_factor
    # If metric is probability, clamp padded_max to 1.0
    if "prob" in metric.lower() or "probability" in metric.lower():
        padded_min, padded_max = 0.0, 1.0
    else:
        # small lower padding (5% of range) if global_min isn't zero
        lower_padding = 0.05 * range_span
        padded_min = global_min - lower_padding
        # avoid negative lower bound unless data actually negative by some margin
        if padded_min > 0 and global_min >= 0:
            padded_min = 0.0
        # final padded max
        padded_max = padded_max

    # Figure size: keep width ~14, height proportional to number of models
    fig_height_per_model = 3.5
    fig, axes = plt.subplots(n_models, 1, figsize=(14, fig_height_per_model * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    x = np.arange(n_labels)

    for i, ax in enumerate(axes):
        # Use mask so bars for missing groups are not plotted
        mask = counts[i] > 0
        bar_positions = x[mask]
        bar_heights = means[i][mask]
        bar_err = cis[i][mask]

        if bar_positions.size == 0:
            ax.text(0.5, 0.5, "No data for this model / metric", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.bar(bar_positions, bar_heights, yerr=bar_err, capsize=4, edgecolor="black")

        # Model subtitle: place on left of subplot
        ax.set_title(model_names[i], fontsize=10, loc="left", pad=6, style="italic")
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        # Set the same y-limits for every subplot (padded)
        ax.set_ylim(padded_min, padded_max)

    # Configure x-axis ticks only on the bottom subplot
    # (set ticks at every label position so alignment remains consistent)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(label_strings, rotation=65, ha="right", fontsize=10)

    # Ensure there's enough bottom margin so rotated labels are not clipped
    plt.subplots_adjust(bottom=0.3, top=0.94, hspace=0.35)

    # Overall title and layout
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if out_prefix:
        safe_metric = metric.replace(" ", "_")
        out_file = f"{out_prefix}_{safe_metric}.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {out_file}")

    plt.show()

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


def main():
    parser = argparse.ArgumentParser(description="Plot stacked cooperation probability and model payoff across multiple models.")
    parser.add_argument("--infile", help="Single input CSV (backwards compatible)")
    parser.add_argument("--model_name", help="Single model name (backwards compatible)")
    parser.add_argument("--infiles", nargs="+", help="List of input CSV files (space separated)")
    parser.add_argument("--model_names", nargs="+", help="List of model names (space separated). Must match the number of infiles.")
    parser.add_argument("--out_prefix", help="Optional prefix for saved figures (e.g. 'stacked' -> stacked_coop_prob.png)", default=None)
    args = parser.parse_args()

    # Resolve inputs: prefer infiles if provided, otherwise fall back to infile
    if args.infiles:
        input_paths = args.infiles
        if not args.model_names:
            print("ERROR: --model_names is required when providing multiple --infiles.", file=sys.stderr)
            raise SystemExit(1)
        model_names = args.model_names
    elif args.infile:
        input_paths = [args.infile]
        model_names = [args.model_name or "Model"]
    else:
        parser.print_help()
        raise SystemExit(1)

    if len(input_paths) != len(model_names):
        print("ERROR: The number of input files must match the number of model names.", file=sys.stderr)
        print(f"Found {len(input_paths)} files and {len(model_names)} names.", file=sys.stderr)
        raise SystemExit(1)

    # Load dataframes
    dfs = []
    for p in input_paths:
        if not os.path.exists(p):
            print(f"ERROR: Input file not found: {p}", file=sys.stderr)
            raise SystemExit(1)
        df = load_and_prepare(p)
        dfs.append(df)
        print(f"Loaded {len(df)} rows from {p}")

    # For each metric, call the stacked plotter. Each plot receives full dfs list so x-axis labels stay consistent.
    plot_stacked_ci(
        dfs,
        metric="coop_prob",
        title="IPD: Average Cooperation Probability For Gemma 3 by Variant × Heuristic × Is Serious",
        ylabel="Cooperation Probability",
        model_names=model_names,
        out_prefix=args.out_prefix,
    )

    plot_stacked_ci(
        dfs,
        metric="model_payoff",
        title="Average Model Payoff by Variant × Heuristic × Is Serious",
        ylabel="Model Payoff",
        model_names=model_names,
        out_prefix=args.out_prefix,
    )


if __name__ == "__main__":
    main()
