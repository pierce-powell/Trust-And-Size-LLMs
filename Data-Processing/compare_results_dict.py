#!/usr/bin/env python3
"""
Usage (example):
    # Single file
    python compare_results_dict.py --infiles cleaned_deepseek_33B_DIC.csv --model_names "DeepSeek 33B"

    # Family
    python compare_results_dict.py --infiles cleaned_olmo_1B_DIC.csv cleaned_olmo_7B_DIC.csv cleaned_olmo_13B_DIC.csv cleaned_olmo_32B_DIC.csv --model_names "OLMo2 1B" "OLMo2 7B" "OLMo2 14B" "OLMo2 32B"
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_prepare(path):
    """Load CSV and normalize expected columns / types."""
    df = pd.read_csv(path)

    required_cols = ("variant", "not_gamified", "given", "kept")
    for c in required_cols:
        if c not in df.columns:
            print(f"ERROR: Required column '{c}' missing in {path}.", file=sys.stderr)
            raise SystemExit(1)

    # Normalize grouping columns to clean strings
    df["variant"] = df["variant"].astype(str).str.strip()

    # Normalize not_gamified: convert booleans/NaN/etc -> "True"/"False"/"nan"
    df["not_gamified"] = df["not_gamified"].astype(str).str.strip()
    df["not_gamified"] = df["not_gamified"].replace({"True": "True", "False": "False", "None": "None", "nan": "nan"})

    # Numeric conversion for metrics
    df["given"] = pd.to_numeric(df["given"], errors="coerce")
    df["kept"] = pd.to_numeric(df["kept"], errors="coerce")

    # drop rows that have both metrics missing (we'll allow rows with one metric present)
    df = df.dropna(subset=["given", "kept"], how="all").copy()

    return df


def build_group_stats_dict(df):
    """
    Build a dict mapping (variant, not_gamified) -> {
        'given_mean','given_std','given_count','given_se','given_ci95',
        'kept_mean','kept_std','kept_count','kept_se','kept_ci95'
    }
    """
    stats = {}
    grouped = df.groupby(["variant", "not_gamified"])
    for name, group in grouped:
        # name is a tuple (variant, not_gamified)
        gm = group["given"].mean(skipna=True)
        gs = group["given"].std(skipna=True)
        gc = int(group["given"].count())
        gse = (gs / np.sqrt(gc)) if gc > 0 and not pd.isna(gs) else 0.0
        gci = 1.96 * gse

        km = group["kept"].mean(skipna=True)
        ks = group["kept"].std(skipna=True)
        kc = int(group["kept"].count())
        kse = (ks / np.sqrt(kc)) if kc > 0 and not pd.isna(ks) else 0.0
        kci = 1.96 * kse

        stats[name] = {
            "given_mean": 0.0 if pd.isna(gm) else float(gm),
            "given_std": 0.0 if pd.isna(gs) else float(gs),
            "given_count": gc,
            "given_se": float(gse),
            "given_ci95": float(gci),
            "kept_mean": 0.0 if pd.isna(km) else float(km),
            "kept_std": 0.0 if pd.isna(ks) else float(ks),
            "kept_count": kc,
            "kept_se": float(kse),
            "kept_ci95": float(kci),
        }
    return stats


def plot_given_kept_stacked(dfs, model_names, out_prefix=None):
    # build per-model stat dicts and global label set
    stats_dicts = []
    global_labels = set()

    for df in dfs:
        stats = build_group_stats_dict(df)
        stats_dicts.append(stats)
        global_labels.update(stats.keys())

    # If some models have no groups (empty), allow labels from raw dfs too
    if not global_labels:
        # try gather labels directly from dfs (in case groups had NaNs)
        for df in dfs:
            labels = set(zip(df["variant"].astype(str).str.strip(), df["not_gamified"].astype(str).str.strip()))
            global_labels.update(labels)

    if not global_labels:
        print("No groups found to plot. Exiting.")
        return

    # stable ordering: sort by variant then not_gamified string
    all_label_tuples = sorted(list(global_labels), key=lambda t: (str(t[0]), str(t[1])))
    label_strings = [f"{v} | {n}" for (v, n) in all_label_tuples]

    n_models = len(dfs)
    n_labels = len(all_label_tuples)

    # build arrays aligned to all_label_tuples
    means_given = np.zeros((n_models, n_labels), dtype=float)
    cis_given = np.zeros_like(means_given)
    counts_given = np.zeros_like(means_given, dtype=int)

    means_kept = np.zeros((n_models, n_labels), dtype=float)
    cis_kept = np.zeros_like(means_kept)
    counts_kept = np.zeros_like(means_kept, dtype=int)

    for i, stats in enumerate(stats_dicts):
        for j, lt in enumerate(all_label_tuples):
            val = stats.get(lt)
            if val is not None:
                means_given[i, j] = val["given_mean"]
                cis_given[i, j] = val["given_ci95"]
                counts_given[i, j] = int(val["given_count"])
                means_kept[i, j] = val["kept_mean"]
                cis_kept[i, j] = val["kept_ci95"]
                counts_kept[i, j] = int(val["kept_count"])
            else:
                # leave zeros
                pass

    # Determine global y-axis max across both metrics (consider mean + ci only where count>0)
    has_data_mask = (counts_given > 0) | (counts_kept > 0)
    if np.any(has_data_mask):
        upper_given = (means_given + cis_given)[has_data_mask]
        upper_kept = (means_kept + cis_kept)[has_data_mask]
        candidate_max = 0.0
        if upper_given.size > 0:
            candidate_max = max(candidate_max, float(np.nanmax(upper_given)))
        if upper_kept.size > 0:
            candidate_max = max(candidate_max, float(np.nanmax(upper_kept)))
        global_max = candidate_max if candidate_max > 0 else 1.0
    else:
        global_max = 1.0

    padding_factor = 1.12
    padded_max = global_max * padding_factor

    # plotting
    fig_height_per_model = 3.5
    fig, axes = plt.subplots(n_models, 1, figsize=(14, fig_height_per_model * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    x = np.arange(n_labels)
    bar_width = 0.38
    left_offsets = x - bar_width / 2
    right_offsets = x + bar_width / 2

    for i, ax in enumerate(axes):
        mask_any = (counts_given[i] > 0) | (counts_kept[i] > 0)

        if not np.any(mask_any):
            ax.text(0.5, 0.5, "No data for this model", ha="center", va="center", transform=ax.transAxes)
        else:
            pos_left = left_offsets[mask_any]
            pos_right = right_offsets[mask_any]
            g_heights = means_given[i][mask_any]
            g_err = cis_given[i][mask_any]
            k_heights = means_kept[i][mask_any]
            k_err = cis_kept[i][mask_any]

            ax.bar(pos_left, g_heights, bar_width, yerr=g_err, capsize=4, label="Given", edgecolor="black")
            ax.bar(pos_right, k_heights, bar_width, yerr=k_err, capsize=4, label="Kept", edgecolor="black")

            if i == 0:
                ax.legend()

        ax.set_ylim(0, padded_max)
        ax.set_ylabel("Points")
        ax.set_title(model_names[i], loc="left", fontsize=10, style="italic")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(label_strings, ha="right", fontsize=9)

    plt.subplots_adjust(bottom=0.32, top=0.94, hspace=0.35)
    fig.suptitle("Dictator: Average Given and Kept Points For by Variant × Is Serious", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if out_prefix:
        out_file = f"{out_prefix}_given_kept.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {out_file}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot side-by-side bars for 'given' and 'kept' across multiple models.")
    parser.add_argument("--infile", help="Single input CSV (backwards compatible)")
    parser.add_argument("--model_name", help="Single model name (backwards compatible)")
    parser.add_argument("--infiles", nargs="+", help="List of input CSV files (space separated)")
    parser.add_argument("--model_names", nargs="+", help="List of model names (space separated). Must match the number of infiles.")
    parser.add_argument("--out_prefix", help="Optional prefix for saved figures (e.g. 'stacked' -> stacked_given_kept.png)", default=None)
    args = parser.parse_args()

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

    plot_given_kept_stacked(dfs, model_names, out_prefix=args.out_prefix)


if __name__ == "__main__":
    main()
