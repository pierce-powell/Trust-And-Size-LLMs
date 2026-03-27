#!/usr/bin/env python3
"""
Aggregated Dictator Plot by Model Size with robust parsing and CI-95 bars.

Save as: compare_results_grouped_dictator_fixed.py

Usage example:
python compare_results_grouped_dictator_fixed.py \
    --infiles qwen_05B_dictator_cleaned.csv dict_gemma_cleaned_1b.csv \
              qwen_7B_dictator_cleaned_1.csv dict_gemma_cleaned_4b.csv \
              qwen_14B_dictator_cleaned.csv dict_gemma_cleaned_12b.csv \
              qwen_32B_dictator_cleaned.csv dict_gemma_cleaned_27b.csv \
    --model_names "Qwen 0.5B" "Gemma 1B" "Qwen 7B" "Gemma 4B" \
                  "Qwen 14B" "Gemma 12B" "Qwen 32B" "Gemma 27B" \
    --group_size 2 \
    --group_labels "Small Models (0.5B+1B)" "Small-Medium (7B+4B)" \
                   "Large-Medium (14B+12B)" "Large (32B+27B)" \
    --out_prefix stacked_grouped_fixed

    python compare_results_between_dict.py --infiles qwen_05B_dictator.csv gemma_1B_dictator.csv olmo_1B_dictator.csv qwen_7B_dictator.csv gemma_4B_dictator.csv olmo_7B_dictator.csv qwen_14B_dictator.csv gemma_12B_dictator.csv olmo_13B_dictator.csv qwen_32B_dictator.csv gemma_27B_dictator.csv olmo_32B_dictator.csv --model_names "Qwen 0.5B" "Gemma 3 1B" "OLMo 2 1B" "Qwen 7B" "Gemma 3 4B" "OLMo 2 7B" "Qwen 14B" "Gemma 3 12B" "OLMo 2 13B" "Qwen 32B" "Gemma 3 27B" "OLMo 2 32B" --group_size 3 --group_labels "Small Models (Qwen 0.5B, Gemma 3 1B, OLMo 1B)" "Small-Medium Models (Qwen 7B, Gemma 3 4B, OLMo 7B)" "Large-Medium Models (Qwen 14B, Gemma 3 12B, OLMo 13B)" "Large Models (Qwen 32B, Gemma 3 27B, OLMo 32B)"
"""
import argparse
import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Helpers: normalize variant and parse isSerious robustly
# -------------------------
def parse_is_serious(val):
    """Robustly convert a not_gamified value into isSerious boolean (True/False) or np.nan if unknown."""
    # Accept real booleans
    if isinstance(val, bool):
        return val
    # Convert NaN to string
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ("true", "t", "1", "yes", "y"):
        return True
    if s in ("false", "f", "0", "no", "n"):
        return False
    # Catch some weird variants
    if s in ("none", "nan", ""):
        return np.nan
    # fallback: try to interpret numerics
    try:
        num = float(s)
        return bool(num)
    except Exception:
        return np.nan

def normalize_variant(v):
    """Turn variant text into canonical keys: 'default' or 'game_theorist'."""
    if pd.isna(v):
        return "unknown"
    s = str(v).strip().lower()
    # simplify whitespace/punctuation
    s_clean = re.sub(r"[^a-z0-9]+", " ", s).strip()
    # keywords for game_theorist
    if "game" in s_clean and "theor" in s_clean:
        return "game_theorist"
    if "game" in s_clean and "theorist" in s_clean:
        return "game_theorist"
    if "gametheorist" in s_clean.replace(" ", ""):
        return "game_theorist"
    # default synonyms
    if s_clean in ("default", "baseline", "control"):
        return "default"
    # sometimes called 'default_player' etc. fallback check
    if "default" in s_clean or "baseline" in s_clean or "control" in s_clean:
        return "default"
    # If nothing matches, return cleaned string (so grouping still possible)
    return s_clean.replace(" ", "_")

# -------------------------
# Data loading
# -------------------------
def load_and_prepare(path):
    """Load CSV and ensure required columns. Normalize variant and parse isSerious."""
    df = pd.read_csv(path)
    required_cols = ["variant", "not_gamified", "given", "kept"]
    for col in required_cols:
        if col not in df.columns:
            print(f"ERROR: Missing required column '{col}' in {path}", file=sys.stderr)
            raise SystemExit(1)
    # normalize
    df["variant_raw"] = df["variant"].astype(str)
    df["variant_norm"] = df["variant"].apply(normalize_variant)
    # not_gamified may be boolean, string, etc.
    df["not_gamified_raw"] = df["not_gamified"]
    df["isSerious"] = df["not_gamified"].apply(parse_is_serious)
    # numeric coercion
    df["given"] = pd.to_numeric(df["given"], errors="coerce")
    df["kept"] = pd.to_numeric(df["kept"], errors="coerce")
    # drop rows with both missing metrics
    df = df.dropna(subset=["given", "kept"], how="all").copy()
    return df

# -------------------------
# Compute aggregated stats per (variant_norm, isSerious)
# -------------------------
def compute_group_stats(dfs):
    """
    Combine list of DataFrames and compute mean, std, count, CI95
    grouped by (variant_norm, isSerious).
    Returns a dict keyed by (variant_norm, isSerious).
    """
    combined = pd.concat(dfs, ignore_index=True)
    # Only keep rows where isSerious is not NaN (your data said none missing, but be safe)
    # We'll still include rows with NaN if present, but they'll be grouped under (something) only if notNaN
    # For this analysis we expect isSerious to be True/False
    grouped = combined.groupby(["variant_norm", "isSerious"], dropna=False)

    stats = {}
    # We want the four canonical buckets: (default, True), (default, False), (game_theorist, True), (game_theorist, False)
    # But also compute any others present (for debugging)
    for (variant, serious), group in grouped:
        # compute given
        gseries = group["given"].dropna()
        gc = int(gseries.count())
        gm = float(gseries.mean()) if gc > 0 else 0.0
        gs = float(gseries.std(ddof=1)) if gc > 1 else 0.0
        gse = (gs / np.sqrt(gc)) if gc > 0 else 0.0
        gci = 1.96 * gse

        # compute kept
        kseries = group["kept"].dropna()
        kc = int(kseries.count())
        km = float(kseries.mean()) if kc > 0 else 0.0
        ks = float(kseries.std(ddof=1)) if kc > 1 else 0.0
        kse = (ks / np.sqrt(kc)) if kc > 0 else 0.0
        kci = 1.96 * kse

        stats[(variant, serious)] = {
            "given_mean": gm,
            "given_std": gs,
            "given_count": gc,
            "given_se": gse,
            "given_ci95": gci,
            "kept_mean": km,
            "kept_std": ks,
            "kept_count": kc,
            "kept_se": kse,
            "kept_ci95": kci,
        }
    return stats

# -------------------------
# Plotting
# -------------------------
def plot_grouped_bars(group_stats_list, group_labels, out_prefix=None):
    """
    group_stats_list: list of dicts (one per size group)
    group_labels: names for each group
    """
    n_groups = len(group_stats_list)
    fig_height_per_group = 3.5
    fig, axes = plt.subplots(n_groups, 1, figsize=(12, fig_height_per_group * n_groups), sharex=False)
    if n_groups == 1:
        axes = [axes]

    # Option 1 order:
    x_order = [
        ("default", True),
        ("default", False),
        ("game_theorist", True),
        ("game_theorist", False)
    ]
    x_labels = [f"{v} | {'Serious' if s else 'Not Serious'}" for v, s in x_order]

    # Determine global y-limit across all groups to keep consistent scaling
    candidate_max = 1.0
    for stats in group_stats_list:
        for cond in x_order:
            val = stats.get(cond)
            if val:
                candidate_max = max(candidate_max, val["given_mean"] + val["given_ci95"], val["kept_mean"] + val["kept_ci95"])
    padded_max = max(1.0, candidate_max * 1.12)

    for ax, stats, label in zip(axes, group_stats_list, group_labels):
        n_conditions = len(x_order)
        x = np.arange(n_conditions)
        width = 0.35

        given_means = [stats.get(cond, {}).get("given_mean", 0.0) for cond in x_order]
        given_cis = [stats.get(cond, {}).get("given_ci95", 0.0) for cond in x_order]
        kept_means = [stats.get(cond, {}).get("kept_mean", 0.0) for cond in x_order]
        kept_cis = [stats.get(cond, {}).get("kept_ci95", 0.0) for cond in x_order]

        # Plot bars; if a bucket had count==0, the mean+ci will be zero; we annotate below
        ax.bar(x - width/2, given_means, width, yerr=given_cis, capsize=4, label="Given", edgecolor="black")
        ax.bar(x + width/2, kept_means, width, yerr=kept_cis, capsize=4, label="Kept", edgecolor="black")

        # annotate missing buckets with faint text if counts are zero
        for idx, cond in enumerate(x_order):
            val = stats.get(cond)
            if val is None or (val.get("given_count", 0) == 0 and val.get("kept_count", 0) == 0):
                ax.text(x[idx], padded_max * 0.18, "no data", ha="center", va="center", color="gray")

        ax.set_title(label, fontsize=10, loc="left", style="italic")
        ax.set_ylabel("Points")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, ha="right")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_ylim(0, padded_max)
        if label == group_labels[0]:
            ax.legend()

    plt.subplots_adjust(bottom=0.3, top=0.94, hspace=0.35)
    fig.suptitle("Dictator: Average Given and Kept Points by Variant × isSerious", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if out_prefix:
        out_file = f"{out_prefix}_given_kept_grouped_fixed.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {out_file}")

    plt.show()

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Aggregated dictator plot by model size (robust)")
    parser.add_argument("--infile", help="Single input CSV (backwards compatible)")
    parser.add_argument("--model_name", help="Single model name (backwards compatible)")
    parser.add_argument("--infiles", nargs="+", help="List of input CSV files (space separated)")
    parser.add_argument("--model_names", nargs="+", help="List of model names (space separated)")
    parser.add_argument("--group_size", type=int, required=True, help="Number of models per size group")
    parser.add_argument("--group_labels", nargs="+", required=True, help="Label for each group")
    parser.add_argument("--out_prefix", help="Prefix for saved figure", default=None)
    args = parser.parse_args()

    # Resolve inputs
    if args.infiles:
        input_paths = args.infiles
        if not args.model_names:
            print("ERROR: --model_names required with multiple --infiles", file=sys.stderr)
            raise SystemExit(1)
        model_names = args.model_names
    elif args.infile:
        input_paths = [args.infile]
        model_names = [args.model_name or "Model"]
    else:
        parser.print_help()
        raise SystemExit(1)

    if len(input_paths) != len(model_names):
        print("ERROR: number of input files must match number of model names", file=sys.stderr)
        raise SystemExit(1)

    # Load CSVs
    dfs = []
    for p in input_paths:
        if not os.path.exists(p):
            print(f"ERROR: Input file not found: {p}", file=sys.stderr)
            raise SystemExit(1)
        df = load_and_prepare(p)
        dfs.append(df)
        print(f"Loaded {len(df)} rows from {p}")

    # Build groups
    n_per_group = args.group_size
    if len(input_paths) % n_per_group != 0:
        print("ERROR: number of input files is not divisible by group_size", file=sys.stderr)
        raise SystemExit(1)
    n_groups = len(input_paths) // n_per_group
    if len(args.group_labels) != n_groups:
        print("ERROR: number of group_labels must match number of groups", file=sys.stderr)
        raise SystemExit(1)

    group_stats_list = []
    # For debugging: print per-group condition counts
    for gi in range(n_groups):
        start = gi * n_per_group
        end = start + n_per_group
        group_dfs = dfs[start:end]
        stats = compute_group_stats(group_dfs)
        group_stats_list.append(stats)

        # Debug: print counts for canonical buckets
        canonical = [("default", True), ("default", False), ("game_theorist", True), ("game_theorist", False)]
        print(f"\nGroup {gi+1} ({args.group_labels[gi]}):", file=sys.stderr)
        for cond in canonical:
            v,s = cond
            val = stats.get(cond)
            if val:
                print(f"  {v} | {'Serious' if s else 'Not Serious'} : given_count={val['given_count']} kept_count={val['kept_count']}", file=sys.stderr)
            else:
                print(f"  {v} | {'Serious' if s else 'Not Serious'} : MISSING", file=sys.stderr)

    # Plot
    plot_grouped_bars(group_stats_list, args.group_labels, out_prefix=args.out_prefix)


if __name__ == "__main__":
    main()
