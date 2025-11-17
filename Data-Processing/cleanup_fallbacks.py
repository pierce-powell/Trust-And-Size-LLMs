#!/usr/bin/env python3
"""
cleanup_fallbacks.py

- Removes unwanted variants (coa, coa_notgamified).
- Removes duplicate condition rows before other cleaning.
- Reads CSV into a pandas DataFrame.
- Detects rows where the model failed to write proper JSON (fallback in model_reason).
- Removes fallback rows and writes a cleaned CSV.
- Normalizes variant names by removing '_notgamified' or '_not_gamified'.
- Recomputes coop_prob per group including the current round.
- Writes removed fallback rows for inspection.

Usage:
    python cleanup_fallbacks.py --infile input.csv --outfile output.csv
"""

import argparse
import os
import sys
import re
from typing import List

import pandas as pd
import numpy as np

DEFAULT_GROUP_COLS = ['seed', 'model', 'variant', 'heuristic', 'not_gamified']
FALLBACK_REGEX = re.compile(r'\bfallback\b', flags=re.IGNORECASE)

# ------------------------ Utility Functions ------------------------ #

def clean_variant_name(variant: str) -> str:
    """Removes '_notgamified' or '_not_gamified' suffixes."""
    if isinstance(variant, str):
        return variant.replace("_notgamified", "").replace("_not_gamified", "").strip()
    return variant

def detect_fallback_rows(df: pd.DataFrame, reason_col: str = 'model_reason') -> pd.Series:
    """Return boolean mask marking rows where model_reason indicates fallback."""
    if reason_col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[reason_col].fillna('').astype(str).str.contains(FALLBACK_REGEX)

def recompute_coop_prob_by_history(df_clean: pd.DataFrame,
                                   group_cols: List[str],
                                   round_col: str = 'round',
                                   choice_col: str = 'model_choice',
                                   coop_prob_col: str = 'coop_prob') -> pd.DataFrame:
    df = df_clean.copy()
    df[round_col] = pd.to_numeric(df[round_col], errors='coerce')
    df[choice_col] = df[choice_col].astype(str).str.strip().str.upper()

    recomputed = pd.Series(index=df.index, dtype=float)
    grouped = df.groupby(group_cols, dropna=False, sort=False)

    for _, group in grouped:
        g = group.sort_values(by=round_col, kind='mergesort')
        is_c = g[choice_col] == 'C'
        cumsum_c = is_c.cumsum()
        cumsum_n = pd.Series(range(1, len(g) + 1), index=g.index)
        recomputed.loc[g.index] = cumsum_c / cumsum_n

    df[coop_prob_col] = recomputed
    return df

# ------------------------ Main Function ------------------------ #

def main(argv):
    parser = argparse.ArgumentParser(description="Clean up fallback rows and recompute coop_prob.")
    parser.add_argument("--infile", "-i", required=True, help="Input CSV file")
    parser.add_argument("--outfile", "-o", default=None, help="Output cleaned CSV file")
    parser.add_argument("--group-cols", "-g", nargs='*', default=DEFAULT_GROUP_COLS,
                        help=f"Grouping columns for 'same conditions' (default: {DEFAULT_GROUP_COLS})")
    parser.add_argument("--reason-col", default='model_reason', help="Column containing model reason")
    parser.add_argument("--choice-col", default='model_choice', help="Column containing model choice ('C'/'D')")
    parser.add_argument("--round-col", default='round', help="Column indicating round number")
    parser.add_argument("--coop-prob-col", default='coop_prob', help="Column storing cooperation probability")
    parser.add_argument("--backup", action='store_true', help="Backup original CSV")
    args = parser.parse_args(argv)

    infile = args.infile
    outfile = args.outfile or os.path.splitext(infile)[0] + ".cleaned.csv"

    if not os.path.exists(infile):
        print(f"ERROR: input file '{infile}' does not exist.", file=sys.stderr)
        sys.exit(2)

    if args.backup:
        import shutil
        backup_path = infile + ".bak"
        shutil.copy2(infile, backup_path)
        print(f"Backup created at {backup_path}")

    # ------------------ Read CSV ------------------ #
    df = pd.read_csv(infile, dtype=str)
    print(f"Loaded {len(df):,} rows")

    # -------- NEW: Remove unwanted variants and duplicate condition rows -------- #

    # Remove unwanted variants
    bad_variants = {"coa", "coa_notgamified"}
    before = len(df)
    df = df[~df["variant"].isin(bad_variants)]
    print(f"Removed {before - len(df):,} rows with bad variants: {bad_variants}")

    # Remove duplicate condition rows BEFORE all other cleaning
    condition_cols = ["seed", "variant", "heuristic", "round", "not_gamified"]
    before = len(df)
    df = df.drop_duplicates(subset=condition_cols, keep="first")
    print(f"Removed {before - len(df):,} duplicate condition rows")
    print(f"Rows after deduplication: {len(df):,}")

    # --------------------------------------------------------------------------- #

    # Normalize variants
    df['variant'] = df['variant'].apply(clean_variant_name)

    # Ensure numeric columns
    df[args.round_col] = pd.to_numeric(df[args.round_col], errors='coerce')
    if args.coop_prob_col not in df.columns:
        df[args.coop_prob_col] = np.nan
    else:
        df[args.coop_prob_col] = pd.to_numeric(df[args.coop_prob_col], errors='coerce')

    # ------------------ Detect fallback rows ------------------ #
    fallback_mask = detect_fallback_rows(df, reason_col=args.reason_col)
    num_failures = int(fallback_mask.sum())
    print(f"Detected fallback rows: {num_failures}")

    df_clean = df.loc[~fallback_mask].copy()
    removed_df = df.loc[fallback_mask].copy()

    # ------------------ Recompute coop_prob ------------------ #
    group_cols = [c for c in args.group_cols if c in df_clean.columns]
    df_clean = recompute_coop_prob_by_history(df_clean,
                                              group_cols=group_cols,
                                              round_col=args.round_col,
                                              choice_col=args.choice_col,
                                              coop_prob_col=args.coop_prob_col)

    # ------------------ Write outputs ------------------ #
    df_clean.to_csv(outfile, index=False)
    print(f"Cleaned CSV written to: {outfile}")

    if num_failures > 0:
        removed_out = os.path.splitext(outfile)[0] + ".removed_rows.csv"
        removed_df.to_csv(removed_out, index=False)
        print(f"Removed fallback rows written to: {removed_out}")

    print(f"Summary: total_rows={len(df):,}, removed={num_failures}, remaining={len(df_clean):,}")

# ------------------------ Entry Point ------------------------ #

if __name__ == "__main__":
    main(sys.argv[1:])
