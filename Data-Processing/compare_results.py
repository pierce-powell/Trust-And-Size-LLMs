#!/usr/bin/env python3
"""
compare_results.py
Visualize cooperation probability and model payoff across variants and heuristics.

python compare_results.py --infile qwen_05B_clean.csv --model_name "Qwen2.5-0.5B" 

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_with_ci(df, metric, title, ylabel, model_name):
    """
    Plot mean metric with 95% confidence intervals (normal approximation).
    Grouped by variant × heuristic × not_gamified.
    """

    title = model_name + ": " + title

    # Compute mean, std, and count across seeds
    summary = (
        df.groupby(["variant", "heuristic", "not_gamified"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Compute standard error and 95% CI
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci95"] = 1.96 * summary["se"]

    # Make a readable label for each bar
    summary["label"] = (
        summary["variant"].astype(str) + " | "
        + summary["heuristic"].astype(str) + " | "
        + summary["not_gamified"].astype(str)
    )

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.bar(
        summary["label"],
        summary["mean"],
        yerr=summary["ci95"],
        capsize=5,
        color="skyblue",
        edgecolor="black"
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot cooperation probability and model payoff.")
    parser.add_argument("--infile", required=True, help="Path to cleaned CSV file")
    parser.add_argument("--model_name", required=True, help="model name to be appened to title")
    args = parser.parse_args()

    df = pd.read_csv(args.infile)

    # Ensure data types
    df["not_gamified"] = df["not_gamified"].astype(str)
    numeric_cols = ["coop_prob", "model_payoff"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any rows with missing metrics
    df = df.dropna(subset=numeric_cols)

    print(f"Loaded {len(df)} rows from {args.infile}")

    # Plot cooperation probability
    plot_with_ci(
        df,
        metric="coop_prob",
        title="Average Cooperation Probability by Variant × Heuristic × Is Serious",
        ylabel="Cooperation Probability", 
        model_name = args.model_name,
    )

    # Plot model payoff
    plot_with_ci(
        df,
        metric="model_payoff",
        title="Average Model Payoff by Variant × Heuristic × Is Serious",
        ylabel="Model Payoff",
        model_name = args.model_name
    )


if __name__ == "__main__":
    main()
