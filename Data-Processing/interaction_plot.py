"""
Create interaction plot for each family.

Usage examples:
    python interaction_plot.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Setting global variables
plt.rcParams.update({"font.size": 14})
df = pd.read_csv("Gemma_Averages.csv") # Change this to match statistics file
heuristic_order = ["AlwaysCooperate", "AlwaysDefect", "Tit4Tat"]

# Converting titles for images
metric_titles = {
    "coop_prob": "Mean Cooperation Probability",
    "model_payoff": "Mean Payoff"
}

for metric in df["metric"].unique():
    metric_df = df[df["metric"] == metric].copy()

    # Organize by heuristic
    metric_df["heuristic"] = pd.Categorical(
        metric_df["heuristic"],
        categories=heuristic_order,
        ordered=True
    )

    models = sorted(metric_df["model"].unique())
    variants = sorted(metric_df["variant"].unique())
    gamified_states = sorted(metric_df["not_gamified"].unique())

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    markers = ['o', 's', '^', 'D', 'P', 'X']

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(heuristic_order))

    total_groups = len(models) * len(variants) * len(gamified_states)
    width = 0.8 / total_groups

    idx = 0

    for m, model in enumerate(models):
        for v, variant in enumerate(variants):
            for g, gamified in enumerate(gamified_states):

                subset = metric_df[
                    (metric_df["model"] == model) &
                    (metric_df["variant"] == variant) &
                    (metric_df["not_gamified"] == gamified)
                ].sort_values("heuristic")

            offsets = x + (idx - total_groups / 2) * width

            ax.errorbar(
                offsets,
                subset["mean"],
                yerr=subset["ci95"],
                fmt=markers[v % len(markers)],
                color=colors[m],
                capsize=3,
                markersize=7,
                linestyle='none',
                label=f"{model} | {variant}"
            )

            idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels(heuristic_order, fontsize=14)
    ax.set_xlabel("Opponent Heuristic", fontsize=14)
    ax.set_ylabel("Mean", fontsize=14)
    ax.set_title(metric_titles.get(metric, metric), fontsize=14)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=14
    )

    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()