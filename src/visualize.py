"""Visualization script for Yelp NLP project outputs."""

from __future__ import annotations

import os
from pathlib import Path

MPL_CACHE_DIR = Path("outputs/.mplconfig")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
XDG_CACHE_DIR = Path("outputs/.cache")
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    tables_dir = Path("outputs/tables")
    figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Step 1: Load confusion matrix from classifier outputs and plot heatmaps.
    confusion_table_path = tables_dir / "confusion_matrix.csv"
    if not confusion_table_path.exists():
        raise FileNotFoundError(
            f"Confusion matrix table not found: {confusion_table_path}. "
            "Run python src/classifier.py first."
        )

    cm_df = pd.read_csv(confusion_table_path, index_col=0)
    labels = [int(c) for c in cm_df.columns]
    cm = cm_df.to_numpy()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix: Yelp Star Prediction")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    confusion_path = figures_dir / "confusion_matrix.png"
    plt.savefig(confusion_path, dpi=300)
    plt.close()
    print(f"Saved figure: {confusion_path}")

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_normalized = cm / row_sums

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0.0,
        vmax=1.0,
    )
    plt.title("Normalized Confusion Matrix (Row-wise)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    confusion_norm_path = figures_dir / "confusion_matrix_normalized.png"
    plt.savefig(confusion_norm_path, dpi=300)
    plt.close()
    print(f"Saved figure: {confusion_norm_path}")

    # Step 2: Create aspect comparison bar chart and print lift
    print("Loading aspect frequency table...")
    aspect_path = tables_dir / "aspect_frequency.csv"
    aspect_df = pd.read_csv(aspect_path)

    pivot_df = aspect_df.pivot(
        index="aspect",
        columns="split",
        values="mention_rate",
    ).fillna(0.0)

    plot_df = pivot_df.reset_index().melt(
        id_vars="aspect",
        value_vars=["positive", "negative"],
        var_name="split",
        value_name="mention_rate",
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="aspect", y="mention_rate", hue="split")
    plt.title("Aspect Mention Rate: Positive vs Negative")
    plt.xlabel("Aspect")
    plt.ylabel("Mention Rate (%)")
    plt.tight_layout()
    aspect_fig_path = figures_dir / "aspect_mention_rate.png"
    plt.savefig(aspect_fig_path, dpi=300)
    plt.close()
    print(f"Saved figure: {aspect_fig_path}")

    print("Lift (negative / positive) by aspect:")
    for aspect in pivot_df.index:
        positive_rate = pivot_df.loc[aspect, "positive"]
        negative_rate = pivot_df.loc[aspect, "negative"]
        if positive_rate > 0:
            lift = negative_rate / positive_rate
        else:
            lift = 0.0
        print(f"- {aspect}: {lift:.3f}")

    # Step 3: Create LDA topic bar charts for positive and negative splits
    print("Loading LDA topic tables...")
    lda_positive_path = tables_dir / "lda_topics_positive.csv"
    lda_negative_path = tables_dir / "lda_topics_negative.csv"
    lda_positive_df = pd.read_csv(lda_positive_path)
    lda_negative_df = pd.read_csv(lda_negative_path)

    for topic_id in sorted(lda_positive_df["topic_id"].unique()):
        topic_df = lda_positive_df[lda_positive_df["topic_id"] == topic_id].copy()
        topic_df = topic_df.sort_values("weight", ascending=True)

        plt.figure(figsize=(8, 5))
        plt.barh(topic_df["word"], topic_df["weight"], color="steelblue")
        plt.title(f"Positive Reviews - Topic {topic_id} Top Words")
        plt.xlabel("Word Weight")
        plt.ylabel("Word")
        plt.tight_layout()
        out_path = figures_dir / f"lda_positive_topic_{topic_id}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved figure: {out_path}")

    for topic_id in sorted(lda_negative_df["topic_id"].unique()):
        topic_df = lda_negative_df[lda_negative_df["topic_id"] == topic_id].copy()
        topic_df = topic_df.sort_values("weight", ascending=True)

        plt.figure(figsize=(8, 5))
        plt.barh(topic_df["word"], topic_df["weight"], color="indianred")
        plt.title(f"Negative Reviews - Topic {topic_id} Top Words")
        plt.xlabel("Word Weight")
        plt.ylabel("Word")
        plt.tight_layout()
        out_path = figures_dir / f"lda_negative_topic_{topic_id}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved figure: {out_path}")

    print("Done generating figures.")
