"""Simple visualization script for Yelp NLP project outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # Step 0: Set paths and make sure output folder exists
    data_path = Path("data/processed/reviews_processed.csv")
    tables_dir = Path("outputs/tables")
    figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # Step 1: Build classifier again and save confusion matrix heatmap
    print("Loading processed data for confusion matrix...")
    df = pd.read_csv(data_path)
    df = df[["cleaned_text", "stars"]].copy()
    df = df.dropna(subset=["cleaned_text", "stars"])
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.dropna(subset=["stars"])
    df["stars"] = df["stars"].astype(int)
    df = df[df["stars"].isin([1, 2, 3, 4, 5])]
    df["cleaned_text"] = df["cleaned_text"].astype(str)

    X = df["cleaned_text"]
    y = df["stars"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[1, 2, 3, 4, 5],
        yticklabels=[1, 2, 3, 4, 5],
    )
    plt.title("Confusion Matrix: Yelp Star Prediction")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    confusion_path = figures_dir / "confusion_matrix.png"
    plt.savefig(confusion_path, dpi=300)
    plt.close()
    print(f"Saved figure: {confusion_path}")

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
