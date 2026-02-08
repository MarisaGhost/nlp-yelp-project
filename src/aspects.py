"""Aspect-based analysis with lift and statistical significance."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

EPSILON = 1e-9


def _count_mentions(tokens: list[str], seed_words: set[str]) -> int:
    return sum(token in seed_words for token in tokens)


if __name__ == "__main__":
    input_path = Path("data/processed/reviews_processed.csv")
    output_dir = Path("outputs/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed review data...")
    df = pd.read_csv(input_path, usecols=["stars", "text", "cleaned_text"])
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.dropna(subset=["stars"]).copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
    df["tokens"] = df["cleaned_text"].str.split()

    aspect_seeds = {
        "Food": {
            "food", "dish", "meal", "taste", "flavor", "delicious", "fresh", "portion",
            "menu", "pizza", "burger", "steak", "sushi", "salad", "dessert", "breakfast",
            "lunch", "dinner", "spicy", "sauce",
        },
        "Service": {
            "service", "staff", "server", "waiter", "waitress", "manager", "host",
            "friendly", "rude", "slow", "fast", "helpful", "attentive", "professional",
            "polite", "courteous", "order", "reservation", "customer", "experience",
        },
        "Price": {
            "price", "prices", "cost", "value", "cheap", "expensive", "affordable",
            "overpriced", "deal", "special", "bill", "tip", "portion", "worth",
            "reasonable", "money", "dollar", "bucks", "paid", "charge",
        },
    }

    positive_df = df[df["stars"].isin([4, 5])].copy()
    negative_df = df[df["stars"].isin([1, 2])].copy()
    split_data = {"positive": positive_df, "negative": negative_df}

    print(f"Positive reviews: {len(positive_df):,}")
    print(f"Negative reviews: {len(negative_df):,}")

    frequency_rows = []
    aspect_flags = {"positive": {}, "negative": {}}

    for split_name, split_df in split_data.items():
        total_reviews = len(split_df)
        print(f"\nAnalyzing {split_name} reviews...")

        for aspect_name, seed_words in aspect_seeds.items():
            mention_counts = split_df["tokens"].apply(lambda tokens: _count_mentions(tokens, seed_words))
            review_has_aspect = mention_counts > 0
            reviews_with_aspect = int(review_has_aspect.sum())
            total_mentions = int(mention_counts.sum())
            mention_rate = 100.0 * reviews_with_aspect / total_reviews if total_reviews else 0.0
            mention_density = total_mentions / max(total_reviews, 1)

            aspect_flags[split_name][aspect_name] = review_has_aspect
            frequency_rows.append(
                {
                    "split": split_name,
                    "aspect": aspect_name,
                    "total_reviews": total_reviews,
                    "reviews_with_aspect": reviews_with_aspect,
                    "mention_rate": mention_rate,
                    "total_mentions": total_mentions,
                    "mention_density_per_review": mention_density,
                }
            )
            print(
                f"{aspect_name}: {mention_rate:.1f}% of {split_name} reviews "
                f"({reviews_with_aspect:,}/{total_reviews:,}), {total_mentions:,} total mentions."
            )

    frequency_df = pd.DataFrame(frequency_rows)
    frequency_path = output_dir / "aspect_frequency.csv"
    frequency_df.to_csv(frequency_path, index=False)
    print(f"\nSaved aspect frequency table: {frequency_path}")

    significance_rows = []
    for aspect_name in aspect_seeds:
        pos_flags = aspect_flags["positive"][aspect_name]
        neg_flags = aspect_flags["negative"][aspect_name]
        pos_total = len(pos_flags)
        neg_total = len(neg_flags)
        pos_with = int(pos_flags.sum())
        neg_with = int(neg_flags.sum())
        pos_without = pos_total - pos_with
        neg_without = neg_total - neg_with

        contingency = np.array(
            [[neg_with, neg_without], [pos_with, pos_without]],
            dtype=float,
        )
        chi2_stat, p_value, _, _ = chi2_contingency(contingency, correction=False)
        neg_rate = neg_with / max(neg_total, 1)
        pos_rate = pos_with / max(pos_total, 1)
        lift = neg_rate / (pos_rate + EPSILON)
        odds_ratio = ((neg_with + EPSILON) * (pos_without + EPSILON)) / (
            (neg_without + EPSILON) * (pos_with + EPSILON)
        )

        significance_rows.append(
            {
                "aspect": aspect_name,
                "negative_with_aspect": neg_with,
                "negative_total": neg_total,
                "positive_with_aspect": pos_with,
                "positive_total": pos_total,
                "negative_rate": neg_rate,
                "positive_rate": pos_rate,
                "lift_negative_over_positive": lift,
                "odds_ratio_negative_vs_positive": odds_ratio,
                "chi2_statistic": chi2_stat,
                "p_value": p_value,
                "is_significant_at_0_05": p_value < 0.05,
            }
        )

    significance_df = pd.DataFrame(significance_rows).sort_values(
        "lift_negative_over_positive",
        ascending=False,
    )
    significance_path = output_dir / "aspect_significance.csv"
    significance_df.to_csv(significance_path, index=False)
    print(f"Saved aspect significance table: {significance_path}")

    # Optional VADER sentiment using raw text for better polarity signal.
    if importlib.util.find_spec("nltk") is not None:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer

        print("\nNLTK detected. Running optional VADER sentiment analysis...")
        sid = None
        try:
            nltk.download("vader_lexicon", quiet=True)
            sid = SentimentIntensityAnalyzer()
        except LookupError:
            print("VADER lexicon unavailable in this environment; skipping sentiment table.")
        sentiment_rows = []

        if sid is not None:
            for split_name, split_df in split_data.items():
                for aspect_name, seed_words in aspect_seeds.items():
                    scores = []
                    for _, row in split_df.iterrows():
                        tokens = row["tokens"]
                        if any(token in seed_words for token in tokens):
                            compound = sid.polarity_scores(row["text"])["compound"]
                            scores.append(compound)

                    sentiment_rows.append(
                        {
                            "split": split_name,
                            "aspect": aspect_name,
                            "mean_compound": float(pd.Series(scores).mean()) if scores else 0.0,
                            "num_reviews_with_aspect": len(scores),
                        }
                    )

            sentiment_df = pd.DataFrame(sentiment_rows)
            sentiment_path = output_dir / "aspect_sentiment.csv"
            sentiment_df.to_csv(sentiment_path, index=False)
            print(f"Saved optional aspect sentiment table: {sentiment_path}")
    else:
        print("\nNLTK not found, so optional VADER sentiment table was skipped.")

    print("\nKey findings (relative comparison + significance):")
    pivot = frequency_df.pivot(index="aspect", columns="split", values="mention_rate").reset_index()
    pivot["lift_negative_over_positive"] = pivot["negative"] / (pivot["positive"] + EPSILON)
    pivot = pivot.sort_values("lift_negative_over_positive", ascending=False)

    print("\nAspect mention rates (% of reviews mentioning seed words):")
    print(
        pivot.to_string(
            index=False,
            formatters={
                "positive": lambda x: f"{x:.1f}",
                "negative": lambda x: f"{x:.1f}",
                "lift_negative_over_positive": lambda x: f"{x:.2f}",
            },
        )
    )

    print("\nStatistical test (chi-square) summary:")
    print(
        significance_df[["aspect", "lift_negative_over_positive", "odds_ratio_negative_vs_positive", "p_value"]]
        .to_string(
            index=False,
            formatters={
                "lift_negative_over_positive": lambda x: f"{x:.2f}",
                "odds_ratio_negative_vs_positive": lambda x: f"{x:.2f}",
                "p_value": lambda x: f"{x:.3e}",
            },
        )
    )
