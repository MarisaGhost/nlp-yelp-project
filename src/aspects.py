"""Simple aspect-based analysis for Yelp restaurant reviews."""

from __future__ import annotations

from pathlib import Path
import importlib.util

import pandas as pd


if __name__ == "__main__":
    # Step 1: Load processed review data
    input_path = Path("data/processed/reviews_processed.csv")
    output_dir = Path("outputs/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed review data...")
    df = pd.read_csv(input_path)
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)

    # Step 2: Define aspect seed words (common restaurant terms)
    # We keep each list short and human-readable for a beginner project.
    aspect_seeds = {
        "Food": {
            "food",
            "dish",
            "meal",
            "taste",
            "flavor",
            "delicious",
            "fresh",
            "portion",
            "menu",
            "pizza",
            "burger",
            "steak",
            "sushi",
            "salad",
            "dessert",
            "breakfast",
            "lunch",
            "dinner",
            "spicy",
            "sauce",
        },
        "Service": {
            "service",
            "staff",
            "server",
            "waiter",
            "waitress",
            "manager",
            "host",
            "friendly",
            "rude",
            "slow",
            "fast",
            "helpful",
            "attentive",
            "professional",
            "polite",
            "courteous",
            "order",
            "reservation",
            "customer",
            "experience",
        },
        "Price": {
            "price",
            "prices",
            "cost",
            "value",
            "cheap",
            "expensive",
            "affordable",
            "overpriced",
            "deal",
            "special",
            "bill",
            "tip",
            "portion",
            "worth",
            "reasonable",
            "money",
            "dollar",
            "bucks",
            "paid",
            "charge",
        },
    }

    # Step 3: Split reviews into positive and negative groups
    positive_df = df[df["stars"].isin([4, 5])].copy()
    negative_df = df[df["stars"].isin([1, 2])].copy()
    split_data = {"positive": positive_df, "negative": negative_df}

    print(f"Positive reviews: {len(positive_df):,}")
    print(f"Negative reviews: {len(negative_df):,}")

    # Step 4: Compute mention rate and total mentions for each aspect and split
    frequency_rows = []

    for split_name, split_df in split_data.items():
        total_reviews = len(split_df)
        print(f"\nAnalyzing {split_name} reviews...")

        for aspect_name, seed_words in aspect_seeds.items():
            reviews_with_aspect = 0
            total_mentions = 0

            for text in split_df["cleaned_text"]:
                tokens = text.split()
                mention_count = sum(token in seed_words for token in tokens)
                if mention_count > 0:
                    reviews_with_aspect += 1
                    total_mentions += mention_count

            if total_reviews > 0:
                mention_rate = 100.0 * reviews_with_aspect / total_reviews
            else:
                mention_rate = 0.0

            frequency_rows.append(
                {
                    "split": split_name,
                    "aspect": aspect_name,
                    "mention_rate": mention_rate,
                    "total_mentions": total_mentions,
                }
            )

            print(
                f"{aspect_name}: mentioned in {mention_rate:.1f}% of {split_name} reviews "
                f"({total_mentions:,} total keyword matches)."
            )

    frequency_df = pd.DataFrame(frequency_rows)
    frequency_path = output_dir / "aspect_frequency.csv"
    frequency_df.to_csv(frequency_path, index=False)
    print(f"\nSaved aspect frequency table: {frequency_path}")

    # Step 5 (Optional): VADER sentiment for reviews that mention each aspect
    # This block runs only when nltk is installed.
    if importlib.util.find_spec("nltk") is not None:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer

        print("\nNLTK detected. Running optional VADER sentiment analysis...")
        nltk.download("vader_lexicon", quiet=True)
        sid = SentimentIntensityAnalyzer()

        sentiment_rows = []

        for split_name, split_df in split_data.items():
            for aspect_name, seed_words in aspect_seeds.items():
                scores = []

                for text in split_df["cleaned_text"]:
                    tokens = text.split()
                    if any(token in seed_words for token in tokens):
                        compound = sid.polarity_scores(text)["compound"]
                        scores.append(compound)

                if len(scores) > 0:
                    mean_compound = float(pd.Series(scores).mean())
                else:
                    mean_compound = 0.0

                sentiment_rows.append(
                    {
                        "split": split_name,
                        "aspect": aspect_name,
                        "mean_compound": mean_compound,
                        "num_reviews_with_aspect": len(scores),
                    }
                )

        sentiment_df = pd.DataFrame(sentiment_rows)
        sentiment_path = output_dir / "aspect_sentiment.csv"
        sentiment_df.to_csv(sentiment_path, index=False)
        print(f"Saved optional aspect sentiment table: {sentiment_path}")
    else:
        print("\nNLTK not found, so optional VADER sentiment table was skipped.")

    # Step 6: Print a short plain-language summary
  # Step 6: Print a more meaningful summary using relative comparison
    print("\nKey findings (relative comparison):")

    pivot = frequency_df.pivot(
        index="aspect",
        columns="split",
        values="mention_rate"
    ).reset_index()

    pivot["lift_negative_over_positive"] = pivot["negative"] / (pivot["positive"] + 1e-9)

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

    top = pivot.iloc[0]
    print(
        f"\nAspect most over-represented in negative reviews: {top['aspect']} "
        f"(negative {top['negative']:.1f}% vs positive {top['positive']:.1f}%, "
        f"lift={top['lift_negative_over_positive']:.2f}x)."
    )

    print(
        "\nInterpretation: Food is frequently mentioned in almost all restaurant reviews "
        "and therefore acts as a baseline aspect. In contrast, Service and Price are "
        "relatively more prominent in negative reviews, suggesting they are more likely "
        "drivers of customer dissatisfaction."
    )
