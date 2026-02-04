"""Simple text classification script for predicting Yelp stars (1-5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # Step 1: Load the processed dataset
    input_path = Path("data/processed/reviews_processed.csv")
    output_dir = Path("outputs/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(input_path)

    # Keep only the columns we need and remove rows with missing values.
    df = df[["cleaned_text", "stars"]].copy()
    df = df.dropna(subset=["cleaned_text", "stars"])

    # Step 2: Build X and y
    # X = cleaned review text, y = integer star label (1 to 5).
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.dropna(subset=["stars"])
    df["stars"] = df["stars"].astype(int)
    df = df[df["stars"].isin([1, 2, 3, 4, 5])]
    df["cleaned_text"] = df["cleaned_text"].astype(str)

    X = df["cleaned_text"]
    y = df["stars"]

    print(f"Total rows used: {len(df):,}")
    print("Class counts:")
    print(y.value_counts().sort_index())

    # Step 3: Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Train size: {len(X_train):,}")
    print(f"Test size: {len(X_test):,}")

    # Step 4: TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF feature size: {X_train_tfidf.shape[1]:,}")

    # Step 5: Train Logistic Regression classifier
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train_tfidf, y_train)

    # Step 6: Evaluate model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # Majority-class baseline: always predict the most frequent training class.
    majority_class = int(y_train.value_counts().idxmax())
    y_baseline = np.full(shape=len(y_test), fill_value=majority_class)
    baseline_accuracy = accuracy_score(y_test, y_baseline)

    print("\nModel performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Majority baseline accuracy: {baseline_accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Step 7: Save metrics table
    metrics_df = pd.DataFrame(
        [
            {"metric": "accuracy", "value": accuracy},
            {"metric": "macro_f1", "value": macro_f1},
            {"metric": "majority_baseline_accuracy", "value": baseline_accuracy},
            {"metric": "majority_class_from_train", "value": majority_class},
        ]
    )
    metrics_path = output_dir / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics: {metrics_path}")

    # Step 8: Save top words comparing class 5 vs class 1
    # score = coef_for_5 - coef_for_1
    class_list = list(model.classes_)
    if 1 not in class_list or 5 not in class_list:
        raise ValueError("Classes 1 and 5 must both exist to compute top feature scores.")

    idx_1 = class_list.index(1)
    idx_5 = class_list.index(5)
    scores = model.coef_[idx_5] - model.coef_[idx_1]
    words = vectorizer.get_feature_names_out()

    sorted_idx = np.argsort(scores)
    top_negative_idx = sorted_idx[:20]
    top_positive_idx = sorted_idx[-20:][::-1]

    rows = []
    for rank, i in enumerate(top_positive_idx, start=1):
        rows.append(
            {
                "group": "top_positive_words_for_class_5",
                "rank": rank,
                "word": words[i],
                "score": scores[i],
            }
        )
    for rank, i in enumerate(top_negative_idx, start=1):
        rows.append(
            {
                "group": "top_negative_words_for_class_5",
                "rank": rank,
                "word": words[i],
                "score": scores[i],
            }
        )

    top_features_df = pd.DataFrame(rows)
    top_features_path = output_dir / "top_features.csv"
    top_features_df.to_csv(top_features_path, index=False)
    print(f"Saved top feature words: {top_features_path}")

    print("\nDone.")
