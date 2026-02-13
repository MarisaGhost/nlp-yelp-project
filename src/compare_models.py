"""Beginner-friendly model comparison using one shared TF-IDF setup.

This script compares three models on the same validation split using only
accuracy, then does a tiny parameter sweep for the best model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier

RANDOM_STATE = 42
STAR_LABELS = [1, 2, 3, 4, 5]
DATA_CANDIDATES = [
    Path("data/processed/reviews_processed.csv"),
    Path("data/processed/merged_reviews.csv"),
]


def resolve_input_path() -> Path:
    """Use the first existing dataset path that matches current project files."""
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No dataset file found. Expected one of: "
        + ", ".join(str(path) for path in DATA_CANDIDATES)
    )


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load and clean label/text columns to match the current rating setup."""
    df = pd.read_csv(input_path, usecols=["text", "stars"])
    df = df.dropna(subset=["text", "stars"]).copy()
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.dropna(subset=["stars"])
    df["stars"] = df["stars"].astype(int)
    df = df[df["stars"].isin(STAR_LABELS)].copy()
    df["text"] = (
        df["text"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df[df["text"] != ""].copy()
    return df


def build_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert stars to binary labels and exclude 3-star neutral reviews.

    label=0: 1-2 stars (negative)
    label=1: 4-5 stars (positive)
    """
    binary_df = df[df["stars"] != 3].copy()
    binary_df["label"] = (binary_df["stars"] >= 4).astype(int)
    return binary_df


def build_shared_features(
    train_text: pd.Series,
    valid_text: pd.Series,
) -> tuple[Any, Any]:
    """Build one shared TF-IDF representation for all compared models.

    Leakage prevention:
    - Fit TF-IDF only on training text.
    - Transform validation text with the fitted vectorizer.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        lowercase=True,
    )
    X_train = vectorizer.fit_transform(train_text)
    X_valid = vectorizer.transform(valid_text)
    return X_train, X_valid


def _fit_predict_knn(
    X_train: Any,
    y_train: pd.Series,
    X_valid: Any,
    *,
    n_neighbors: int,
) -> tuple[pd.Series, str]:
    """Fit KNN with cosine metric if available, else fall back to default metric."""
    try:
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = pd.Series(model.predict(X_valid))
        return preds, "metric=cosine"
    except Exception as exc:
        fallback = KNeighborsClassifier(n_neighbors=n_neighbors)
        fallback.fit(X_train, y_train)
        preds = pd.Series(fallback.predict(X_valid))
        return preds, f"metric=default (cosine unsupported: {exc.__class__.__name__})"


def compare_models(
    X_train: Any,
    y_train: pd.Series,
    X_valid: Any,
    y_valid: pd.Series,
) -> pd.DataFrame:
    """Compare SGD, Naive Bayes, and KNN using validation accuracy only."""
    rows: list[dict[str, Any]] = []

    sgd = SGDClassifier(
        random_state=RANDOM_STATE,
        max_iter=1000,
        tol=1e-3,
    )
    sgd.fit(X_train, y_train)
    sgd_preds = pd.Series(sgd.predict(X_valid))
    rows.append(
        {
            "model_name": "sgd",
            "accuracy": accuracy_score(y_valid, sgd_preds),
            "notes": "default SGDClassifier",
        }
    )

    nb = ComplementNB()
    nb.fit(X_train, y_train)
    nb_preds = pd.Series(nb.predict(X_valid))
    rows.append(
        {
            "model_name": "naive_bayes",
            "accuracy": accuracy_score(y_valid, nb_preds),
            "notes": "ComplementNB",
        }
    )

    knn_preds, knn_note = _fit_predict_knn(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        n_neighbors=5,
    )
    rows.append(
        {
            "model_name": "knn",
            "accuracy": accuracy_score(y_valid, knn_preds),
            "notes": f"KNeighborsClassifier(n_neighbors=5, {knn_note})",
        }
    )

    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def tune_best_model(
    best_model_name: str,
    X_train: Any,
    y_train: pd.Series,
    X_valid: Any,
    y_valid: pd.Series,
) -> dict[str, Any]:
    """Run a tiny, model-specific parameter sweep for the best model only."""
    best_accuracy = -1.0
    best_params: dict[str, Any] = {}
    best_notes = ""

    if best_model_name == "sgd":
        for loss in ["hinge", "log_loss"]:
            for alpha in [1e-4, 1e-3, 1e-2]:
                model = SGDClassifier(
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                    tol=1e-3,
                    loss=loss,
                    alpha=alpha,
                )
                model.fit(X_train, y_train)
                preds = pd.Series(model.predict(X_valid))
                score = accuracy_score(y_valid, preds)
                if score > best_accuracy:
                    best_accuracy = score
                    best_params = {"loss": loss, "alpha": alpha}
                    best_notes = "tiny grid over loss + alpha"

    elif best_model_name == "naive_bayes":
        for alpha in [0.1, 0.5, 1.0]:
            model = ComplementNB(alpha=alpha)
            model.fit(X_train, y_train)
            preds = pd.Series(model.predict(X_valid))
            score = accuracy_score(y_valid, preds)
            if score > best_accuracy:
                best_accuracy = score
                best_params = {"alpha": alpha}
                best_notes = "tiny grid over alpha (ComplementNB)"

    elif best_model_name == "knn":
        for n_neighbors in [3, 5, 7, 9]:
            preds, note = _fit_predict_knn(
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid,
                n_neighbors=n_neighbors,
            )
            score = accuracy_score(y_valid, preds)
            if score > best_accuracy:
                best_accuracy = score
                best_params = {"n_neighbors": n_neighbors}
                best_notes = f"tiny grid over n_neighbors ({note})"
    else:
        raise ValueError(f"Unsupported model for tuning: {best_model_name}")

    return {
        "model_name": best_model_name,
        "validation_accuracy": best_accuracy,
        "best_params": json.dumps(best_params, sort_keys=True),
        "notes": best_notes,
    }


def main() -> None:
    input_path = resolve_input_path()
    output_dir = Path("outputs/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset(input_path)
    df = build_binary_labels(raw_df)
    X = df["text"]
    y = df["label"]

    X_train_text, X_valid_text, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_valid = build_shared_features(X_train_text, X_valid_text)
    comparison_df = compare_models(X_train, y_train, X_valid, y_valid)
    comparison_path = output_dir / "model_accuracy.csv"
    comparison_df.to_csv(comparison_path, index=False)

    best_model_name = str(comparison_df.iloc[0]["model_name"])
    tuning_result = tune_best_model(best_model_name, X_train, y_train, X_valid, y_valid)
    tuning_df = pd.DataFrame([tuning_result])
    tuning_path = output_dir / "best_model_tuning.csv"
    tuning_df.to_csv(tuning_path, index=False)

    print(f"Dataset used: {input_path}")
    print("Task: binary sentiment-style classification (1-2 vs 4-5 stars)")
    print("3-star reviews are excluded from this comparison.")
    print(f"Rows used after excluding 3-star: {len(df):,}")
    print("\nValidation accuracy comparison:")
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nBest model after tiny tuning:")
    print(
        f"- model={tuning_result['model_name']}, "
        f"accuracy={tuning_result['validation_accuracy']:.4f}, "
        f"best_params={tuning_result['best_params']}"
    )
    print("\nSaved files:")
    print(f"- {comparison_path}")
    print(f"- {tuning_path}")


if __name__ == "__main__":
    main()
