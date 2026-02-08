"""Improved text classification for predicting Yelp stars (1-5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
LABELS = [1, 2, 3, 4, 5]


def _prepare_dataset(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, usecols=["text", "stars"])
    df = df.dropna(subset=["text", "stars"]).copy()
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.dropna(subset=["stars"])
    df["stars"] = df["stars"].astype(int)
    df = df[df["stars"].isin(LABELS)].copy()
    df["text"] = (
        df["text"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df[df["text"] != ""].copy()
    return df


def _build_feature_matrices(
    train_text: pd.Series,
    test_text: pd.Series,
    *,
    word_max_features: int,
    char_max_features: int,
    char_ngram_range: tuple[int, int],
    min_df: int,
):
    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=word_max_features,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.98,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram_range,
        min_df=5,
        max_features=char_max_features,
        sublinear_tf=True,
    )

    train_word = word_vectorizer.fit_transform(train_text)
    test_word = word_vectorizer.transform(test_text)
    train_char = char_vectorizer.fit_transform(train_text)
    test_char = char_vectorizer.transform(test_text)

    train_matrix = hstack([train_word, train_char]).tocsr()
    test_matrix = hstack([test_word, test_char]).tocsr()
    return train_matrix, test_matrix, word_vectorizer, char_vectorizer


if __name__ == "__main__":
    input_path = Path("data/processed/reviews_processed.csv")
    output_dir = Path("outputs/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = _prepare_dataset(input_path)
    X = df["text"]
    y = df["stars"]

    print(f"Total rows used: {len(df):,}")
    print("Class counts:")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train size: {len(X_train):,}")
    print(f"Test size: {len(X_test):,}")

    # Hold out a small validation split from training data for model selection.
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    candidate_models = [
        {
            "name": "sgd_log_loss_balanced_alpha_3e-5_char35",
            "model": {
                "loss": "log_loss",
                "alpha": 3e-5,
                "class_weight": "balanced",
            },
            "features": {
                "word_max_features": 100_000,
                "char_max_features": 120_000,
                "char_ngram_range": (3, 5),
                "min_df": 3,
            },
        },
        {
            "name": "sgd_log_loss_balanced_alpha_4e-5_char25",
            "model": {
                "loss": "log_loss",
                "alpha": 4e-5,
                "class_weight": "balanced",
            },
            "features": {
                "word_max_features": 120_000,
                "char_max_features": 150_000,
                "char_ngram_range": (2, 5),
                "min_df": 3,
            },
        },
        {
            "name": "sgd_log_loss_unweighted_alpha_3e-5_char35",
            "model": {
                "loss": "log_loss",
                "alpha": 3e-5,
                "class_weight": None,
            },
            "features": {
                "word_max_features": 100_000,
                "char_max_features": 120_000,
                "char_ngram_range": (3, 5),
                "min_df": 3,
            },
        },
        {
            "name": "sgd_modified_huber_balanced_alpha_1e-5_char35",
            "model": {
                "loss": "modified_huber",
                "alpha": 1e-5,
                "class_weight": "balanced",
            },
            "features": {
                "word_max_features": 100_000,
                "char_max_features": 120_000,
                "char_ngram_range": (3, 5),
                "min_df": 3,
            },
        },
    ]

    selection_rows = []
    best_model_name = ""
    best_model_params = {}
    best_feature_params = {}
    best_val_f1 = -1.0

    print("\nSelecting model on validation split...")
    for candidate in candidate_models:
        model_name = candidate["name"]
        model_params = candidate["model"]
        feature_params = candidate["features"]
        X_subtrain_mat, X_val_mat, _, _ = _build_feature_matrices(
            X_subtrain,
            X_val,
            **feature_params,
        )
        clf = SGDClassifier(
            loss=model_params["loss"],
            alpha=model_params["alpha"],
            penalty="l2",
            class_weight=model_params["class_weight"],
            max_iter=3000,
            tol=1e-3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        clf.fit(X_subtrain_mat, y_subtrain)
        y_val_pred = clf.predict(X_val_mat)
        val_macro_f1 = f1_score(y_val, y_val_pred, average="macro")
        val_accuracy = accuracy_score(y_val, y_val_pred)

        selection_rows.append(
            {
                "model_name": model_name,
                "validation_accuracy": val_accuracy,
                "validation_macro_f1": val_macro_f1,
                "word_max_features": feature_params["word_max_features"],
                "char_max_features": feature_params["char_max_features"],
                "char_ngram_range": str(feature_params["char_ngram_range"]),
                "min_df": feature_params["min_df"],
                "loss": model_params["loss"],
                "alpha": model_params["alpha"],
                "class_weight": str(model_params["class_weight"]),
            }
        )
        print(
            f"- {model_name}: val_accuracy={val_accuracy:.4f}, "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_model_name = model_name
            best_model_params = model_params
            best_feature_params = feature_params

    selection_df = pd.DataFrame(selection_rows).sort_values(
        "validation_macro_f1",
        ascending=False,
    )
    selection_path = output_dir / "model_selection.csv"
    selection_df.to_csv(selection_path, index=False)
    print(f"Saved model selection results: {selection_path}")
    print(f"Selected model: {best_model_name} (val macro F1={best_val_f1:.4f})")

    # Refit vectorizers on full training split and train selected classifier.
    X_train_mat, X_test_mat, word_vectorizer, _ = _build_feature_matrices(
        X_train,
        X_test,
        **best_feature_params,
    )
    print(f"Feature size (word+char): {X_train_mat.shape[1]:,}")

    model = SGDClassifier(
        loss=best_model_params["loss"],
        alpha=best_model_params["alpha"],
        penalty="l2",
        class_weight=best_model_params["class_weight"],
        max_iter=3000,
        tol=1e-3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_mat, y_train)

    y_pred = model.predict(X_test_mat)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    majority_class = int(y_train.value_counts().idxmax())
    y_baseline = np.full(shape=len(y_test), fill_value=majority_class)
    baseline_accuracy = accuracy_score(y_test, y_baseline)

    print("\nModel performance (test split):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Majority baseline accuracy: {baseline_accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=LABELS, digits=4))

    metrics_df = pd.DataFrame(
        [
            {"metric": "selected_model", "value": best_model_name},
            {"metric": "selected_model_validation_macro_f1", "value": best_val_f1},
            {"metric": "selected_word_max_features", "value": best_feature_params["word_max_features"]},
            {"metric": "selected_char_max_features", "value": best_feature_params["char_max_features"]},
            {"metric": "selected_char_ngram_range", "value": str(best_feature_params["char_ngram_range"])},
            {"metric": "selected_alpha", "value": best_model_params["alpha"]},
            {"metric": "selected_class_weight", "value": str(best_model_params["class_weight"])},
            {"metric": "accuracy", "value": accuracy},
            {"metric": "macro_f1", "value": macro_f1},
            {"metric": "weighted_f1", "value": weighted_f1},
            {"metric": "majority_baseline_accuracy", "value": baseline_accuracy},
            {"metric": "majority_class_from_train", "value": majority_class},
        ]
    )
    metrics_path = output_dir / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics: {metrics_path}")

    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
    report_path = output_dir / "classification_report.csv"
    report_df.to_csv(report_path, index=True)
    print(f"Saved per-class report: {report_path}")

    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    cm_df.index.name = "true_label"
    cm_df.columns.name = "predicted_label"
    cm_path = output_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_path, index=True)
    print(f"Saved confusion matrix table: {cm_path}")

    # Keep interpretability focused on human-readable word ngrams only.
    class_list = list(model.classes_)
    if 1 not in class_list or 5 not in class_list:
        raise ValueError("Classes 1 and 5 must exist to compute top feature scores.")
    idx_1 = class_list.index(1)
    idx_5 = class_list.index(5)
    word_count = len(word_vectorizer.get_feature_names_out())
    word_scores = model.coef_[idx_5, :word_count] - model.coef_[idx_1, :word_count]
    words = word_vectorizer.get_feature_names_out()
    weak_tokens = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "for",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "with",
    }

    informative_idx = [
        i
        for i, token in enumerate(words)
        if token not in weak_tokens and len(token) >= 3
    ]
    informative_scores = word_scores[informative_idx]
    informative_words = words[informative_idx]

    sorted_idx = np.argsort(informative_scores)
    top_negative_idx = sorted_idx[:25]
    top_positive_idx = sorted_idx[-25:][::-1]

    rows = []
    for rank, i in enumerate(top_positive_idx, start=1):
        rows.append(
            {
                "group": "top_positive_words_for_class_5",
                "rank": rank,
                "word": informative_words[i],
                "score": informative_scores[i],
            }
        )
    for rank, i in enumerate(top_negative_idx, start=1):
        rows.append(
            {
                "group": "top_negative_words_for_class_5",
                "rank": rank,
                "word": informative_words[i],
                "score": informative_scores[i],
            }
        )
    top_features_df = pd.DataFrame(rows)
    top_features_path = output_dir / "top_features.csv"
    top_features_df.to_csv(top_features_path, index=False)
    print(f"Saved top feature words: {top_features_path}")

    error_df = pd.DataFrame(
        {
            "text": X_test.reset_index(drop=True),
            "true_star": y_test.reset_index(drop=True),
            "pred_star": pd.Series(y_pred),
        }
    )
    error_df["absolute_error"] = (error_df["true_star"] - error_df["pred_star"]).abs()
    error_df = error_df.sort_values("absolute_error", ascending=False)
    error_path = output_dir / "largest_rating_errors.csv"
    error_df.head(500).to_csv(error_path, index=False)
    print(f"Saved largest prediction errors: {error_path}")

    print("\nDone.")
