"""Extended experimentation for binary Yelp triage (1-2 vs 4-5 stars)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split

RANDOM_STATE = 42
INPUT_PATH = Path("data/processed/reviews_processed.csv")
OUTPUT_DIR = Path("outputs/tables")


def _load_binary_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["text", "stars"])
    df = df.dropna(subset=["text", "stars"]).copy()
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.dropna(subset=["stars"])
    df["stars"] = df["stars"].astype(int)
    df = df[df["stars"].isin([1, 2, 4, 5])].copy()
    df["label"] = (df["stars"] >= 4).astype(int)
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)
    return df


def _build_vectorizers():
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        max_features=120_000,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=5,
        sublinear_tf=True,
        max_features=120_000,
    )
    return word, char


def _fit_features(train_text: pd.Series, test_text: pd.Series, variant: str):
    word_vec, char_vec = _build_vectorizers()

    if variant == "word_only":
        x_train = word_vec.fit_transform(train_text)
        x_test = word_vec.transform(test_text)
        return x_train, x_test

    if variant == "char_only":
        x_train = char_vec.fit_transform(train_text)
        x_test = char_vec.transform(test_text)
        return x_train, x_test

    train_word = word_vec.fit_transform(train_text)
    test_word = word_vec.transform(test_text)
    train_char = char_vec.fit_transform(train_text)
    test_char = char_vec.transform(test_text)
    x_train = hstack([train_word, train_char]).tocsr()
    x_test = hstack([test_word, test_char]).tocsr()
    return x_train, x_test


def _metrics_from_scores(y_true: pd.Series, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }


def run_cross_validation(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    x = df["text"].reset_index(drop=True)
    y = df["label"].reset_index(drop=True)

    for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        print(f"CV fold {fold_id}/5...", flush=True)
        x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        x_train_mat, x_valid_mat = _fit_features(x_train, x_valid, variant="hybrid")

        model = SGDClassifier(
            loss=params["loss"],
            alpha=params["alpha"],
            class_weight=params["class_weight"],
            random_state=RANDOM_STATE,
            max_iter=3000,
            tol=1e-3,
            n_jobs=-1,
        )
        model.fit(x_train_mat, y_train)
        pred = model.predict(x_valid_mat)
        scores = model.decision_function(x_valid_mat)
        metrics = _metrics_from_scores(y_valid, pred, scores)
        metrics["fold"] = fold_id
        rows.append(metrics)

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(OUTPUT_DIR / "binary_cv_metrics.csv", index=False)

    summary = cv_df.drop(columns=["fold"]).agg(["mean", "std"]).T.reset_index()
    summary.columns = ["metric", "mean", "std"]
    summary.to_csv(OUTPUT_DIR / "binary_cv_summary.csv", index=False)
    return cv_df


def run_grid_search(
    x_train: pd.Series,
    y_train: pd.Series,
    x_valid: pd.Series,
    y_valid: pd.Series,
) -> pd.DataFrame:
    param_grid = {
        "loss": ["hinge", "log_loss", "modified_huber"],
        "alpha": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
        "class_weight": [None, "balanced"],
    }
    rows = []
    x_subtrain_mat, x_valid_mat = _fit_features(x_train, x_valid, variant="hybrid")

    for params in ParameterGrid(param_grid):
        model = SGDClassifier(
            loss=params["loss"],
            alpha=params["alpha"],
            class_weight=params["class_weight"],
            random_state=RANDOM_STATE,
            max_iter=3000,
            tol=1e-3,
            n_jobs=-1,
        )
        model.fit(x_subtrain_mat, y_train)
        pred = model.predict(x_valid_mat)
        scores = model.decision_function(x_valid_mat)
        metrics = _metrics_from_scores(y_valid, pred, scores)
        rows.append(
            {
                "loss": params["loss"],
                "alpha": params["alpha"],
                "class_weight": str(params["class_weight"]),
                **metrics,
            }
        )

    grid_df = pd.DataFrame(rows).sort_values(["f1", "roc_auc", "accuracy"], ascending=False)
    grid_df.to_csv(OUTPUT_DIR / "binary_hyperparameter_grid.csv", index=False)
    return grid_df


def run_ablation(
    x_train: pd.Series,
    y_train: pd.Series,
    x_test: pd.Series,
    y_test: pd.Series,
    best_params: dict,
) -> pd.DataFrame:
    variants = ["word_only", "char_only", "hybrid"]
    rows = []

    for variant in variants:
        x_train_mat, x_test_mat = _fit_features(x_train, x_test, variant=variant)
        model = SGDClassifier(
            loss=best_params["loss"],
            alpha=best_params["alpha"],
            class_weight=best_params["class_weight"],
            random_state=RANDOM_STATE,
            max_iter=3000,
            tol=1e-3,
            n_jobs=-1,
        )
        model.fit(x_train_mat, y_train)
        pred = model.predict(x_test_mat)
        scores = model.decision_function(x_test_mat)
        rows.append({"feature_variant": variant, **_metrics_from_scores(y_test, pred, scores)})

    ablation_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    ablation_df.to_csv(OUTPUT_DIR / "binary_ablation.csv", index=False)
    return ablation_df


def run_final_holdout_and_error_analysis(
    x_train: pd.Series,
    y_train: pd.Series,
    x_test: pd.Series,
    y_test: pd.Series,
    best_params: dict,
) -> dict[str, float]:
    x_train_mat, x_test_mat = _fit_features(x_train, x_test, variant="hybrid")

    model = SGDClassifier(
        loss=best_params["loss"],
        alpha=best_params["alpha"],
        class_weight=best_params["class_weight"],
        random_state=RANDOM_STATE,
        max_iter=3000,
        tol=1e-3,
        n_jobs=-1,
    )
    model.fit(x_train_mat, y_train)
    pred = model.predict(x_test_mat)
    scores = model.decision_function(x_test_mat)
    metrics = _metrics_from_scores(y_test, pred, scores)

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["true_neg", "true_pos"], columns=["pred_neg", "pred_pos"])
    cm_df.to_csv(OUTPUT_DIR / "binary_confusion_matrix.csv")

    results_df = pd.DataFrame(
        {
            "text": x_test.values,
            "true_label": y_test.values,
            "pred_label": pred,
            "margin_score": scores,
        }
    )
    errors = results_df[results_df["true_label"] != results_df["pred_label"]].copy()

    false_pos = errors[(errors["true_label"] == 0) & (errors["pred_label"] == 1)].copy()
    false_neg = errors[(errors["true_label"] == 1) & (errors["pred_label"] == 0)].copy()
    false_pos = false_pos.sort_values("margin_score", ascending=False).head(20)
    false_neg = false_neg.sort_values("margin_score", ascending=True).head(20)
    error_examples = pd.concat([false_pos.assign(error_type="false_positive"), false_neg.assign(error_type="false_negative")])
    error_examples["text"] = error_examples["text"].str.slice(0, 280)
    error_examples.to_csv(OUTPUT_DIR / "binary_error_analysis_examples.csv", index=False)

    summary = {
        "test_size": int(len(y_test)),
        "error_count": int(len(errors)),
        "false_positive_count": int(len(results_df[(results_df["true_label"] == 0) & (results_df["pred_label"] == 1)])),
        "false_negative_count": int(len(results_df[(results_df["true_label"] == 1) & (results_df["pred_label"] == 0)])),
        **metrics,
    }
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "binary_holdout_metrics.csv", index=False)
    return summary


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_binary_data(INPUT_PATH)

    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )

    # Train/validation split for grid search (inside training partition only).
    x_subtrain, x_valid, y_subtrain, y_valid = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    print("Running hyperparameter grid...")
    grid = run_grid_search(x_subtrain, y_subtrain, x_valid, y_valid)
    best_row = grid.iloc[0]
    best_params = {
        "loss": best_row["loss"],
        "alpha": float(best_row["alpha"]),
        "class_weight": None if best_row["class_weight"] == "None" else "balanced",
    }

    print("Running 5-fold cross-validation with best params...")
    run_cross_validation(data, best_params)

    print("Running holdout evaluation and error analysis...")
    holdout = run_final_holdout_and_error_analysis(x_train, y_train, x_test, y_test, best_params)

    print("Running ablation study...")
    run_ablation(x_train, y_train, x_test, y_test, best_params)

    print("Saved outputs:")
    print("- outputs/tables/binary_hyperparameter_grid.csv")
    print("- outputs/tables/binary_cv_metrics.csv")
    print("- outputs/tables/binary_cv_summary.csv")
    print("- outputs/tables/binary_holdout_metrics.csv")
    print("- outputs/tables/binary_confusion_matrix.csv")
    print("- outputs/tables/binary_error_analysis_examples.csv")
    print("- outputs/tables/binary_ablation.csv")
    print("Holdout summary:")
    print(holdout)
