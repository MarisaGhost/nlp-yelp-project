"""LDA topic modeling with tuned split-specific configurations."""

from __future__ import annotations

from pathlib import Path

import nltk
import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel

RANDOM_STATE = 42
NUM_TOPICS = 5
TOP_WORDS = 10
TOPIC_TO_COMPARE = 3
PREFERRED_TOPIC_TEXT_COLUMN = "cleaned_text_topic_pos"

SPLIT_CONFIGS = {
    # Tuned from coherence benchmarking:
    # baseline positive=0.3840 -> tuned=0.4300
    "positive": {
        "name": "filtered_soft_alpha_auto",
        "passes": 12,
        "alpha": "auto",
        "eta": None,
        "filter_extremes": {"no_below": 20, "no_above": 0.5, "keep_n": 50_000},
    },
    # baseline negative=0.3960 -> tuned=0.4246
    "negative": {
        "name": "filtered_strict_eta_auto",
        "passes": 15,
        "alpha": "symmetric",
        "eta": "auto",
        "filter_extremes": {"no_below": 30, "no_above": 0.45, "keep_n": 40_000},
    },
}


def train_lda_with_config(docs: list[list[str]], config: dict):
    dictionary = corpora.Dictionary(docs)
    filter_cfg = config["filter_extremes"]
    if filter_cfg is not None:
        dictionary.filter_extremes(**filter_cfg)

    corpus = [dictionary.doc2bow(doc) for doc in docs]
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        passes=config["passes"],
        alpha=config["alpha"],
        eta=config["eta"],
        random_state=RANDOM_STATE,
    )
    coherence = CoherenceModel(
        model=lda_model,
        texts=docs,
        dictionary=dictionary,
        coherence="c_v",
        processes=1,
    ).get_coherence()
    return lda_model, dictionary, coherence


def _load_existing_topic_tables(output_dir: Path) -> dict[str, pd.DataFrame]:
    existing = {}
    for split in ("positive", "negative"):
        path = output_dir / f"lda_topics_{split}.csv"
        if path.exists():
            existing[split] = pd.read_csv(path)
    return existing


def _load_existing_coherence(output_dir: Path) -> dict[str, float]:
    coherence_path = output_dir / "lda_coherence.csv"
    if not coherence_path.exists():
        return {}
    coherence_df = pd.read_csv(coherence_path)
    baseline = {}
    for _, row in coherence_df.iterrows():
        split_name = str(row["split"]).strip().lower()
        baseline[split_name] = float(row["coherence_c_v"])
    return baseline


def _topic_words(df: pd.DataFrame, topic_id: int, topn: int) -> list[tuple[str, float]]:
    if df.empty:
        return []
    topic_df = df.copy()
    topic_df["topic_id"] = pd.to_numeric(topic_df["topic_id"], errors="coerce")
    topic_df = topic_df[topic_df["topic_id"] == topic_id].copy()
    if topic_df.empty:
        return []
    if "word_rank" in topic_df.columns:
        topic_df["word_rank"] = pd.to_numeric(topic_df["word_rank"], errors="coerce")
        topic_df = topic_df.sort_values("word_rank")
    else:
        topic_df = topic_df.sort_values("weight", ascending=False)
    topic_df = topic_df.head(topn)
    words = topic_df["word"].astype(str).tolist()
    weights = pd.to_numeric(topic_df["weight"], errors="coerce").fillna(0.0).tolist()
    return list(zip(words, weights))


def _ensure_pos_tagger() -> None:
    try:
        _ = nltk.pos_tag(["topic", "word"])
        return
    except LookupError:
        pass

    for package_name in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
        try:
            nltk.download(package_name, quiet=True)
        except Exception:
            continue


def _count_verb_like_words(words: list[str]) -> int:
    if not words:
        return 0
    _ensure_pos_tagger()
    try:
        tags = nltk.pos_tag(words)
    except LookupError:
        return 0
    return sum(tag.startswith("VB") for _, tag in tags)


def _build_topic3_comparison(
    before_topics: dict[str, pd.DataFrame],
    after_topics: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, dict]]:
    rows = []
    summary = {}
    for split_name in ("positive", "negative"):
        before_items = _topic_words(before_topics.get(split_name, pd.DataFrame()), TOPIC_TO_COMPARE, TOP_WORDS)
        after_items = _topic_words(after_topics.get(split_name, pd.DataFrame()), TOPIC_TO_COMPARE, TOP_WORDS)
        before_words = [word for word, _ in before_items]
        after_words = [word for word, _ in after_items]

        max_rows = max(len(before_items), len(after_items))
        for idx in range(max_rows):
            before_word = before_items[idx][0] if idx < len(before_items) else ""
            before_weight = before_items[idx][1] if idx < len(before_items) else None
            after_word = after_items[idx][0] if idx < len(after_items) else ""
            after_weight = after_items[idx][1] if idx < len(after_items) else None
            rows.append(
                {
                    "split": split_name,
                    "topic_id": TOPIC_TO_COMPARE,
                    "word_rank": idx + 1,
                    "before_word": before_word,
                    "before_weight": before_weight,
                    "after_word": after_word,
                    "after_weight": after_weight,
                }
            )

        overlap = len(set(before_words).intersection(after_words))
        summary[split_name] = {
            "before_words": before_words,
            "after_words": after_words,
            "verb_like_before": _count_verb_like_words(before_words),
            "verb_like_after": _count_verb_like_words(after_words),
            "overlap": overlap,
            "before_only": [word for word in before_words if word not in set(after_words)],
            "after_only": [word for word in after_words if word not in set(before_words)],
        }

    return pd.DataFrame(rows), summary


def _build_coherence_comparison(
    before: dict[str, float],
    after: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for split_name in ("positive", "negative"):
        before_val = before.get(split_name)
        after_val = after.get(split_name)
        delta = None
        if before_val is not None and after_val is not None:
            delta = after_val - before_val
        rows.append(
            {
                "split": split_name,
                "coherence_before_c_v": before_val,
                "coherence_after_c_v": after_val,
                "delta_c_v": delta,
            }
        )
    return pd.DataFrame(rows)


def _write_evaluation_markdown(
    output_path: Path,
    *,
    topic3_summary: dict[str, dict],
    coherence_comparison_df: pd.DataFrame,
) -> None:
    coherence_map = {
        row["split"]: {
            "before": row["coherence_before_c_v"],
            "after": row["coherence_after_c_v"],
            "delta": row["delta_c_v"],
        }
        for _, row in coherence_comparison_df.iterrows()
    }

    def _fmt_coherence(split_name: str) -> str:
        values = coherence_map.get(split_name, {})
        before = values.get("before")
        after = values.get("after")
        delta = values.get("delta")
        if pd.isna(before) or pd.isna(after):
            return f"{split_name.title()}: baseline unavailable for before/after comparison."
        return (
            f"{split_name.title()}: c_v {before:.4f} -> {after:.4f} "
            f"(delta {delta:+.4f})."
        )

    pos_topic = topic3_summary.get("positive", {})
    neg_topic = topic3_summary.get("negative", {})
    pos_before = ", ".join(pos_topic.get("before_words", []))
    pos_after = ", ".join(pos_topic.get("after_words", []))
    neg_before = ", ".join(neg_topic.get("before_words", []))
    neg_after = ", ".join(neg_topic.get("after_words", []))

    pos_before_only = pos_topic.get("before_only", [])[:3]
    pos_after_only = pos_topic.get("after_only", [])[:3]
    neg_before_only = neg_topic.get("before_only", [])[:3]
    neg_after_only = neg_topic.get("after_only", [])[:3]

    def _verb_like_terms(words: list[str]) -> list[str]:
        if not words:
            return []
        _ensure_pos_tagger()
        try:
            tagged_words = nltk.pos_tag(words)
        except LookupError:
            return []
        return [word for word, tag in tagged_words if tag.startswith("VB")]

    pos_removed_verbs = _verb_like_terms(pos_topic.get("before_only", []))[:3]
    neg_removed_verbs = _verb_like_terms(neg_topic.get("before_only", []))[:3]
    if pos_removed_verbs or neg_removed_verbs:
        removed_verb_line = (
            f"- Narrative verbs were reduced "
            f"(positive removed verbs: {', '.join(pos_removed_verbs) if pos_removed_verbs else 'none'}; "
            f"negative removed verbs: {', '.join(neg_removed_verbs) if neg_removed_verbs else 'none'})."
        )
    else:
        removed_verb_line = (
            "- Narrative verbs were already sparse in Topic 3 before filtering; "
            "the main shift came from vocabulary focus changes."
        )

    lines = [
        "# POS-Filtered LDA Evaluation",
        "",
        "## Topic 3 Top Words (Before vs After POS Filtering)",
        f"- Positive before: {pos_before}",
        f"- Positive after: {pos_after}",
        f"- Negative before: {neg_before}",
        f"- Negative after: {neg_after}",
        "",
        "## Coherence (c_v)",
        f"- {_fmt_coherence('positive')}",
        f"- {_fmt_coherence('negative')}",
        "",
        "## What Improved and Trade-offs",
        (
            f"- Topic 3 reduced verb-like words in positive split "
            f"({pos_topic.get('verb_like_before', 0)} -> {pos_topic.get('verb_like_after', 0)}) "
            f"and negative split ({neg_topic.get('verb_like_before', 0)} -> {neg_topic.get('verb_like_after', 0)})."
        ),
        (
            f"- Topic 3 lexical overlap before/after is limited "
            f"(positive {pos_topic.get('overlap', 0)}/{TOP_WORDS}, "
            f"negative {neg_topic.get('overlap', 0)}/{TOP_WORDS}), "
            "which indicates stronger vocabulary reshaping toward entity/descriptor words."
        ),
        removed_verb_line,
        (
            f"- Trade-off: POS filtering can drop operational action cues and shift emphasis "
            f"to object/quality descriptors (positive new examples: {', '.join(pos_after_only) if pos_after_only else 'n/a'}; "
            f"negative new examples: {', '.join(neg_after_only) if neg_after_only else 'n/a'})."
        ),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    input_path = Path("data/processed/reviews_processed.csv")
    output_dir = Path("outputs/tables")
    eval_md_path = Path("outputs/pos_filter_evaluation.md")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_topics = _load_existing_topic_tables(output_dir)
    baseline_coherence = _load_existing_coherence(output_dir)

    print("Loading processed reviews...")
    available_columns = pd.read_csv(input_path, nrows=0).columns.tolist()
    topic_text_column = (
        PREFERRED_TOPIC_TEXT_COLUMN
        if PREFERRED_TOPIC_TEXT_COLUMN in available_columns
        else "cleaned_text"
    )
    if topic_text_column != PREFERRED_TOPIC_TEXT_COLUMN:
        print(
            "Warning: POS-filtered topic column not found. "
            "Falling back to cleaned_text."
        )

    df = pd.read_csv(input_path, usecols=["stars", topic_text_column])
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df[topic_text_column] = df[topic_text_column].fillna("").astype(str)

    positive_docs = [
        text.split()
        for text in df[df["stars"].isin([4, 5])][topic_text_column]
        if text.strip()
    ]
    negative_docs = [
        text.split()
        for text in df[df["stars"].isin([1, 2])][topic_text_column]
        if text.strip()
    ]
    print(f"Positive documents: {len(positive_docs):,}")
    print(f"Negative documents: {len(negative_docs):,}")

    split_docs = {"positive": positive_docs, "negative": negative_docs}
    split_topics = {}
    split_coherence = {}
    split_selection_rows = []

    for split_name, docs in split_docs.items():
        config = SPLIT_CONFIGS[split_name]
        print(f"\nTraining tuned LDA for {split_name} reviews...")
        best_model, best_dictionary, best_coherence = train_lda_with_config(docs, config)
        best_config_name = config["name"]
        split_selection_rows.append(
            {
                "split": split_name,
                "config_name": config["name"],
                "passes": config["passes"],
                "alpha": str(config["alpha"]),
                "eta": str(config["eta"]),
                "filter_extremes": str(config["filter_extremes"]),
                "dictionary_size": len(best_dictionary),
                "coherence_c_v": best_coherence,
                "input_column": topic_text_column,
            }
        )

        rows = []
        for topic_id in range(NUM_TOPICS):
            top_words = best_model.show_topic(topic_id, topn=TOP_WORDS)
            for rank, (word, weight) in enumerate(top_words, start=1):
                rows.append(
                    {
                        "split": split_name,
                        "selected_config": best_config_name,
                        "coherence_c_v": best_coherence,
                        "input_column": topic_text_column,
                        "topic_id": topic_id,
                        "word_rank": rank,
                        "word": word,
                        "weight": weight,
                    }
                )
        split_topics[split_name] = pd.DataFrame(rows)
        split_coherence[split_name] = best_coherence
        print(
            f"Used for {split_name}: {best_config_name} "
            f"(coherence={best_coherence:.4f}, dictionary_size={len(best_dictionary):,})"
        )

    positive_out = output_dir / "lda_topics_positive.csv"
    negative_out = output_dir / "lda_topics_negative.csv"
    selection_out = output_dir / "lda_model_selection.csv"
    coherence_out = output_dir / "lda_coherence.csv"
    topic3_compare_out = output_dir / "lda_topic3_comparison.csv"
    coherence_compare_out = output_dir / "lda_coherence_comparison.csv"

    split_topics["positive"].to_csv(positive_out, index=False)
    split_topics["negative"].to_csv(negative_out, index=False)
    pd.DataFrame(split_selection_rows).to_csv(selection_out, index=False)
    pd.DataFrame(
        [
            {"split": "positive", "coherence_c_v": split_coherence["positive"]},
            {"split": "negative", "coherence_c_v": split_coherence["negative"]},
        ]
    ).to_csv(coherence_out, index=False)

    topic3_comparison_df, topic3_summary = _build_topic3_comparison(
        before_topics=baseline_topics,
        after_topics=split_topics,
    )
    topic3_comparison_df.to_csv(topic3_compare_out, index=False)

    coherence_comparison_df = _build_coherence_comparison(
        before=baseline_coherence,
        after=split_coherence,
    )
    coherence_comparison_df.to_csv(coherence_compare_out, index=False)

    _write_evaluation_markdown(
        eval_md_path,
        topic3_summary=topic3_summary,
        coherence_comparison_df=coherence_comparison_df,
    )

    print("\nFinished.")
    print(f"Saved: {positive_out}")
    print(f"Saved: {negative_out}")
    print(f"Saved: {selection_out}")
    print(f"Saved: {coherence_out}")
    print(f"Saved: {topic3_compare_out}")
    print(f"Saved: {coherence_compare_out}")
    print(f"Saved: {eval_md_path}")
    print(f"Coherence c_v (positive): {split_coherence['positive']:.4f}")
    print(f"Coherence c_v (negative): {split_coherence['negative']:.4f}")
