"""LDA topic modeling with tuned split-specific configurations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel

RANDOM_STATE = 42
NUM_TOPICS = 5
TOP_WORDS = 10

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


if __name__ == "__main__":
    input_path = Path("data/processed/reviews_processed.csv")
    output_dir = Path("outputs/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed reviews...")
    df = pd.read_csv(input_path, usecols=["stars", "cleaned_text"])
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)

    positive_docs = [
        text.split()
        for text in df[df["stars"].isin([4, 5])]["cleaned_text"]
        if text.strip()
    ]
    negative_docs = [
        text.split()
        for text in df[df["stars"].isin([1, 2])]["cleaned_text"]
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

    split_topics["positive"].to_csv(positive_out, index=False)
    split_topics["negative"].to_csv(negative_out, index=False)
    pd.DataFrame(split_selection_rows).to_csv(selection_out, index=False)
    pd.DataFrame(
        [
            {"split": "positive", "coherence_c_v": split_coherence["positive"]},
            {"split": "negative", "coherence_c_v": split_coherence["negative"]},
        ]
    ).to_csv(coherence_out, index=False)

    print("\nFinished.")
    print(f"Saved: {positive_out}")
    print(f"Saved: {negative_out}")
    print(f"Saved: {selection_out}")
    print(f"Saved: {coherence_out}")
    print(f"Coherence c_v (positive): {split_coherence['positive']:.4f}")
    print(f"Coherence c_v (negative): {split_coherence['negative']:.4f}")
