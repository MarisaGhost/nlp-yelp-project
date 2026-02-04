"""Beginner-friendly LDA topic modeling for positive and negative Yelp reviews."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel


if __name__ == "__main__":
    # Step 1: Load processed review data
    input_path = Path("data/processed/reviews_processed.csv")
    output_dir = Path("outputs/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed reviews...")
    df = pd.read_csv(input_path)
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)

    # Step 2: Split into positive (4-5 stars) and negative (1-2 stars)
    positive_df = df[df["stars"].isin([4, 5])].copy()
    negative_df = df[df["stars"].isin([1, 2])].copy()

    # Use cleaned_text and split on whitespace to get tokens.
    positive_docs = [text.split() for text in positive_df["cleaned_text"] if text.strip()]
    negative_docs = [text.split() for text in negative_df["cleaned_text"] if text.strip()]

    print(f"Positive documents: {len(positive_docs):,}")
    print(f"Negative documents: {len(negative_docs):,}")

    # Step 3: Build dictionary/corpus and train LDA for each split
    split_docs = {"positive": positive_docs, "negative": negative_docs}
    split_topics = {}
    split_coherence = {}

    for split_name, docs in split_docs.items():
        print(f"\nTraining LDA for {split_name} reviews...")
        dictionary = corpora.Dictionary(docs)
        corpus = [dictionary.doc2bow(doc) for doc in docs]

        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=5,
            passes=10,
            random_state=42,
        )

        rows = []
        for topic_id in range(5):
            top_words = lda_model.show_topic(topic_id, topn=10)
            for rank, (word, weight) in enumerate(top_words, start=1):
                rows.append(
                    {
                        "split": split_name,
                        "topic_id": topic_id,
                        "word_rank": rank,
                        "word": word,
                        "weight": weight,
                    }
                )

        coherence_model = CoherenceModel(
            model=lda_model,
            texts=docs,
            dictionary=dictionary,
            coherence="c_v",
        )
        coherence_value = coherence_model.get_coherence()

        split_topics[split_name] = pd.DataFrame(rows)
        split_coherence[split_name] = coherence_value

    # Step 4: Save topic tables
    positive_out = output_dir / "lda_topics_positive.csv"
    negative_out = output_dir / "lda_topics_negative.csv"

    split_topics["positive"].to_csv(positive_out, index=False)
    split_topics["negative"].to_csv(negative_out, index=False)

    print("\nFinished.")
    print(f"Saved: {positive_out}")
    print(f"Saved: {negative_out}")
    print(f"Coherence c_v (positive): {split_coherence['positive']:.4f}")
    print(f"Coherence c_v (negative): {split_coherence['negative']:.4f}")
