"""Simple preprocessing script for Yelp reviews (NLP class version)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import spacy


if __name__ == "__main__":
    # Step 0: Set paths and basic settings
    input_path = Path("data/processed/merged_reviews.csv")
    output_path = Path("data/processed/reviews_processed.csv")
    text_column = "text"

    # Step 1: Read merged review data
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if text_column not in df.columns:
        raise ValueError(f"Expected '{text_column}' column in {input_path}")

    # Step 2: Load spaCy model (disable parser + NER for speed)
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Run: python -m spacy download en_core_web_sm"
        ) from exc

    # Step 3: Process each review text
    # Required operations:
    # - lowercase
    # - remove URLs/emails
    # - tokenize + lemmatize
    # - remove stopwords/punctuation/numbers
    # - keep alphabetic tokens only
    url_pattern = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
    email_pattern = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", flags=re.IGNORECASE)

    raw_texts = df[text_column].fillna("").astype(str)
    normalized_texts = []
    for text in raw_texts:
        text = text.lower()
        text = url_pattern.sub(" ", text)
        text = email_pattern.sub(" ", text)
        normalized_texts.append(text)

    cleaned_texts = []
    token_lists = []

    # nlp.pipe is faster than calling nlp(text) one-by-one.
    for doc in nlp.pipe(normalized_texts, batch_size=1000):
        lemmas = []
        for token in doc:
            if token.is_stop or token.is_punct or token.like_num:
                continue
            if not token.is_alpha:
                continue

            lemma = token.lemma_.strip().lower()
            if not lemma or lemma == "-pron-":
                continue

            lemmas.append(lemma)

        token_lists.append(lemmas)
        cleaned_texts.append(" ".join(lemmas))

    # Step 4: Create output columns
    df["cleaned_text"] = cleaned_texts
    df["tokens"] = token_lists

    # Step 5: Save processed data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved processed file: {output_path}")
    print(f"Rows processed: {len(df):,}")
