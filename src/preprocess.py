"""Simple preprocessing script for Yelp reviews (NLP class version)."""

from __future__ import annotations

import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

INCLUDE_ADVERBS_FOR_TOPIC_MODELING = True


def _ensure_nltk_resources() -> None:
    """Download required NLTK resources if they are not already available."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("corpora/stopwords", "stopwords"),
    ]
    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(package_name, quiet=True)
            except Exception:
                # Some resources are version-specific aliases; keep going.
                pass


def _penn_to_wordnet_pos(pos_tag: str) -> str:
    if pos_tag.startswith("NN"):
        return "n"
    if pos_tag.startswith("VB"):
        return "v"
    if pos_tag.startswith("JJ"):
        return "a"
    if pos_tag.startswith("RB"):
        return "r"
    return "n"


def _tokenize_and_tag(text: str) -> list[tuple[str, str]]:
    tokens = nltk.word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    normalized = []
    for token, pos_tag in tagged_tokens:
        token = token.lower().strip()
        if not token or not token.isalpha():
            continue
        normalized.append((token, pos_tag))
    return normalized


def _build_base_tokens(
    tagged_tokens: list[tuple[str, str]],
    *,
    lemmatizer: WordNetLemmatizer,
    english_stopwords: set[str],
) -> list[str]:
    lemmas = []
    for token, pos_tag in tagged_tokens:
        if token in english_stopwords:
            continue
        lemma = lemmatizer.lemmatize(token, pos=_penn_to_wordnet_pos(pos_tag)).lower().strip()
        if not lemma or lemma in english_stopwords:
            continue
        lemmas.append(lemma)
    return lemmas


def _build_topic_tokens(
    tagged_tokens: list[tuple[str, str]],
    *,
    lemmatizer: WordNetLemmatizer,
    english_stopwords: set[str],
    include_adverbs: bool,
) -> list[str]:
    allowed_prefixes = ("NN", "JJ", "RB") if include_adverbs else ("NN", "JJ")

    lemmas = []
    for token, pos_tag in tagged_tokens:
        if token in english_stopwords:
            continue
        if not pos_tag.startswith(allowed_prefixes):
            continue

        lemma = lemmatizer.lemmatize(token, pos=_penn_to_wordnet_pos(pos_tag)).lower().strip()
        if not lemma or lemma in english_stopwords:
            continue
        lemmas.append(lemma)

    return lemmas


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

    # Step 2: Prepare NLTK resources.
    _ensure_nltk_resources()
    try:
        english_stopwords = set(stopwords.words("english"))
        _ = nltk.word_tokenize("resource check")
        _ = nltk.pos_tag(["resource", "check"])
    except LookupError as exc:
        raise RuntimeError(
            "Required NLTK resources are missing for POS-filtered topic text. "
            "Please install/download punkt, averaged_perceptron_tagger, wordnet, and stopwords."
        ) from exc
    lemmatizer = WordNetLemmatizer()

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
    cleaned_texts_topic_pos = []
    token_lists_topic_pos = []

    # Build both general cleaned text and a dedicated POS-filtered topic text.
    for text in normalized_texts:
        tagged_tokens = _tokenize_and_tag(text)
        base_lemmas = _build_base_tokens(
            tagged_tokens,
            lemmatizer=lemmatizer,
            english_stopwords=english_stopwords,
        )
        topic_lemmas = _build_topic_tokens(
            tagged_tokens,
            lemmatizer=lemmatizer,
            english_stopwords=english_stopwords,
            include_adverbs=INCLUDE_ADVERBS_FOR_TOPIC_MODELING,
        )

        token_lists.append(base_lemmas)
        cleaned_texts.append(" ".join(base_lemmas))
        token_lists_topic_pos.append(topic_lemmas)
        cleaned_texts_topic_pos.append(" ".join(topic_lemmas))

    # Step 4: Create output columns
    df["cleaned_text"] = cleaned_texts
    df["tokens"] = token_lists
    df["cleaned_text_topic_pos"] = cleaned_texts_topic_pos
    df["tokens_topic_pos"] = token_lists_topic_pos

    # Step 5: Save processed data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved processed file: {output_path}")
    print(f"Rows processed: {len(df):,}")
    topic_nonempty = sum(bool(tokens) for tokens in token_lists_topic_pos)
    print(f"Non-empty POS-filtered topic docs: {topic_nonempty:,}")
