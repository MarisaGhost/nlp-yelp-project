# Yelp Restaurant Reviews NLP Project

This project turns Yelp review text into actionable insight for restaurant operations.
It combines topic modeling, document classification, and aspect-level statistical testing to explain why ratings move, not just how many stars customers gave.

## Why this project matters
Star averages hide root causes. Free-text reviews contain operational signals (service quality, wait-time friction, value perception, and food consistency) that managers can act on quickly.

## Research Questions
1. Which themes dominate positive vs negative restaurant experiences?
2. Which aspects are disproportionately associated with negative reviews?
3. Can review text predict 1-5 star ratings with meaningful improvement over baseline?
4. How can interpretable NLP outputs support managerial decisions?

## Dataset and Scope
- Source: Yelp Academic Dataset
- Scope: Philadelphia restaurant businesses
- Size: downsampled to approximately 100,000 reviews for reproducible computation

Core processed files:
- `data/processed/merged_reviews.csv`
- `data/processed/reviews_processed.csv`

## Analytics Pipeline
### 1) Text Extraction and Processing
- Extract and merge review/business data from raw JSON.
- Normalize text (lowercase, URL/email removal).
- Build two lemmatized channels with NLTK:
- `cleaned_text`: general cleaned tokens for keyword/aspect analysis.
- `cleaned_text_topic_pos`: POS-filtered tokens for topic modeling.
- POS filter keeps nouns/proper nouns (`NN*`), adjectives (`JJ*`), and optionally adverbs (`RB*`).

### 2) Topic Modeling (LDA)
- Train separate LDA models for positive (4-5 stars) and negative (1-2 stars) reviews.
- Use POS-filtered text channel (`cleaned_text_topic_pos`) as LDA input.
- 5 topics per split, top 10 words per topic, split-specific vocabulary filtering.
- Evaluate with coherence `c_v`.

### 3) Document Classification
- Predict 1-5 stars from raw review text.
- Hybrid TF-IDF features:
- word n-grams (1,2)
- character n-grams (`char_wb`, 3-5)
- Classifier: linear SGD with validation-based model selection.

### 4) Aspect-Based Statistical Analysis
- Seed-word retrieval for Food, Service, and Price.
- Compare mention rates by split.
- Compute lift, odds ratio, and chi-square significance.

## Current Results
From `outputs/tables/model_metrics.csv`:
- Accuracy: **0.6482**
- Macro F1: **0.5949**
- Weighted F1: **0.6430**
- Majority baseline accuracy: **0.4051**

From `outputs/tables/aspect_significance.csv`:
- Service lift (negative/positive): **1.1749**, odds ratio: **1.7053**
- Price lift (negative/positive): **1.1594**, odds ratio: **1.2833**
- Aspect-rate differences are statistically significant.

From `outputs/tables/lda_coherence.csv`:
- Positive coherence `c_v`: **0.4851**
- Negative coherence `c_v`: **0.4314**

POS-filtering evaluation artifacts:
- `outputs/tables/lda_topic3_comparison.csv`
- `outputs/tables/lda_coherence_comparison.csv`
- `outputs/pos_filter_evaluation.md`

## Project Structure
```text
nlp-yelp-project/
|-- data/
|   |-- raw/
|   `-- processed/
|-- outputs/
|   |-- tables/
|   |-- figures/
|   |-- pos_filter_evaluation.md
|   `-- report.md
|-- src/
|   |-- data_loader.py
|   |-- preprocess.py
|   |-- topics.py
|   |-- classifier.py
|   |-- aspects.py
|   `-- visualize.py
|-- main.py
|-- README.md
`-- requirements.txt
```

## Reproduce
### 1) Environment
```bash
conda create -n nlp_project python=3.9
conda activate nlp_project
pip install -r requirements.txt
```

Note: preprocessing uses NLTK resources (`punkt`, `averaged_perceptron_tagger`, `wordnet`, `omw-1.4`, `stopwords`).
The script attempts to download missing resources automatically.

### 2) Add raw data
Place these files in `data/raw/`:
- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_review.json`

### 3) Run full pipeline
```bash
python main.py
```

### 4) Run step-by-step
```bash
python src/data_loader.py
python src/preprocess.py
python src/topics.py
python src/classifier.py
python src/aspects.py
python src/visualize.py
```

## Outputs
- Tables: `outputs/tables/`
- Figures: `outputs/figures/`
- Evaluation note: `outputs/pos_filter_evaluation.md`
- Final write-up: `outputs/report.md`

Results are deterministic due to fixed random seeds.
