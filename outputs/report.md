# Yelp NLP Project Report

## 1) Executive Summary
This project converts unstructured Yelp reviews into operational insight for restaurant decision-making. It combines:
- POS-filtered topic modeling for interpretable themes,
- 5-class rating prediction from free text,
- aspect-level statistical testing for Food, Service, and Price.

Key outcomes from current outputs:
- Rating prediction achieved **0.6482 accuracy** and **0.5949 macro F1** versus **0.4051 majority baseline** (`outputs/tables/model_metrics.csv`).
- Service and price language are disproportionately concentrated in negative reviews (Service lift **1.1749**, Price lift **1.1594**) with strong significance (`outputs/tables/aspect_significance.csv`).
- POS-filtered LDA coherence improved to **0.4851** (positive) and **0.4314** (negative) (`outputs/tables/lda_coherence.csv`).

## 2) Project Objective and Questions
Objective: build a practical NLP workflow that identifies what customers discuss, what predicts star ratings, and which operational dimensions are most associated with dissatisfaction.

Research questions:
1. Which themes dominate positive vs negative restaurant experiences?
2. Which aspects are overrepresented in negative reviews?
3. How well can review text predict 1-5 star ratings?
4. How can interpretable NLP outputs translate into manager-facing action?

## 3) Data and Scope
- Source: Yelp Academic Dataset
- Scope: Philadelphia restaurant businesses
- Size: approximately 100,000 reviews

Primary artifacts:
- `data/processed/reviews_processed.csv`
- `outputs/tables/model_metrics.csv`
- `outputs/tables/classification_report.csv`
- `outputs/tables/lda_topics_positive.csv`
- `outputs/tables/lda_topics_negative.csv`
- `outputs/tables/lda_coherence.csv`
- `outputs/tables/lda_topic3_comparison.csv`
- `outputs/tables/lda_coherence_comparison.csv`
- `outputs/tables/aspect_significance.csv`

## 4) Methods
### 4.1 Preprocessing and text channels
`src/preprocess.py` builds two lemmatized representations with NLTK:
- `cleaned_text`: normalized, stopword-filtered tokens.
- `cleaned_text_topic_pos`: POS-filtered tokens for topic modeling.

POS filtering rules for topic modeling:
- Keep nouns/proper nouns (`NN*`), adjectives (`JJ*`), and optional adverbs (`RB*`).
- Lemmatize kept tokens with WordNet lemmatization.
- Remove stopwords and non-alphabetic tokens.

### 4.2 Topic modeling
`src/topics.py` trains split LDA models:
- Positive split: stars 4-5
- Negative split: stars 1-2
- Topics per split: 5
- Top words per topic: 10
- Coherence metric: `c_v`

The script also writes evaluation artifacts:
- `outputs/tables/lda_topic3_comparison.csv` (before vs after POS filtering)
- `outputs/tables/lda_coherence_comparison.csv` (coherence deltas)
- `outputs/pos_filter_evaluation.md`

### 4.3 Rating prediction
`src/classifier.py` uses raw review text with hybrid TF-IDF features:
- word n-grams `(1, 2)`
- character n-grams `char_wb (3, 5)`

Classifier: `SGDClassifier` with validation-based selection.

### 4.4 Aspect significance
`src/aspects.py` computes mention-rate differences and significance for Food, Service, Price:
- lift (negative over positive)
- odds ratio
- chi-square p-value

## 5) Results
### 5.1 Classification quality
From `outputs/tables/model_metrics.csv`:
- Accuracy: **0.64815**
- Macro F1: **0.59486**
- Weighted F1: **0.64297**
- Majority baseline accuracy: **0.40505**

From `outputs/tables/classification_report.csv`:
- Class 1 F1: **0.7244**
- Class 2 F1: **0.4134**
- Class 3 F1: **0.5065**
- Class 4 F1: **0.5492**
- Class 5 F1: **0.7808**

Interpretation: extreme sentiment classes are easier to identify than middle ratings.

### 5.2 Aspect-level risk signals
From `outputs/tables/aspect_significance.csv`:
- Service: lift **1.1749**, odds ratio **1.7053**, p-value **8.17e-174**
- Price: lift **1.1594**, odds ratio **1.2833**, p-value **1.74e-48**
- Food: lift **0.9212**, odds ratio **0.6300**, p-value **1.15e-105**

Interpretation: service and value language are the strongest dissatisfaction indicators in this dataset.

### 5.3 Topic modeling with POS filtering
From `outputs/tables/lda_coherence.csv`:
- Positive coherence `c_v`: **0.4851**
- Negative coherence `c_v`: **0.4314**

From `outputs/tables/lda_coherence_comparison.csv`:
- Positive coherence delta: **+0.0551**
- Negative coherence delta: **+0.0069**

From `outputs/tables/lda_topic3_comparison.csv`:
- Positive Topic 3 before: `coffee, breakfast, brunch, egg, cream, toast, sweet, french, tea, chocolate`
- Positive Topic 3 after: `bar, drink, beer, night, table, happy, hour, great, selection, friend`
- Negative Topic 3 before: `come, table, wait, service, ask, drink, time, order, minute, go`
- Negative Topic 3 after: `cheese, sandwich, good, fry, chicken, burger, salad, order, place, time`

Interpretation:
- POS-filtering reduced narrative verb presence in negative Topic 3 and increased descriptor/entity concentration.
- Topic vocabulary became more sharply thematic, with improved coherence on both splits.

## 6) Operational Implications
- Prioritize service-reliability interventions (wait-time, order handling, staff communication).
- Treat price/value messaging as second-priority risk mitigation.
- Use topic tables and model features together for weekly triage and root-cause tracking.
- Keep middle-rating reviews under targeted review because they remain the hardest prediction segment.

## 7) Limitations
- Aspect seed words are interpretable but may miss synonyms/context.
- LDA topic labels still require human interpretation.
- Findings are Philadelphia-focused and may not generalize without revalidation.
- Results are predictive/associative, not causal.

## 8) Reproducibility
Run the complete workflow:

```bash
python main.py
```

Or run step-by-step:

```bash
python src/data_loader.py
python src/preprocess.py
python src/topics.py
python src/classifier.py
python src/aspects.py
python src/visualize.py
```
