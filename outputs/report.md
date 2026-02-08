# Yelp NLP Project Report

## 1) Background & Motivation
Star ratings are useful, but they do not explain *why* customers feel positive or negative.  
Review text gives richer signals about food quality, service behavior, wait time, and pricing concerns.  
In this project, I used NLP methods to turn free-text Yelp reviews into interpretable insights for restaurant decision-making.

## 2) Data
I used the Yelp Academic Dataset and focused on restaurant businesses in Philadelphia.  
The pipeline filters businesses to the restaurant category, merges reviews with business metadata, and builds a processed review table.  
The project setup targets a sample size around 100k reviews for manageable modeling in class settings.

Main data files used in this report:
- `data/processed/reviews_processed.csv`
- `outputs/tables/model_metrics.csv`
- `outputs/tables/top_features.csv`
- `outputs/tables/lda_topics_positive.csv`
- `outputs/tables/lda_topics_negative.csv`
- `outputs/tables/lda_coherence.csv`
- `outputs/tables/aspect_frequency.csv`
- `outputs/tables/aspect_significance.csv`
- `outputs/tables/aspect_sentiment.csv`

## 3) Methods

### Preprocessing
- Input text: `cleaned_text` pipeline built from Yelp review text.
- Steps: lowercase, remove URLs/emails, spaCy tokenization + lemmatization, remove stopwords/punctuation/numbers, keep alphabetic tokens.
- Output columns include normalized text and tokenized forms for downstream analysis.

### Topic Modeling
- I trained separate LDA models for:
  - Positive reviews (stars 4-5)
  - Negative reviews (stars 1-2)
- Settings: 5 topics per split, top 10 words per topic.
- Vocabulary was filtered with split-specific thresholds (`filter_extremes`) to reduce noise and improve coherence.
- Coherence (`c_v`) from the run:
  - Positive: **0.4300**
  - Negative: **0.4246**

### Rating Prediction
- Task: predict star rating (1-5) from raw review `text`.
- Features: hybrid TF-IDF
  - word n-grams (`ngram_range=(1,2)`)
  - character n-grams (`char_wb`, `ngram_range=(3,5)`)
- Model: linear SGD classifier (log-loss) with class balancing.
- Model selection: validation split over multiple SGD settings; best model selected by validation macro F1.
- Evaluation split: 80/20 train-test, `random_state=42`.

### Aspect Analysis
- Seed-word method for three aspects: **Food, Service, Price**.
- For positive and negative splits, I computed:
  - mention rate (% reviews mentioning an aspect at least once)
  - total mentions (count of matched seed words)
- I ran chi-square tests and odds-ratio calculations to check whether aspect-rate gaps are statistically significant.
- I also used VADER sentiment (optional extension) to compare average compound polarity for aspect-related reviews.

## 4) Results & Insights

### Classifier Performance
From `outputs/tables/model_metrics.csv`:
- **Accuracy = 0.6482**
- **Macro F1 = 0.5949**
- **Weighted F1 = 0.6430**
- **Majority-class baseline accuracy = 0.4051**

Interpretation:
- The model clearly improves over baseline, so text features add predictive signal.
- Compared with the earlier baseline (~0.586 accuracy, ~0.546 macro F1), this setup gives a meaningful lift.
- Mid ratings (especially 2-4) are harder to separate than extreme ratings, as seen in the confusion matrix figure.

Figures:
- `outputs/figures/confusion_matrix.png`
- `outputs/figures/confusion_matrix_normalized.png`

### Aspect Frequency and Lift
From `outputs/tables/aspect_frequency.csv`, negative/positive mention-rate lift:
- **Service lift ~1.175**
- **Price lift ~1.159**
- **Food lift ~0.921** (used as a baseline-style reference aspect)

Interpretation:
- Service and price are discussed relatively more in negative reviews.
- Food appears heavily in both splits, but proportionally less dominant in negatives than service/price.

Figure:
- `outputs/figures/aspect_mention_rate.png`

From `outputs/tables/aspect_significance.csv`:
- All three aspect differences are statistically significant at the 0.05 level.
- Strongest over-representation in negative reviews is **Service** (odds ratio ~1.71, p-value ~8.17e-174).

### Optional Aspect Sentiment
From `outputs/tables/aspect_sentiment.csv`:
- Positive reviews mentioning aspects show high mean compound scores (~0.89-0.91).
- Negative reviews mentioning aspects are much lower (~0.26-0.36).
- Service has one of the lowest sentiment levels on the negative side, consistent with complaint-driven reviews.

### LDA Topic Themes (Plain-Language Summary)
From `outputs/tables/lda_topics_positive.csv`, `outputs/tables/lda_topics_negative.csv`, and `outputs/tables/lda_coherence.csv`:
- Positive topics emphasize words related to tasty food, friendly service, drinks, and overall good experience.
- Negative topics emphasize waiting/order issues, customer-service friction, and dissatisfaction with taste/value.
- Across both splits, recurring themes include service speed, ordering process, menu/food quality, and pricing/value perception.

Figures:
- Positive topic bars: `outputs/figures/lda_positive_topic_0.png` to `outputs/figures/lda_positive_topic_4.png`
- Negative topic bars: `outputs/figures/lda_negative_topic_0.png` to `outputs/figures/lda_negative_topic_4.png`

## 5) Limitations
- Seed-word aspect analysis depends on manual keyword choices and can miss synonyms/context.
- LDA topics are useful summaries but not perfectly interpretable; topic labels are subjective.
- The analysis is city-specific (Philadelphia), so patterns may not generalize to other markets.
- Text-only modeling ignores useful metadata (user history, business attributes, time effects).

## 6) Business Recommendations
- Prioritize service operations (speed, staff responsiveness, order accuracy), since service terms are overrepresented in negative feedback.
- Monitor pricing/value perception closely (especially words like overpriced/charge/value) and align portion/quality with price expectations.
- Use food-related positive terms (e.g., delicious, fresh, favorite) in marketing language and menu highlights.
- Track aspect mention rates monthly to detect early shifts in complaint patterns before ratings drop.
- Combine model predictions with manual review sampling for high-risk reviews to improve actionability.

## 7) Reproducibility
Run the pipeline scripts in order:

```bash
python src/data_loader.py
python src/preprocess.py
python src/topics.py
python src/classifier.py
python src/aspects.py
python src/visualize.py
```

Or run a single orchestrator script if you maintain one (for example, `python main.py`).
