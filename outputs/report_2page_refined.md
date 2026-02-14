# Yelp NLP Report (Two-Page Version)

## Business problem and motivation
Restaurant teams often track performance with star averages, but a single number hides why the score moved. This project converts Yelp review text into actionable operational insight by combining topic modeling (to surface dominant themes), interpretable text classification (to triage sentiment/ratings), and aspect-level statistical testing (to detect whether Food, Service, and Price are disproportionately associated with negative experiences).

This problem is highly relevant to today’s restaurant and service industries, where customer feedback volume exceeds what managers can manually review. The solution is designed for practical decision support: faster complaint escalation, clearer root-cause prioritization, and better evidence for staffing/training/pricing actions.

## Data processing and text engineering
Data source: Yelp Academic Dataset (https://business.yelp.com/data/resources/open-dataset/). Review and business records were merged to focus on Philadelphia restaurants, producing approximately 100,000 reviews.

Reviews were normalized using lowercase conversion and URL/email removal to reduce noise and stabilize tokenization. NLTK lemmatization was applied for interpretable comparisons and aspect retrieval. For topic modeling, POS-filtered tokens were used to retain nouns/proper nouns and adjectives, with optional adverbs. This reduced narrative noise and improved topic coherence.

## Methods and tools
### 1) Topic modeling (LDA)
Two separate LDA models were trained: positive reviews (4-5 stars) and negative reviews (1-2 stars). Each model produced 5 topics with top 10 words per topic. Coherence (`c_v`) was used for quality evaluation.

A split-model design was used because praise and complaint language differ; a single mixed model can produce vague or contradictory topics. Split models yielded clearer, more operationally useful themes.

Tuned coherence results:
- Positive split: **0.4851**
- Negative split: **0.4314**
- Coherence deltas vs prior baseline: **+0.0551** (positive), **+0.0069** (negative)

### 2) Document classification
Reviews were represented with hybrid TF-IDF features combining word n-grams (1-2) and character n-grams (`char_wb`, 3-5), which fits Yelp text noise and spelling variation while preserving semantic signals.

Primary operational target: binary triage (1-2 vs 4-5 stars), excluding 3-star reviews because neutral/mixed feedback weakens strict risk triage.

Compared models (same split and feature setup):
- SGDClassifier
- ComplementNB
- KNN (cosine, k=5)

Validation accuracy:
- **SGDClassifier (best; hinge, alpha=1e-4): 0.95470**
- ComplementNB: **0.93335**
- KNN: **0.87241**

A targeted hyperparameter sweep was run for the winning model (`loss` in {hinge, log_loss}, `alpha` in {1e-4, 1e-3, 1e-2}). The best remained `hinge`, `alpha=1e-4`, indicating a stable optimum under this setup.

Secondary exploratory task: 5-class star prediction. A validation-driven model selection process compared multiple SGD/feature configurations and selected `log_loss`, `alpha=3e-5`, class-weight balanced, with hybrid TF-IDF.

Held-out test metrics (5-class):
- Accuracy: **0.64815**
- Macro F1: **0.59486**
- Weighted F1: **0.64297**
- Majority baseline accuracy: **0.40505**

The gap between weighted and macro F1 indicates stronger performance on frequent patterns and weaker performance on harder edge classes; confusion is concentrated in adjacent ratings.

Implementation controls used to improve reliability:
- Stratified data splits with fixed `random_state=42` for reproducibility.
- TF-IDF fit on training data only, then applied to validation/test (leakage prevention).
- Class-weight balancing in the selected 5-class model to reduce majority-class bias.

### 2.1) Experimentation and classifier quality (binary triage)
To strengthen experimental rigor, we added cross-validation, broader tuning, confusion-matrix analysis, ROC-AUC reporting, error analysis, and feature ablation:
- **5-fold cross-validation (best SGD setup)**: accuracy **0.9657 +/- 0.0007**, F1 **0.9786 +/- 0.0005**, ROC-AUC **0.9912 +/- 0.0004**.
- **Hyperparameter grid**: 30 combinations (`loss` in {hinge, log_loss, modified_huber}, `alpha` in {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}, `class_weight` in {None, balanced}). Best validation setting: `loss=hinge`, `alpha=3e-5`, `class_weight=None`.
- **Holdout confusion matrix** (n=17,329): TN=3,209, FP=325, FN=225, TP=13,570. Holdout metrics: accuracy **0.9683**, F1 **0.9801**, ROC-AUC **0.9924**.
- **Error analysis**: hardest cases are very short texts (e.g., one-word reviews), mixed-sentiment reviews containing both praise and complaints, and contrast-heavy language.
- **Ablation study** (same classifier settings): hybrid features performed best overall (F1 **0.9801**) vs word-only (F1 **0.9790**) and char-only (F1 **0.9756**), showing value from combining semantic and subword signals.

### 3) Aspect-based statistical testing
Seed-word retrieval identified aspect mentions for Food, Service, and Price. Mention-rate differences were tested between negative (1-2 stars) and positive (4-5 stars) subsets using lift, odds ratio, and chi-square significance.

Results:
- **Service**: lift **1.1749**, odds ratio **1.7053**, chi-square **789.97**, p-value **8.17E-174**
- **Price**: lift **1.1594**, odds ratio **1.2833**, chi-square **214.11**, p-value **1.74E-48**
- **Food**: lift **0.9212**, odds ratio **0.6300**, chi-square **476.64**, p-value **1.15E-105**

Service is the strongest disproportionate negative driver; price/value also skews negative. Food is frequently mentioned in both classes but is relatively less concentrated in negative reviews.

## Results and business impact
The solution provides immediate business value by converting unstructured feedback into prioritized actions:
- Real-time triage of high-risk reviews via high-accuracy binary classification.
- Theme-level diagnosis from split LDA topics for positive vs negative experience drivers.
- Statistically supported prioritization of operational interventions (service first, then value/price).

Expected business effects include reduced manual review time, faster service recovery, lower impact of negative word-of-mouth, and better ROI on staffing/training/process redesign. The approach also supports consistent monitoring across locations and earlier detection of recurring service failures.

## Conclusion
This analytic workflow turns Yelp text into concrete operational guidance rather than passive reporting. The combination of interpretable topics, high-performing text classification, and statistically significant aspect analysis provides a repeatable feedback loop for management decisions. In this dataset, the strongest evidence supports prioritizing service reliability and price/value perception improvements before broader, less targeted interventions.

## Limitations
Findings are associative, not causal. Aspect seed words may miss synonyms/context, and results are Philadelphia-specific, so external deployment should include local revalidation.
