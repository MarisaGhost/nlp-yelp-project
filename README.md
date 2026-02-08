# Yelp Restaurant Review NLP Project

This project applies Natural Language Processing (NLP) techniques to Yelp restaurant reviews in order to understand customer sentiment and explain star ratings using review text.

Instead of relying only on numerical ratings, the project analyzes free-text reviews to uncover common themes, predict ratings, and identify which aspects of the dining experience are most strongly associated with positive and negative feedback.

---

## Project Goals

The main objectives of this project are:

- Understand what customers talk about in positive vs. negative restaurant reviews
- Identify key topics and themes using topic modeling
- Predict Yelp star ratings (1–5) from review text
- Analyze which aspects (Food, Service, Price) are most associated with negative reviews
- Produce interpretable, business-relevant insights

This project was designed for an INSY669

---

## Dataset

The project uses the **Yelp Academic Dataset**.

Data processing steps:
- Filter businesses to **restaurants in Philadelphia**
- Merge review text with business information
- Downsample to ~100,000 reviews for manageable computation

Key processed files:
- `data/processed/merged_reviews.csv`
- `data/processed/reviews_processed.csv`

---

## Methods Overview

### Text Preprocessing
- Lowercasing
- URL and email removal
- spaCy tokenization and lemmatization
- Stopword, punctuation, and number removal
- Alphabetic tokens only

### Topic Modeling
- Latent Dirichlet Allocation (LDA)
- Separate models for:
  - Positive reviews (4–5 stars)
  - Negative reviews (1–2 stars)
- 5 topics per split
- Tuned vocabulary filtering (`filter_extremes`) per split
- Topic coherence (c_v) reported

### Rating Prediction
- Raw review text (minimal normalization)
- Hybrid TF-IDF features:
  - word n-grams (1-2)
  - character n-grams (3-5, `char_wb`)
- Linear SGD classifier (log-loss) with class balancing
- Small validation split for model selection
- 80/20 train-test split
- Evaluation with accuracy, macro F1, weighted F1, confusion matrix
- Word-level interpretability via model coefficients

### Aspect-Based Analysis
- Rule-based seed-word approach
- Aspects analyzed:
  - Food
  - Service
  - Price
- Comparison of mention rates in positive vs. negative reviews
- Lift analysis (negative / positive mention rate)
- Statistical significance testing (chi-square + odds ratio)
- Optional VADER sentiment analysis for aspect-related reviews

---

## Key Results

- **Text-based model performance**
  - Accuracy: ~0.65
  - Macro F1: ~0.59
  - Significantly better than majority-class baseline

- **Aspect insights**
  - Food is frequently mentioned in all reviews (baseline aspect)
  - Service and Price are relatively more prominent in negative reviews
  - These aspects are more likely drivers of dissatisfaction

- **Topic modeling**
  - Positive topics emphasize food quality and friendly service
  - Negative topics emphasize service issues, wait times, and value concerns

---

## Project Structure

nlp-yelp-project/
├── data/
│ ├── raw/ # Yelp JSON files
│ └── processed/ # Processed CSV files
├── outputs/
│ ├── tables/ # Result tables (CSV)
│ ├── figures/ # Saved plots
│ └── report.md # Final project report
├── src/
│ ├── data_loader.py
│ ├── preprocess.py
│ ├── topics.py
│ ├── classifier.py
│ ├── aspects.py
│ └── visualize.py
├── main.py # Run full pipeline
├── README.md
└── requirements.txt


---

# How to Run

This section explains how to set up the environment and reproduce all results in this project from scratch.

# Step 1: Clone or Download the Project

Download the project folder and navigate to the project root directory:

cd nlp-yelp-project


The directory should contain folders such as src/, data/, and outputs/.

# Step 2: Create and Activate Python Environment

This project was developed using Python 3.9.

Using Conda (recommended):

conda create -n nlp_project python=3.9
conda activate nlp_project

# Step 3: Install Required Packages

Install all required Python libraries using requirements.txt:

pip install -r requirements.txt


Download the spaCy English language model:

python -m spacy download en_core_web_sm

# Step 4: Prepare the Dataset

Download the Yelp Academic Dataset and place the raw files in:

data/raw/


The following files are required:

yelp_academic_dataset_business.json

yelp_academic_dataset_review.json

No preprocessing is needed at this stage.

# Step 5: Run the Full Pipeline (Recommended)

To reproduce all results, tables, figures, and the final report, run:

python main.py


This single command will execute the entire pipeline in the correct order:

Load and filter Yelp restaurant data

Preprocess review text

Train topic models (LDA)

Train rating prediction model

Run aspect-based analysis

Generate all figures

All outputs will be saved automatically.

# Step 6: (Optional) Run Scripts Individually

If you prefer to run each step manually, execute the scripts in the following order:

python src/data_loader.py
python src/preprocess.py
python src/topics.py
python src/classifier.py
python src/aspects.py
python src/visualize.py


This produces the same outputs as main.py.

# Step 7: Check Outputs

After running the pipeline, results can be found in:

Tables: outputs/tables/

Figures: outputs/figures/

Final Report: outputs/report.md

# Notes

Runtime depends on machine performance; preprocessing and topic modeling may take several minutes.

Results are deterministic due to fixed random seeds.

The analysis focuses on Philadelphia restaurants and may not generalize to other cities.
