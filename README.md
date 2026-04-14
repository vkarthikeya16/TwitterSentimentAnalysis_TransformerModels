# Twitter Sentiment Analysis — Indian Political Tweets

> **MSc Data Science Dissertation** · University of Roehampton  
> Comparing DistilBERT fine-tuning against a Naive Bayes TF-IDF baseline for three-class sentiment classification on Indian political Twitter data.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Models](#models)
  - [DistilBERT (Transformer)](#distilbert-transformer)
  - [Naive Bayes (Baseline)](#naive-bayes-baseline)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Author](#author)

---

## Overview

This project investigates the effectiveness of transformer-based language models versus classical machine learning approaches for sentiment analysis on politically charged, domain-specific social media text. The study frames sentiment classification as a three-class problem — **Negative**, **Neutral**, and **Positive** — using a curated dataset of Indian political tweets.

The core contribution is a rigorous side-by-side evaluation of:

- **DistilBERT** (`distilbert-base-uncased`) fine-tuned for sequence classification
- **Multinomial Naive Bayes** with TF-IDF feature extraction as an interpretable baseline

---

## Dataset

| Property | Value |
|---|---|
| Source | `Twitter_Data.csv` |
| Domain | Indian political discourse |
| Working sample | 20,000 tweets (stratified random sample, `random_state=42`) |
| Target column | `category` (`-1` Negative · `0` Neutral · `1` Positive) |
| Preprocessing | Regex cleaning, NLTK stopword removal, text length feature engineering |

---

## Project Architecture

```
Raw CSV
   │
   ▼
Data Loading & EDA
   │  ├── Shape, column inspection, summary statistics
   │  ├── Sentiment class distribution (bar chart)
   │  └── Text length distribution (histogram, per-class mean)
   │
   ▼
Text Preprocessing
   │  ├── Regex-based noise removal
   │  └── NLTK stopword filtering
   │
   ├──────────────────────────────────────┐
   ▼                                      ▼
DistilBERT Pipeline                 Naive Bayes Pipeline
   │                                      │
   ├── Tokenization (max_length=32)        ├── TF-IDF Vectorization (max_features=5000)
   ├── 80 / 10 / 10 stratified split       ├── 80 / 20 stratified split
   ├── TensorDataset + DataLoader          ├── MultinomialNB fit
   ├── Fine-tuning (AdamW, lr=2e-5)        └── Evaluation
   └── Evaluation (Val + Test)
   │
   ▼
Comparative Evaluation
   └── Accuracy · Precision · Recall · F1 · Confusion Matrices
```

---

## Models

### DistilBERT (Transformer)

| Hyperparameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Classification head | `DistilBertForSequenceClassification` (3 labels) |
| Tokenizer max length | 32 tokens |
| Optimizer | AdamW |
| Learning rate | `2e-5` |
| Epochs | 1 |
| Train / Val / Test split | 80% / 10% / 10% (stratified) |
| Batch computation | GPU if available, CPU fallback |

**Training loop summary:**

```python
optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(input_ids=..., attention_mask=..., labels=...)
        outputs.loss.backward()
        optimizer.step()
```

Evaluation uses `torch.no_grad()` inference with `argmax` over logits for class prediction.

---

### Naive Bayes (Baseline)

| Hyperparameter | Value |
|---|---|
| Vectorizer | `TfidfVectorizer` |
| Max features | 5,000 |
| Classifier | `MultinomialNB` |
| Train / Test split | 80% / 20% (stratified) |

Provides a computationally lightweight, interpretable reference point to quantify the representational advantage of the transformer architecture.

---

## Results

Metrics reported on held-out test sets using **weighted averaging** across all three sentiment classes.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| DistilBERT (fine-tuned) | — | — | — | — |
| Naive Bayes (TF-IDF) | — | — | — | — |

> **Note:** Fill in the metric values after running the notebook. Confusion matrices for both models are generated inline within the notebook cells.

---

## Repository Structure

```
├── A00051441_KarthikeyaVaitla_Dissertation.ipynb   # Main analysis notebook
├── Twitter_Data.csv                                 # Dataset (add to project root)
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place `Twitter_Data.csv` in the project root directory before running the notebook.

---

## Usage

Launch Jupyter and open the dissertation notebook:

```bash
jupyter notebook A00051441_KarthikeyaVaitla_Dissertation.ipynb
```

Run cells sequentially. The notebook is self-contained and covers:

1. Data loading and exploratory analysis
2. Text preprocessing
3. DistilBERT tokenisation and training
4. DistilBERT validation and test evaluation
5. Naive Bayes training and evaluation
6. Side-by-side metric comparison and confusion matrices

GPU acceleration is detected automatically via `torch.cuda.is_available()`. CPU execution is supported but will be significantly slower for the DistilBERT training cell.

---

## Dependencies

```
torch
transformers
scikit-learn
pandas
numpy
matplotlib
nltk
tqdm
```

Generate a `requirements.txt` with:

```bash
pip freeze > requirements.txt
```

---

## Author

**Karthikeya Vaitla**  
MSc Data Science · University of Roehampton  
Student ID: A00051441

---

*This project was completed as part of the MSc Data Science dissertation requirement. All design choices, training configurations, and evaluation methodologies are documented within the notebook.*
