# Twitter Sentiment Analysis on Indian Political Discourse
### MSc Data Science — University of Roehampton
**Student:** Karthikeya Vaitla | **ID:** M10710 | 

---

## 📌 Project Overview

This project investigates whether transformer-based deep learning models significantly outperform traditional machine learning baselines in classifying sentiment from Indian political Twitter data. Three models are implemented and compared on the same held-out test set:

| Model | Type | Accuracy |
|-------|------|----------|
| Naive Bayes + TF-IDF | Traditional ML (Baseline) | 74% |
| DistilBERT | Transformer (Fine-tuned) | 92% |
| RoBERTa | Transformer (Feature Extraction) | 50%* |

> *RoBERTa was trained on a 30k subset with frozen base layers due to GPU memory constraints. Both transformer models were evaluated on the identical test set ensuring fair comparison.

---

## 📁 File Structure

```
project/
├── 22_02_Need_Assignment_Karthikeya_Disse_M10710.ipynb   ← Main notebook
├── Twitter_Data.csv                                       ← Dataset (162,980 tweets)

```

---

## 📊 Dataset

- **Source:** Twitter_Data.csv
- **Size:** 162,980 tweets (after cleaning: 162,973 valid entries)
- **Columns:** `clean_text`, `category`
- **Domain:** Indian political discourse (BJP, Congress, Modi, Elections)
- **Language:** English (with political terminology)

### Label Distribution

| Label | Original Value | Mapped Value | Count | Percentage |
|-------|---------------|--------------|-------|------------|
| Negative | -1 | 0 | 35,510 | 21.8% |
| Neutral | 0 | 1 | 55,213 | 33.9% |
| Positive | 1 | 2 | 72,250 | 44.3% |

> Labels were shifted from (-1, 0, 1) → (0, 1, 2) using `category + 1` to satisfy PyTorch's CrossEntropyLoss requirement for non-negative integer labels.

---

## ⚙️ Environment Setup

### Running on Google Colab (Recommended)

```python
# Cell 0 — Run this first
from google.colab import drive
drive.mount('/content/drive')

# Set CSV path (upload Twitter_Data.csv to your Drive first)
CSV_PATH = "/content/drive/MyDrive/Twitter_Data.csv"

# Install dependencies
!pip install -q transformers torch tqdm certifi

# Model paths — downloaded from HuggingFace Hub automatically
DISTILBERT_PATH = "distilbert-base-uncased"
ROBERTA_PATH    = "roberta-base"
```

### Runtime Settings
- **Runtime type:** T4 GPU (Runtime → Change runtime type → T4 GPU)
- **Python version:** 3.12
- **Key libraries:** `transformers`, `torch`, `sklearn`, `pandas`, `seaborn`

### Install All Dependencies

```bash
pip install transformers torch tqdm certifi scikit-learn pandas numpy matplotlib seaborn nltk
```

---

## 🔄 Pipeline Overview

```
Raw CSV
   ↓
Text Preprocessing
   ↓
Train / Val / Test Split  (80% / 10% / 10%, Stratified)
   ↓
        ┌─────────────────────────────┐
        │                             │
   TF-IDF + NB            DistilBERT Tokenizer + RoBERTa Tokenizer
        │                             │
   Pipeline.fit()          TensorDataset → DataLoader
        │                             │
   Cross-Validation (5-fold)     Training Loop (5 epochs)
        │                             │
        └──────────┬──────────────────┘
                   ↓
           Evaluation on Same Test Set
                   ↓
        Accuracy / Precision / Recall / F1
                   ↓
           Confusion Matrix + Error Analysis
```

---

## 🧹 Preprocessing Steps

```python
def clean_text(text):
    text = text.lower()                          # lowercase
    text = re.sub(r'http\S+|www\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)     # remove special chars & numbers
    text = re.sub(r'\s+', ' ', text).strip()    # remove extra whitespace
    text = remove_stopwords(text)               # remove stopwords (negation-aware)
    return text
```

### Key Preprocessing Decisions

| Decision | Reason |
|----------|--------|
| Lowercase conversion | Normalises vocabulary — "Modi" and "modi" are same token |
| URL removal | URLs carry no sentiment signal |
| Special character removal | Reduces noise; transformers handle punctuation internally |
| Negation-aware stopwords | Words like "not", "no", "never" preserved — critical for sentiment |
| Label mapping -1→0, 0→1, 1→2 | PyTorch CrossEntropyLoss requires non-negative integers |

---

## 🤖 Model Details

### 1. Naive Bayes (Baseline)

```python
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ('nb', MultinomialNB())
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

| Parameter | Value | Reason |
|-----------|-------|--------|
| max_features | 5000 | Sufficient vocabulary without curse of dimensionality |
| ngram_range | (1,2) | Captures two-word sentiment phrases like "not good" |
| n_splits | 5 | Standard cross-validation for reliable baseline estimate |

---

### 2. DistilBERT

```python
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3
)
```

| Parameter | Value | Reason |
|-----------|-------|--------|
| max_length | 32 | Covers 95%+ of tweet lengths; reduces memory quadratically |
| batch_size | 64 | Fits T4 VRAM comfortably for DistilBERT's 66M parameters |
| epochs | 5 | Sufficient convergence; val accuracy plateaus after epoch 3 |
| learning_rate | 2e-5 | Standard from original BERT paper for fine-tuning |
| optimizer | AdamW | Correct weight decay decoupling — standard for transformers |
| warmup_steps | 0 | Short training run; warmup overhead not justified |
| grad_clip | 1.0 | Prevents exploding gradients in transformer training |
| Training samples | 130,378 | Full training set used |

**Architecture:** 6 transformer encoder layers, 12 attention heads, 768 hidden dimensions, 66M parameters. Distilled from BERT via knowledge distillation.

---

### 3. RoBERTa

```python
model_rb = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=3
)
for param in model_rb.roberta.parameters():
    param.requires_grad = False   # freeze base layers
```

| Parameter | Value | Reason |
|-----------|-------|--------|
| max_length | 24 | Reduced for speed; tweets well within this limit |
| batch_size | 128 | Larger batch possible since base layers frozen (less gradient memory) |
| epochs | 5 | Same as DistilBERT for consistency |
| learning_rate | 3e-5 | Slightly higher — only classification head trained |
| warmup_steps | 10% of total | Protects head initialisation in early training |
| Base layers | Frozen | Prevents catastrophic forgetting with limited data |
| Training samples | 30,000 | Reduced due to GPU memory constraints |
| Mixed precision | Yes (autocast) | Halves memory usage; enables larger batch size |

**Architecture:** 12 transformer encoder layers, 12 attention heads, 768 hidden dimensions, 125M parameters. Trained on 160GB of data with dynamic masking, no NSP task.

---

## 📈 Results

### Classification Reports

**DistilBERT — Test Set (16,298 samples)**
```
              precision  recall  f1-score  support
Negative          0.89    0.88      0.89     3551
Neutral           0.90    0.96      0.93     5522
Positive          0.95    0.91      0.93     7225
accuracy                            0.92    16298
weighted avg      0.92    0.92      0.92    16298
```

**Naive Bayes — Test Set (16,298 samples)**
```
              precision  recall  f1-score  support
Negative          0.85    0.47      0.61     3551
Neutral           0.81    0.69      0.75     5522
Positive          0.68    0.90      0.78     7225
accuracy                            0.74    16298
weighted avg      0.76    0.74      0.73    16298
```

**RoBERTa — Test Set (5,000 samples, frozen base)**
```
              precision  recall  f1-score  support
Negative          0.00    0.00      0.00     1049
Neutral           0.65    0.27      0.39     1691
Positive          0.48    0.91      0.63     2260
accuracy                            0.51     5000
```

---

## ⚠️ Known Limitations

1. **Unequal training data** — DistilBERT trained on 130k, RoBERTa on 30k. Both evaluated on the same test set. Training data difference is a compute constraint, not a methodological choice.

2. **RoBERTa base layers fully frozen** — With only 30k samples and a frozen encoder, the classification head had insufficient signal to adapt to political sentiment. Full fine-tuning on the complete dataset is the recommended next step.

3. **No early stopping** — DistilBERT val loss diverges slightly after epoch 2. A checkpoint-based early stopping mechanism would yield a slightly better generalising model.

4. **Class imbalance not handled** — Dataset has a 1:1.5:2 ratio (Negative:Neutral:Positive). SMOTE or class-weighted loss would be valid extensions.

5. **No hyperparameter search** — All learning rates and batch sizes taken from literature defaults. A grid search or Optuna sweep would likely improve results.

---

## 🔑 Key Design Choices Justified

**Why DistilBERT over full BERT?**
DistilBERT retains 97% of BERT performance at 40% fewer parameters. For max_length=32 (tweet-length sequences), distillation loss is negligible. Full BERT would OOM on T4 at batch_size=64.

**Why freeze RoBERTa base layers?**
With a reduced training set of 30k samples, full fine-tuning of 125M parameters risks catastrophic forgetting and overfitting. Freezing is a form of parameter-efficient transfer learning validated in low-data regimes.

**Why AdamW over Adam?**
Regular Adam's L2 regularisation is mathematically incorrect — weight decay is coupled with the gradient update. AdamW decouples them correctly. This is the standard optimiser for all transformer fine-tuning since 2018.

**Why negation-aware stopword removal?**
Standard NLP pipelines remove all stopwords including "not", "no", "never". For sentiment analysis, removing negation completely flips meaning — "not good" becomes "good". This is a deliberate, domain-aware preprocessing decision.

**Why 5-fold CV for Naive Bayes but not transformers?**
Naive Bayes trains in seconds — 5-fold CV costs negligible time and gives a more reliable estimate. Transformer cross-validation would require 5× the GPU hours (approximately 150 hours on T4). Both are evaluated on the same held-out test set for final comparison.

---

## 👨‍💻 Author

**Karthikeya**
MSc Data Science, University of Roehampton
Student ID: A00051441 / M10710
Notebook: `A00051441_KarthikeyaVaitla_Code.ipynb`
