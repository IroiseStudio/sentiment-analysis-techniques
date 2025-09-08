---
title: Sentiment Analysis Techniques
emoji: 🧪
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.44.0
python_version: '3.10'
app_file: app.py
pinned: false
license: cc
---

# 🧪 Sentiment Analysis Techniques

A modular Gradio app for **sentiment analysis** that runs locally or on **Hugging Face Spaces**.  
It supports **binary** and **multi-class** datasets and lets you compare four families of techniques side-by-side:

1. **ML Classifiers** — scikit-learn + TF-IDF
2. **Main NLP Baselines** — VADER and simple rules
3. **Transformers** — Hugging Face pipelines (sentiment & zero-shot)
4. **Prompt Engineering (Local)** — zero/few-shot via a strict prompt template using a **local Transformers pipeline**

A **Results & Comparison** section at the bottom summarizes metrics from each family using a shared test split when available.

---

## What’s Included (by Tab)

### Data

- Load CSV/TXT (or the sample `data/dataset.txt`).
- Choose your **text** and **label** columns.
- Live status banner (e.g., _dataset loaded: True | total: 350 | positive: 120 | negative: 230_).
- Quick stats: number of classes and **class distribution**.
- Optional preprocessing (persisted in app state): lowercase, punctuation removal, stopwords, stemming/lemmatization.
- Apply the selection to store the dataset in state (auto-updates the banner).

### ML Classifiers

- Pipeline: `TfidfVectorizer → Classifier`.
- Models: Logistic Regression, Linear SVM, Multinomial NB, Random Forest.
- Stratified **train/test split** with seed; shared across tabs for fair comparisons.
- Metrics after training: **Accuracy**, **Macro F1**, **Micro F1** and a **confusion matrix heatmap**.
- Predict subtab: test single texts on the trained model.

### Main NLP (Baselines)

- **VADER** or **keyword rules** (you can edit per-class keyword lists).
- Predict subtab: single-text predictions and **Evaluate on dataset** (uses the shared test split if present).

### Transformers

- Two modes:
  - **Sentiment**: load a finetuned `text-classification` model (e.g., `distilbert-base-uncased-finetuned-sst-2-english`, or a 3-class model like `cardiffnlp/twitter-roberta-base-sentiment-latest`).
  - **Zero-shot**: label texts without training using an NLI model (e.g., `facebook/bart-large-mnli`) and your label list/hypothesis template.
- Predict subtab: single-text predictions and **Evaluate on dataset**.

### Prompt Engineering (Local only)

- Design a strict **prompt template** that enumerates the allowed labels and (optionally) includes **few-shot examples**.
- Backend: **Local Transformers pipeline** only (no external APIs or tokens).
- Controls: model id (e.g., `google/flan-t5-small`), task (`text2text-generation` for seq2seq or `text-generation` for causal LMs), temperature, max new tokens.
- Predict subtab: shows both the model’s **raw output** and the **parsed label**. Evaluate on dataset to log metrics into **Results & Comparison**.

---

## Project Layout

```text
app.py                 # Gradio app entry
state.py               # Shared state (dataset, splits, configs, metrics)

tabs/
  data_tab.py          # Upload, map columns, preprocessing
  ml/
    train_tab.py       # Train TF-IDF + classical models
    predict_tab.py     # Predict with trained model
  nlp/
    select_tab.py      # VADER / Rules config
    predict_tab.py     # Baseline predictions + evaluation
  transformers/
    select_tab.py      # Sentiment / Zero-shot config
    predict_tab.py     # Transformer predictions + evaluation
  prompts/
    design_tab.py      # Local prompt template + few-shot setup
    predict_tab.py     # Local prompting predictions + evaluation

data/
  dataset.txt          # Sample dataset (text, label)

requirements.txt
README.md
```

---

## Dataset Format

Two columns: **text** + **label**. Labels may be **2+ classes**.

**CSV example**

```csv
sentence,label
"I loved this movie so much",positive
"Terrible plot and acting",negative
"It's fine, nothing special",neutral
```

**TXT example (one item per line)**

```text
I loved this movie so much	positive
Terrible plot and acting	negative
It's fine, nothing special	neutral
```

The repo includes a small sample at `data/dataset.txt`.

---

## Running Locally

```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

The app loads at `http://127.0.0.1:7860/`.  
(Use `share=True` in `demo.launch()` for a temporary public link while developing.)

---

## Running on Hugging Face Spaces

1. Create a Space (SDK: **Gradio**, Hardware: **CPU Basic** is fine for small models).
2. Push the repo files (`app.py`, `state.py`, `tabs/…`, `requirements.txt`, `README.md`, `data/dataset.txt`).
3. Build logs will show dependency install and the app will come up automatically.

**No secrets required** — the app uses local pipelines only.  
If a rebuild seems stuck, push a new commit or use **Settings → Factory reset** in the Space.

---

## Metrics & Comparison

Wherever possible, tabs use the same **stratified test split** created in the ML tab. Each family reports:

- **Accuracy**, **Macro F1**, **Micro F1**
- **Confusion matrix heatmap** (C×C)

The **Results & Comparison** section aggregates the latest metrics for quick side-by-side review.

---

## Tips for Prompt Engineering

- Keep outputs short and deterministic: `temperature=0.0`, `max_new_tokens` around 16–32.
- End your template with a strict instruction, e.g.:
  ```
  Return exactly one label from {labels} and nothing else.
  Text: {text}
  Final label:
  ```
- Add 1–3 few-shot examples per class if available.
- For multi-class (e.g., positive/neutral/negative), ensure `{labels}` contains **all** classes.

---

## Requirements

Minimal set for the current app:

```text
gradio>=4.0.0
pandas
numpy
scikit-learn
matplotlib
nltk
transformers>=4.41
torch
```

> If you previously installed LangChain or HF Hub clients, they’re not required for this version of the Prompt Engineering tab.

---

## License

Creative Commons (CC). See the Space header for details.
