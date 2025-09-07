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

# 🧪 [Sentiment Analysis Techniques](https://huggingface.co/spaces/AlbanDelamarre/Sentiment_Analysis_Techniques)

A modular Gradio app for **sentiment analysis** built to run on **Hugging Face Spaces**.
It supports **binary or multi‑class** datasets out of the box and lets you compare four families of techniques:

1. **ML Classifiers** — scikit‑learn + TF‑IDF
2. **Main NLP Baselines** — VADER and simple rules
3. **Transformers** — Hugging Face pipelines
4. **Prompt Engineering** — LLM zero/few‑shot classification via prompt templates (LangChain optional)

This project aligns with the practice tasks in the companion brief (load data, build ML baseline, try TF‑IDF, test a transformer, and explore prompt engineering).

---

## App Features

- **Data Tab**

  - Upload CSV or TXT, or load the included sample (`data/dataset.txt`).
  - Map **text** and **label** columns. Labels can be **2+ classes** (e.g., `positive`, `negative`, `neutral`, …).
  - Auto‑infer the set of classes and show **class distribution** and **imbalance warnings** (e.g., min samples per class).
  - Preprocessing toggles: lowercase, punctuation removal, stopwords, stemming **or** lemmatization.
  - Persist the preprocessing config in app state.

- **ML Classifiers**

  - Pipeline: `TfidfVectorizer → Model`.
  - Models: Logistic Regression, Linear SVM, Multinomial Naive Bayes, Random Forest.
  - **Multi‑class**: handled via native or **One‑vs‑Rest** as appropriate.
  - Train/test split with **stratify** and seed; warns if a class is too rare for the chosen split.
  - Metrics:
    - **Macro F1** (primary) and **micro F1**
    - Accuracy, macro/micro **Precision** and **Recall**
    - **Per‑class report** and **Confusion Matrix** (C×C)
  - Linear models: **top features per class**.
  - Save/load trained models (Space filesystem).

- **Main NLP Baselines**

  - **VADER** (best as a binary or 3‑way positive/neutral/negative baseline).
  - **Rules**: editable keyword lists per class (works for multi‑class by adding lists for each class).
  - Show raw scores and chosen label.

- **Transformers**

  - Pick a HF model and run inference via `transformers` pipeline (`text-classification`).
  - Works for **multi‑label** or **multi‑class** models (we expose label mapping and thresholding when needed).
  - Example defaults:
    - `distilbert-base-uncased-finetuned-sst-2-english` (binary)
    - `cardiffnlp/twitter-roberta-base-sentiment-latest` (3‑class; mapped as is)
  - Controls: max length, truncation, batch size.

- **Prompt Engineering**
  - Zero/few‑shot classification using an LLM with a **prompt template**.
  - **Multi‑class aware**: you specify the allowed class list; the prompt enumerates them and the parser enforces one of those outputs.
  - Two subtabs:
    - **Prompt Design**: base prompt, few‑shot examples, temperature, max tokens.
    - **Predict**: free‑form input; display raw LLM output + parsed label.
  - **Backends** (choose one):
    - **Local HF models** (via `text-generation` pipeline) where feasible.
    - **HF Inference API** (requires token).
    - **OpenAI** (requires API key).

---

## Project Layout

```
app.py                 # Gradio entry point
state.py               # Shared AppState container
tabs/
  data_tab.py          # Upload, preview, preprocessing
  ml/
    train_tab.py       # Train classic models
    predict_tab.py     # Predict using trained model
  nlp/
    select_tab.py      # VADER / Rules options
    predict_tab.py     # Baseline predictions
  transformers/
    select_tab.py      # HF model picker and params
    predict_tab.py     # Transformer predictions
  prompts/
    design_tab.py      # Prompt template + few‑shot examples
    predict_tab.py     # Run LLM with the prompt
models/
  saved/               # Persisted sklearn models (joblib)
data/
  dataset.txt          # Sample dataset (sentence, label)
requirements.txt
README.md
```

---

## Dataset Format

Text column + label column. Labels can be any number of classes:

- Binary: `positive`, `negative`
- Multi‑class: e.g., `very_negative`, `negative`, `neutral`, `positive`, `very_positive`

**CSV example**

```
sentence,label
"I loved this movie so much","positive"
"Terrible plot and acting","negative"
"It's fine, nothing special","neutral"
```

**Provided sample**: `data/dataset.txt` (tuple‑style rows, binary).

---

## Running on Hugging Face Spaces

### 1) Create a Space

- Space SDK: **Gradio**.
- Hardware: **CPU Basic** is enough for classical/TF‑IDF and small transformer inference. For larger models, pick a bigger instance or use the **HF Inference API**.

### 2) Repo files

Push this repo (or drag‑and‑drop in the Space UI). Include:

- `app.py`, `state.py`, `tabs/…`, `requirements.txt`, `README.md`, `data/dataset.txt`

### 3) Dependencies

Spaces auto‑install from `requirements.txt`. Suggested minimal set:

```
gradio>=4.0.0
pandas
scikit-learn
numpy
nltk
transformers
torch   # if you run local transformer pipelines
langchain>=0.2.0    # optional, for Prompt Engineering tab
huggingface_hub     # optional, if using Inference API
```

If you use VADER, we’ll download the lexicon at startup in code.

### 4) Secrets (optional)

Add secrets in **Settings → Repository secrets** if you plan to call hosted APIs:

- `HUGGINGFACEHUB_API_TOKEN` — to use the **HF Inference API**
- `OPENAI_API_KEY` — to use OpenAI via LangChain

The app will detect available keys and enable backends accordingly.

### 5) Launch

Spaces will run `app.py` automatically. The Gradio UI will appear after build.

---

## Evaluation

On the **ML Train** subtab we compute:

- **Macro F1** (primary) and **Micro F1**
- Accuracy, macro/micro Precision and Recall
- **Per‑class classification report**
- **Confusion matrix** sized **C×C** (C = number of classes)

Optional: ROC‑AUC (one‑vs‑rest) when decision scores are available.

---

## Model Options (Transformers)

- **Binary**: `distilbert-base-uncased-finetuned-sst-2-english`
- **3‑class**: `cardiffnlp/twitter-roberta-base-sentiment-latest`  
  (We keep its 3 labels; for binary comparisons you can map or merge in the UI.)

You can also paste any public `text-classification` model ID. Larger models may be slow on CPU Spaces—prefer the HF Inference API or smaller checkpoints.

---

## Prompt Engineering Tips (Multi‑class)

- **Enumerate allowed labels** in the prompt:
  ```
  You are a sentiment classifier. Return exactly one of: {labels}.
  Text: "{text}"
  Answer:
  ```
- Add **few‑shot examples** for each class to stabilize outputs.
- Use an **output parser** (regex / LangChain parser) to coerce the response to the set of labels.
- For ambiguous text, consider **self‑consistency** (sample N times, majority vote).

---

## Roadmap

- Batch prediction upload (CSV/TXT) across all tabs.
- Token attributions for transformers.
- Exportable experiment reports (Markdown/CSV).
- Optional fine‑tuning flow for transformers.
- Better guardrails for prompt outputs (self‑consistency, majority vote).

---
