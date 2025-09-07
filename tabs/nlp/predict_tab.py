from __future__ import annotations
from typing import Callable, Optional, Dict, Any, List, Tuple
import re
import gradio as gr
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

_state_getter: Optional[Callable] = None
_vader: Optional[SentimentIntensityAnalyzer] = None

def bind_state(getter: Callable):
    global _state_getter
    _state_getter = getter

def _ensure_vader():
    global _vader
    if _vader is None:
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon")
        _vader = SentimentIntensityAnalyzer()
    return _vader

def nlp_status_text() -> str:
    st = _state_getter()
    method = getattr(st, "nlp_method", None) or "VADER"
    cfg = getattr(st, "nlp_config", {}) or {}
    if method == "VADER":
        pt = cfg.get("pos_threshold", 0.05)
        nt = cfg.get("neg_threshold", -0.05)
        en = bool(cfg.get("enable_neutral", False))
        mp = cfg.get("map_positive", "positive")
        mn = cfg.get("map_negative", "negative")
        mu = cfg.get("map_neutral", None)
        neu_txt = f"enabled → {mu}" if en and mu else "disabled"
        return (
            "#### NLP settings: VADER  \n"
            f"• pos_threshold: {pt} | neg_threshold: {nt} | neutral: {neu_txt}  \n"
            f"• mapping: POS→{mp}, NEG→{mn}" + (f", NEU→{mu}" if en and mu else "")
        )
    elif method == "Rules":
        rules = (cfg.get("rules", {}) or {})
        cs = bool(cfg.get("case_sensitive", False))
        mode = cfg.get("mode", "most_matches")
        parts = [f"{lab}({len(kws) if isinstance(kws, list) else 0})" for lab, kws in rules.items()]
        return (
            "#### NLP settings: Rules  \n"
            f"• keywords per label: {', '.join(parts) if parts else '—'}  \n"
            f"• case_sensitive: {cs} | mode: {mode}"
        )
    else:
        return f"#### NLP settings: {method}"

def _guess_label(labels: List[str], target: str) -> Optional[str]:
    t = target.lower()
    for lab in labels:
        if lab.lower() == t:
            return lab
    for lab in labels:
        if t in lab.lower() or lab.lower() in t:
            return lab
    return None

def _vader_predict_one(text: str, cfg: Dict[str, Any], labels_fallback: List[str]) -> str:
    sia = _ensure_vader()
    s = sia.polarity_scores(text or "")
    c = s.get("compound", 0.0)
    pos_t = float(cfg.get("pos_threshold", 0.05))
    neg_t = float(cfg.get("neg_threshold", -0.05))
    use_neu = bool(cfg.get("enable_neutral", False))
    pos_lbl = cfg.get("map_positive") or _guess_label(labels_fallback, "positive") or labels_fallback[0]
    neg_lbl = cfg.get("map_negative") or _guess_label(labels_fallback, "negative") or labels_fallback[-1]
    neu_lbl = cfg.get("map_neutral")
    if c >= pos_t:
        return pos_lbl
    elif c <= neg_t:
        return neg_lbl
    else:
        if use_neu and neu_lbl:
            return neu_lbl
        return pos_lbl if c >= 0 else neg_lbl

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text or "")

def _rules_predict_one(text: str, cfg: Dict[str, Any]) -> Optional[str]:
    rules: Dict[str, List[str]] = cfg.get("rules", {}) or {}
    case_sensitive = bool(cfg.get("case_sensitive", False))
    mode = cfg.get("mode", "most_matches")
    text_proc = text if case_sensitive else (text or "").lower()
    tok_set = set(_tokenize(text_proc))
    counts: Dict[str, int] = {}
    for label, kw_list in rules.items():
        if not isinstance(kw_list, list):
            continue
        kws = [str(k) if case_sensitive else str(k).lower() for k in kw_list]
        counts[label] = sum(1 for k in kws if k in tok_set)
    if mode == "first_match":
        for label in rules.keys():
            if counts.get(label, 0) > 0:
                return label
        return None
    # most_matches
    chosen, best = None, -1
    for label in rules.keys():
        c = counts.get(label, 0)
        if c > best:
            best = c; chosen = label
    return chosen if best > 0 else None

def make_tab():
    status_md = gr.Markdown(nlp_status_text())
    gr.Markdown("### NLP Baseline Predict")
    inp = gr.Textbox(label="Input text", lines=3, placeholder="Type a sentence to classify…")
    btn = gr.Button("Predict")
    out = gr.JSON(value={"label": None, "scores": {}, "note": "Choose settings in the NLP Select tab and try again."})

    # Evaluate on dataset
    gr.Markdown("### Evaluate on dataset (uses ML test split if available)")
    eval_btn = gr.Button("Evaluate NLP on dataset")
    eval_json = gr.JSON(value={"status": "Not evaluated yet"})

    def on_predict(text: str):
        st = _state_getter()
        method = getattr(st, "nlp_method", None) or "VADER"
        cfg = getattr(st, "nlp_config", {}) or {}
        labels_fb = getattr(st, "labels_", []) or ["positive", "negative", "neutral"]

        if not text or not text.strip():
            return {"label": None, "scores": {}, "note": "Empty input."}

        if method == "VADER":
            label = _vader_predict_one(text, cfg, labels_fb)
            return {"label": label, "scores": {}, "note": None}
        elif method == "Rules":
            label = _rules_predict_one(text, cfg)
            note = None if label else "No rule matched; consider adding keywords or using VADER/ML."
            return {"label": label, "scores": {}, "note": note}
        else:
            return {"label": None, "scores": {}, "note": f"Unknown method '{method}'."}

    btn.click(on_predict, [inp], [out])

    def _eval_dataset():
        st = _state_getter()
        df = getattr(st, "df_clean", None)
        tcol = getattr(st, "text_col", None)
        lcol = getattr(st, "label_col", None)
        method = getattr(st, "nlp_method", None) or "VADER"
        cfg = getattr(st, "nlp_config", {}) or {}

        if df is None or tcol is None or lcol is None:
            return {"error": "Load dataset in Data tab first."}

        # choose indices: ML test split if available, else full dataset
        split = getattr(st, "eval_split", {}) or {}
        idx_eval = split.get("test", None)
        if not idx_eval:
            idx_eval = df.index.tolist()

        X = df.loc[idx_eval, tcol].astype(str).tolist()
        y_true = df.loc[idx_eval, lcol].astype(str).tolist()
        labels_all = sorted(df[lcol].astype(str).unique().tolist())

        preds: List[Optional[str]] = []
        if method == "VADER":
            labels_fb = getattr(st, "labels_", []) or labels_all
            for text in X:
                preds.append(_vader_predict_one(text, cfg, labels_fb))
        elif method == "Rules":
            for text in X:
                preds.append(_rules_predict_one(text, cfg))
        else:
            return {"error": f"Unknown method '{method}'."}

        # Replace None with a fallback (e.g., most frequent class) to avoid metric errors
        if any(p is None for p in preds):
            fallback = pd.Series(y_true).mode().iloc[0]
            preds = [p if p is not None else fallback for p in preds]

        rpt = classification_report(np.array(y_true).astype(str), np.array(preds).astype(str),
                                    output_dict=True, zero_division=0)
        mets = {
            "macro_f1": rpt.get("macro avg", {}).get("f1-score", None),
            "micro_f1": rpt.get("micro avg", {}).get("f1-score", None),
            "accuracy": rpt.get("accuracy", None),
        }
        st.nlp_metrics = {"metrics": mets, "n_eval": len(idx_eval), "method": method, "config": cfg}
        return {"status": "ok", **st.nlp_metrics}

    eval_event = eval_btn.click(_eval_dataset, inputs=None, outputs=eval_json)

    return {"status_md": status_md, "eval_event": eval_event}
