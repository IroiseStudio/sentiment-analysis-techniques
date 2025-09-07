from __future__ import annotations
from typing import Callable, Optional, Dict, Any, List
import gradio as gr
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

_state_getter: Optional[Callable] = None

def bind_state(getter: Callable):
    global _state_getter
    _state_getter = getter

def tr_status_text() -> str:
    st = _state_getter()
    mode = (st.transformer_config or {}).get("mode", "sentiment")
    mid = st.transformer_id or "—"
    cfg = st.transformer_config or {}
    if mode == "sentiment":
        mp = cfg.get("map_positive") or "positive"
        mn = cfg.get("map_negative") or "negative"
        return (
            "#### Transformers settings: Sentiment\n"
            f"• model: `{mid}`  \n"
            f"• mapping: POS→{mp}, NEG→{mn}"
        )
    else:
        hyp = cfg.get("hypothesis_template", "This text is {}.")
        st_labels = getattr(st, "labels_", []) or []
        return (
            "#### Transformers settings: Zero-shot\n"
            f"• model: `{mid}`  \n"
            f"• hypothesis_template: “{hyp}”  \n"
            f"• candidate labels: {', '.join(st_labels) if st_labels else '—'}"
        )

def _predict_one_sentiment(pipe, text: str, pos_map: str, neg_map: str) -> str:
    out = pipe(text)  # [{'label': 'POSITIVE', 'score': 0.99}]
    lab = (out[0]["label"] if isinstance(out, list) else out["label"]).upper()
    if "POS" in lab:
        return pos_map
    elif "NEG" in lab:
        return neg_map
    # Fallbacks for LABEL_0/LABEL_1, assume LABEL_1 = positive
    if lab == "LABEL_1":
        return pos_map
    return neg_map

def _predict_one_zeroshot(pipe, text: str, labels: List[str], hyp: str) -> str:
    res = pipe(text, candidate_labels=labels, hypothesis_template=hyp, multi_label=False)
    # res: {'sequence':..., 'labels': [...], 'scores': [...]}
    return res["labels"][0] if isinstance(res, dict) else res[0]["labels"][0]

def make_tab():
    status_md = gr.Markdown(tr_status_text())

    gr.Markdown("### Transformers Predict")
    inp = gr.Textbox(label="Input text", lines=3, placeholder="Type a sentence to classify…")
    btn = gr.Button("Predict")
    out = gr.JSON(value={"label": None, "note": "Load a Transformer in the Select subtab first."})

    gr.Markdown("### Evaluate on dataset")
    eval_btn = gr.Button("Evaluate Transformer on dataset")
    eval_json = gr.JSON(value={"status": "Not evaluated yet"})

    def on_predict(text: str):
        st = _state_getter()
        if not text or not text.strip():
            return {"label": None, "note": "Empty input."}
        pipe = getattr(st, "transformer_pipeline", None)
        cfg = getattr(st, "transformer_config", {}) or {}
        if pipe is None:
            return {"label": None, "note": "No transformer loaded. Go to Select subtab."}

        mode = cfg.get("mode", "sentiment")
        if mode == "sentiment":
            pos_map = cfg.get("map_positive") or "positive"
            neg_map = cfg.get("map_negative") or "negative"
            label = _predict_one_sentiment(pipe, text, pos_map, neg_map)
            return {"label": label}
        else:
            labels = getattr(st, "labels_", []) or None
            if not labels:
                return {"label": None, "note": "Dataset labels not available. Load & apply labels in Data tab."}
            hyp = cfg.get("hypothesis_template", "This text is {}.")
            label = _predict_one_zeroshot(pipe, text, labels, hyp)
            return {"label": label}

    btn.click(on_predict, [inp], [out])

    def _eval_dataset():
        st = _state_getter()
        df = getattr(st, "df_clean", None)
        tcol = getattr(st, "text_col", None)
        lcol = getattr(st, "label_col", None)
        pipe = getattr(st, "transformer_pipeline", None)
        cfg = getattr(st, "transformer_config", {}) or {}

        if df is None or tcol is None or lcol is None:
            return {"error": "Load dataset in Data tab first."}
        if pipe is None:
            return {"error": "Load a transformer in the Select subtab first."}

        # indices to evaluate: reuse ML split if available, else whole dataset
        split = getattr(st, "eval_split", {}) or {}
        idx_eval = split.get("test", None) or df.index.tolist()

        texts = df.loc[idx_eval, tcol].astype(str).tolist()
        y_true = df.loc[idx_eval, lcol].astype(str).tolist()
        labels_all = sorted(df[lcol].astype(str).unique().tolist())

        preds: List[str] = []
        mode = cfg.get("mode", "sentiment")
        if mode == "sentiment":
            pos_map = cfg.get("map_positive") or "positive"
            neg_map = cfg.get("map_negative") or "negative"
            for tx in texts:
                preds.append(_predict_one_sentiment(pipe, tx, pos_map, neg_map))
        else:
            cand_labels = getattr(st, "labels_", []) or labels_all
            hyp = cfg.get("hypothesis_template", "This text is {}.")
            for tx in texts:
                preds.append(_predict_one_zeroshot(pipe, tx, cand_labels, hyp))

        rpt = classification_report(np.array(y_true).astype(str), np.array(preds).astype(str),
                                    output_dict=True, zero_division=0)
        mets = {
            "macro_f1": rpt.get("macro avg", {}).get("f1-score", None),
            "micro_f1": rpt.get("micro avg", {}).get("f1-score", None),
            "accuracy": rpt.get("accuracy", None),
        }
        st.tr_metrics = {"metrics": mets, "n_eval": len(idx_eval), "mode": mode, "model": st.transformer_id, "config": cfg}
        return {"status": "ok", **st.tr_metrics}

    eval_event = eval_btn.click(_eval_dataset, inputs=None, outputs=eval_json)

    return {"status_md": status_md, "eval_event": eval_event}
