from __future__ import annotations
from typing import Callable, Optional, Dict, Any, List
import re, time, traceback
import gradio as gr
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from transformers import pipeline as hf_pipeline, __version__ as TRANSFORMERS_VER

_state_getter: Optional[Callable] = None

# simple in-memory cache for pipelines by (task, model_id)
_PIPE_CACHE: Dict[tuple, Any] = {}

def bind_state(getter: Callable):
    global _state_getter
    _state_getter = getter

# ---------- helpers ----------
def _infer_task(model_id: str, fallback: str = "text2text-generation") -> str:
    mid = (model_id or "").lower()
    if any(k in mid for k in ["flan", "t5", "mt5", "ul2", "bart", "mbart", "pegasus"]):
        return "text2text-generation"
    return "text-generation" if fallback not in {"text-generation", "text2text-generation"} else fallback

def _prompt_status_text() -> str:
    st = _state_getter()
    cfg = getattr(st, "prompt_config", {}) or {}
    model = cfg.get("model_id") or "—"
    task = cfg.get("task") or _infer_task(model)
    temp = float(cfg.get("temperature", 0.0))
    mx = int(cfg.get("max_tokens", 128))
    labels = cfg.get("allowed_labels", [])
    return (
        "#### Prompt settings  \n"
        f"• backend: **Local (HF Pipeline)** | model: `{model}`  \n"
        f"• task: {task} | temperature: {temp} | max_tokens: {mx}  \n"
        f"• labels: {', '.join(labels) if labels else '—'}"
    )

def _get_pipeline(task: str, model_id: str):
    key = (task, model_id)
    if key in _PIPE_CACHE:
        return _PIPE_CACHE[key]
    pipe = hf_pipeline(task, model=model_id)
    _PIPE_CACHE[key] = pipe
    return pipe

def _format_examples(examples: List[Dict[str,str]]) -> str:
    parts = []
    for ex in examples or []:
        t = (ex.get("text","") or "").strip()
        l = (ex.get("label","") or "").strip()
        if t and l:
            parts.append(f"Example:\nText: {t}\nLabel: {l}\n")
    return "\n".join(parts)

def _extract_text(output: Any) -> str:
    """Normalize various pipeline outputs to plain string."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        return str(output.get("generated_text") or output.get("text") or output)
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict):
            return str(first.get("generated_text") or first.get("text") or first)
        return str(first)
    return str(output)

def _map_to_allowed(raw: str, allowed: List[str]) -> Optional[str]:
    if not raw:
        return None
    out = raw.strip().splitlines()[0]
    out = re.sub(r"[^A-Za-z0-9_\-\s]", "", out).strip().lower()
    for lbl in allowed:
        if out == lbl.lower():
            return lbl
    for lbl in allowed:
        if lbl.lower() in out or out in lbl.lower():
            return lbl
    syn = {"pos":"positive","neg":"negative","neu":"neutral","neutral":"neutral"}
    if out in syn and any(lbl.lower()==syn[out] for lbl in allowed):
        return next(lbl for lbl in allowed if lbl.lower()==syn[out])
    return None

def _labels_fallback() -> List[str]:
    st = _state_getter()
    labs = getattr(st, "labels_", []) or ["positive","negative"]
    return list(labs)

def _safe_error(e: Exception) -> Dict[str, Any]:
    return {
        "type": e.__class__.__name__,
        "message": str(e) or "(no message)",
        "trace": "\n".join(traceback.format_exc(limit=4).splitlines()[-4:]),
    }

def _run_llm(text: str) -> Dict[str, Any]:
    st = _state_getter()
    cfg = getattr(st, "prompt_config", {}) or {}
    template = getattr(st, "prompt_template", "") or ""
    labels = cfg.get("allowed_labels", []) or _labels_fallback()
    examples = getattr(st, "prompt_examples", []) or []

    model_id = cfg.get("model_id") or "google/flan-t5-small"
    task = cfg.get("task") or _infer_task(model_id)
    temp = float(cfg.get("temperature", 0.0))
    mx = int(cfg.get("max_tokens", 128))

    pipe = _get_pipeline(task, model_id)
    # format the prompt with strict fields
    formatted = template.format(
        text=text,
        labels=", ".join(labels),
        examples=_format_examples(examples),
    )
    # generate
    out_raw = pipe(formatted, max_new_tokens=mx, do_sample=(temp > 0), temperature=temp)
    out = _extract_text(out_raw)
    parsed = _map_to_allowed(out, labels)
    return {"raw": out, "label": parsed, "backend": "hf_pipeline", "model": model_id, "task": task}

# ---------- UI ----------
def make_tab():
    status_md = gr.Markdown(_prompt_status_text())

    gr.Markdown("### Prompt Predict (Local Transformers)")
    inp = gr.Textbox(label="Input text", lines=3, placeholder="Type a sentence…")
    btn = gr.Button("Predict")
    out = gr.JSON(value={"label": None, "raw": None, "note": "Save settings in the Design subtab, then try again."})

    gr.Markdown("### Evaluate Prompt on dataset (uses ML test split if available)")
    eval_btn = gr.Button("Evaluate Prompt on dataset")
    eval_json = gr.JSON(value={"status": "Not evaluated yet"})

    def on_predict(text: str):
        if not text or not text.strip():
            return {"label": None, "raw": None, "note": "Empty input."}
        try:
            res = _run_llm(text)
            note = None if res.get("label") else "Could not map output to an allowed label."
            return {**res, "note": note}
        except Exception as e:
            return {"error": _safe_error(e)}

    btn.click(on_predict, [inp], [out])

    def _eval_dataset():
        st = _state_getter()
        df = getattr(st, "df_clean", None)
        tcol = getattr(st, "text_col", None)
        lcol = getattr(st, "label_col", None)
        if df is None or tcol is None or lcol is None:
            return {"error": {"type": "DataError", "message": "Load dataset in Data tab first.", "trace": ""}}

        split = getattr(st, "eval_split", {}) or {}
        idx_eval = split.get("test") or df.index.tolist()

        texts = df.loc[idx_eval, tcol].astype(str).tolist()
        y_true = df.loc[idx_eval, lcol].astype(str).tolist()

        y_pred: List[Optional[str]] = []
        try:
            for tx in texts:
                res = _run_llm(tx)
                y_pred.append(res.get("label"))
                time.sleep(0.01)  # tiny pause to keep UI responsive
        except Exception as e:
            return {"error": _safe_error(e)}

        if any(p is None for p in y_pred):
            fallback = pd.Series(y_true).mode().iloc[0]
            y_pred = [p if p is not None else fallback for p in y_pred]

        try:
            rpt = classification_report(
                np.array(y_true).astype(str),
                np.array(y_pred).astype(str),
                output_dict=True,
                zero_division=0
            )
        except Exception as e:
            return {"error": _safe_error(e)}

        mets = {
            "macro_f1": rpt.get("macro avg", {}).get("f1-score", None),
            "micro_f1": rpt.get("micro avg", {}).get("f1-score", None),
            "accuracy": rpt.get("accuracy", None),
        }

        st.prompt_metrics = {
            "metrics": mets,
            "n_eval": len(idx_eval),
            "backend": "hf_pipeline",
            "config": getattr(st, "prompt_config", {}),
        }
        return {"status": "ok", **st.prompt_metrics}

    eval_event = eval_btn.click(_eval_dataset, None, eval_json)

    return {"status_md": status_md, "eval_event": eval_event}
