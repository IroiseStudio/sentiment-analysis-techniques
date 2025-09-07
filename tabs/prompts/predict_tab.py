from __future__ import annotations
from typing import Callable, Optional, Dict, Any, List
import os, re, time
import gradio as gr
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from transformers import pipeline as hf_pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import (
    HuggingFacePipeline,   # local Transformers pipeline
    HuggingFaceEndpoint,   # Hugging Face Hub / Inference API
)

_state_getter: Optional[Callable] = None

def bind_state(getter: Callable):
    """Bind the global state getter provided by app/state."""
    global _state_getter
    _state_getter = getter

# ---------- helpers ----------
def _token_detected() -> bool:
    return bool(
        os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )

def _infer_task(model_id: str, fallback: str = "text2text-generation") -> str:
    mid = (model_id or "").lower()
    # seq2seq families → text2text-generation
    if any(k in mid for k in ["flan", "t5", "mt5", "ul2", "bart", "mbart", "pegasus"]):
        return "text2text-generation"
    # otherwise default to causal LM generation
    return "text-generation" if fallback not in {"text-generation", "text2text-generation"} else fallback

def _prompt_status_text() -> str:
    st = _state_getter()
    bk = getattr(st, "prompt_backend", "hf_pipeline")
    cfg = getattr(st, "prompt_config", {}) or {}
    model = cfg.get("model_id") or "—"
    task = cfg.get("task") or _infer_task(model)
    temp = float(cfg.get("temperature", 0.0))
    mx = int(cfg.get("max_tokens", 128))
    labels = cfg.get("allowed_labels", [])
    bk_name = {"hf_pipeline": "Local (HF Pipeline)", "hf_inference": "Hugging Face Hub"}.get(bk, bk)
    token_note = " | token: " + ("detected" if _token_detected() else "missing") if bk == "hf_inference" else ""
    return (
        "#### Prompt settings  \n"
        f"• backend: **{bk_name}** | model: `{model}`{token_note}  \n"
        f"• task: {task} | temperature: {temp} | max_tokens: {mx}  \n"
        f"• labels: {', '.join(labels) if labels else '—'}"
    )

def _build_llm_from_state():
    """Instantiate a LangChain LLM for the current settings (local or Hub)."""
    st = _state_getter()
    bk = getattr(st, "prompt_backend", "hf_pipeline")
    cfg = getattr(st, "prompt_config", {}) or {}
    model_id = cfg.get("model_id") or "google/flan-t5-small"
    temperature = float(cfg.get("temperature", 0.0))
    max_tokens = int(cfg.get("max_tokens", 128))
    task = cfg.get("task") or _infer_task(model_id)

    if bk == "hf_pipeline":
        # Local Transformers pipeline
        pipe = hf_pipeline(
            task,
            model=model_id,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm, {"backend": "hf_pipeline", "model": model_id, "task": task}

    if bk == "hf_inference":
        token = (
            os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )
        if not token:
            raise RuntimeError("Missing HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN). Set it in your environment/Space secrets.")

        # Hugging Face Inference via new Endpoint class
        llm = HuggingFaceEndpoint(
            repo_id=model_id,
            task=task,                         # important: explicit task
            temperature=temperature,
            max_new_tokens=max_tokens,
            huggingfacehub_api_token=token,    # passes auth header
        )
        return llm, {"backend": "hf_inference", "model": model_id, "task": task}

    # Fallback (shouldn't happen)
    pipe = hf_pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=max_tokens)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm, {"backend": "hf_pipeline", "model": "google/flan-t5-small", "task": "text2text-generation"}

def _format_examples(examples: List[Dict[str, str]]) -> str:
    parts = []
    for ex in examples or []:
        t = (ex.get("text", "") or "").strip()
        l = (ex.get("label", "") or "").strip()
        if t and l:
            parts.append(f"Example:\nText: {t}\nLabel: {l}\n")
    return "\n".join(parts)

def _map_to_allowed(raw: str, allowed: List[str]) -> Optional[str]:
    """Map free-form model output to one of the allowed labels."""
    if not raw:
        return None
    out = raw.strip().splitlines()[0]                 # take first line
    out = re.sub(r"[^A-Za-z0-9_\-\s]", "", out)       # clean punctuation
    out = out.strip().lower()
    # exact match
    for lbl in allowed:
        if out == lbl.lower():
            return lbl
    # contains / contained-in for lenient mapping
    for lbl in allowed:
        if lbl.lower() in out or out in lbl.lower():
            return lbl
    # common short synonyms
    syn = {"pos": "positive", "neg": "negative", "neu": "neutral"}
    if out in syn and any(lbl.lower() == syn[out] for lbl in allowed):
        return next(lbl for lbl in allowed if lbl.lower() == syn[out])
    return None

def _labels_fallback() -> List[str]:
    st = _state_getter()
    labs = getattr(st, "labels_", []) or ["positive", "negative"]
    return list(labs)

def _run_llm(text: str) -> Dict[str, Any]:
    """Run the prompt chain on a single input and return {'raw','label',...}."""
    st = _state_getter()
    cfg = getattr(st, "prompt_config", {}) or {}
    template = getattr(st, "prompt_template", "") or ""
    labels = cfg.get("allowed_labels", []) or _labels_fallback()
    examples = getattr(st, "prompt_examples", []) or []

    llm, meta = _build_llm_from_state()
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    output = chain.invoke({
        "text": text,
        "labels": ", ".join(labels),
        "examples": _format_examples(examples),
    })
    parsed = _map_to_allowed(output, labels)
    return {"raw": output, "label": parsed, **meta}

# ---------- UI ----------
def make_tab():
    status_md = gr.Markdown(_prompt_status_text())

    gr.Markdown("### Prompt Predict (LangChain)")
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
            return {
                "label": res.get("label"),
                "raw": res.get("raw"),
                "backend": res.get("backend"),
                "model": res.get("model"),
                "task": res.get("task"),
                "note": note,
            }
        except Exception as e:
            return {"label": None, "raw": None, "error": str(e)}

    btn.click(on_predict, [inp], [out])

    def _eval_dataset():
        st = _state_getter()
        df = getattr(st, "df_clean", None)
        tcol = getattr(st, "text_col", None)
        lcol = getattr(st, "label_col", None)
        if df is None or tcol is None or lcol is None:
            return {"error": "Load dataset in Data tab first."}

        split = getattr(st, "eval_split", {}) or {}
        idx_eval = split.get("test") or df.index.tolist()

        texts = df.loc[idx_eval, tcol].astype(str).tolist()
        y_true = df.loc[idx_eval, lcol].astype(str).tolist()

        y_pred: List[Optional[str]] = []
        try:
            for tx in texts:
                res = _run_llm(tx)
                y_pred.append(res.get("label"))
                # Be gentle to the Hub if using remote backend
                if getattr(st, "prompt_backend", "hf_pipeline") == "hf_inference":
                    time.sleep(0.15)
        except Exception as e:
            return {"error": str(e)}

        # Replace None with majority class to avoid metric errors
        if any(p is None for p in y_pred):
            fallback = pd.Series(y_true).mode().iloc[0]
            y_pred = [p if p is not None else fallback for p in y_pred]

        rpt = classification_report(
            np.array(y_true).astype(str),
            np.array(y_pred).astype(str),
            output_dict=True,
            zero_division=0
        )
        mets = {
            "macro_f1": rpt.get("macro avg", {}).get("f1-score", None),
            "micro_f1": rpt.get("micro avg", {}).get("f1-score", None),
            "accuracy": rpt.get("accuracy", None),
        }

        st.prompt_metrics = {
            "metrics": mets,
            "n_eval": len(idx_eval),
            "backend": getattr(st, "prompt_backend", "hf_pipeline"),
            "config": getattr(st, "prompt_config", {}),
        }
        return {"status": "ok", **st.prompt_metrics}

    eval_event = eval_btn.click(_eval_dataset, None, eval_json)

    return {"status_md": status_md, "eval_event": eval_event}
