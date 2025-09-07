from __future__ import annotations
from typing import Callable, Optional, Dict, Any, List
import os
import pandas as pd
import gradio as gr

_state_getter: Optional[Callable] = None

def bind_state(getter: Callable):
    global _state_getter
    _state_getter = getter

DEFAULT_TEMPLATE = (
    "You are a strict sentiment classifier.\n"
    "Allowed labels (choose exactly one): {labels}\n\n"
    "{examples}\n"
    "Text: {text}\n"
    "Answer:"
)

def _labels_from_state() -> List[str]:
    st = _state_getter()
    labels = getattr(st, "labels_", []) or ["positive", "negative"]
    return list(labels)

def _examples_table_from_state() -> List[List[str]]:
    st = _state_getter()
    ex = getattr(st, "prompt_examples", []) or []
    rows = [[d.get("text", ""), d.get("label", "")] for d in ex]
    return rows or [["", ""], ["", ""], ["", ""]]

def _normalize_rows(rows) -> List[List[str]]:
    if rows is None:
        return []
    if isinstance(rows, pd.DataFrame):
        return rows.fillna("").astype(str).values.tolist()
    if isinstance(rows, list):
        return [[str(c or "") for c in r] for r in rows]
    return []

def prompt_status_text() -> str:
    st = _state_getter()
    backend = getattr(st, "prompt_backend", "none")
    cfg = getattr(st, "prompt_config", {}) or {}
    template = (getattr(st, "prompt_template", "") or DEFAULT_TEMPLATE).strip()
    labels = cfg.get("allowed_labels", []) or _labels_from_state()
    model = cfg.get("model_id") or "—"
    temp = cfg.get("temperature", 0.0)
    mx = cfg.get("max_tokens", 128)
    task = cfg.get("task", "auto")
    bk = {"hf_pipeline": "Local (HF Pipeline)", "hf_inference": "Hugging Face Hub", "none": "—"}.get(backend, backend)
    token_note = ""
    if backend == "hf_inference":
        token_note = " | token: " + ("detected" if (os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")) else "missing")
    return (
        "#### Prompt settings  \n"
        f"• backend: **{bk}** | model: `{model}`{token_note}  \n"
        f"• task: {task} | temperature: {temp} | max_tokens: {mx}  \n"
        f"• labels: {', '.join(labels)}  \n"
        f"• template preview (first 120 chars): `{template[:120]}{'…' if len(template) > 120 else ''}`"
    )

def make_tab():
    status_md = gr.Markdown(prompt_status_text())

    gr.Markdown("### Prompt Engineering — Select Backend & Design Prompt")

    backend = gr.Radio(
        ["Local (HF Pipeline)", "Hugging Face Hub"],
        value="Local (HF Pipeline)",
        label="Backend"
    )

    # One shared TASK dropdown (applies to both backends)
    task_dd = gr.Dropdown(
        ["text2text-generation", "text-generation"],
        value="text2text-generation",
        label="Transformers task"
    )

    with gr.Group(visible=True) as grp_local:
        local_model = gr.Textbox(
            value="google/flan-t5-small",
            label="Local model id (HF Transformers)",
            info="Seq2seq (e.g., FLAN-T5) → text2text-generation; causal LMs → text-generation."
        )

    with gr.Group(visible=False) as grp_hub:
        hub_model = gr.Textbox(
            value="google/flan-t5-small",
            label="Hub repo_id",
            info="Requires HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN) in your environment."
        )

    with gr.Accordion("Prompt Template & Labels", open=True):
        template_tb = gr.Textbox(
            value=DEFAULT_TEMPLATE, lines=10,
            label="Prompt template (must include {text}; optional {labels} and {examples})"
        )
        labels_tb = gr.Textbox(value=", ".join(_labels_from_state()), label="Allowed labels (comma-separated)")
        temp = gr.Slider(0.0, 1.5, value=0.0, step=0.1, label="temperature")
        max_tokens = gr.Slider(16, 512, value=128, step=8, label="max_new_tokens / max_tokens")

    with gr.Accordion("Few-shot Examples (optional)", open=False):
        ex_df = gr.Dataframe(
            headers=["text", "label"],
            value=_examples_table_from_state(),
            datatype=["str", "str"],
            row_count=(3, "dynamic"),
            type="array",
            label="Examples"
        )
        seed_btn = gr.Button("Seed examples from dataset (1 per class if available)")

    save_btn = gr.Button("Save settings")
    out = gr.JSON(value={"status": "No settings saved yet."})

    def on_backend_change(b):
        return gr.update(visible=(b == "Local (HF Pipeline)")), gr.update(visible=(b == "Hugging Face Hub"))
    backend.change(on_backend_change, backend, [grp_local, grp_hub])

    def on_seed_examples():
        st = _state_getter()
        df = getattr(st, "df_clean", None)
        tcol = getattr(st, "text_col", None)
        lcol = getattr(st, "label_col", None)
        if df is None or tcol is None or lcol is None:
            return _examples_table_from_state()
        rows: List[List[str]] = []
        for lab, sub in df.groupby(lcol):
            try:
                rows.append([str(sub.iloc[0][tcol]), str(lab)])
            except Exception:
                pass
        return rows or _examples_table_from_state()

    seed_btn.click(on_seed_examples, None, ex_df)

    def on_save(
        backend_v,
        task_v,
        local_model_v,
        hub_model_v,
        template_v, labels_v, temp_v, max_tokens_v,
        examples_rows
    ):
        st = _state_getter()

        # backend + model
        if backend_v == "Local (HF Pipeline)":
            bk = "hf_pipeline"
            model_id = (local_model_v or "").strip() or "google/flan-t5-small"
        else:
            bk = "hf_inference"
            model_id = (hub_model_v or "").strip() or "google/flan-t5-small"

        # labels + examples
        allowed = [x.strip() for x in (labels_v or "").split(",") if x.strip()] or _labels_from_state()
        examples = []
        for r in _normalize_rows(examples_rows):
            if not r or (not r[0] and not r[1]): 
                continue
            examples.append({"text": str(r[0]), "label": str(r[1])})

        # persist
        st.prompt_backend = bk
        st.prompt_template = (template_v or DEFAULT_TEMPLATE)
        st.prompt_examples = examples
        st.prompt_config = {
            "temperature": float(temp_v),
            "max_tokens": int(max_tokens_v),
            "allowed_labels": allowed,
            "model_id": model_id,
            "task": task_v or "text2text-generation",
        }

        return {
            "status": "saved",
            "backend": bk,
            "model": model_id,
            "task": st.prompt_config["task"],
            "labels": allowed,
            "n_examples": len(examples),
            "temperature": float(temp_v),
            "max_tokens": int(max_tokens_v),
        }, prompt_status_text()

    save_event = save_btn.click(
        on_save,
        inputs=[backend, task_dd, local_model, hub_model, template_tb, labels_tb, temp, max_tokens, ex_df],
        outputs=[out, status_md]
    )

    return {"status_md": status_md, "save_event": save_event}
