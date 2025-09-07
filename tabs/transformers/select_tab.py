from __future__ import annotations
from typing import Callable, Optional, Dict, Any, List
import gradio as gr

from transformers import pipeline

_state_getter: Optional[Callable] = None

def bind_state(getter: Callable):
    global _state_getter
    _state_getter = getter

# ---------- helpers ----------

def _labels_from_state() -> List[str]:
    st = _state_getter()
    labs = getattr(st, "labels_", []) or []
    return list(labs) if labs else ["positive", "negative"]

def _guess(labels: List[str], target: str) -> Optional[str]:
    t = target.lower()
    for lab in labels:
        if lab.lower() == t:
            return lab
    for lab in labels:
        if t in lab.lower() or lab.lower() in t:
            return lab
    return None

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
        return (
            "#### Transformers settings: Zero-shot\n"
            f"• model: `{mid}`  \n"
            f"• hypothesis_template: “{hyp}”  \n"
            f"• candidate labels: {', '.join(_labels_from_state())}"
        )

# ---------- UI ----------

def make_tab():
    status_md = gr.Markdown(tr_status_text())

    gr.Markdown("### Select a Transformer approach")
    mode = gr.Radio(["sentiment", "zero-shot"], value="sentiment", label="Mode")

    with gr.Group(visible=True) as grp_sentiment:
        model_id_sent = gr.Textbox(
            label="Model id",
            value="distilbert-base-uncased-finetuned-sst-2-english",
            info="Any text-classification model that outputs POSITIVE/NEGATIVE"
        )
        labs = _labels_from_state()
        pos_map = gr.Dropdown(labs, value=_guess(labs, "positive"), label="Map POSITIVE to")
        neg_map = gr.Dropdown(labs, value=_guess(labs, "negative"), label="Map NEGATIVE to")

    with gr.Group(visible=False) as grp_zeroshot:
        model_id_zn = gr.Textbox(
            label="Model id",
            value="facebook/bart-large-mnli",
            info="Zero-shot classification model"
        )
        hyp = gr.Textbox(value="This text is {}.", label="Hypothesis template")

    load_btn = gr.Button("Load / Save settings")
    out = gr.JSON(value={"status": "No transformer loaded yet"})

    def on_mode_change(m):
        return gr.update(visible=(m == "sentiment")), gr.update(visible=(m == "zero-shot"))
    mode.change(on_mode_change, inputs=mode, outputs=[grp_sentiment, grp_zeroshot])

    def on_save(m, mid_s, pos_lbl, neg_lbl, mid_z, hyp_t):
        st = _state_getter()
        if m == "sentiment":
            model_id = mid_s.strip() or "distilbert-base-uncased-finetuned-sst-2-english"
            cfg = {"mode": "sentiment", "map_positive": pos_lbl, "map_negative": neg_lbl}
            pipe = pipeline("text-classification", model=model_id, return_all_scores=False)
        else:
            model_id = mid_z.strip() or "facebook/bart-large-mnli"
            cfg = {"mode": "zero-shot", "hypothesis_template": hyp_t or "This text is {}."}
            pipe = pipeline("zero-shot-classification", model=model_id)

        st.transformer_id = model_id
        st.transformer_config = cfg
        st.transformer_pipeline = pipe
        return {"status": "ok", "model": model_id, "config": cfg}, tr_status_text()

    save_event = load_btn.click(
        on_save,
        inputs=[mode, model_id_sent, pos_map, neg_map, model_id_zn, hyp],
        outputs=[out, status_md]
    )

    # return handles so app.py can refresh banners if needed
    return {"status_md": status_md, "save_event": save_event}
