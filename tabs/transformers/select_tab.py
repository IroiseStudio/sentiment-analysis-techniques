from __future__ import annotations
from typing import Callable, Optional
import gradio as gr

_state_getter: Optional[Callable] = None

def bind_state(getter: Callable):
    global _state_getter
    _state_getter = getter

def make_tab():
    gr.Markdown("### Transformer Model (Stub)")
    model_id = gr.Dropdown(
        choices=[
            "distilbert-base-uncased-finetuned-sst-2-english",
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        ],
        value="distilbert-base-uncased-finetuned-sst-2-english",
        label="Model ID"
    )
    max_len = gr.Slider(16, 512, value=128, step=8, label="Max length")
    batch = gr.Slider(1, 32, value=8, step=1, label="Batch size")
    gr.Markdown("_Model will be loaded and cached in implementation phase._")
