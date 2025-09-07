from __future__ import annotations
from typing import Callable, Optional
import gradio as gr

_state_getter: Optional[Callable] = None

def bind_state(getter: Callable):
    global _state_getter
    _state_getter = getter

def make_tab():
    gr.Markdown("### Transformer Predict (Stub)")
    inp = gr.Textbox(label="Input text")
    btn = gr.Button("Predict")
    out = gr.JSON(value={"label": None, "score": None, "note": "Transformer inference not implemented"})
    btn.click(lambda x: {"label": None, "score": None, "note": "Stub"}, [inp], [out])
