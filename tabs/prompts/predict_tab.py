from __future__ import annotations
from typing import Callable, Optional
import gradio as gr

_state_getter: Optional[Callable] = None

def bind_state(getter: Callable):
    global _state_getter
    _state_getter = getter

def make_tab():
    gr.Markdown("### Prompt Predict (Stub)")
    inp = gr.Textbox(label="Input text")
    btn = gr.Button("Run Prompt")
    out = gr.JSON(value={"raw": None, "parsed_label": None, "note": "LLM call not implemented"})
    btn.click(lambda x: {"raw": None, "parsed_label": None, "note": "Stub"}, [inp], [out])
