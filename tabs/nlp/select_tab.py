from __future__ import annotations
from typing import Callable, Optional
import gradio as gr

_state_getter: Optional[Callable] = None

def bind_state(getter: Callable):
    global _state_getter
    _state_getter = getter

def make_tab():
    gr.Markdown("### NLP Baselines (Stub)")
    method = gr.Radio(["VADER", "Rules"], value="VADER", label="Method")
    threshold = gr.Slider(-1.0, 1.0, value=0.05, step=0.01, label="VADER threshold")
    pos = gr.Textbox(lines=3, label="Positive keywords (comma-separated)")
    neg = gr.Textbox(lines=3, label="Negative keywords (comma-separated)")
    gr.Markdown("_Settings will be saved to state in the implementation phase._")
