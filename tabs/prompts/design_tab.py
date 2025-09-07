from __future__ import annotations
from typing import Callable, Optional, List
import gradio as gr

_state_getter: Optional[Callable] = None

def bind_state(getter: Callable):
    global _state_getter
    _state_getter = getter

DEFAULT_TEMPLATE = (
    'You are a sentiment classifier. Return exactly one of: {labels}.\n'
    'Text: "{text}"\n'
    'Answer:'
)

def make_tab():
    gr.Markdown("### Prompt Design (Stub)")
    tmpl = gr.Textbox(label="Prompt template", value=DEFAULT_TEMPLATE, lines=6)
    labels = gr.Textbox(label="Allowed labels (comma-separated)", value="positive, negative")
    temp = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature")
    max_toks = gr.Slider(16, 512, value=128, step=8, label="Max tokens")
    gr.Markdown("_We will wire saving to state and backend selection later._")
