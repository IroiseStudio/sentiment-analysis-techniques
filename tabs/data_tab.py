from __future__ import annotations
from typing import Callable, Optional
import gradio as gr

_state_getter: Optional[Callable] = None

def bind_state(getter: Callable):
    global _state_getter
    _state_getter = getter

def make_tab():
    """Renders the Data tab with placeholders. Hook real logic later."""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""### Load Dataset
Upload CSV or TXT, or load the included sample. Map the **text** and **label** columns.
_This is a scaffold. We'll wire functionality in the next step._
""")
            file_in = gr.File(label="Upload CSV/TXT", file_count="single")
            load_sample_btn = gr.Button("Load Sample (data/dataset.txt)")
            text_col = gr.Dropdown(choices=[], label="Text column")
            label_col = gr.Dropdown(choices=[], label="Label column")
            apply_btn = gr.Button("Apply Preprocessing & Cache")

        with gr.Column(scale=2):
            gr.Markdown("### Preview & Stats")
            head_html = gr.HTML(value="<i>Head preview will appear here.</i>")
            info_html = gr.HTML(value="<i>Rows, class balance, and warnings will appear here.</i>")

    # Placeholder no-op callbacks to avoid missing handlers
    def _noop(*args, **kwargs):
        return gr.update(), gr.update(), gr.update()

    load_sample_btn.click(_noop, [], [head_html, text_col, label_col])  # stubs
    apply_btn.click(_noop, [file_in, text_col, label_col], [head_html, info_html])
