from __future__ import annotations
from typing import Callable, Optional
import gradio as gr

_state_getter: Optional[Callable] = None

def bind_state(getter: Callable):
    global _state_getter
    _state_getter = getter

def make_tab():
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Model & Vectorizer (Stub)")
            model = gr.Radio(["LogisticRegression","LinearSVM","MultinomialNB","RandomForest"],
                             value="LogisticRegression", label="Model")
            ngram = gr.Slider(1, 3, value=1, step=1, label="TF-IDF ngram max")
            test_size = gr.Slider(0.1, 0.5, value=0.2, step=0.05, label="Test size")
            seed = gr.Number(value=42, label="Random state", precision=0)
            train_btn = gr.Button("Train")
        with gr.Column(scale=2):
            gr.Markdown("### Results (Stub)")
            metrics = gr.JSON(value={"status": "No training yet"})
            cm = gr.HTML(value="<i>Confusion matrix will render here.</i>")

    def _noop(*args, **kwargs):
        return {"status": "Training not implemented"}, "<i>CM not implemented</i>"

    train_btn.click(_noop, [model, ngram, test_size, seed], [metrics, cm])
