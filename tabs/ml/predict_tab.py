from __future__ import annotations
from typing import Callable, Optional, Dict, Any, List

import gradio as gr
import numpy as np

_state_getter: Optional[Callable] = None

def bind_state(getter: Callable):
    global _state_getter
    _state_getter = getter


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _predict_with_proba(pipe, text: str) -> Dict[str, Any]:
    """Return label and confidence. Use predict_proba when available, else decision_function→softmax."""
    if not text or not text.strip():
        return {"label": None, "score": None, "probs": None, "note": "Empty input."}

    try:
        # Single-item prediction arrays
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba([text])[0]
            labels = getattr(pipe, "classes_", None)
        else:
            # Try decision_function then softmax
            if hasattr(pipe, "decision_function"):
                scores = pipe.decision_function([text])
                # decision_function returns array shape (1, C) or (1,) for binary
                if scores.ndim == 1:
                    scores = scores.reshape(1, -1)
                probs = _softmax(scores)[0]
                labels = getattr(pipe, "classes_", None)
            else:
                # Fallback: no probabilities
                pred = pipe.predict([text])[0]
                return {"label": str(pred), "score": None, "probs": None, "note": "Confidence not available."}

        # If labels not present on pipeline, try to get from last step
        if labels is None:
            try:
                labels = pipe.steps[-1][1].classes_
            except Exception:
                labels = None

        if labels is None:
            # Unknown label order; return top only
            top_idx = int(np.argmax(probs))
            return {"label": str(top_idx), "score": float(probs[top_idx]), "probs": None, "note": "Labels missing."}

        top_idx = int(np.argmax(probs))
        top_label = str(labels[top_idx])
        # make a tidy probs dict sorted desc
        prob_dict = {str(labels[i]): float(probs[i]) for i in range(len(labels))}
        prob_dict = dict(sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True))
        return {"label": top_label, "score": float(probs[top_idx]), "probs": prob_dict, "note": None}
    except Exception as e:
        return {"label": None, "score": None, "probs": None, "error": str(e)}


def make_tab():
    gr.Markdown("### Predict with Trained ML Model")
    inp = gr.Textbox(label="Input text", lines=3, placeholder="Type a sentence to classify…")
    btn = gr.Button("Predict")
    out = gr.JSON(value={"label": None, "score": None, "probs": None, "note": "Load and train a model first."})

    def on_predict(text: str):
        st = _state_getter()
        pipe = getattr(st, "ml_pipeline", None)
        if pipe is None:
            return {"label": None, "score": None, "probs": None, "error": "No trained model in state. Train one in the ML Train tab."}
        return _predict_with_proba(pipe, text)

    btn.click(on_predict, [inp], [out])
