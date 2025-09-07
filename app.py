import gradio as gr
from state import get_state

# Data
from tabs.data_tab import (
    bind_state as data_bind,
    make_tab as make_data_tab,
    _dataset_status_text as ds_status_text,  # type: ignore
)

# ML
from tabs.ml.train_tab import bind_state as ml_train_bind, make_tab as make_ml_train_tab

# NLP
from tabs.nlp.select_tab import (
    bind_state as nlp_sel_bind,
    make_tab as make_nlp_select_tab,
    nlp_status_text as nlp_select_status_text,
)
from tabs.nlp.predict_tab import (
    bind_state as nlp_pred_bind,
    make_tab as make_nlp_predict_tab,
    nlp_status_text as nlp_predict_status_text,
)

# Transformers
from tabs.transformers.select_tab import bind_state as tr_sel_bind, make_tab as make_tr_select_tab
from tabs.transformers.predict_tab import bind_state as tr_pred_bind, make_tab as make_tr_predict_tab

# Prompt Engineering (HF-only)
from tabs.prompts.design_tab import bind_state as pr_des_bind, make_tab as make_pr_design_tab, prompt_status_text as pr_status_text
from tabs.prompts.predict_tab import bind_state as pr_pred_bind, make_tab as make_pr_predict_tab

def bind_all_state():
    getter = get_state
    data_bind(getter)
    ml_train_bind(getter)
    nlp_sel_bind(getter)
    nlp_pred_bind(getter)
    tr_sel_bind(getter)
    tr_pred_bind(getter)
    pr_des_bind(getter)
    pr_pred_bind(getter)

def _dataset_status_text() -> str:
    return ds_status_text()

def _fmt_metrics(m: dict) -> str:
    if not m:
        return "—"
    parts = []
    for k in ["macro_f1", "micro_f1", "accuracy"]:
        if k in m and m[k] is not None:
            parts.append(f"{k}: {m[k]:.3f}")
    return " | ".join(parts) if parts else "—"

def _global_results_md() -> str:
    st = get_state()
    lines = ["## Results & Comparison"]

    # ML
    if st.ml_metrics:
        lines.append("### ML Classifiers")
        lines.append(f"- Model: **{st.ml_model_name}**")
        lines.append(f"- Metrics: {_fmt_metrics(st.ml_metrics)}")
        ntest = len((st.eval_split or {}).get("test", []))
        if ntest:
            lines.append(f"- Evaluated on test set: **n={ntest}**")
    else:
        lines.append("### ML Classifiers")
        lines.append("- Not run yet. Go to **ML → Train** and run training.")

    # NLP
    if st.nlp_metrics:
        nm = st.nlp_metrics
        lines.append("### Main NLP")
        lines.append(f"- Method: **{nm.get('method','?')}**")
        lines.append(f"- Metrics: {_fmt_metrics(nm.get('metrics', {}))}")
        lines.append(f"- Evaluated on: **n={nm.get('n_eval','?')}**")
    else:
        lines.append("### Main NLP")
        lines.append("- Not evaluated. Go to **Main NLP → Predict** and click **Evaluate NLP on dataset**.")

    # Transformers
    if getattr(st, "tr_metrics", {}):
        tm = st.tr_metrics
        lines.append("### Transformers")
        lines.append(f"- Model: **{st.transformer_id or 'unknown'}** (mode: {tm.get('mode','?')})")
        lines.append(f"- Metrics: {_fmt_metrics(tm.get('metrics', {}))}")
        lines.append(f"- Evaluated on: **n={tm.get('n_eval','?')}**")
    else:
        lines.append("### Transformers")
        lines.append("- Not evaluated. Go to **Transformers → Predict** and click **Evaluate NLP on dataset**.")

    # Prompt Eng.
    if getattr(st, "prompt_metrics", {}):
        pm = st.prompt_metrics
        lines.append("### Prompt Engineering")
        lines.append(f"- Backend: **{pm.get('backend','?')}**")
        lines.append(f"- Metrics: {_fmt_metrics(pm.get('metrics', {}))}")
        lines.append(f"- Evaluated on: **n={pm.get('n_eval','?')}**")
    else:
        lines.append("### Prompt Engineering")
        lines.append("- Not evaluated. Go to **Prompt Engineering → Predict** and click **Evaluate NLP on dataset**.")

    return "\n".join(lines)

def build_ui():
    bind_all_state()
    with gr.Blocks(title="🧪 Sentiment Analysis Wizard") as demo:
        gr.Markdown("# Sentiment Analysis Wizard")

        # Global dataset status
        with gr.Row():
            status_md = gr.Markdown(value=_dataset_status_text())

        # --- Data ---
        with gr.Tab("Data") as data_tab:
            with gr.Accordion("Data", open=True):
                gr.Markdown(
                    "Use this tab to **load a dataset** (CSV/TXT) and map the **text** and **label** columns. "
                    "Apply your selection to store it in app state; the status banner at the top updates automatically."
                )
            data_handles = make_data_tab(status_md=status_md)

        # --- ML Classifiers ---
        with gr.Tab("ML Classifiers"):
            with gr.Accordion("Machine Learning Classifiers", open=True):
                gr.Markdown(
                    "Train classical models (TF-IDF + classifier). Pick a model, adjust vectorizer & hyperparameters, "
                    "and click **Train**. This creates a shared **test split** reused by other tabs for fair comparisons."
                )
            with gr.Tabs():
                with gr.Tab("Train"):
                    ml_handles = make_ml_train_tab()   # {'train_event': ...}
                with gr.Tab("Predict"):
                    from tabs.ml.predict_tab import make_tab as make_ml_predict_tab
                    make_ml_predict_tab()

        # --- Main NLP ---
        with gr.Tab("Main NLP"):
            with gr.Accordion("Main NLP Techniques", open=True):
                gr.Markdown(
                    "Configure **VADER** or simple **keyword rules** (Select), then try single-text predictions and "
                    "**Evaluate on dataset** (Predict). Scores appear below in **Results & Comparison**."
                )
            with gr.Tabs():
                with gr.Tab("Select") as nlp_sel_tab:
                    sel_handles = make_nlp_select_tab()   # {'status_md', 'save_event'}
                with gr.Tab("Predict") as nlp_pred_tab:
                    pred_handles = make_nlp_predict_tab() # {'status_md', 'eval_event'}

            # Safe guards before wiring
            if isinstance(sel_handles, dict) and sel_handles.get("status_md") is not None:
                nlp_sel_tab.select(lambda: nlp_select_status_text(), None, sel_handles["status_md"])
            if isinstance(pred_handles, dict) and pred_handles.get("status_md") is not None:
                nlp_pred_tab.select(lambda: nlp_predict_status_text(), None, pred_handles["status_md"])
            if isinstance(sel_handles, dict) and sel_handles.get("save_event") is not None and isinstance(pred_handles, dict) and pred_handles.get("status_md") is not None:
                sel_handles["save_event"].then(lambda: nlp_predict_status_text(), None, pred_handles["status_md"])

        # --- Transformers ---
        with gr.Tab("Transformers"):
            with gr.Accordion("Transformers", open=True):
                gr.Markdown(
                    "**Sentiment** uses a finetuned classifier (fast, strong for POS/NEG).  \n"
                    "**Zero-shot** is prompt-based NLI: it scores how well the text entails each of your labels, "
                    "via a hypothesis template, with no training."
                )
            with gr.Tabs():
                with gr.Tab("Select") as tr_sel_tab:
                    tr_sel_handles = make_tr_select_tab()     # {'status_md', 'save_event'}
                with gr.Tab("Predict") as tr_pred_tab:
                    tr_pred_handles = make_tr_predict_tab()   # {'status_md', 'eval_event'}

            if isinstance(tr_sel_handles, dict) and tr_sel_handles.get("status_md") is not None:
                tr_sel_tab.select(lambda: tr_sel_handles["status_md"].value, None, tr_sel_handles["status_md"])
            if isinstance(tr_pred_handles, dict) and tr_pred_handles.get("status_md") is not None:
                tr_pred_tab.select(lambda: tr_pred_handles["status_md"].value, None, tr_pred_handles["status_md"])

        # --- Prompt Engineering (LangChain, HF-only) ---
        with gr.Tab("Prompt Engineering"):
            with gr.Accordion("Prompt Engineering", open=True):
                gr.Markdown(
                    "HF-only prompt flows (LangChain). Design an **instruction + few-shot** template and run it with "
                    "either a **local Transformers pipeline** or **Hugging Face Inference**. We post-process free-form "
                    "LLM outputs back into your allowed label set."
                )
            with gr.Tabs():
                with gr.Tab("Prompt Design") as pr_des_tab:
                    pr_des_handles = make_pr_design_tab()   # should return {'status_md','save_event'}
                with gr.Tab("Predict") as pr_pred_tab:
                    pr_pred_handles = make_pr_predict_tab() # should return {'status_md','eval_event'}

            # Null-safe wiring: only connect if handles exist
            if isinstance(pr_des_handles, dict) and pr_des_handles.get("status_md") is not None:
                pr_des_tab.select(lambda: pr_status_text(), None, pr_des_handles["status_md"])
            if isinstance(pr_pred_handles, dict) and pr_pred_handles.get("status_md") is not None:
                pr_pred_tab.select(lambda: "", None, pr_pred_handles["status_md"])  # no-op refresh

        # --- Results & Comparison ---
        gr.Markdown("---")
        results_md = gr.Markdown(value=_global_results_md())

        def refresh_results():
            return _global_results_md()

        # Auto-refresh results after key actions (guard each)
        if isinstance(data_handles, dict) and data_handles.get("apply_event") is not None:
            data_handles["apply_event"].then(refresh_results, None, results_md)
        if isinstance(ml_handles, dict) and ml_handles.get("train_event") is not None:
            ml_handles["train_event"].then(refresh_results, None, results_md)
        if isinstance(pred_handles, dict) and pred_handles.get("eval_event") is not None:
            pred_handles["eval_event"].then(refresh_results, None, results_md)
        if isinstance(tr_pred_handles, dict) and tr_pred_handles.get("eval_event") is not None:
            tr_pred_handles["eval_event"].then(refresh_results, None, results_md)
        if isinstance(pr_pred_handles, dict) and pr_pred_handles.get("eval_event") is not None:
            pr_pred_handles["eval_event"].then(refresh_results, None, results_md)

    return demo

if __name__ == "__main__":
    demo = build_ui()
    # For a public link when running locally: demo.launch(share=True)
    demo.launch()
