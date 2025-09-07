import gradio as gr
from state import get_state

# Data
from tabs.data_tab import bind_state as data_bind, make_tab as make_data_tab, _dataset_status_text as ds_status_text  # type: ignore

# ML
from tabs.ml.train_tab import bind_state as ml_train_bind, make_tab as make_ml_train_tab

# NLP
from tabs.nlp.select_tab import bind_state as nlp_sel_bind, make_tab as make_nlp_select_tab, nlp_status_text as nlp_select_status_text
from tabs.nlp.predict_tab import bind_state as nlp_pred_bind, make_tab as make_nlp_predict_tab, nlp_status_text as nlp_predict_status_text

# (Transformers / Prompts tabs unchanged for now)
from tabs.transformers.select_tab import bind_state as tr_sel_bind, make_tab as make_tr_select_tab
from tabs.transformers.predict_tab import bind_state as tr_pred_bind, make_tab as make_tr_predict_tab
from tabs.prompts.design_tab import bind_state as pr_des_bind, make_tab as make_pr_design_tab
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
    # reuse helper from data_tab if desired
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
        split = st.eval_split or {}
        ntest = len(split.get("test", []))
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
        lines.append(f"- Model: **{st.transformer_id or 'unknown'}**")
        lines.append(f"- Metrics: {_fmt_metrics(tm.get('metrics', {}))}")
    else:
        lines.append("### Transformers")
        lines.append("- Not evaluated yet. Go to **Transformers → Predict** and click **Evaluate NLP on dataset**.")

    # Prompt
    if getattr(st, "prompt_metrics", {}):
        pm = st.prompt_metrics
        lines.append("### Prompt Engineering")
        lines.append(f"- Backend: **{st.prompt_backend}**")
        lines.append(f"- Metrics: {_fmt_metrics(pm.get('metrics', {}))}")
    else:
        lines.append("### Prompt Engineering")
        lines.append("- Not evaluated yet.")

    return "\n".join(lines)

def build_ui():
    bind_all_state()
    with gr.Blocks(title="🧪 Sentiment Analysis Wizard") as demo:
        gr.Markdown("# Sentiment Analysis Wizard")

        # Global dataset status
        with gr.Row():
            status_md = gr.Markdown(value=_dataset_status_text())

        # Data tab (pass banner so it can update directly)
        with gr.Tab("Data") as data_tab:
            data_handles = make_data_tab(status_md=status_md)

        # ML
        with gr.Tab("ML Classifiers"):
            with gr.Tabs():
                with gr.Tab("Train"):
                    ml_handles = make_ml_train_tab()   # returns {"train_event"}
                with gr.Tab("Predict"):
                    from tabs.ml.predict_tab import make_tab as make_ml_predict_tab  # lazy import ok
                    make_ml_predict_tab()

        # NLP
        with gr.Tab("Main NLP"):
            with gr.Tabs():
                with gr.Tab("Select") as nlp_sel_tab:
                    sel_handles = make_nlp_select_tab()
                with gr.Tab("Predict") as nlp_pred_tab:
                    pred_handles = make_nlp_predict_tab()  # returns {"eval_event", "status_md"}

            # Sync banners on subtab select
            nlp_sel_tab.select(lambda: nlp_select_status_text(), None, sel_handles["status_md"])
            nlp_pred_tab.select(lambda: nlp_predict_status_text(), None, pred_handles["status_md"])

        # Transformers
        with gr.Tab("Transformers"):
                    with gr.Tabs():
                        with gr.Tab("Select") as tr_sel_tab:
                            tr_sel_handles = make_tr_select_tab()
                        with gr.Tab("Predict") as tr_pred_tab:
                            tr_pred_handles = make_tr_predict_tab()
                    # refresh banners on subtab select
                    tr_sel_tab.select(lambda: tr_sel_handles["status_md"].value, None, tr_sel_handles["status_md"])
                    tr_pred_tab.select(lambda: tr_pred_handles["status_md"].value, None, tr_pred_handles["status_md"])


        # Prompt Engineering
        with gr.Tab("Prompt Engineering"):
            with gr.Tabs():
                with gr.Tab("Prompt Design"):
                    make_pr_design_tab()
                with gr.Tab("Predict"):
                    make_pr_predict_tab()

        # --- Global Results & Comparison ---
        gr.Markdown("---")
        results_md = gr.Markdown(value=_global_results_md())

        def refresh_results():
            return _global_results_md()

        # Auto-refresh results after key actions
        data_handles["apply_event"].then(refresh_results, None, results_md)
        ml_handles["train_event"].then(refresh_results, None, results_md)
        pred_handles["eval_event"].then(refresh_results, None, results_md)
        tr_pred_handles["eval_event"].then(refresh_results, None, results_md)


        # Optional manual refresh button if you want:
        # refresh_btn = gr.Button("Refresh results")
        # refresh_btn.click(refresh_results, None, results_md)

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
