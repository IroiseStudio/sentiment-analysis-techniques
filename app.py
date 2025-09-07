import gradio as gr
import pandas as pd  # not strictly needed now, but fine to keep
from state import get_state

# Tabs
from tabs.data_tab import bind_state as data_bind, make_tab as make_data_tab, refresh_outputs as data_refresh
from tabs.ml.train_tab import bind_state as ml_train_bind, make_tab as make_ml_train_tab
from tabs.ml.predict_tab import bind_state as ml_pred_bind, make_tab as make_ml_predict_tab
from tabs.nlp.select_tab import bind_state as nlp_sel_bind, make_tab as make_nlp_select_tab
from tabs.nlp.predict_tab import bind_state as nlp_pred_bind, make_tab as make_nlp_predict_tab
from tabs.transformers.select_tab import bind_state as tr_sel_bind, make_tab as make_tr_select_tab
from tabs.transformers.predict_tab import bind_state as tr_pred_bind, make_tab as make_tr_predict_tab
from tabs.prompts.design_tab import bind_state as pr_des_bind, make_tab as make_pr_design_tab
from tabs.prompts.predict_tab import bind_state as pr_pred_bind, make_tab as make_pr_predict_tab

def bind_all_state():
    getter = get_state
    data_bind(getter)
    ml_train_bind(getter)
    ml_pred_bind(getter)
    nlp_sel_bind(getter)
    nlp_pred_bind(getter)
    tr_sel_bind(getter)
    tr_pred_bind(getter)
    pr_des_bind(getter)
    pr_pred_bind(getter)

def _dataset_status_text() -> str:
    st = get_state()
    loaded = (st.df_raw is not None) and (st.text_col is not None) and (st.label_col is not None)
    if not loaded:
        return "### Dataset status: not loaded. Go to the **Data** tab to load a sample or upload your CSV/TXT."
    df = st.df_raw
    total = len(df)
    # counts for any number of classes
    try:
        counts = df[st.label_col].astype(str).value_counts()
        parts = [f"{k}: {v}" for k, v in counts.items()]
        counts_str = " | ".join(parts)
    except Exception:
        counts_str = "n/a"
    return f"### Dataset loaded: true  |  total: {total}  |  {counts_str}"

def build_ui():
    bind_all_state()
    with gr.Blocks(title="🧪 Sentiment Analysis Wizard") as demo:
        gr.Markdown("# Sentiment Analysis Wizard")

        # Global status bar visible on every tab
        with gr.Row():
            status_md = gr.Markdown(value=_dataset_status_text())

        # --- Data tab (capture handle and refresh on select + after apply) ---
        with gr.Tab("Data") as data_tab:
            data_handles = make_data_tab()

        # When user selects the Data tab, repopulate its preview + dropdowns from state
        def on_data_tab_select():
            head, info, text_upd, label_upd = data_refresh()
            status = _dataset_status_text()
            return head, info, text_upd, label_upd, status

        data_tab.select(
            on_data_tab_select,
            inputs=None,
            outputs=[
                data_handles["head_html"],
                data_handles["info_html"],
                data_handles["text_dropdown"],
                data_handles["label_dropdown"],
                status_md,
            ],
        )

        # After Apply Selections, auto-refresh the status banner
        if isinstance(data_handles, dict) and "apply_event" in data_handles:
            data_handles["apply_event"].then(lambda: _dataset_status_text(), inputs=None, outputs=status_md)

        # --- Other tabs ---
        with gr.Tab("ML Classifiers"):
            with gr.Tabs():
                with gr.Tab("Train"):
                    make_ml_train_tab()
                with gr.Tab("Predict"):
                    make_ml_predict_tab()

        with gr.Tab("Main NLP"):
            with gr.Tabs():
                with gr.Tab("Select"):
                    make_nlp_select_tab()
                with gr.Tab("Predict"):
                    make_nlp_predict_tab()

        with gr.Tab("Transformers"):
            with gr.Tabs():
                with gr.Tab("Select"):
                    make_tr_select_tab()
                with gr.Tab("Predict"):
                    make_tr_predict_tab()

        with gr.Tab("Prompt Engineering"):
            with gr.Tabs():
                with gr.Tab("Prompt Design"):
                    make_pr_design_tab()
                with gr.Tab("Predict"):
                    make_pr_predict_tab()

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
