import gradio as gr
from state import get_state

# Tabs
from tabs.data_tab import bind_state as data_bind, make_tab as make_data_tab
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

def build_ui():
    bind_all_state()
    with gr.Blocks(title="🧪 Sentiment Analysis Wizard") as demo:
        gr.Markdown("""# Sentiment Analysis Wizard
Clean scaffold: navigation and wiring are in place. We'll add the real logic next.
""")
        with gr.Tab("Data"):
            make_data_tab()

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
