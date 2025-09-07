from __future__ import annotations
from typing import Callable, Optional, Tuple, Dict, Any, List

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-friendly (HF Spaces)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

_state_getter: Optional[Callable] = None

def bind_state(getter: Callable):
    """Bind the global state getter (get_state)."""
    global _state_getter
    _state_getter = getter


def _build_clf(model_name: str, params: Dict[str, Any]):
    """Return an sklearn classifier according to selection."""
    if model_name == "LogisticRegression":
        return LogisticRegression(
            C=float(params.get("lr_C", 1.0)),
            max_iter=int(params.get("lr_max_iter", 1000)),
            n_jobs=None,
        )
    if model_name == "LinearSVM":
        # LinearSVC + calibration to get predict_proba-like scores
        base = LinearSVC(C=float(params.get("svm_C", 1.0)))
        return CalibratedClassifierCV(base, cv=3)
    if model_name == "MultinomialNB":
        return MultinomialNB(alpha=float(params.get("nb_alpha", 1.0)))
    if model_name == "RandomForest":
        md = params.get("rf_max_depth", 0)
        max_depth = None if (md is None or int(md) <= 0) else int(md)
        return RandomForestClassifier(
            n_estimators=int(params.get("rf_n_estimators", 200)),
            max_depth=max_depth,
            n_jobs=-1,
            random_state=int(params.get("seed", 42)),
        )
    raise ValueError(f"Unknown model: {model_name}")


def _tfidf_vec(vec_params: Dict[str, Any]) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer from UI parameters."""
    return TfidfVectorizer(
        ngram_range=(1, int(vec_params.get("ngram_max", 1))),
        min_df=int(vec_params.get("min_df", 1)),
        max_df=float(vec_params.get("max_df", 1.0)),
        sublinear_tf=bool(vec_params.get("sublinear_tf", True)),
        strip_accents="unicode",
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)
    rpt = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "macro_f1": rpt.get("macro avg", {}).get("f1-score", None),
        "micro_f1": rpt.get("micro avg", {}).get("f1-score", None),
        "accuracy": rpt.get("accuracy", None),
        "precision_macro": rpt.get("macro avg", {}).get("precision", None),
        "recall_macro": rpt.get("macro avg", {}).get("recall", None),
        "per_class": {lbl: rpt.get(lbl, {}) for lbl in sorted(set(labels))},
    }


def _cm_heatmap_figure(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]):
    all_labels = sorted(set(labels))
    cm = confusion_matrix(y_true.astype(str), y_pred.astype(str), labels=all_labels)

    fig, ax = plt.subplots(figsize=(5 + 0.3*len(all_labels), 4 + 0.2*len(all_labels)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Ticks and labels
    ax.set_xticks(np.arange(len(all_labels)))
    ax.set_yticks(np.arange(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=45, ha="right")
    ax.set_yticklabels(all_labels)

    # Colorbar
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    return fig


def make_tab():
    """Render the ML Train tab with model-specific hyperparam panels."""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Model and Vectorizer")

            model = gr.Radio(
                ["LogisticRegression", "LinearSVM", "MultinomialNB", "RandomForest"],
                value="LogisticRegression",
                label="Model"
            )

            with gr.Accordion("Vectorizer (TF-IDF)", open=True):
                ngram_max = gr.Slider(1, 3, value=1, step=1, label="ngram max")
                min_df = gr.Slider(1, 10, value=1, step=1, label="min_df (docs)")
                max_df = gr.Slider(0.5, 1.0, value=1.0, step=0.05, label="max_df (fraction)")
                sublinear_tf = gr.Checkbox(True, label="sublinear_tf")

            with gr.Accordion("Train/Test Split", open=True):
                test_size = gr.Slider(0.1, 0.5, value=0.2, step=0.05, label="test_size")
                seed = gr.Number(value=42, label="random_state", precision=0)
                stratify = gr.Checkbox(True, label="stratify by label")

            with gr.Accordion("Model Hyperparameters", open=True):
                # One group per model (toggle visibility)
                with gr.Group(visible=True) as grp_lr:
                    lr_C = gr.Number(value=1.0, label="LR: C")
                    lr_max_iter = gr.Number(value=1000, label="LR: max_iter", precision=0)

                with gr.Group(visible=False) as grp_svm:
                    svm_C = gr.Number(value=1.0, label="LinearSVM: C")

                with gr.Group(visible=False) as grp_nb:
                    nb_alpha = gr.Number(value=1.0, label="MultinomialNB: alpha")

                with gr.Group(visible=False) as grp_rf:
                    rf_n_estimators = gr.Slider(50, 500, value=200, step=50, label="RandomForest: n_estimators")
                    rf_max_depth = gr.Number(value=0, label="RandomForest: max_depth (0 = None)", precision=0)

            train_btn = gr.Button("Train")

        with gr.Column(scale=2):
            gr.Markdown("### Results")
            metrics_json = gr.JSON(value={"status": "No training yet"})
            cm_plot = gr.Plot()  # matplotlib figure output

    # Toggle which hyperparam group is visible
    def on_model_change(name: str):
        return (
            gr.update(visible=(name == "LogisticRegression")),
            gr.update(visible=(name == "LinearSVM")),
            gr.update(visible=(name == "MultinomialNB")),
            gr.update(visible=(name == "RandomForest")),
        )
    model.change(on_model_change, inputs=model, outputs=[grp_lr, grp_svm, grp_nb, grp_rf])

    def on_train(
        model_name,
        ngram_max_v, min_df_v, max_df_v, sublinear_tf_v,
        test_size_v, seed_v, stratify_v,
        # pass all possible params; we'll pick the ones for the selected model
        lr_C_v=1.0, lr_max_iter_v=1000,
        svm_C_v=1.0,
        nb_alpha_v=1.0,
        rf_n_estimators_v=200, rf_max_depth_v=0
    ):
        st = _state_getter()
        df = getattr(st, "df_clean", None)
        tcol = getattr(st, "text_col", None)
        lcol = getattr(st, "label_col", None)

        if df is None or tcol is None or lcol is None:
            return {"error": "Load a dataset and apply selections in the Data tab first."}, None

        # Vectorizer params
        vec_params = {
            "ngram_max": int(ngram_max_v),
            "min_df": int(min_df_v),
            "max_df": float(max_df_v),
            "sublinear_tf": bool(sublinear_tf_v),
        }

        # Model-specific params
        clf_params = {"seed": seed_v}
        if model_name == "LogisticRegression":
            clf_params.update({"lr_C": lr_C_v, "lr_max_iter": lr_max_iter_v})
        elif model_name == "LinearSVM":
            clf_params.update({"svm_C": svm_C_v})
        elif model_name == "MultinomialNB":
            clf_params.update({"nb_alpha": nb_alpha_v})
        elif model_name == "RandomForest":
            clf_params.update({"rf_n_estimators": rf_n_estimators_v, "rf_max_depth": rf_max_depth_v})

        # Data and labels
        X = df[tcol].astype(str).values
        y = df[lcol].astype(str).values
        label_set = sorted(pd.Series(y).unique().tolist())

        # Split with stratify fallback
        try:
            strat = y if bool(stratify_v) else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=float(test_size_v),
                random_state=int(seed_v),
                stratify=strat
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=float(test_size_v),
                random_state=int(seed_v),
                stratify=None
            )

        pipe = Pipeline([
            ("tfidf", _tfidf_vec(vec_params)),
            ("clf", _build_clf(model_name, clf_params))
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mets = _metrics(y_test, y_pred, label_set)
        fig = _cm_heatmap_figure(y_test, y_pred, label_set)

        # Persist to state
        st.ml_pipeline = pipe
        st.ml_model_name = model_name
        st.ml_metrics = mets
        st.train_test_split_config.update({
            "test_size": float(test_size_v),
            "random_state": int(seed_v),
            "stratify": bool(stratify_v),
        })

        return mets, fig

    train_btn.click(
        on_train,
        inputs=[
            model,
            ngram_max, min_df, max_df, sublinear_tf,
            test_size, seed, stratify,
            # all potential hyperparams (unused ones are ignored)
            lr_C, lr_max_iter,
            svm_C,
            nb_alpha,
            rf_n_estimators, rf_max_depth
        ],
        outputs=[metrics_json, cm_plot]
    )