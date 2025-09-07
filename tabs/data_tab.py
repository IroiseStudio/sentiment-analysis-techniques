from __future__ import annotations
from typing import Callable, Optional, Tuple, List
import gradio as gr
import pandas as pd
import os, ast

_state_getter: Optional[Callable] = None

def bind_state(getter: Callable):
    global _state_getter
    _state_getter = getter

TXT_SAMPLE_PATH = "data/dataset.txt"

def _dataset_status_text() -> str:
    st = _state_getter()
    loaded = (getattr(st, "df_raw", None) is not None) and (st.text_col is not None) and (st.label_col is not None)
    if not loaded:
        return "### Dataset status: not loaded. Go to the **Data** tab to load a sample or upload your CSV/TXT."
    df = st.df_raw
    total = len(df)
    try:
        counts = df[st.label_col].astype(str).value_counts()
        parts = [f"{k}: {v}" for k, v in counts.items()]
        counts_str = " | ".join(parts)
    except Exception:
        counts_str = "n/a"
    return f"### Dataset loaded: true  |  total: {total}  |  {counts_str}"

def _parse_tuple_txt(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                tup = ast.literal_eval(line)
                if isinstance(tup, tuple) and len(tup) == 2:
                    rows.append({"text": str(tup[0]), "label": str(tup[1])})
            except Exception:
                pass
    if not rows:
        raise ValueError("No valid (text, label) tuples found in TXT.")
    return pd.DataFrame(rows)

def _load_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext == ".txt":
        try:
            return _parse_tuple_txt(path)
        except Exception:
            return pd.read_csv(path, sep=None, engine="python")
    else:
        return pd.read_csv(path)

def _infer_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = [c.lower() for c in df.columns]
    text_candidates = ["text", "sentence", "review", "comment", "content", "body", "message"]
    label_candidates = ["label", "sentiment", "class", "target", "y"]
    text_col = next((df.columns[cols.index(c)] for c in text_candidates if c in cols), None)
    label_col = next((df.columns[cols.index(c)] for c in label_candidates if c in cols), None)
    if text_col is None:
        text_col = max(df.columns, key=lambda c: df[c].astype(str).str.len().mean() if str(df[c].dtype) == "object" else -1)
    if label_col is None:
        uniq_sorted = sorted(df.columns, key=lambda c: df[c].nunique())
        label_col = uniq_sorted[0] if uniq_sorted else df.columns[0]
        if label_col == text_col and len(df.columns) > 1:
            label_col = uniq_sorted[1]
    return text_col, label_col

def _class_balance_html(df: pd.DataFrame, label_col: str) -> str:
    vc = df[label_col].astype(str).value_counts(dropna=False)
    total = vc.sum()
    rows = ['<table><tr><th>Class</th><th>Count</th><th>Percent</th></tr>']
    for k, v in vc.items():
        pct = 100.0 * v / total if total else 0.0
        rows.append(f"<tr><td>{k}</td><td>{v}</td><td>{pct:.1f}%</td></tr>")
    rows.append("</table>")
    return "\n".join(rows)

def _warnings_html(df: pd.DataFrame, text_col: str, label_col: str) -> str:
    msgs: List[str] = []
    if df[text_col].isna().any():
        msgs.append("Missing values in text column.")
    if df[label_col].isna().any():
        msgs.append("Missing values in label column.")
    vc = df[label_col].value_counts()
    if not vc.empty and vc.min() < 3:
        msgs.append("One or more classes have fewer than 3 samples; consider merging or adding data.")
    if not vc.empty and (vc.max() / max(1, vc.min())) > 10:
        msgs.append("Severe class imbalance detected (max/min ratio > 10).")
    return "<br/>".join(f"⚠️ {m}" for m in msgs) if msgs else "<i>No warnings.</i>"

def make_tab(status_md: Optional[gr.Markdown] = None):
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Load Dataset\nUpload CSV or TXT, or load the included sample. Map the **text** and **label** columns.")
            file_in = gr.File(label="Upload CSV/TXT", file_count="single", file_types=[".csv", ".txt"])
            load_sample_btn = gr.Button("Load Sample (data/dataset.txt)")
            text_col = gr.Dropdown(choices=[], label="Text column")
            label_col = gr.Dropdown(choices=[], label="Label column")
            apply_btn = gr.Button("Apply Selections")

        with gr.Column(scale=2):
            gr.Markdown("### Preview & Stats")
            head_html = gr.HTML(value="<i>No dataset loaded yet. Use the controls on the left to load data.</i>")
            info_html = gr.HTML(value="<i>Rows, class balance, and warnings will appear here.</i>")

    def _load_path(path: str):
        st = _state_getter()
        df = _load_any(path)
        st.df_raw = df
        tcol, lcol = _infer_columns(df)
        head = df.head(10).to_html(index=False)
        choices = list(df.columns)
        text_upd = gr.update(choices=choices, value=tcol)
        label_upd = gr.update(choices=choices, value=lcol)
        info = "<br/>".join([
            f"<b>Shape:</b> {df.shape[0]} rows × {df.shape[1]} cols",
            "<b>Class balance:</b> " + _class_balance_html(df, lcol),
            "<b>Warnings:</b> " + _warnings_html(df, tcol, lcol),
        ])
        return head, info, text_upd, label_upd

    def on_file_change(fobj):
        if fobj is None:
            return gr.update(value="<i>No file.</i>"), gr.update(value=""), gr.update(), gr.update()
        return _load_path(fobj.name)

    def on_load_sample():
        if not os.path.exists(TXT_SAMPLE_PATH):
            return ("<b>Sample not found at data/dataset.txt</b>", "", gr.update(), gr.update())
        return _load_path(TXT_SAMPLE_PATH)

    def on_apply(tcol: str, lcol: str):
        st = _state_getter()
        df = st.df_raw
        if df is None:
            return gr.update(value="<b>No dataset loaded.</b>"), gr.update(value="")
        if tcol not in df.columns or lcol not in df.columns:
            return gr.update(value="<b>Invalid column selection.</b>"), gr.update(value="")
        # Save to state
        st.text_col = tcol
        st.label_col = lcol
        st.labels_ = sorted([str(x) for x in df[lcol].astype(str).unique().tolist()])
        st.df_clean = df.copy()

        # Reset any stale results/splits
        st.eval_split = {}
        st.ml_pipeline = None
        st.ml_model_name = None
        st.ml_metrics = {}
        st.nlp_metrics = {}
        st.tr_metrics = {}
        st.prompt_metrics = {}

        head = df.head(10).to_html(index=False)
        info = "<br/>".join([
            f"<b>Text column:</b> {tcol}",
            f"<b>Label column:</b> {lcol}",
            "<b>Class balance:</b> " + _class_balance_html(df, lcol),
            "<b>Warnings:</b> " + _warnings_html(df, tcol, lcol),
        ])
        if status_md is None:
            return head, info
        else:
            return head, info, _dataset_status_text()

    # wire
    file_in.change(on_file_change, [file_in], [head_html, info_html, text_col, label_col])
    load_sample_btn.click(on_load_sample, [], [head_html, info_html, text_col, label_col])

    if status_md is None:
        apply_event = apply_btn.click(on_apply, [text_col, label_col], [head_html, info_html])
    else:
        apply_event = apply_btn.click(on_apply, [text_col, label_col], [head_html, info_html, status_md])

    return {
        "apply_event": apply_event,
        "head_html": head_html,
        "info_html": info_html,
        "text_dropdown": text_col,
        "label_dropdown": label_col,
    }