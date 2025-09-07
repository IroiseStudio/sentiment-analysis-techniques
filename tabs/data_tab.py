from __future__ import annotations
from typing import Callable, Optional, Tuple, List
import gradio as gr
import pandas as pd
import os, ast

_state_getter: Optional[Callable] = None

def bind_state(getter: Callable):
    global _state_getter
    _state_getter = getter

# --------------------------
# Helpers
# --------------------------
TXT_SAMPLE_PATH = "data/dataset.txt"

def _parse_tuple_txt(path: str) -> pd.DataFrame:
    """
    Parse a TXT file where each line is a Python tuple: ("text", "label"),
    and return a DataFrame with columns ['text', 'label'].
    """
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
                # ignore malformed lines
                pass
    if not rows:
        raise ValueError("No valid (text, label) tuples found in TXT.")
    return pd.DataFrame(rows)

def _load_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext == ".txt":
        # try tuple-per-line, else fallback to CSV-ish
        try:
            return _parse_tuple_txt(path)
        except Exception:
            return pd.read_csv(path, sep=None, engine="python")
    else:
        # last resort
        return pd.read_csv(path)

def _infer_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = [c.lower() for c in df.columns]
    # text candidates
    text_candidates = ["text", "sentence", "review", "comment", "content", "body", "message"]
    # label candidates
    label_candidates = ["label", "sentiment", "class", "target", "y"]
    text_col = None
    label_col = None
    for cand in text_candidates:
        if cand in cols:
            text_col = df.columns[cols.index(cand)]
            break
    for cand in label_candidates:
        if cand in cols:
            label_col = df.columns[cols.index(cand)]
            break
    # if still None, try heuristics: longest text-like column for text, smallest unique set for label
    if text_col is None:
        text_col = max(
            df.columns,
            key=lambda c: df[c].astype(str).str.len().mean() if str(df[c].dtype) == "object" else -1
        )
    if label_col is None:
        uniq_sorted = sorted(df.columns, key=lambda c: df[c].nunique())
        label_col = uniq_sorted[0] if uniq_sorted else df.columns[0]
        if label_col == text_col and len(df.columns) > 1:
            label_col = uniq_sorted[1]
    return text_col, label_col

def _class_balance_html(df: pd.DataFrame, label_col: str) -> str:
    vc = df[label_col].astype(str).value_counts(dropna=False)
    total = vc.sum()
    lines = ['<table><tr><th>Class</th><th>Count</th><th>Percent</th></tr>']
    for k, v in vc.items():
        pct = 100.0 * v / total if total else 0.0
        lines.append(f"<tr><td>{k}</td><td>{v}</td><td>{pct:.1f}%</td></tr>")
    lines.append("</table>")
    return "\n".join(lines)

def _warnings_html(df: pd.DataFrame, text_col: str, label_col: str) -> str:
    msgs: List[str] = []
    if df[text_col].isna().any():
        msgs.append("Missing values in text column.")
    if df[label_col].isna().any():
        msgs.append("Missing values in label column.")
    # rare class warning
    vc = df[label_col].value_counts()
    if not vc.empty and vc.min() < 3:
        msgs.append("One or more classes have fewer than 3 samples; consider merging or adding data.")
    # imbalance warning
    if not vc.empty and (vc.max() / max(1, vc.min())) > 10:
        msgs.append("Severe class imbalance detected (max/min ratio > 10).")
    if not msgs:
        return "<i>No warnings.</i>"
    return "<br/>".join(f"⚠️ {m}" for m in msgs)

# --------------------------
# Public refresh (used by app.py on tab-select)
# --------------------------
def refresh_outputs():
    """Return (head_html, info_html, text_dropdown_update, label_dropdown_update) from current state."""
    st = _state_getter()
    df = getattr(st, "df_raw", None)
    tcol = getattr(st, "text_col", None)
    lcol = getattr(st, "label_col", None)

    if df is None or tcol is None or lcol is None:
        # Nothing loaded → gentle message
        return (
            "<i>No dataset loaded yet. Use the controls on the left to load a sample or upload your data.</i>",
            "<i>Rows, class balance, and warnings will appear here.</i>",
            gr.update(), gr.update()
        )

    head = df.head(10).to_html(index=False)
    info = "<br/>".join([
        f"<b>Text column:</b> {tcol}",
        f"<b>Label column:</b> {lcol}",
        "<b>Class balance:</b> " + _class_balance_html(df, lcol),
        "<b>Warnings:</b> " + _warnings_html(df, tcol, lcol),
    ])
    # Keep user-chosen columns selected in dropdowns
    choices = list(df.columns)
    text_upd = gr.update(choices=choices, value=tcol)
    label_upd = gr.update(choices=choices, value=lcol)
    return head, info, text_upd, label_upd

# --------------------------
# UI
# --------------------------
def make_tab():
    """Renders the Data tab with real loading + mapping logic and exposes components for tab-select refresh."""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""### Load Dataset
Upload CSV or TXT, or load the included sample. Map the **text** and **label** columns.
""")
            file_in = gr.File(label="Upload CSV/TXT", file_count="single", file_types=[".csv", ".txt"])
            load_sample_btn = gr.Button("Load Sample (data/dataset.txt)")
            text_col = gr.Dropdown(choices=[], label="Text column")
            label_col = gr.Dropdown(choices=[], label="Label column")
            apply_btn = gr.Button("Apply Selections")

        with gr.Column(scale=2):
            gr.Markdown("### Preview & Stats")
            head_html = gr.HTML(value="<i>No dataset loaded yet. Use the controls on the left to load data.</i>")
            info_html = gr.HTML(value="<i>Rows, class balance, and warnings will appear here.</i>")

    # --------------------------
    # Callbacks
    # --------------------------
    def _load_path(path: str):
        st = _state_getter()
        df = _load_any(path)
        st.df_raw = df
        # infer columns
        tcol, lcol = _infer_columns(df)
        # build outputs
        head = df.head(10).to_html(index=False)
        # dropdown updates
        choices = list(df.columns)
        text_upd = gr.update(choices=choices, value=tcol)
        label_upd = gr.update(choices=choices, value=lcol)
        # info: rows/cols + class table + warnings
        info_parts = [f"<b>Shape:</b> {df.shape[0]} rows × {df.shape[1]} cols"]
        if lcol in df.columns:
            info_parts.append("<b>Class balance:</b> " + _class_balance_html(df, lcol))
            info_parts.append("<b>Warnings:</b> " + _warnings_html(df, tcol, lcol))
        info = "<br/>".join(info_parts)
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
        st.df_clean = df.copy()  # preprocessing comes later
        # Build info
        head, info, _, _ = refresh_outputs()
        return head, info

    # Wire events
    file_in.change(on_file_change, [file_in], [head_html, info_html, text_col, label_col])
    load_sample_btn.click(on_load_sample, [], [head_html, info_html, text_col, label_col])
    apply_event = apply_btn.click(on_apply, [text_col, label_col], [head_html, info_html])

    # Return handles (for app.py to refresh on tab select & after apply)
    return {
        "apply_event": apply_event,
        "head_html": head_html,
        "info_html": info_html,
        "text_dropdown": text_col,
        "label_dropdown": label_col,
    }