from __future__ import annotations
from typing import Callable, Optional, Dict, Any, List
import json
import gradio as gr

_state_getter: Optional[Callable] = None

def bind_state(getter: Callable):
    global _state_getter
    _state_getter = getter

# -------- helpers --------

def _labels_from_state() -> List[str]:
    st = _state_getter()
    labels = getattr(st, "labels_", []) or []
    # fallback if no dataset loaded
    if not labels:
        labels = ["positive", "negative", "neutral"]
    return list(labels)

def _guess_label(labels: List[str], target: str) -> Optional[str]:
    # find a label in labels matching target (case-insensitive substring)
    t = target.lower()
    for lab in labels:
        if lab.lower() == t:
            return lab
    for lab in labels:
        if t in lab.lower() or lab.lower() in t:
            return lab
    return None

def _default_rules_json(labels: List[str]) -> str:
    # seed with empty lists for each label (and basic defaults if pos/neg exist)
    template: Dict[str, List[str]] = {lab: [] for lab in labels}
    pos = _guess_label(labels, "positive")
    neg = _guess_label(labels, "negative")
    if pos:
        template[pos] = ["good", "great", "excellent", "love", "amazing", "awesome"]
    if neg:
        template[neg] = ["bad", "terrible", "awful", "hate", "worst", "poor"]
    return json.dumps(template, indent=2)

def _fmt_rules_summary(rules: Dict[str, List[str]]) -> str:
    parts = []
    for lab, kws in rules.items():
        n = len(kws) if isinstance(kws, list) else 0
        parts.append(f"{lab}({n})")
    return ", ".join(parts) if parts else "—"

# Public: build the NLP status line from state
def nlp_status_text() -> str:
    st = _state_getter()
    method = getattr(st, "nlp_method", None) or "VADER"
    cfg = getattr(st, "nlp_config", {}) or {}

    if method == "VADER":
        pt = cfg.get("pos_threshold", 0.05)
        nt = cfg.get("neg_threshold", -0.05)
        en = bool(cfg.get("enable_neutral", False))
        mp = cfg.get("map_positive", "positive")
        mn = cfg.get("map_negative", "negative")
        mu = cfg.get("map_neutral", None)
        neu_txt = f"enabled → {mu}" if en and mu else "disabled"
        return (
            "#### NLP settings: VADER  \n"
            f"• pos_threshold: {pt} | neg_threshold: {nt} | neutral: {neu_txt}  \n"
            f"• mapping: POS→{mp}, NEG→{mn}" + (f", NEU→{mu}" if en and mu else "")
        )
    elif method == "Rules":
        rules = cfg.get("rules", {}) or {}
        cs = bool(cfg.get("case_sensitive", False))
        mode = cfg.get("mode", "most_matches")
        return (
            "#### NLP settings: Rules  \n"
            f"• keywords per label: {_fmt_rules_summary(rules)}  \n"
            f"• case_sensitive: {cs} | mode: {mode}"
        )
    else:
        return f"#### NLP settings: {method}"

# -------- UI --------

def make_tab():
    labels_now = _labels_from_state()

    status_md = gr.Markdown(nlp_status_text())  # <<< banner at top

    gr.Markdown("### NLP Baselines — Select & Configure")
    method = gr.Radio(["VADER", "Rules"], value="VADER", label="Method")

    # VADER controls
    with gr.Group(visible=True) as grp_vader:
        gr.Markdown(
            "VADER returns **compound ∈ [-1, 1]** and pos/neu/neg scores. "
            "We map compound to your dataset labels."
        )
        pos_thresh = gr.Slider(0.0, 0.5, value=0.05, step=0.01, label="Positive threshold (compound ≥)")
        neg_thresh = gr.Slider(-0.5, 0.0, value=-0.05, step=0.01, label="Negative threshold (compound ≤)")
        enable_neutral = gr.Checkbox(label="Enable neutral class (requires a neutral mapping)", value=(_guess_label(labels_now, "neutral") is not None))
        # mapping to dataset labels
        lbl_choices = labels_now
        pos_label = gr.Dropdown(lbl_choices, value=_guess_label(lbl_choices, "positive"), label="Map POSITIVE to")
        neg_label = gr.Dropdown(lbl_choices, value=_guess_label(lbl_choices, "negative"), label="Map NEGATIVE to")
        neu_label = gr.Dropdown(lbl_choices, value=_guess_label(lbl_choices, "neutral"), label="Map NEUTRAL to (optional)")

    # Rules controls
    with gr.Group(visible=False) as grp_rules:
        gr.Markdown(
            "Provide a JSON mapping of **label → list of keywords**. "
            "Prediction picks the label with the most matches (or first match)."
        )
        rules_json = gr.Textbox(
            value=_default_rules_json(labels_now),
            lines=12,
            label="Rules JSON (label → keywords[])"
        )
        case_sensitive = gr.Checkbox(False, label="Case sensitive matching")
        mode = gr.Radio(["most_matches", "first_match"], value="most_matches", label="Decision mode")
        seed_btn = gr.Button("Re-seed defaults for current labels")

    # Save & feedback
    save_btn = gr.Button("Save settings")
    out_msg = gr.JSON(value={"status": "No settings saved yet."})

    # Refresh labels if dataset changed
    refresh_lbl_btn = gr.Button("Refresh labels from Data")
    cur_labels_md = gr.Markdown(f"**Current labels:** {', '.join(labels_now)}")

    # ----- callbacks -----

    def on_method_change(name: str):
        return (
            gr.update(visible=(name == "VADER")),
            gr.update(visible=(name == "Rules")),
        )

    method.change(on_method_change, inputs=method, outputs=[grp_vader, grp_rules])

    def on_refresh_labels():
        labs = _labels_from_state()
        text = f"**Current labels:** {', '.join(labs)}"
        # update dropdown choices
        return (
            gr.update(value=text),
            gr.update(choices=labs, value=_guess_label(labs, "positive")),
            gr.update(choices=labs, value=_guess_label(labs, "negative")),
            gr.update(choices=labs, value=_guess_label(labs, "neutral")),
            gr.update(value=_default_rules_json(labs)),
        )

    refresh_lbl_btn.click(
        on_refresh_labels,
        inputs=None,
        outputs=[cur_labels_md, pos_label, neg_label, neu_label, rules_json]
    )

    def on_seed_rules():
        labs = _labels_from_state()
        return _default_rules_json(labs)

    seed_btn.click(on_seed_rules, inputs=None, outputs=rules_json)

    def on_save(
        method_v,
        pos_t, neg_t, en_neu, pos_lbl, neg_lbl, neu_lbl,
        rules_txt, case_sens, mode_v
    ):
        st = _state_getter()
        st.nlp_method = method_v

        if method_v == "VADER":
            config = {
                "pos_threshold": float(pos_t),
                "neg_threshold": float(neg_t),
                "enable_neutral": bool(en_neu),
                "map_positive": pos_lbl,
                "map_negative": neg_lbl,
                "map_neutral": neu_lbl if en_neu else None,
            }
        else:
            # validate rules JSON
            try:
                parsed = json.loads(rules_txt or "{}")
                if not isinstance(parsed, dict):
                    raise ValueError("Rules must be a dict of label → keywords[].")
                # normalize: ensure list of strings
                rules_norm = {}
                for k, v in parsed.items():
                    if isinstance(v, list):
                        rules_norm[str(k)] = [str(x) for x in v]
                config = {
                    "rules": rules_norm,
                    "case_sensitive": bool(case_sens),
                    "mode": mode_v,
                }
            except Exception as e:
                return {"error": f"Invalid Rules JSON: {e}"}, nlp_status_text()

        st.nlp_config = config
        return {"status": "saved", "method": method_v, "config": config}, nlp_status_text()

    save_event = save_btn.click(
        on_save,
        inputs=[
            method,
            pos_thresh, neg_thresh, enable_neutral, pos_label, neg_label, neu_label,
            rules_json, case_sensitive, mode
        ],
        outputs=[out_msg, status_md]
    )

    # Return handles so app.py can chain updates into Predict tab
    return {"status_md": status_md, "save_event": save_event}
