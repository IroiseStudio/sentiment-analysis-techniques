from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class AppState:
    # Data
    df_raw: Any = None
    df_clean: Any = None
    text_col: Optional[str] = None
    label_col: Optional[str] = None
    labels_: List[str] = field(default_factory=list)
    preproc_config: Dict[str, Any] = field(default_factory=lambda: {
        "lowercase": True,
        "remove_punct": False,
        "remove_numbers": False,
        "stopwords": False,
        "stem_or_lemma": "none",  # 'none' | 'stem' | 'lemma'
    })

    # Shared evaluation split (indices into df_clean/df_raw)
    eval_split: Dict[str, List[int]] = field(default_factory=dict)  # {"train":[...], "test":[...]}

    # ML
    train_test_split_config: Dict[str, Any] = field(default_factory=lambda: {
        "test_size": 0.2,
        "random_state": 42,
        "stratify": True,
    })
    ml_pipeline: Any = None
    ml_metrics: Dict[str, Any] = field(default_factory=dict)
    ml_model_name: Optional[str] = None

    # NLP baselines
    nlp_method: Optional[str] = None
    nlp_config: Dict[str, Any] = field(default_factory=dict)
    nlp_metrics: Dict[str, Any] = field(default_factory=dict)

    # Transformers
    transformer_id: Optional[str] = None
    transformer_config: Dict[str, Any] = field(default_factory=dict)
    transformer_pipeline: Any = None
    tr_metrics: Dict[str, Any] = field(default_factory=dict)

    # Prompt Engineering
    prompt_template: str = ""
    prompt_examples: List[Dict[str, str]] = field(default_factory=list)
    prompt_backend: str = "none"  # 'none' | 'hf_pipeline' | 'hf_inference' | 'openai'
    prompt_config: Dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.0,
        "max_tokens": 128,
        "allowed_labels": [],  # list of strings for multi-class
    })
    prompt_metrics: Dict[str, Any] = field(default_factory=dict)

# simple singleton
_state: Optional[AppState] = None

def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
    return _state
