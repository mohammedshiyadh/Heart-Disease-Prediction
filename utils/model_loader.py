from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = {
    "Playground LightGBM (primary)": PROJECT_ROOT / "models" / "lightgbm_playground.pkl",
    "UCI LightGBM (experimental)": PROJECT_ROOT / "models" / "lightgbm_uci.pkl",
}
FEATURES = [
    "Age", "Sex", "Chest pain type", "BP", "Cholesterol", "FBS over 120",
    "EKG results", "Max HR", "Exercise angina", "ST depression", "Slope of ST",
    "Number of vessels fluro", "Thallium",
]


@st.cache_resource(show_spinner="Loading the selected LightGBM pipeline…")
def load_model(model_name: str) -> Any:
    """Load and sanity-check one of the original serialized pipelines."""
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model selection: {model_name}")
    path = MODEL_PATHS[model_name]
    if not path.is_file():
        raise FileNotFoundError(f"Model file is missing: {path}")
    model = joblib.load(path)
    if not callable(getattr(model, "predict", None)) or not callable(getattr(model, "predict_proba", None)):
        raise TypeError(f"The saved object at {path.name} does not expose predict() and predict_proba().")
    fitted_features = getattr(model, "feature_names_in_", None)
    if fitted_features is None:
        raise ValueError(f"The saved pipeline at {path.name} does not expose its fitted feature schema.")
    if list(fitted_features) != FEATURES:
        raise ValueError("The saved pipeline's feature names or order do not match the inspected training notebook.")
    classes = list(getattr(model, "classes_", []))
    if len(classes) != 2 or set(classes) != {0, 1}:
        raise ValueError(f"Expected binary classes 0 and 1; found {classes}.")
    return model
