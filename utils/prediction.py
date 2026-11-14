from __future__ import annotations

import warnings
from collections.abc import Mapping
from math import isfinite
from numbers import Real

import pandas as pd

from utils.model_loader import FEATURES

NUMERIC_BOUNDS = {
    "Age": (29, 77),
    "BP": (94, 200),
    "Cholesterol": (126, 564),
    "Max HR": (71, 202),
    "ST depression": (0.0, 6.2),
}
CATEGORY_CHOICES = {
    "Sex": (0, 1),
    "Chest pain type": (1, 2, 3, 4),
    "FBS over 120": (0, 1),
    "EKG results": (0, 1, 2),
    "Exercise angina": (0, 1),
    "Slope of ST": (1, 2, 3),
    "Number of vessels fluro": (0, 1, 2, 3),
    "Thallium": (3, 6, 7),
}
INTEGER_NUMERIC_FEATURES = {"Age", "BP", "Cholesterol", "Max HR"}


def build_feature_frame(values: Mapping[str, int | float]) -> pd.DataFrame:
    """Validate user entries and preserve the notebook's model input order."""
    missing = [feature for feature in FEATURES if feature not in values]
    extra = [feature for feature in values if feature not in FEATURES]
    if missing or extra:
        raise ValueError(f"Input columns mismatch. Missing: {missing}; unexpected: {extra}.")
    row: dict[str, int | float] = {}
    for feature in FEATURES:
        value = values[feature]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{feature} must be numeric.")
        if not isfinite(value):
            raise ValueError(f"{feature} must be a finite number.")
        if feature in NUMERIC_BOUNDS:
            low, high = NUMERIC_BOUNDS[feature]
            if not low <= value <= high:
                raise ValueError(f"{feature} must be between {low} and {high} (the observed training-data range).")
        if feature in CATEGORY_CHOICES or feature in INTEGER_NUMERIC_FEATURES:
            if not float(value).is_integer():
                raise ValueError(f"{feature} must be a whole number.")
            value = int(value)
        if feature in CATEGORY_CHOICES and value not in CATEGORY_CHOICES[feature]:
            raise ValueError(f"{feature} has an unsupported category selection.")
        row[feature] = float(value) if feature == "ST depression" else int(value)
    return pd.DataFrame([row], columns=FEATURES)


def predict_one(model, features: pd.DataFrame) -> tuple[int, dict[int, float]]:
    """Return the predicted class and probabilities keyed by target class."""
    classes = list(model.classes_)
    if len(classes) != 2 or set(classes) != {0, 1}:
        raise ValueError(f"Expected model classes 0 and 1; received {classes}.")
    # ColumnTransformer returns a transformed matrix without column labels.
    # LightGBM warns about that internal matrix even though the caller supplied
    # the original named columns to the complete fitted pipeline.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        prediction = int(model.predict(features)[0])
        raw = model.predict_proba(features)
    if getattr(raw, "ndim", None) != 2 or raw.shape != (1, len(classes)):
        raise ValueError(f"Expected one probability row for classes {classes}; received shape {getattr(raw, 'shape', None)}.")
    raw = raw[0]
    probabilities = {int(label): float(probability) for label, probability in zip(classes, raw)}
    if prediction not in {0, 1}:
        raise ValueError(f"The selected pipeline predicted unexpected class {prediction}.")
    if set(probabilities) != {0, 1} or any(not isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("The selected pipeline returned invalid binary class probabilities.")
    if abs(sum(probabilities.values()) - 1.0) > 1e-6:
        raise ValueError("The selected pipeline returned probabilities that do not sum to 1.")
    return prediction, probabilities
