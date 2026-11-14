from __future__ import annotations

from math import isclose, isfinite

import numpy as np
import pandas as pd

from utils.model_loader import FEATURES


def explain_prediction(model, features: pd.DataFrame) -> pd.DataFrame:
    """Return grouped local SHAP contributions for one fitted LightGBM pipeline.

    The saved preprocessing step transforms the same named raw inputs used for
    prediction. Additivity is checked against LightGBM's raw class-1 output so
    unsupported or incompatible explainers fail closed instead of fabricating
    a contribution summary.
    """
    if list(features.columns) != FEATURES:
        raise ValueError("SHAP input columns do not match the trained feature schema.")
    if set(getattr(model, "classes_", [])) != {0, 1}:
        raise ValueError("SHAP explanation requires the model's binary 0/1 target classes.")
    if not hasattr(model, "named_steps") or not {"preprocessor", "classifier"}.issubset(model.named_steps):
        raise TypeError("The selected model does not contain the expected preprocessing and classifier steps.")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "booster_"):
        raise TypeError("The selected classifier does not expose a fitted LightGBM booster.")

    transformed = preprocessor.transform(features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)
    feature_names = list(preprocessor.get_feature_names_out())
    if transformed.shape != (1, len(feature_names)):
        raise ValueError("The fitted preprocessor returned an unexpected feature matrix shape.")

    import shap

    explainer = shap.TreeExplainer(classifier)
    explanation = explainer(transformed)
    values = np.asarray(explanation.values)
    if values.shape != (1, len(feature_names)):
        raise ValueError(f"The SHAP explainer returned an unsupported output shape: {values.shape}.")
    contributions = values[0]
    if not np.isfinite(contributions).all():
        raise ValueError("The SHAP explainer returned non-finite contributions.")

    expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    raw_output = float(classifier.booster_.predict(transformed, raw_score=True)[0])
    reconstructed = expected + float(contributions.sum())
    if not isclose(reconstructed, raw_output, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError("SHAP contributions failed the model-output additivity check.")

    grouped = {feature: 0.0 for feature in FEATURES}
    for transformed_name, contribution in zip(feature_names, contributions):
        if transformed_name.startswith("num__"):
            original_name = transformed_name.removeprefix("num__")
            if original_name not in grouped:
                raise ValueError(f"Unexpected numerical SHAP feature: {transformed_name}.")
        elif transformed_name.startswith("cat__"):
            encoded_name = transformed_name.removeprefix("cat__")
            matches = [name for name in FEATURES if encoded_name.startswith(f"{name}_")]
            if not matches:
                raise ValueError(f"Could not map encoded SHAP feature to an original column: {transformed_name}.")
            original_name = max(matches, key=len)
        else:
            raise ValueError(f"Unexpected transformed SHAP feature: {transformed_name}.")
        grouped[original_name] += float(contribution)

    if not all(isfinite(value) for value in grouped.values()):
        raise ValueError("Grouped SHAP contributions contain non-finite values.")
    result = pd.DataFrame(
        {"Feature": list(grouped), "Contribution": list(grouped.values())}
    )
    result["Abs contribution"] = result["Contribution"].abs()
    return result.sort_values("Abs contribution", ascending=False).reset_index(drop=True)
