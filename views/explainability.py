from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.visualizations import shap_figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEESWARM_PATH = PROJECT_ROOT / "assets" / "shap_beeswarm.png"
SHAP_IMPORTANCE = {
    "Thallium": 1.19808,
    "Chest pain type": 0.95884,
    "Number of vessels fluro": 0.69684,
    "Max HR": 0.67120,
    "Exercise angina": 0.47393,
    "Slope of ST": 0.44356,
    "Sex": 0.41523,
    "ST depression": 0.36148,
    "Age": 0.31306,
    "EKG results": 0.19031,
    "Cholesterol": 0.12164,
    "BP": 0.04706,
    "FBS over 120": 0.00324,
}


def render() -> None:
    st.markdown('<div class="eyebrow">Model behaviour</div>', unsafe_allow_html=True)
    st.title("Explainability")
    st.write("These explanations describe the Playground-trained LightGBM model. SHAP values represent how each feature moves the model output relative to its expected output; they describe model behaviour, not causes or medical importance.")

    st.markdown("### Ranked feature importance")
    st.write("Grouped mean absolute SHAP values from the notebook's 1,000-row Playground validation sample. One-hot encoded category contributions were aggregated back to their original feature names.")
    importance = pd.DataFrame({"Feature": list(SHAP_IMPORTANCE), "SHAP Importance": list(SHAP_IMPORTANCE.values())})
    st.plotly_chart(shap_figure(importance), use_container_width=True)
    with st.expander("View the recorded grouped SHAP values"):
        display = importance.rename(columns={"SHAP Importance": "Grouped mean |SHAP|"})
        st.dataframe(display, hide_index=True, use_container_width=True)

    st.markdown("### SHAP summary / beeswarm")
    st.write("This is the beeswarm plot saved in the original research notebook. Each dot represents one validation row for an encoded feature; its horizontal position shows the SHAP value, and its colour shows whether the encoded feature value is low or high.")
    if BEESWARM_PATH.is_file():
        st.image(str(BEESWARM_PATH), use_container_width=True, caption="Original notebook SHAP beeswarm · sample of 1,000 Playground validation records")
    else:
        st.warning("The notebook's saved SHAP beeswarm image is missing. The recorded feature ranking above is still available.")

    st.markdown("### Reading contribution direction")
    st.markdown(
        "- **Positive SHAP values** push the model output toward the dataset's *Presence* label.\n"
        "- **Negative SHAP values** push the model output toward the dataset's *Absence* label.\n"
        "- The vertical zero line is the point where a feature has no contribution in that direction for the plotted row.\n"
        "- For example, the encoded `Chest pain type = 4` row has positive and negative contributions across records; the chart shows variation in this model, not a causal or clinical effect."
    )
    st.caption("These are model-specific explanations for the Playground LightGBM pipeline and the notebook's validation sample. The notebook did not save per-person SHAP values, so this page does not claim to explain an individual prediction.")
