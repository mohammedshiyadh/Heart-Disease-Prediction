from __future__ import annotations

import streamlit as st


def render() -> None:
    st.markdown('<div class="eyebrow">Project notes</div>', unsafe_allow_html=True)
    st.title("About HeartSense AI")
    st.write("HeartSense AI turns the original research project into an interactive educational dashboard while preserving its trained models, preprocessing pipelines, notebook, and recorded results.")

    st.markdown("### Project objectives")
    st.markdown("- Explore the UCI and Kaggle Playground datasets and their differences.\n- Compare five classifier families using their recorded evaluation designs.\n- Assess transfer from the synthetic Playground training data to UCI records.\n- Make model predictions and notebook-derived explanations easier to inspect.")

    st.markdown("### Datasets, models, and preprocessing")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Datasets")
        st.write("The project investigated 630,000 synthetic Kaggle Playground training records and 303 records from the UCI Cleveland heart disease dataset.")
        st.markdown("#### Algorithms")
        st.write("Logistic Regression, Random Forest, XGBoost, Linear SVM, and LightGBM were evaluated. The Playground-trained LightGBM pipeline is the default model; the smaller UCI-trained pipeline is available for comparison.")
    with right:
        st.markdown("#### Preprocessing")
        st.write("Numeric values use the saved median imputer and standard scaler. Categorical values use the saved most-frequent imputer and one-hot encoder. These transformations run inside each serialized pipeline.")
        st.markdown("#### Interpretability")
        st.write("The feature ranking is based on grouped mean absolute SHAP values recorded for a 1,000-row validation sample from the Playground experiment.")

    st.markdown("### Evaluation methodology")
    st.write("The Playground comparison reports five-fold cross-validation. The UCI holdout experiment uses a stratified 80/20 split (242 training and 61 test records). The cross-dataset experiment evaluates Playground-trained models against all 303 UCI records; it is not an independent UCI holdout. The recorded comparison table provides accuracy, precision, recall, F1-score, and ROC-AUC where available.")
    with st.expander("Metric definitions"):
        st.markdown("- **Accuracy:** fraction of predictions that match the recorded labels.\n- **Precision:** among predicted positive cases, fraction that are positive in the labels.\n- **Recall:** among positive cases in the labels, fraction predicted positive.\n- **F1-score:** harmonic mean of precision and recall.\n- **ROC-AUC:** how well model scores rank positive cases above negative cases across thresholds.")

    st.markdown("### Technology")
    st.write("Python, Pandas, NumPy, scikit-learn, LightGBM, SHAP, Joblib, Streamlit, and Plotly. XGBoost was also used in the research comparison experiments.")

    st.markdown("### Research limitations")
    st.write("The Playground data is synthetic; the UCI dataset and its 61-record holdout are small; and the cross-dataset evaluation uses the full set of 303 UCI records rather than a separate test split. None of the models have independent clinical validation.")
    st.markdown("### Future improvements")
    st.write("Potential research steps include external evaluation on larger, diverse datasets, probability calibration, and validation of the preprocessing and model behaviour with independent cohorts. These are future research directions, not capabilities claimed by this demo.")
    st.markdown('<div class="disclaimer">This application is for educational and research purposes only. It has not been clinically validated and must not be used for medical diagnosis, treatment decisions, or emergency assessment.</div>', unsafe_allow_html=True)
