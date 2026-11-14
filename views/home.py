from __future__ import annotations

import streamlit as st


# =========================================================
# HERO SECTION
# =========================================================

def render_hero():

    st.markdown(
        """
        <div class="home-hero">

            <h1>HeartSense AI</h1>

            <p>
                Understand heart health through machine learning.
                Explore predictions, compare algorithms and
                investigate model explainability.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# PROJECT OVERVIEW
# =========================================================

def render_overview():

    st.subheader("Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Playground Records",
            "630,000",
        )

    with col2:
        st.metric(
            "UCI Records",
            "303",
        )

    with col3:
        st.metric(
            "Algorithms Evaluated",
            "5",
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

def render_model_results():

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Playground Accuracy",
            "88.80%",
        )
        st.caption("ROC-AUC: 95.48%")

    with col2:
        st.metric(
            "UCI Holdout Accuracy",
            "93.44%",
        )
        st.caption("ROC-AUC: 96.43%")

    with col3:
        st.metric(
            "Cross-Dataset Accuracy",
            "82.18%",
        )
        st.caption("ROC-AUC: 91.82%")

    st.caption(
        "These results come from separate experiments. "
        "The UCI holdout and cross-dataset results are not interchangeable."
    )


def _navigate_to_prediction():
    st.session_state["heartsense_active_page"] = "Prediction"


# =========================================================
# QUICK PREDICTION
# =========================================================

def render_quick_prediction():

    st.divider()

    st.markdown("## Prediction workspace")

    st.write(
        "Enter information using human-readable choices, review your selections, "
        "then request an output from the saved LightGBM pipeline."
    )
    st.button(
        "Open prediction workspace",
        type="primary",
        on_click=_navigate_to_prediction,
    )


# =========================================================
# PROJECT MODULES
# =========================================================

def render_modules():

    st.divider()

    st.subheader("Explore HeartSense AI")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="section-card">
                <h3>🫀 Prediction</h3>
                <p>
                    Enter patient features and explore
                    the LightGBM model's predictions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="section-card">
                <h3>📊 Model Comparison</h3>
                <p>
                    Compare Logistic Regression, Random Forest,
                    XGBoost, Linear SVM and LightGBM.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            """
            <div class="section-card">
                <h3>🔍 Explainability</h3>
                <p>
                    Investigate feature importance and
                    model explanations using SHAP.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# METHODOLOGY
# =========================================================

def render_methodology():

    st.divider()

    st.subheader("Model Development")

    st.write(
        """
        HeartSense AI evaluates five machine-learning algorithms:

        - Logistic Regression
        - Random Forest
        - XGBoost
        - Linear Support Vector Machine
        - LightGBM

        The prediction interface uses a saved LightGBM pipeline.
        The pipeline includes numerical preprocessing, categorical
        encoding and the trained classifier.
        """
    )


# =========================================================
# DISCLAIMER
# =========================================================

def render_disclaimer():

    st.warning(
        """
        HeartSense AI is an educational machine-learning demonstration and has not been clinically validated.
        Predictions are not medical diagnoses and must not be used to make treatment decisions or assess medical emergencies.
        Consult a qualified healthcare professional for medical advice.
        """
    )


# =========================================================
# MAIN HOMEPAGE
# =========================================================

def render():

    render_hero()

    render_overview()

    render_model_results()

    render_quick_prediction()

    render_modules()

    render_methodology()

