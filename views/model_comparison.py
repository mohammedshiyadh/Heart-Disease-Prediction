from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.visualizations import comparison_figure

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "model_comparison.csv"
METRICS = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
EXPERIMENTS = {
    "Playground CV": (
        "A. Playground five-fold cross-validation",
        "Five-fold cross-validation on the 630,000-record synthetic Playground training dataset.",
    ),
    "UCI Holdout": (
        "B. UCI holdout evaluation",
        "Stratified 80/20 split of the 303-record UCI dataset: 242 training records and 61 holdout records.",
    ),
    "Cross-Dataset": (
        "C. Cross-dataset evaluation · Playground → UCI",
        "The Playground-trained models were evaluated on all 303 UCI records. This transfer experiment is separate from, and does not use, the 61-record UCI holdout evaluation above.",
    ),
}


def _load_results() -> pd.DataFrame:
    data = pd.read_csv(RESULTS_PATH)
    required = {"Model", "Experiment", *METRICS}
    if not required.issubset(data.columns):
        raise ValueError(f"Results CSV is missing columns: {sorted(required - set(data.columns))}")
    if data.empty:
        raise ValueError("The results CSV contains no experiment rows.")
    for metric in METRICS:
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
    if data[METRICS].isna().any().any():
        raise ValueError("One or more recorded metric values are missing or non-numeric.")
    if not data[METRICS].apply(lambda column: column.between(0, 1).all()).all():
        raise ValueError("One or more recorded metric values are outside the expected 0–1 range.")
    unknown = set(data["Experiment"]) - set(EXPERIMENTS)
    if unknown:
        raise ValueError(f"Unrecognized experiment labels in results CSV: {sorted(unknown)}")
    return data


def _render_experiment(data: pd.DataFrame, experiment: str) -> None:
    title, description = EXPERIMENTS[experiment]
    st.markdown(f"### {title}")
    st.write(description)
    subset = data.loc[data["Experiment"] == experiment].copy()
    if subset.empty:
        st.info("No recorded results are available for this experiment.")
        return
    st.plotly_chart(comparison_figure(data, experiment), use_container_width=True)
    display = subset.copy()
    for metric in METRICS:
        display[metric] = display[metric].map(lambda value: f"{value:.2%}")
    st.dataframe(display, hide_index=True, use_container_width=True)
    if experiment == "Playground CV":
        st.caption("LinearSVC does not expose predict_proba in this notebook; its ROC-AUC was computed from decision scores.")


def render() -> None:
    st.markdown('<div class="eyebrow">Evaluation results</div>', unsafe_allow_html=True)
    st.title("Model comparison")
    st.write("The three experiments have different datasets and evaluation designs. Review each result group in its own context; there is no single shared test set across these charts.")
    with st.expander("What do these metrics mean?"):
        st.markdown(
            "- **Accuracy:** fraction of predictions that match the recorded labels.\n"
            "- **Precision:** among predicted positive cases, fraction that are positive in the labels.\n"
            "- **Recall:** among positive cases in the labels, fraction predicted positive.\n"
            "- **F1-score:** harmonic mean of precision and recall.\n"
            "- **ROC-AUC:** ability of model scores to rank positive cases above negative cases across thresholds."
        )
    try:
        data = _load_results()
    except Exception as exc:
        st.error(f"Could not load the preserved experiment results: {exc}")
        return

    for index, experiment in enumerate(EXPERIMENTS):
        if index:
            st.divider()
        _render_experiment(data, experiment)
