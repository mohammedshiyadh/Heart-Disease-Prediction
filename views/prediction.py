from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from utils.model_loader import FEATURES, MODEL_PATHS, load_model
from utils.explainability import explain_prediction
from utils.prediction import (
    INTEGER_NUMERIC_FEATURES,
    NUMERIC_BOUNDS,
    build_feature_frame,
    predict_one,
)
from utils.theme import theme_palette
from utils.visualizations import contribution_figure, probability_figure


# UCI Cleveland data dictionary encodings. The project notebook confirms the
# same category values in both datasets and trains directly on these columns.
CATEGORY_ENCODINGS: dict[str, dict[str, int]] = {
    "Sex": {"Female": 0, "Male": 1},
    "Chest pain type": {
        "Typical angina": 1,
        "Atypical angina": 2,
        "Non-anginal pain": 3,
        "Asymptomatic": 4,
    },
    "FBS over 120": {"No": 0, "Yes": 1},
    "EKG results": {
        "Normal": 0,
        "ST-T abnormality": 1,
        "Probable or definite LVH pattern": 2,
    },
    "Exercise angina": {"No": 0, "Yes": 1},
    "Slope of ST": {"Upsloping": 1, "Flat": 2, "Downsloping": 3},
    "Number of vessels fluro": {
        "0 major vessels": 0,
        "1 major vessel": 1,
        "2 major vessels": 2,
        "3 major vessels": 3,
    },
    "Thallium": {
        "Normal": 3,
        "Fixed defect": 6,
        "Reversible defect": 7,
    },
}

FIELD_LABELS = {
    "Age": "Age (years)",
    "Sex": "Sex",
    "Chest pain type": "Chest pain type",
    "BP": "Resting blood pressure (mmHg)",
    "Cholesterol": "Serum cholesterol (mg/dL)",
    "FBS over 120": "Fasting blood sugar above 120 mg/dL",
    "EKG results": "ECG (EKG) at rest",
    "Max HR": "Highest heart rate during the test (bpm)",
    "Exercise angina": "Chest discomfort during exercise",
    "ST depression": "ST line change during exercise",
    "Slope of ST": "ST line shape during exercise",
    "Number of vessels fluro": "Major vessels noted on imaging",
    "Thallium": "Thallium scan result",
}

NUMERIC_HELP = {
    "Age": "Enter your age in years.",
    "BP": "Copy the resting blood pressure from your report, in mmHg.",
    "Cholesterol": "Copy the serum cholesterol measurement from your report, in mg/dL.",
    "Max HR": "Copy the highest heart rate recorded during the test, in beats per minute (bpm).",
    "ST depression": "Copy this measurement from an exercise ECG report. Do not estimate it. The dataset does not specify a separate unit.",
}
CATEGORY_HELP = {
    "Sex": "Choose the category that matches your report.",
    "Chest pain type": "Angina is the dataset's term for chest discomfort. Use the wording in your report; do not guess if it is different.",
    "FBS over 120": "Was fasting blood sugar above 120 mg/dL in your report? Choose the recorded result.",
    "EKG results": "ECG (also called EKG) records the heart's electrical activity. LVH means left ventricular hypertrophy. These are report categories, not diagnoses.",
    "Exercise angina": "Choose whether chest discomfort during exercise was recorded in your report.",
    "Slope of ST": "The ST segment is part of the line on an ECG tracing. Copy the shape recorded during exercise in your report; do not guess.",
    "Number of vessels fluro": "Copy the number of major vessels noted in the report. Fluoroscopy is an imaging method used during some heart tests.",
    "Thallium": "Copy the thallium scan result from your report. If you do not have a result, ask a healthcare professional rather than guessing.",
}

INTEGER_FIELDS = set(CATEGORY_ENCODINGS) | INTEGER_NUMERIC_FEATURES
FORM_KEYS = {
    feature: f"hs_readable_choice_v2_{index}" if feature in CATEGORY_ENCODINGS else f"hs_exact_measurement_v2_{index}"
    for index, feature in enumerate(FEATURES)
}
SLIDER_KEYS = {feature: f"hs_measurement_slider_v2_{index}" for index, feature in enumerate(FEATURES) if feature not in CATEGORY_ENCODINGS}
MODEL_KEY = "hs_prediction_model_choice"
MEASUREMENT_MODE_KEY = "hs_measurement_input_mode"
PREVIOUS_MEASUREMENT_MODE_KEY = "hs_previous_measurement_input_mode"
REVIEW_KEY = "hs_reviewed_input"
RESULT_KEY = "hs_prediction_result"
DEMO_KEY = "hs_demo_profile_loaded"
PLACEHOLDER = "Choose an option"

DEMO_PROFILE = {
    "Age": 54,
    "Sex": "Male",
    "Chest pain type": "Asymptomatic",
    "BP": 130,
    "Cholesterol": 243,
    "FBS over 120": "No",
    "EKG results": "Normal",
    "Max HR": 157,
    "Exercise angina": "No",
    "ST depression": 0.10,
    "Slope of ST": "Upsloping",
    "Number of vessels fluro": "0 major vessels",
    "Thallium": "Normal",
}
MEASUREMENT_MODES = ("Type exact values", "Adjust sliders")


def _slider_start(feature: str) -> int | float:
    low, high = NUMERIC_BOUNDS[feature]
    midpoint = (low + high) / 2
    return round(midpoint, 2) if feature == "ST depression" else round(midpoint)


def _switch_measurement_mode() -> None:
    previous = st.session_state.get(PREVIOUS_MEASUREMENT_MODE_KEY, MEASUREMENT_MODES[0])
    current = st.session_state.get(MEASUREMENT_MODE_KEY, MEASUREMENT_MODES[0])
    if previous != current:
        for feature, exact_key in FORM_KEYS.items():
            if feature in CATEGORY_ENCODINGS:
                continue
            slider_key = SLIDER_KEYS[feature]
            if previous == MEASUREMENT_MODES[1]:
                value = st.session_state.get(slider_key, _slider_start(feature))
            else:
                value = st.session_state.get(exact_key)
            if value is not None:
                target_key = slider_key if current == MEASUREMENT_MODES[1] else exact_key
                st.session_state[target_key] = value
        st.session_state[PREVIOUS_MEASUREMENT_MODE_KEY] = current
        st.session_state["hs_measurement_confirmation_v2"] = False
        st.session_state.pop(REVIEW_KEY, None)
        st.session_state.pop(RESULT_KEY, None)


def encode_user_input(user_values: dict[str, Any]) -> pd.DataFrame:
    """Map readable labels to the pipeline's exact ordered feature schema."""
    missing = [
        FIELD_LABELS[feature]
        for feature in FEATURES
        if user_values.get(feature) is None
        or user_values.get(feature) == PLACEHOLDER
    ]
    if missing:
        raise ValueError("Please complete: " + ", ".join(missing) + ".")

    encoded: dict[str, int | float] = {}
    for feature in FEATURES:
        value = user_values[feature]
        if feature in CATEGORY_ENCODINGS:
            try:
                encoded[feature] = CATEGORY_ENCODINGS[feature][value]
            except KeyError as exc:
                raise ValueError(f"Choose a valid option for {FIELD_LABELS[feature]}.") from exc
        else:
            encoded[feature] = value

    try:
        frame = build_feature_frame(encoded)
    except ValueError as exc:
        message = str(exc)
        for feature, friendly_label in FIELD_LABELS.items():
            message = message.replace(feature, friendly_label)
        raise ValueError(message) from exc
    if list(frame.columns) != FEATURES:
        raise ValueError("The prepared input does not match the trained model's feature order.")
    for feature in INTEGER_FIELDS:
        if not pd.api.types.is_integer_dtype(frame[feature].dtype):
            raise ValueError(f"{FIELD_LABELS[feature]} did not produce the required whole-number value.")
    if not pd.api.types.is_float_dtype(frame["ST depression"].dtype):
        raise ValueError("ST depression did not produce the required numeric measurement.")
    return frame


def _validate_model_category_mappings(model: Any) -> None:
    """Fail closed if a saved pipeline no longer matches the readable choices."""
    named_steps = getattr(model, "named_steps", {})
    preprocessor = named_steps.get("preprocessor")
    if preprocessor is None or not hasattr(preprocessor, "transformers_"):
        raise ValueError("The selected saved pipeline does not expose its fitted category mapping.")

    categorical = next(
        (step for step in preprocessor.transformers_ if step[0] == "cat"),
        None,
    )
    if categorical is None:
        raise ValueError("The selected saved pipeline is missing its fitted category mapping.")
    transformer, configured_features = categorical[1], list(categorical[2])
    configured = getattr(transformer, "named_steps", {}).get("encoder")
    categories = getattr(configured, "categories_", None)
    if categories is None or len(categories) != len(configured_features):
        raise ValueError("The selected saved pipeline has an invalid fitted category mapping.")
    if set(configured_features) != set(CATEGORY_ENCODINGS) or len(configured_features) != len(CATEGORY_ENCODINGS):
        raise ValueError("The selected saved pipeline's category features do not match the reviewed form.")

    for feature, readable_choices in CATEGORY_ENCODINGS.items():
        category_index = configured_features.index(feature)
        fitted_values = set(categories[category_index])
        if not set(readable_choices.values()).issubset(fitted_values):
            raise ValueError(
                f"The readable choices for {FIELD_LABELS[feature]} do not match the selected saved pipeline."
            )


def _reset_form() -> None:
    st.session_state[MEASUREMENT_MODE_KEY] = MEASUREMENT_MODES[0]
    st.session_state[PREVIOUS_MEASUREMENT_MODE_KEY] = MEASUREMENT_MODES[0]
    for feature, key in FORM_KEYS.items():
        if feature in CATEGORY_ENCODINGS:
            st.session_state[key] = PLACEHOLDER
        else:
            st.session_state[key] = None
            st.session_state[SLIDER_KEYS[feature]] = _slider_start(feature)
    st.session_state.pop(REVIEW_KEY, None)
    st.session_state.pop(RESULT_KEY, None)
    st.session_state["hs_measurement_confirmation_v2"] = False
    st.session_state[DEMO_KEY] = False


def _load_demo_profile() -> None:
    for feature, value in DEMO_PROFILE.items():
        st.session_state[FORM_KEYS[feature]] = value
        if feature in SLIDER_KEYS:
            st.session_state[SLIDER_KEYS[feature]] = value
    st.session_state[DEMO_KEY] = True
    st.session_state["hs_measurement_confirmation_v2"] = False
    st.session_state.pop(REVIEW_KEY, None)
    st.session_state.pop(RESULT_KEY, None)


def _clear_review_for_model_change() -> None:
    st.session_state.pop(REVIEW_KEY, None)
    st.session_state.pop(RESULT_KEY, None)


def _format_value(feature: str, value: Any) -> str:
    if feature == "Age":
        return f"{int(value)} years"
    if feature == "BP":
        return f"{int(value)} mmHg"
    if feature == "Cholesterol":
        return f"{int(value)} mg/dL"
    if feature == "Max HR":
        return f"{int(value)} bpm"
    if feature == "ST depression":
        return f"{float(value):.2f}"
    return str(value)


def _summary_frame(user_values: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Information": FIELD_LABELS[feature], "Your selection": _format_value(feature, user_values[feature])}
            for feature in FEATURES
        ]
    )


def _render_numeric_input(feature: str, input_mode: str) -> int | float | None:
    low, high = NUMERIC_BOUNDS[feature]
    if input_mode == MEASUREMENT_MODES[1]:
        return st.slider(
            FIELD_LABELS[feature],
            min_value=low,
            max_value=high,
            value=_slider_start(feature),
            step=0.01 if feature == "ST depression" else 1,
            format="%.2f" if feature == "ST depression" else None,
            key=SLIDER_KEYS[feature],
            help=(
                f"{NUMERIC_HELP[feature]} This slider starts at the middle of the observed training-data span. "
                "That starting point is not your measurement or a normal range. Move it to the exact value in your report. "
                f"Observed training-data span: {low}–{high}."
            ),
        )

    kwargs: dict[str, Any] = {
        "label": FIELD_LABELS[feature],
        "min_value": low,
        "max_value": high,
        "value": None,
        "step": 0.01 if feature == "ST depression" else 1,
        "key": FORM_KEYS[feature],
        "help": f"{NUMERIC_HELP[feature]} Observed training-data range: {low}–{high}.",
        "placeholder": "Enter a measurement",
    }
    if feature == "ST depression":
        kwargs["format"] = "%.2f"
    return st.number_input(**kwargs)


def _render_category_input(feature: str) -> str:
    options = [PLACEHOLDER, *CATEGORY_ENCODINGS[feature].keys()]
    return st.selectbox(
        FIELD_LABELS[feature],
        options=options,
        index=0,
        key=FORM_KEYS[feature],
        help=CATEGORY_HELP[feature],
    )


def render_input_form(model_name: str, input_mode: str) -> tuple[bool, dict[str, Any] | None]:
    """Collect the three logical input groups; do not run inference here."""
    with st.form("heartsense_prediction_form", clear_on_submit=False):
        st.markdown("#### 1 · Personal information")
        personal_left, personal_right = st.columns(2)
        with personal_left:
            age = _render_numeric_input("Age", input_mode)
        with personal_right:
            sex = _render_category_input("Sex")

        st.markdown("#### 2 · Cardiovascular measurements")
        measure_columns = st.columns(2)
        with measure_columns[0]:
            bp = _render_numeric_input("BP", input_mode)
            cholesterol = _render_numeric_input("Cholesterol", input_mode)
            fbs = _render_category_input("FBS over 120")
        with measure_columns[1]:
            max_hr = _render_numeric_input("Max HR", input_mode)

        st.markdown("#### 3 · Heart test information")
        test_columns = st.columns(2)
        with test_columns[0]:
            chest_pain = _render_category_input("Chest pain type")
            ekg = _render_category_input("EKG results")
            exercise_angina = _render_category_input("Exercise angina")
            thallium = _render_category_input("Thallium")
        with test_columns[1]:
            st_depression = _render_numeric_input("ST depression", input_mode)
            slope = _render_category_input("Slope of ST")
            vessels = _render_category_input("Number of vessels fluro")

        measurements_confirmed = st.checkbox(
            "I reviewed these inputs and understand this is an educational model output, not a diagnosis.",
            key="hs_measurement_confirmation_v2",
            help="For your own information, copy measurements from your report rather than estimating them. The demonstration profile is illustrative only.",
        )
        submitted = st.form_submit_button(
            "Review my information",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return False, None
    return True, {
        "Age": age,
        "Sex": sex,
        "Chest pain type": chest_pain,
        "BP": bp,
        "Cholesterol": cholesterol,
        "FBS over 120": fbs,
        "EKG results": ekg,
        "Max HR": max_hr,
        "Exercise angina": exercise_angina,
        "ST depression": st_depression,
        "Slope of ST": slope,
        "Number of vessels fluro": vessels,
        "Thallium": thallium,
        "_measurements_confirmed": measurements_confirmed,
        "_model_name": model_name,
    }


def _render_input_review(user_values: dict[str, Any], model_name: str) -> None:
    st.markdown("### Review your information")
    st.table(_summary_frame(user_values))
    st.caption(f"Selected model: **{model_name}**")


def _render_result(result: dict[str, Any]) -> None:
    predicted_class = result["predicted_class"]
    class_name = "Presence" if predicted_class == 1 else "Absence"
    probabilities = result["probabilities"]
    model_name = result["model_name"]
    palette = theme_palette()

    st.markdown("### Prediction result")
    st.markdown(
        f"""
        <style>
        .hs-prediction-result {{
            background: var(--hs-card);
            color: var(--hs-card-text);
            border: 1px solid color-mix(in srgb, var(--hs-card-text) 18%, transparent);
            border-left: 6px solid var(--hs-accent);
            border-radius: 16px;
            padding: 1.1rem 1.35rem;
            box-shadow: 0 8px 24px rgba(20, 43, 58, .08);
            margin-bottom: .8rem;
        }}
        .hs-prediction-result * {{ color: var(--hs-card-text) !important; }}
        .hs-prediction-result .hs-prediction-label {{ color: var(--hs-accent) !important; font-size: .77rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }}
        .hs-prediction-result h3 {{ margin: .25rem 0; font-size: 1.55rem; }}
        </style>
        <div class="hs-prediction-result">
            <div class="hs-prediction-label">Output from {model_name}</div>
            <h3>Model label: {class_name}</h3>
            <div>The model matched the “{class_name}” label used in its training data. This is not a medical assessment.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Model output probabilities")
    probability_left, probability_right = st.columns(2)
    with probability_left:
        st.metric("Presence label · model probability", f"{probabilities[1]:.2%}")
    with probability_right:
        st.metric("Absence label · model probability", f"{probabilities[0]:.2%}")
    st.caption(
        "These percentages show how the model split its output between the two training labels. "
        "They are not the chance that this prediction is correct."
    )
    st.plotly_chart(
        probability_figure(
            probabilities,
            accent=palette["accent"],
            card=palette["card"],
            text_color=palette["card_text"],
        ),
        use_container_width=True,
    )

    with st.expander("What this output means"):
        st.write(
            "Presence and Absence are the two labels used in the training data. "
            "These percentages are the selected model's class probabilities for this input; "
            "they do not tell us how likely this model is to be correct for one person and are not clinically validated estimates of disease risk. "
            "A Presence output does not confirm disease, and an Absence output does not rule it out."
        )

    st.markdown("#### What influenced this result most?")
    contributions = result.get("contributions")
    if contributions is not None:
        strongest = contributions.iloc[0]
        strongest_feature = FIELD_LABELS[strongest["Feature"]]
        strongest_value = float(strongest["Contribution"])
        if strongest_value > 0:
            direction = "toward the Presence label"
        elif strongest_value < 0:
            direction = "toward the Absence label"
        else:
            direction = "in neither direction"
        st.write(
            f"**Strongest model influence:** {strongest_feature} moved this model's score {direction}. "
            "This describes how the model responded to this input. It does not identify a medical cause or the main health problem."
        )
        st.write(
            "Each bar estimates how one answer moved this model's score toward or away from its Presence label. "
            "Longer bars mean a larger effect on this model's score, not greater medical importance."
        )
        visible = contributions.head(8).copy()
        visible["Feature"] = visible["Feature"].map(FIELD_LABELS)
        with st.expander("View this model explanation", expanded=True):
            st.plotly_chart(
                contribution_figure(
                    visible,
                    accent=palette["accent"],
                    card=palette["card"],
                    text_color=palette["card_text"],
                ),
                use_container_width=True,
            )
            st.caption(
                "SHAP contributions describe this model's behavior for the submitted input. "
                "They do not establish medical importance or causation, and are not a treatment recommendation."
            )
    else:
        st.info(
            "The model output is available, but an individual explanation could not be verified for this run. "
            "No feature contribution values are shown."
        )
        if result.get("explanation_error"):
            with st.expander("Explanation details"):
                st.code(result["explanation_error"])

    st.markdown("#### Submitted information")
    st.table(_summary_frame(result["user_values"]))

    export_rows = [
        {"Information": FIELD_LABELS[feature], "Your selection": _format_value(feature, result["user_values"][feature])}
        for feature in FEATURES
    ]
    export_rows.extend(
        [
            {"Information": "Model used", "Your selection": model_name},
            {"Information": "Predicted dataset class", "Your selection": class_name},
            {"Information": "Model probability · Presence", "Your selection": f"{probabilities[1]:.8f}"},
            {"Information": "Model probability · Absence", "Your selection": f"{probabilities[0]:.8f}"},
        ]
    )
    csv_data = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download prediction summary (CSV)",
        data=csv_data,
        file_name="heartsense_prediction_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown(
        '<div class="hs-disclaimer"><strong>Educational and research use only.</strong> '
        'HeartSense AI is an educational machine-learning demonstration and has not been clinically validated. '
        'Predictions are not medical diagnoses and must not be used to make treatment decisions or assess medical emergencies. '
        'Consult a qualified healthcare professional for medical advice.</div>',
        unsafe_allow_html=True,
    )


def render() -> None:
    st.markdown('<div class="hs-page-kicker">Saved model · Human-readable inputs</div>', unsafe_allow_html=True)
    st.title("Prediction workspace")
    st.write(
        "Use your own measurements and the wording in your test report. Review the summary before you choose to run the demonstration model. "
        "If a measurement or test result is unfamiliar, do not guess—ask a healthcare professional for help reading the report."
    )

    model_name = st.selectbox(
        "Model for this demonstration",
        options=list(MODEL_PATHS),
        index=0,
        key=MODEL_KEY,
        on_change=_clear_review_for_model_change,
        help="The Playground-trained model is primary. The UCI model is an experimental option trained on a smaller dataset.",
    )
    if model_name.startswith("UCI"):
        st.info("Experimental model: the UCI pipeline was trained on 242 records. Its output is not a clinical assessment.")

    input_mode = st.radio(
        "How would you like to enter measurements?",
        options=list(MEASUREMENT_MODES),
        horizontal=True,
        key=MEASUREMENT_MODE_KEY,
        on_change=_switch_measurement_mode,
        help="Type exact numbers from your report, or move sliders to those same recorded values.",
    )
    st.caption("These are measurements from medical tests, not everyday estimates. Slider starting points are only controls, not normal values or medical advice.")

    demo_column, clear_column = st.columns([1, 1])
    with demo_column:
        st.button(
            "Load demonstration profile",
            on_click=_load_demo_profile,
            help="Fills the form with illustrative values only. This is not a real patient profile.",
        )
    with clear_column:
        st.button("Clear form", on_click=_reset_form)

    if st.session_state.get(DEMO_KEY):
        st.info("Illustrative demonstration profile loaded. These values do not represent a real person.")

    with st.expander(
        "Enter your information · 13 fields",
        expanded=st.session_state.get(REVIEW_KEY) is None,
    ):
        submitted, user_values = render_input_form(model_name, input_mode)
    st.caption(
        "Category wording follows the "
        "[UCI Heart Disease dataset dictionary](https://archive.ics.uci.edu/dataset/45/heart+disease). "
        "These labels describe dataset categories, not medical diagnoses."
    )
    if submitted and user_values is not None:
        selected_model = user_values.pop("_model_name")
        measurements_confirmed = user_values.pop("_measurements_confirmed")
        if not measurements_confirmed:
            st.session_state.pop(REVIEW_KEY, None)
            st.session_state.pop(RESULT_KEY, None)
            st.error("Please confirm that the measurements match your report before reviewing them.")
            return
        try:
            encode_user_input(user_values)
        except (TypeError, ValueError) as exc:
            st.session_state.pop(REVIEW_KEY, None)
            st.session_state.pop(RESULT_KEY, None)
            st.error(str(exc))
        else:
            st.session_state[REVIEW_KEY] = {
                "user_values": user_values,
                "model_name": selected_model,
            }
            st.session_state.pop(RESULT_KEY, None)
            st.caption("Review your selections below. The model has not run yet.")

    reviewed = st.session_state.get(REVIEW_KEY)
    if not reviewed:
        st.caption("Your input summary will appear here after you select ‘Review my information’.")
        return

    user_values = reviewed["user_values"]
    model_name = reviewed["model_name"]
    _render_input_review(user_values, model_name)

    result = st.session_state.get(RESULT_KEY)
    button_label = "Predict using reviewed information" if result is None else "Run prediction again"
    if st.button(
        button_label,
        type="primary",
        use_container_width=True,
        help="Uses the readable summary shown above. To use changed form entries, submit ‘Review my information’ first.",
    ):
        try:
            with st.spinner("Running the saved pipeline and checking its model explanation…"):
                input_frame = encode_user_input(user_values)
                model = load_model(model_name)
                _validate_model_category_mappings(model)
                predicted_class, probabilities = predict_one(model, input_frame)
                result = {
                    "predicted_class": predicted_class,
                    "probabilities": probabilities,
                    "model_name": model_name,
                    "user_values": user_values,
                }
                try:
                    result["contributions"] = explain_prediction(model, input_frame)
                except Exception as explanation_error:
                    result["contributions"] = None
                    result["explanation_error"] = f"{type(explanation_error).__name__}: {explanation_error}"
                st.session_state[RESULT_KEY] = result
        except FileNotFoundError as exc:
            st.error(f"The selected model file is missing. No alternate model was used. {exc}")
            result = None
        except (TypeError, ValueError) as exc:
            st.error(f"Input or model validation failed: {exc}")
            result = None
        except Exception as exc:
            st.error(f"The selected model could not be loaded or did not return a valid prediction. No alternate model was used.")
            with st.expander("Technical details"):
                st.exception(exc)
            result = None

    if result is not None:
        _render_result(result)
        st.button("Edit inputs for another prediction", on_click=_clear_review_for_model_change)
