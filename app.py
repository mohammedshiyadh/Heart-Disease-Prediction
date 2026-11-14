from __future__ import annotations

from importlib import import_module
from typing import Final

import streamlit as st


# =========================================================
# APPLICATION CONFIGURATION
# =========================================================

APP_NAME: Final[str] = "HeartSense AI"
APP_TAGLINE: Final[str] = "Machine Learning · Research Explorer"

PAGES: Final[dict[str, tuple[str, str]]] = {
    "Overview": ("home", "home"),
    "Prediction": ("prediction", "activity"),
    "Model Comparison": ("model_comparison", "bar_chart"),
    "Explainability": ("explainability", "insights"),
    "About": ("about", "info"),
}

st.set_page_config(
    page_title=APP_NAME,
    page_icon="♥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# THEME AND STYLING
# =========================================================

def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --hs-ink: #132b3a;
            --hs-muted: #627783;
            --hs-teal: #087f82;
            --hs-mint: #e5f4f1;
            --hs-paper: #f5f8f8;
            --hs-border: #e5ecec;
        }

        html, body, [class*="css"] {
            font-family: "DM Sans", "Segoe UI", Arial, sans-serif;
            color: var(--hs-ink);
        }

        .stApp {
            background: var(--hs-paper);
        }

        [data-testid="stHeader"] {
            background: rgba(245, 248, 248, 0.94);
        }

        [data-testid="stSidebar"] {
            background: #102a3a;
        }

        [data-testid="stSidebar"] * {
            color: #edf6f5;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            padding: 0.35rem 0.15rem;
            border-radius: 8px;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(255, 255, 255, 0.08);
        }

        h1, h2, h3 {
            color: var(--hs-ink);
            letter-spacing: -0.035em;
            font-family: "Manrope", "Segoe UI", Arial, sans-serif;
        }

        .hs-brand {
            font-family: "Manrope", "Segoe UI", Arial, sans-serif;
            font-weight: 800;
            font-size: 1.42rem;
            letter-spacing: -0.045em;
            color: #f4ffff;
            margin-bottom: 0.15rem;
        }

        .hs-brand-accent {
            color: #59d2c1;
        }

        .hs-tagline {
            color: #b9cece;
            font-size: 0.82rem;
            margin-top: 0;
        }

        .hs-page-kicker {
            color: var(--hs-teal);
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .hs-hero {
            background: linear-gradient(
                120deg,
                #112d40 0%,
                #174958 72%,
                #087f82 135%
            );
            border-radius: 22px;
            padding: 1.55rem 1.8rem;
            color: #f6ffff;
            margin: 0.15rem 0 1.15rem 0;
        }

        .hs-hero-title {
            color: #ffffff !important;
            font-size: 1.65rem;
            font-weight: 800;
            margin: 0 0 0.3rem 0;
        }

        .hs-hero-copy {
            color: #d4e7e7;
            margin: 0;
            font-size: 0.96rem;
        }

        .hs-disclaimer {
            border-left: 4px solid #d29b40;
            background: #fff8e9;
            padding: 0.85rem 1rem;
            border-radius: 8px;
            color: #5e4a24;
            font-size: 0.88rem;
            line-height: 1.5;
            margin-top: 1.1rem;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            padding: 0.95rem 1.05rem;
            border: 1px solid var(--hs-border);
            border-radius: 14px;
        }

        .stButton > button,
        .stFormSubmitButton > button {
            border-radius: 10px;
            font-weight: 700;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.16);
        }

        @media (max-width: 760px) {
            .hs-hero {
                padding: 1.2rem;
                border-radius: 16px;
            }

            .hs-hero-title {
                font-size: 1.35rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

def render_sidebar() -> str:
    with st.sidebar:

        st.markdown(
            '<div class="hs-brand">♥ '
            '<span class="hs-brand-accent">HeartSense</span> AI</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<p class="hs-tagline">{APP_TAGLINE}</p>',
            unsafe_allow_html=True,
        )

        st.divider()

        selected_page = st.radio(
            "Navigate",
            options=list(PAGES.keys()),
            index=0,
            label_visibility="collapsed",
            key="heartsense_active_page",
        )

        st.divider()

        st.caption("PRIMARY DEMONSTRATION MODEL")
        st.markdown("**LightGBM · Playground dataset**")
        st.caption("Educational research prototype")

    return selected_page


# =========================================================
# DISCLAIMER
# =========================================================

def render_disclaimer() -> None:
    st.markdown(
        """
        <div class="hs-disclaimer">
            <strong>Research demonstration only.</strong>
            HeartSense AI has not been clinically validated.
            Its predictions must not be used for medical diagnosis,
            treatment decisions, or emergency assessment.
            If you have health concerns, consult a qualified
            healthcare professional.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# PAGE LOADING
# =========================================================

def render_page(page_name: str) -> None:

    module_name, _icon = PAGES[page_name]

    try:
        page_module = import_module(f"views.{module_name}")

    except ModuleNotFoundError as exc:
        st.error(
            f"The {page_name} page could not be loaded. "
            f"Check that views/{module_name}.py exists "
            "and that its dependencies are installed."
        )

        with st.expander("Technical details"):
            st.exception(exc)

        return

    render_function = getattr(page_module, "render", None)

    if not callable(render_function):
        st.error(
            f"views/{module_name}.py must define a callable render() function."
        )
        return

    try:
        render_function()

    except Exception as exc:
        st.error(
            f"Something went wrong while rendering the {page_name} page."
        )

        st.caption("Error details:")

        with st.expander("Technical details", expanded=True):
            st.exception(exc)


# =========================================================
# MAIN APPLICATION
# =========================================================

def main() -> None:

    apply_theme()

    selected_page = render_sidebar()

    render_page(selected_page)

    render_disclaimer()


if __name__ == "__main__":
    main()