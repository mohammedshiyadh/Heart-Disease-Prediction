from __future__ import annotations

import streamlit as st

DEFAULT_COLORS = {
    "hs_accent": "#087F82",
    "hs_background": "#F5F8F8",
    "hs_card": "#FFFFFF",
}


def _luminance(color: str) -> float:
    """Return WCAG relative luminance for a color-picker hex value."""
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _foreground(background: str) -> str:
    return "#142B3A" if _luminance(background) > 0.42 else "#F5FBFB"


def theme_palette() -> dict[str, str]:
    """Read the shared appearance controls and choose readable foregrounds."""
    accent = st.session_state.get("hs_accent", DEFAULT_COLORS["hs_accent"])
    background = st.session_state.get("hs_background", DEFAULT_COLORS["hs_background"])
    card = st.session_state.get("hs_card", DEFAULT_COLORS["hs_card"])
    return {
        "accent": accent,
        "background": background,
        "card": card,
        "text": _foreground(background),
        "card_text": _foreground(card),
        "accent_text": _foreground(accent),
    }


def reset_colors() -> None:
    for key, value in DEFAULT_COLORS.items():
        st.session_state[key] = value


def apply_user_theme() -> None:
    palette = theme_palette()
    scheme = "dark" if _foreground(palette["background"]) == "#F5FBFB" else "light"
    st.markdown(
        f"""
        <style>
        :root {{
            --hs-accent: {palette['accent']};
            --hs-background: {palette['background']};
            --hs-card: {palette['card']};
            --hs-text: {palette['text']};
            --hs-card-text: {palette['card_text']};
            --hs-accent-text: {palette['accent_text']};
            color-scheme: {scheme};
        }}
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
            background-color: var(--hs-background) !important;
            color: var(--hs-text) !important;
        }}
        [data-testid="stMain"] p, [data-testid="stMain"] label,
        [data-testid="stMain"] h1, [data-testid="stMain"] h2,
        [data-testid="stMain"] h3, [data-testid="stMain"] h4,
        [data-testid="stMain"] li, [data-testid="stMain"] [data-testid="stMarkdownContainer"] {{
            color: var(--hs-text) !important;
        }}
        div[data-testid="stMetric"], [data-testid="stDataFrame"] {{
            background: var(--hs-card) !important;
            color: var(--hs-card-text) !important;
            border-color: color-mix(in srgb, var(--hs-card-text) 18%, transparent) !important;
        }}
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {{ color: var(--hs-card-text) !important; }}
        [data-testid="stMain"] input, [data-testid="stMain"] textarea,
        [data-testid="stMain"] [data-baseweb="select"] > div {{
            background-color: var(--hs-card) !important;
            color: var(--hs-card-text) !important;
            border-color: color-mix(in srgb, var(--hs-card-text) 30%, transparent) !important;
        }}
        [role="option"], [role="listbox"] {{
            background-color: var(--hs-card) !important;
            color: var(--hs-card-text) !important;
        }}
        .stButton > button, .stFormSubmitButton > button,
        [data-testid="stDownloadButton"] > button,
        [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {{
            background-color: var(--hs-accent) !important;
            color: var(--hs-accent-text) !important;
            border: 0 !important;
            border-radius: 10px;
            font-weight: 700;
        }}
        .stButton > button *, .stFormSubmitButton > button *,
        [data-testid="stDownloadButton"] > button * {{ color: var(--hs-accent-text) !important; }}
        .stButton > button:hover, .stFormSubmitButton > button:hover,
        [data-testid="stDownloadButton"] > button:hover,
        [data-testid="stBaseButton-primary"]:hover,
        [data-testid="stBaseButton-secondary"]:hover {{ filter: brightness(.93); }}
        [data-testid="stSlider"] [role="slider"] {{ border-color: var(--hs-accent) !important; }}
        .section-card {{
            background: var(--hs-card);
            color: var(--hs-card-text);
            border: 1px solid color-mix(in srgb, var(--hs-card-text) 18%, transparent);
            border-radius: 14px;
            padding: 1.2rem;
            min-height: 140px;
        }}
        .section-card h3, .section-card p {{ color: var(--hs-card-text) !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_theme_controls() -> None:
    with st.sidebar.expander("Customize appearance", expanded=False):
        st.color_picker("Accent color", value=DEFAULT_COLORS["hs_accent"], key="hs_accent")
        st.color_picker("Page background", value=DEFAULT_COLORS["hs_background"], key="hs_background")
        st.color_picker("Card background", value=DEFAULT_COLORS["hs_card"], key="hs_card")
        st.button("Reset colors", on_click=reset_colors, use_container_width=True)
    apply_user_theme()
