from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go

ABSENCE_COLOR = "#8795A1"


def probability_figure(
    probabilities: dict[int, float],
    *,
    accent: str = "#087F82",
    card: str = "#FFFFFF",
    text_color: str = "#132B3A",
) -> go.Figure:
    labels = ["Absence label", "Presence label"]
    values = [probabilities[0], probabilities[1]]
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=[ABSENCE_COLOR, accent],
        text=[f"{value:.1%}" for value in values],
        textposition="outside",
        hovertemplate="%{y}<br>Model probability: %{x:.2%}<extra></extra>",
    ))
    fig.update_layout(
        height=230,
        margin=dict(l=5, r=38, t=10, b=5),
        xaxis=dict(range=[0, 1], tickformat=".0%", title=None, gridcolor="rgba(128, 145, 153, .2)"),
        yaxis_title=None,
        showlegend=False,
        font=dict(color=text_color),
        plot_bgcolor=card,
        paper_bgcolor=card,
    )
    return fig


def comparison_figure(data, experiment: str) -> go.Figure:
    subset = data[data["Experiment"] == experiment]
    long_data = subset.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"], var_name="Metric", value_name="Score")
    fig = px.bar(long_data, x="Metric", y="Score", color="Model", barmode="group", text=long_data["Score"].map(lambda value: f"{value:.1%}"), color_discrete_sequence=["#087f82", "#39728e", "#d19b4a", "#64748b", "#9b6b91"])
    fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate="%{fullData.name}<br>%{x}: %{y:.2%}<extra></extra>")
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=25, b=10), yaxis=dict(range=[0, 1.08], tickformat=".0%", title=None), xaxis_title=None, legend_title=None, plot_bgcolor="white", paper_bgcolor="white")
    return fig


def shap_figure(importance) -> go.Figure:
    ordered = importance.sort_values("SHAP Importance", ascending=True)
    fig = px.bar(ordered, x="SHAP Importance", y="Feature", orientation="h", color="SHAP Importance", color_continuous_scale=["#b9ded7", "#087f82"], text=ordered["SHAP Importance"].map(lambda value: f"{value:.3f}"))
    fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate="%{y}<br>Grouped mean absolute SHAP: %{x:.5f}<extra></extra>")
    fig.update_layout(height=490, margin=dict(l=10, r=45, t=10, b=10), coloraxis_showscale=False, xaxis_title="Grouped mean absolute SHAP value", yaxis_title=None, plot_bgcolor="white", paper_bgcolor="white")
    return fig


def contribution_figure(
    contributions,
    *,
    accent: str = "#087F82",
    card: str = "#FFFFFF",
    text_color: str = "#132B3A",
) -> go.Figure:
    """Show signed local SHAP contributions toward/away from class 1."""
    ordered = contributions.sort_values("Contribution", ascending=True)
    positive = ordered["Contribution"].clip(lower=0)
    negative = ordered["Contribution"].clip(upper=0)
    limit = max(float(ordered["Contribution"].abs().max()), 0.01) * 1.2
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=negative,
        y=ordered["Feature"],
        name="Toward Absence",
        orientation="h",
        marker_color=ABSENCE_COLOR,
        text=[f"{value:.3f}" if value < 0 else "" for value in negative],
        textposition="outside",
        hovertemplate="%{y}<br>Contribution toward Absence: %{x:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=positive,
        y=ordered["Feature"],
        name="Toward Presence",
        orientation="h",
        marker_color=accent,
        text=[f"+{value:.3f}" if value > 0 else "" for value in positive],
        textposition="outside",
        hovertemplate="%{y}<br>Contribution toward Presence: +%{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        barmode="relative",
        height=max(300, 36 * len(ordered)),
        margin=dict(l=10, r=35, t=10, b=10),
        xaxis=dict(range=[-limit, limit], zeroline=True, zerolinecolor="rgba(128,145,153,.55)", title="Change in the model's score"),
        yaxis_title=None,
        legend_title=None,
        font=dict(color=text_color),
        plot_bgcolor=card,
        paper_bgcolor=card,
    )
    return fig
