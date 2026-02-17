"""
Historical data charts and EDA visualizations using Plotly.
Dark-themed, interactive charts for AQI trends and pollutant analysis.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


DARK_TEMPLATE = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(15,15,35,0.5)",
    "font": {"family": "Inter, sans-serif", "color": "#e0e0e0"},
}


def create_aqi_trend_chart(df: pd.DataFrame, days: int = 30) -> go.Figure:
    """Create an interactive AQI trend line chart."""
    if df is None or df.empty or "aqi" not in df.columns:
        return _empty_chart("No AQI data available")

    df = df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        if days:
            cutoff = df["datetime"].max() - pd.Timedelta(days=days)
            df = df[df["datetime"] >= cutoff]
        x_col = "datetime"
    else:
        x_col = df.index

    fig = go.Figure()

    # AQI area fill
    fig.add_trace(go.Scatter(
        x=df[x_col], y=df["aqi"],
        mode="lines",
        name="AQI",
        line=dict(color="#00d4ff", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.1)",
    ))

    # AQI threshold lines
    thresholds = [(50, "#00e400", "Good"), (100, "#ffff00", "Moderate"),
                  (150, "#ff7e00", "USG"), (200, "#ff0000", "Unhealthy")]
    for val, color, label in thresholds:
        fig.add_hline(y=val, line_dash="dot", line_color=color,
                      opacity=0.3, annotation_text=label,
                      annotation_font_color=color, annotation_font_size=10)

    fig.update_layout(
        title="AQI Trend",
        xaxis_title="Date",
        yaxis_title="AQI (US EPA)",
        height=380,
        **DARK_TEMPLATE,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


def create_pollutant_breakdown(df: pd.DataFrame) -> go.Figure:
    """Create a stacked bar chart of pollutant concentrations."""
    pollutants = ["pm2_5", "pm10", "no2", "so2", "o3", "co"]
    available = [p for p in pollutants if p in df.columns]

    if not available:
        return _empty_chart("No pollutant data available")

    # Get last 7 days daily average
    df = df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").resample("D").mean().tail(7).reset_index()

    colors = ["#ff6384", "#36a2eb", "#ffce56", "#4bc0c0", "#9966ff", "#ff9f40"]

    fig = go.Figure()
    for i, pol in enumerate(available):
        fig.add_trace(go.Bar(
            x=df["datetime"] if "datetime" in df.columns else df.index,
            y=df[pol],
            name=pol.upper(),
            marker_color=colors[i % len(colors)],
            opacity=0.85,
        ))

    fig.update_layout(
        title="Pollutant Concentrations (7-Day)",
        barmode="group",
        height=350,
        **DARK_TEMPLATE,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create a feature correlation heatmap."""
    cols = ["temp", "humidity", "pressure", "wind_speed", "clouds",
            "pm2_5", "pm10", "no2", "so2", "o3", "aqi"]
    available = [c for c in cols if c in df.columns]

    if len(available) < 3:
        return _empty_chart("Not enough features for correlation")

    corr = df[available].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))

    fig.update_layout(
        title="Feature Correlations",
        height=450,
        **DARK_TEMPLATE,
    )
    return fig


def create_weather_vs_aqi(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of temperature vs AQI with humidity as color."""
    if "temp" not in df.columns or "aqi" not in df.columns:
        return _empty_chart("Need temp and AQI data")

    fig = px.scatter(
        df, x="temp", y="aqi",
        color="humidity" if "humidity" in df.columns else None,
        color_continuous_scale="Viridis",
        opacity=0.6,
        title="Temperature vs AQI",
    )

    fig.update_layout(
        height=380,
        **DARK_TEMPLATE,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Temperature (°C)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="AQI"),
    )
    return fig


def _empty_chart(message: str) -> go.Figure:
    """Create an empty chart with a message."""
    fig = go.Figure()
    fig.update_layout(
        height=300,
        **DARK_TEMPLATE,
        annotations=[{
            "text": message,
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5,
            "font": {"size": 16, "color": "#888"},
            "showarrow": False,
        }],
    )
    return fig
