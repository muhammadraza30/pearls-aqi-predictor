"""
AQI Accelerometer / Gauge component using Plotly.
Creates a stunning circular gauge with color-coded AQI zones.
Light theme variant for the dashboard.
"""
import plotly.graph_objects as go


# AQI color zones (US EPA standard)
AQI_ZONES = [
    (0, 51, "#00e400", "Good"),
    (51, 101, "#ffff00", "Moderate"),
    (101, 151, "#ff7e00", "Unhealthy for Sensitive Groups"),
    (151, 201, "#ff0000", "Unhealthy"),
    (201, 301, "#8f3f97", "Very Unhealthy"),
    (301, 501, "#7e0023", "Hazardous"),
]

def create_aqi_gauge(aqi_value: float, title: str = "Current AQI", height: int = 350) -> go.Figure:
    """
    Create a premium accelerometer-style AQI gauge (light theme).

    Args:
        aqi_value: The AQI value to display (0-500)
        title: Title shown below the gauge
        height: Height of the figure in pixels

    Returns:
        Plotly Figure object
    """
    aqi_value = max(0, min(500, aqi_value))

    # Build color steps for the gauge bar
    steps = []
    for low, high, color, label in AQI_ZONES:
        steps.append({
            "range": [low, high],
            "color": color,
            "name": label,
        })

    # Determine category text and color for needle
    category = "Good"
    needle_color = "#00e400"
    for low, high, color, label in AQI_ZONES:
        if low <= aqi_value < high or (high == 500 and aqi_value >= 300):
            category = label
            needle_color = color
            break

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=aqi_value,
        number={
            "font": {"size": 56, "color": "#1f2937", "family": "Inter, sans-serif"},
            "suffix": "",
        },
        title={
            "text": f"<b>{title}</b><br><span style='font-size:16px;color:{needle_color}'>{category}</span>",
            "font": {"size": 20, "color": "#374151", "family": "Inter, sans-serif"},
        },
        gauge={
            "axis": {
                "range": [0, 500],
                "tickwidth": 2,
                "tickcolor": "#9ca3af",
                "tickfont": {"size": 11, "color": "#6b7280"},
                "dtick": 50,
            },
            "bar": {
                "color": needle_color,
                "thickness": 0.3,
            },
            "bgcolor": "#f3f4f6",
            "borderwidth": 2,
            "bordercolor": "#e5e7eb",
            "steps": steps,
            "threshold": {
                "line": {"color": "#1f2937", "width": 4},
                "thickness": 0.85,
                "value": aqi_value,
            },
        },
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=30, r=30, t=60, b=30),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Inter, sans-serif"},
    )

    return fig


def create_mini_gauge(aqi_value: float, day_label: str, height: int = 220) -> go.Figure:
    """
    Create a smaller gauge for multi-day forecast display (light theme).
    """
    aqi_value = max(0, min(500, aqi_value))

    category = "Good"
    color = "#00e400"
    for low, high, c, label in AQI_ZONES:
        if low <= aqi_value < high or (high == 500 and aqi_value >= 300):
            category = label
            color = c
            break

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        number={
            "font": {"size": 36, "color": "#1f2937", "family": "Inter"},
        },
        title={
            "text": f"<b>{day_label}</b>",
            "font": {"size": 14, "color": "#6b7280", "family": "Inter"},
        },
        gauge={
            "axis": {"range": [0, 500], "tickfont": {"size": 9, "color": "#9ca3af"}, "dtick": 100},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "#f3f4f6",
            "borderwidth": 1,
            "bordercolor": "#e5e7eb",
            "steps": [
                {"range": [0, 50], "color": "rgba(0,228,0,0.15)"},
                {"range": [50, 100], "color": "rgba(255,255,0,0.12)"},
                {"range": [100, 150], "color": "rgba(255,126,0,0.12)"},
                {"range": [150, 200], "color": "rgba(255,0,0,0.12)"},
                {"range": [200, 300], "color": "rgba(143,63,151,0.12)"},
                {"range": [300, 500], "color": "rgba(126,0,35,0.12)"},
            ],
        },
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=45, b=15),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )

    return fig
