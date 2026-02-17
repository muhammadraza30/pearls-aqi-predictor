"""
Premium forecast cards for 4-day AQI display (Light Theme).
Renders color-coded cards with category info.
"""
import streamlit as st
from datetime import datetime, timedelta


def get_day_label(date_str: str, index: int) -> str:
    """Return human-readable day label like 'Today', 'Tomorrow', etc."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        today = datetime.now().date()
        delta = (d - today).days
        if delta == 0:
            return "Today"
        elif delta == 1:
            return "Tomorrow"
        else:
            return d.strftime("%A")  # Day name like 'Thursday'
    except Exception:
        labels = ["Today", "Tomorrow", "Day 3", "Day 4"]
        return labels[min(index, len(labels) - 1)]


def render_forecast_cards(predictions: list):
    """
    Render premium forecast cards (Light Theme).

    Args:
        predictions: List of dicts with keys: date, aqi_pred, category, color, message, emoji
    """
    if not predictions:
        st.warning("No predictions available.")
        return

    cols = st.columns(len(predictions))

    for i, (col, pred) in enumerate(zip(cols, predictions)):
        with col:
            aqi = pred.get("aqi_pred", 0)
            color = pred.get("color", "#666")
            category = pred.get("category", "Unknown")
            message = pred.get("message", "")
            emoji = pred.get("emoji", "🔵")
            date_str = pred.get("date", "")
            is_hazardous = pred.get("is_hazardous", False)
            day_label = get_day_label(date_str, i)

            # Determine card border/shadow
            shadow = f"0 4px 12px {color}30" if is_hazardous else "0 2px 8px rgba(0,0,0,0.08)"
            border = f"2px solid {color}" if is_hazardous else "1px solid #e5e7eb"

            st.markdown(f"""
            <div style="
                background: #ffffff;
                border: {border};
                border-radius: 16px;
                padding: 20px 16px;
                text-align: center;
                box-shadow: {shadow};
                transition: transform 0.2s ease;
                min-height: 220px;
                color: #1f2937;
            ">
                <div style="font-size: 0.85rem; color: #6b7280; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">
                    {day_label}
                </div>
                <div style="font-size: 0.75rem; color: #9ca3af; margin-bottom: 12px;">
                    {date_str}
                </div>
                <div style="font-size: 2.5rem; font-weight: 800; color: {color};
                            margin-bottom: 4px;">
                    {emoji} {aqi:.0f}
                </div>
                <div style="font-size: 0.95rem; font-weight: 700; color: {color};
                            margin-bottom: 8px;">
                    {category}
                </div>
                <div style="font-size: 0.75rem; color: #4b5563; line-height: 1.4;">
                    {message[:80]}{'...' if len(message) > 80 else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_alert_banner(predictions: list):
    """Show hazardous AQI alert banner if any day is unhealthy (Light Theme)."""
    hazardous = [p for p in predictions if p.get("is_hazardous")]
    if hazardous:
        worst = max(hazardous, key=lambda p: p.get("aqi_pred", 0))
        st.markdown(f"""
        <div style="
            background-color: #fef2f2;
            border: 1px solid #fca5a5;
            border-left: 5px solid #ef4444;
            border-radius: 8px;
            padding: 16px;
            margin: 10px 0 20px 0;
            display: flex;
            align-items: flex-start;
            gap: 16px;
        ">
            <span style="font-size: 1.5rem;">⚠️</span>
            <div>
                <div style="color: #991b1b; font-weight: 700; font-size: 1rem; margin-bottom: 4px;">
                    Health Advisory
                </div>
                <div style="color: #7f1d1d; font-size: 0.9rem;">
                    {worst['category']} conditions expected on {worst.get('date', 'upcoming day')}
                    (AQI: {worst['aqi_pred']:.0f}). {worst.get('message', '')}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
