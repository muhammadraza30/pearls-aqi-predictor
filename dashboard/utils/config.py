"""
Dashboard configuration — page config, custom CSS, and AQI thresholds.
Clean Light Theme (Off-white background, High Contrast Text) for maximum readability.
"""
import streamlit as st


PAGE_CONFIG = {
    "page_title": "Karachi AQI Predictor",
    "page_icon": "🌍",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# AQI thresholds: (max_val, label, color, emoji)
AQI_THRESHOLDS = [
    (50, "Good", "#00e400", "🟢"),
    (100, "Moderate", "#ffff00", "🟡"),
    (150, "Unhealthy for Sensitive Groups", "#ff7e00", "🟠"),
    (200, "Unhealthy", "#ff0000", "🔴"),
    (300, "Very Unhealthy", "#8f3f97", "🟣"),
    (500, "Hazardous", "#7e0023", "⚫"),
]


CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* ---- Global Light Theme ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        /* color: #31333f; */ /* Let Streamlit default handle text color mostly, but enforce primary */
    }
    .stApp {
        background-color: #f8f9fa; /* Off-white background */
    }

    /* ---- Header ---- */
    /* Leaving header visible as requested */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 2rem;
    }

    /* ---- Cards (Light Mode) ---- */
    .glass-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 24px;
    }

    /* ---- Section headers ---- */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #111827;
        margin-top: 10px;
        margin-bottom: 16px;
        padding-left: 10px;
        border-left: 4px solid #3b82f6;
    }

    /* ---- Sidebar styling ---- */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    .sidebar-deploy-btn {
        display: block;
        width: 100%;
        text-align: center;
        background-color: #2563eb;
        color: white !important;
        padding: 10px;
        border-radius: 6px;
        text-decoration: none;
        font-weight: 600;
        margin-top: 10px;
        transition: background-color 0.2s;
    }
    .sidebar-deploy-btn:hover {
        background-color: #1d4ed8;
    }

    /* ---- Metrics ---- */
    [data-testid="metric-container"] {
        background: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #e5e7eb;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #6b7280;
        font-weight: 600;
        padding: 8px 16px;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 2px solid #2563eb !important;
        background: rgba(37, 99, 235, 0.05);
    }
    
    /* ---- DataFrame ---- */
    [data-testid="stDataFrame"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
    }
</style>
"""


def apply_page_config():
    """Apply Streamlit page configuration."""
    st.set_page_config(**PAGE_CONFIG)


def apply_custom_css():
    """Apply the light theme CSS."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
