"""
Dashboard configuration — page config, custom CSS, and AQI thresholds.
Premium Light Theme with glassmorphism, micro-animations, and gradient accents.
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
    }
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e8edf5 50%, #f5f0ff 100%);
    }

    /* ---- Animated gradient header bar ---- */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(135deg, #2563eb, #7c3aed, #2563eb);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s ease-in-out infinite;
        margin-bottom: 0;
    }
    @keyframes shimmer {
        0%, 100% { background-position: 0% center; }
        50% { background-position: 100% center; }
    }

    .hero-subtitle {
        color: #6b7280;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }

    /* ---- Glassmorphism Cards ---- */
    .glass-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(229, 231, 235, 0.6);
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(37, 99, 235, 0.1);
    }

    /* ---- Stat Cards ---- */
    .stat-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(8px);
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.08);
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ---- Section headers ---- */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #111827;
        margin-top: 10px;
        margin-bottom: 16px;
        padding-left: 12px;
        border-left: 4px solid #2563eb;
    }

    /* ---- Gradient Sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: none;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaption"] p {
        color: #94a3b8 !important;
    }

    /* Sidebar brand badge */
    .brand-badge {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin: 12px 0;
    }
    .brand-badge .brand-name {
        color: #ffffff !important;
        font-size: 0.95rem;
        font-weight: 700;
        margin: 0;
    }
    .brand-badge .brand-sub {
        color: rgba(255,255,255,0.75) !important;
        font-size: 0.72rem;
        font-weight: 500;
        margin: 0;
    }

    /* Feature pill badges */
    .feature-pill {
        display: inline-block;
        background: rgba(37, 99, 235, 0.15);
        color: #93c5fd !important;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.72rem;
        font-weight: 600;
        margin: 3px 2px;
    }

    /* ---- Metrics ---- */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(8px);
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
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
        padding: 10px 20px;
        border: none;
        border-radius: 8px 8px 0 0;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(37, 99, 235, 0.04);
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 3px solid #2563eb !important;
        background: rgba(37, 99, 235, 0.06);
    }

    /* ---- DataFrame ---- */
    [data-testid="stDataFrame"] {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---- Smooth fade-in animation ---- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stTabs [data-baseweb="tab-panel"] {
        animation: fadeInUp 0.4s ease-out;
    }

    /* ---- Map container ---- */
    [data-testid="stDeckGlJsonChart"], iframe[title="streamlit_map"] {
        border-radius: 16px !important;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }

    /* ---- Hide Streamlit Elements ---- */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stDecoration"] {visibility: hidden;}
    
    /* ---- Footer ---- */
    .app-footer {
        text-align: center;
        padding: 24px 0 12px 0;
        color: #9ca3af;
        font-size: 0.78rem;
        border-top: 1px solid #e5e7eb;
        margin-top: 40px;
    }
    .app-footer a {
        color: #2563eb;
        text-decoration: none;
        font-weight: 600;
    }
</style>
"""


def apply_page_config():
    """Apply Streamlit page configuration."""
    st.set_page_config(**PAGE_CONFIG)


def apply_custom_css():
    """Apply the premium light theme CSS."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
