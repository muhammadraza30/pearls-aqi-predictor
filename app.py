"""
Pearls AQI Predictor — Karachi
Light Theme Dashboard with Map and Health Advice.
"""
import sys
import pandas as pd
from pathlib import Path

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import warnings
warnings.filterwarnings("ignore")

from dashboard.utils.config import apply_page_config, apply_custom_css
from dashboard.utils.data_loader import (
    load_predictions, 
    load_historical_data, 
    get_model_version,
    load_comparison_data,
    load_model_metrics
)
from dashboard.components.gauge import create_aqi_gauge, create_mini_gauge
from dashboard.components.forecast_cards import render_forecast_cards, render_alert_banner
from dashboard.components.charts import (
    create_aqi_trend_chart,
    create_pollutant_breakdown,
    create_correlation_heatmap,
    create_weather_vs_aqi,
)


def _render_stat_cards(df):
    """Render overview stat cards with styled backgrounds."""
    if df is None or df.empty:
        return

    # Compute stats
    total_records = f"{len(df):,}"
    days_of_data = "0"
    avg_aqi = "0"
    start_str, end_str = "", ""

    if "datetime" in df.columns:
        dates = pd.to_datetime(df["datetime"])
        days_of_data = str((dates.max() - dates.min()).days)
        start_str = dates.min().strftime("%b %Y")
        end_str = dates.max().strftime("%b %Y")
    if "aqi" in df.columns:
        avg_aqi = f"{df['aqi'].mean():.0f}"

    # Render styled HTML cards
    st.markdown(f"""
    <div style="display: flex; gap: 16px; margin-bottom: 8px;">
        <div class="stat-card" style="flex: 1;">
            <div class="stat-value">{total_records}</div>
            <div class="stat-label">HOURLY RECORDS</div>
        </div>
        <div class="stat-card" style="flex: 1;">
            <div class="stat-value">{days_of_data}</div>
            <div class="stat-label">DAYS OF DATA</div>
        </div>
        <div class="stat-card" style="flex: 1;">
            <div class="stat-value">{avg_aqi}</div>
            <div class="stat-label">AVG. AQI</div>
        </div>
        <div class="stat-card" style="flex: 1;">
            <div class="stat-value">3</div>
            <div class="stat-label">ML MODELS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if start_str and end_str:
        st.caption(f"📅 Historical data range: {start_str} — {end_str} · Source: Open-Meteo API (hourly)")


def main():
    # ── Load data ──
    predictions = load_predictions()
    df_hist = load_historical_data()
    comp_df = load_comparison_data()
    metrics_data = load_model_metrics()
    today_pred = predictions[0] if predictions else {}
    today_aqi = today_pred.get("aqi_pred", 0)
    best_model_name = today_pred.get("model_used", "Best Model")

    apply_page_config()
    apply_custom_css()
    
    # ── Sidebar ──
    with st.sidebar:
        st.title("Settings")
        st.markdown("### 🌆 Karachi, Pakistan")
        st.caption("Lat: 24.8608 | Lon: 67.0104")
        st.markdown("---")

        # Model version
        version_info = get_model_version()
        v = version_info.get("version", 0)
        trained_at = version_info.get("trained_at", "N/A")
        st.markdown(f"## Best Model: {best_model_name}")
        if v > 0:
            st.markdown(f"**Model v{v}**")
            st.caption(f"Trained: {trained_at if trained_at != 'N/A' else 'N/A'}")
        else:
            st.info("No models trained yet")

        # Metrics
        metrics_data = load_model_metrics()
        if metrics_data and "best_model" in metrics_data:
            best_m = metrics_data["best_model"]
            if best_m in metrics_data.get("models", {}):
                rmse = metrics_data["models"][best_m]["RMSE"]
                st.metric("Model RMSE", f"{rmse:.2f}", help="Lower is better")

        st.markdown("---")
        
        # Credit
        st.markdown("""
        <div style="text-align: center; padding: 8px 0;">
            <div style="font-size: 0.78rem; color: #94a3b8 !important; font-weight: 600;">
                Made by
            </div>
            <div style="font-size: 1rem; font-weight: 700; color: #f1f5f9 !important; margin: 4px 0;">
                Muhammad Raza
            </div>
            <div style="font-size: 0.7rem; color: #64748b !important; font-weight: 500;">
                10Pearls Shine Cohort 7 Internship
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Main Content ──
    st.markdown('<div class="hero-title">🌍 Karachi AQI Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Real-time Air Quality Forecasting — Powered by Machine Learning & Hopsworks Feature Store</div>', unsafe_allow_html=True)

   
    # ── Data Stats Overview ──
    _render_stat_cards(df_hist)

    # ── Tabs ──
    tab1, tab2, tab3 = st.tabs([
        "📊 Dashboard",
        "🔬 Trends & Analysis",
        "🧠 Model Insights",
    ])

    # ================================================================
    #  TAB 1: Dashboard Overview
    # ================================================================
    with tab1:
        if predictions:
            # Alert banner
            render_alert_banner(predictions)

            # Today's AQI — large gauge
            
            col_gauge, col_info = st.columns([2, 1])
            with col_gauge:
                fig_gauge = create_aqi_gauge(today_aqi, title="Current Air Quality Index", height=380)
                st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

            with col_info:
                # Health Advice Card
                category = today_pred.get('category', 'Unknown')
                advice = "Enjoy the fresh air! Great for outdoor activities. 🌿"
                if "Moderate" in category:
                    advice = "Acceptable for most. Sensitive groups should limit prolonged outdoor exertion. 😷"
                elif "Sensitive" in category:
                    advice = "⚠️ Sensitive groups should reduce outdoor activities and consider wearing masks."
                elif "Unhealthy" in category:
                    advice = "🚨 Everyone may experience health effects. Limit outdoor exposure."
                elif "Hazardous" in category:
                    advice = "🛑 Health emergency! Stay indoors. Close windows and doors."
                
                st.markdown(f"""
                <div class="glass-card">
                    <div style="font-size: 1.3rem; font-weight: 800; color: #1f2937; margin-bottom: 12px;">
                        💡 Health Advice
                    </div>
                    <div style="font-size: 1rem; color: #4b5563; line-height: 1.6; margin-bottom: 18px;">
                        {advice}
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                        <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%;
                                     background: {today_pred.get('color', '#333')};"></span>
                        <span style="font-size: 0.95rem; font-weight: 700; color: {today_pred.get('color', '#333')};">
                            {category}
                        </span>
                    </div>
                    <div style="margin-top: 16px; padding-top: 12px; border-top: 1px solid #e5e7eb;">
                        <span style="font-size: 0.78rem; color: #9ca3af;">
                            Predicted by <b>{best_model_name}</b> model
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # 3-Day Forecast Cards
            st.markdown("### 📅 Today + 3-Day Forecast")
            if len(predictions) > 0:
                render_forecast_cards(predictions)
            else:
                st.info("Insufficient forecast data available.")

            # Comparison Table
            if comp_df is not None and not comp_df.empty:
                st.markdown("### 🔍 Model Forecast Comparison")
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

        else:
            st.warning("No prediction data available. Run the prediction pipeline first.")
       
        # ── Historical Data Table ──
        if df_hist is not None and not df_hist.empty:
            with st.expander("📋 View Historical Data", expanded=False):
                st.dataframe(df_hist, use_container_width=True, hide_index=True)

    # ================================================================
    #  TAB 2: Trends & Analysis
    # ================================================================
    with tab2:
        col_map, col_trend = st.columns([1, 2])
        
        with col_map:
            st.markdown("### 📍 Monitoring Location")
            map_data = pd.DataFrame({'lat': [24.8608], 'lon': [67.0104]})
            st.map(map_data, zoom=10, use_container_width=True)
            st.caption("Karachi, Pakistan · Open-Meteo Grid Point")

        with col_trend:
            st.markdown("### 📈 30-Day AQI Trend")
            if df_hist is not None and not df_hist.empty:
                fig_trend = create_aqi_trend_chart(df_hist, days=30)
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No historical data available.")

        st.markdown("---")
        if df_hist is not None and not df_hist.empty:
            col_a, col_b = st.columns(2)
            with col_a:
                fig_pol = create_pollutant_breakdown(df_hist)
                st.plotly_chart(fig_pol, use_container_width=True)
            with col_b:
                fig_corr = create_correlation_heatmap(df_hist)
                st.plotly_chart(fig_corr, use_container_width=True)

    # ================================================================
    #  TAB 3: Model Insights
    # ================================================================
    with tab3:
        st.markdown("### 🧠 Model Performance Metrics")

        if metrics_data and "models" in metrics_data:
            metrics_dict = metrics_data["models"]
            
            # Metric cards for each model
            model_cols = st.columns(len(metrics_dict))
            for col, (name, m) in zip(model_cols, metrics_dict.items()):
                with col:
                    rmse = m.get("RMSE", 0)
                    mae = m.get("MAE", 0)
                    r2 = m.get("R2", 0)
                    is_best = (name == metrics_data.get("best_model"))
                    badge = ' 🏆' if is_best else ''
                    
                    st.markdown(f"""
                    <div class="glass-card" style="text-align: center; {'border: 2px solid #2563eb;' if is_best else ''}">
                        <div style="font-size: 1.1rem; font-weight: 800; color: #1f2937; margin-bottom: 12px;">
                            {name}{badge}
                        </div>
                        <div style="margin-bottom: 8px;">
                            <span style="font-size: 0.75rem; color: #6b7280;">RMSE</span><br>
                            <span style="font-size: 1.4rem; font-weight: 700; color: #2563eb;">{rmse:.2f}</span>
                        </div>
                        <div style="margin-bottom: 8px;">
                            <span style="font-size: 0.75rem; color: #6b7280;">MAE</span><br>
                            <span style="font-size: 1.1rem; font-weight: 600; color: #374151;">{mae:.2f}</span>
                        </div>
                        <div>
                            <span style="font-size: 0.75rem; color: #6b7280;">R²</span><br>
                            <span style="font-size: 1.1rem; font-weight: 600; color: #374151;">{r2:.4f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            best_model = metrics_data.get("best_model", "Unknown")
            st.success(f"🏆 Best model selected by lowest RMSE: **{best_model}**")
        else:
            st.info("No metrics available. Run the training pipeline first.")

        st.markdown("---")
        st.markdown("### 📊 Feature Importance")
        try:
            from src.inference.predictor import AQIInferenceEngine
            best_model = metrics_data.get("best_model", "LightGBM") if metrics_data else "LightGBM"
            models_dir = PROJECT_ROOT / "models"
            model_path = models_dir / f"{best_model}_model.pkl"
            
            if model_path.exists():
                engine = AQIInferenceEngine(str(model_path))
                fi = engine.get_feature_importance()
                if fi is not None and not fi.empty:
                    import plotly.express as px
                    fig_fi = px.bar(
                        fi.head(10), x="importance", y="feature",
                        orientation="h", title=f"Top Features ({best_model})",
                        color="importance", color_continuous_scale="Blues"
                    )
                    fig_fi.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_fi, use_container_width=True)
                else:
                    st.info("Feature importance not available.")
            else:
                st.info("Model file not found.")
        except Exception as e:
            st.error(f"Error loading feature importance: {e}")


if __name__ == "__main__":
    main()
