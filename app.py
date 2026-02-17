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


def main():
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
        if v > 0:
            st.markdown(f"**Model v{v}**")
            st.caption(f"Trained: {trained_at[:16] if trained_at != 'N/A' else 'N/A'}")
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
        
        # Deploy Button
        st.markdown(
            """
            <a href="https://share.streamlit.io" target="_blank" class="sidebar-deploy-btn">
                🚀 Deploy to Cloud
            </a>
            """, 
            unsafe_allow_html=True
        )

    # ── Main Content ──
    st.title("🌍 Karachi AQI Predictor")
    st.markdown("**Real-time Air Quality Forecasting — Powered by ML & Hopsworks**")
    st.markdown("---")

    # ── Load data ──
    predictions = load_predictions()
    df_hist = load_historical_data()
    comp_df = load_comparison_data()
    metrics_data = load_model_metrics()

    # ── Tabs ──
    tab1, tab2, tab3 = st.tabs([
        "📊 Dashboard",
        "🔬 Trends & Map",
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
            today_pred = predictions[0] if predictions else {}
            today_aqi = today_pred.get("aqi_pred", 0)
            best_model_name = today_pred.get("model_used", "Best Model")

            col_gauge, col_info = st.columns([2, 1])
            with col_gauge:
                fig_gauge = create_aqi_gauge(today_aqi, title="Current Air Quality Index", height=350)
                st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

            with col_info:
                # Health Advice Card
                category = today_pred.get('category', 'Unknown')
                advice = "Enjoy the fresh air!"
                if "Moderate" in category:
                    advice = "Sensitive groups should reduce outdoor exertion."
                elif "Unhealthy" in category or "Hazardous" in category:
                    advice = "⚠️ Avoid outdoor activities. Wear a mask if necessary."
                
                st.markdown(f"""
                <div class="glass-card">
                    <div style="font-size: 1.2rem; font-weight: 700; color: #1f2937; margin-bottom: 8px;">
                        Health Advice
                    </div>
                    <div style="font-size: 1rem; color: #4b5563; line-height: 1.5; margin-bottom: 16px;">
                        {advice}
                    </div>
                    <div style="font-size: 0.9rem; font-weight: 600; color: {today_pred.get('color', '#333')};">
                        Condition: {category}
                    </div>
                    <div style="margin-top: 12px; font-size: 0.8rem; color: #9ca3af;">
                        Prediction by {best_model_name}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # 3-Day Forecast Cards (Including Today)
            st.markdown("### 📅 4-Day Forecast")
            if len(predictions) > 0:
                render_forecast_cards(predictions)
            else:
                st.info("Insufficient forecast data available.")

            # Comparison Table (Including Today)
            if not comp_df.empty:
                st.markdown("### 🔍 Model Forecast Comparison")
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

        else:
            st.warning("No prediction data available. Run the prediction pipeline first.")

    # ================================================================
    #  TAB 2: Trends & Map
    # ================================================================
    with tab2:
        col_map, col_trend = st.columns([1, 2])
        
        with col_map:
            st.markdown("### 📍 Location")
            map_data = pd.DataFrame({'lat': [24.8608], 'lon': [67.0104]})
            st.map(map_data, zoom=10, use_container_width=True)
            st.caption("Monitoring Station: Karachi, Pakistan")

        with col_trend:
            st.markdown("### 📈 30-Day AQI Trend")
            if df_hist is not None and not df_hist.empty:
                fig_trend = create_aqi_trend_chart(df_hist, days=30)
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No historical data.")

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
            metrics_df = pd.DataFrame(metrics_dict).T
            st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)
            best_model = metrics_data.get("best_model", "Unknown")
            st.success(f"🏆 Best Performing Model: **{best_model}**")
        else:
            st.info("No metrics available.")

        st.markdown("---")
        st.markdown("### 📊 Feature Importance")
        try:
            from src.inference.predictor import AQIInferenceEngine
            best_model = metrics_data.get("best_model", "RandomForest") if metrics_data else "RandomForest"
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
