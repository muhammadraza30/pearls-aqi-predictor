"""
Pre-compute AQI predictions using Recursive Hourly Forecasting (SVR/LGBM/LSTM).
Generates 72-hour hourly forecast and aggregates to Daily Averages for dashboard.
"""
import os
import sys
import json
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, date, timedelta
import warnings

# Suppress LightGBM warning about feature names
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.inference.predictor import AQIInferenceEngine, categorize_aqi
from src.features.utils import fetch_historical_weather  # Reuse this

LOG = logging.getLogger("predict_3days")

LAT = float(os.getenv("LATITUDE", "24.8608"))
LON = float(os.getenv("LONGITUDE", "67.0104"))
METRICS_FILE = PROJECT_ROOT / "data" / "model_metrics.json"

MODELS_TO_RUN = ["SVR", "LightGBM", "LSTM"]
FEATURES = ['temp', 'humidity', 'wind_speed', 'clouds', 'aqi_lag_1', 'aqi_lag_3', 'aqi_lag_6', 'aqi_lag_24']


def fetch_hourly_forecast(lat, lon):
    """Fetch 72-hour hourly weather forecast."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover",
        "forecast_days": 4,
        "timezone": "UTC"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("hourly", {})
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['time'])
        # Rename to match training schema
        df = df.rename(columns={
            "temperature_2m": "temp",
            "relative_humidity_2m": "humidity",
            "wind_speed_10m": "wind_speed",
            "cloud_cover": "clouds"
        })
        return df[['datetime', 'temp', 'humidity', 'wind_speed', 'clouds']]
    except Exception as e:
        LOG.error("Failed to fetch hourly forecast: %s", e)
        return pd.DataFrame()


def get_lags(history_series, idx):
    """
    Compute lags from a history series (list or array).
    idx is the index of the 'current' time step we are predicting for.
    Wait, if history is [t-24 ... t], and we predict t+1.
    Lag 1 = history[-1]
    Lag 3 = history[-3]
    """
    # History series contains [past_24h ... predicted_so_far]
    n = len(history_series)
    # lag 1 is the last element
    l1 = history_series[-1] if n >= 1 else 0
    l3 = history_series[-3] if n >= 3 else l1
    l6 = history_series[-6] if n >= 6 else l3
    l24 = history_series[-24] if n >= 24 else l6
    return l1, l3, l6, l24


def predict_recursive_72h(engine, history_aqi, weather_forecast_df):
    """
    Recursive forecasting loop.
    history_aqi: list of last 24h real AQI values.
    weather_forecast_df: dataframe of next 72 hours weather.
    """
    predictions = []
    current_history = list(history_aqi)
    
    # Iterate through forecast hours
    for _, row in weather_forecast_df.iterrows():
        # 1. Build Input Feature Row
        l1, l3, l6, l24 = get_lags(current_history, 0)
        
        # Match feature order: ['temp', 'humidity', 'wind_speed', 'clouds', 'aqi_lag_1', 'aqi_lag_3', 'aqi_lag_6', 'aqi_lag_24']
        input_data = {
            'temp': row['temp'],
            'humidity': row['humidity'],
            'wind_speed': row['wind_speed'],
            'clouds': row['clouds'],
            'aqi_lag_1': l1,
            'aqi_lag_3': l3,
            'aqi_lag_6': l6,
            'aqi_lag_24': l24
        }
        
        df_input = pd.DataFrame([input_data])
        
        # 2. Predict
        pred = engine.predict(df_input)
        val = max(0.0, float(pred[0])) if len(pred) > 0 else 0.0
        
        # 3. Update History & Store
        current_history.append(val)
        predictions.append(val)
        
        # Keep history manageable? No need, list append is fine for 72 steps.
        
    return predictions


def load_recent_history():
    """Fetch last 24-48h of AQI history to bootstrap lags."""
    # Use fetch_historical_weather from utils
    # Fetch 3 days to be safe
    df = fetch_historical_weather(LAT, LON, days=3)
    if 'aqi' in df.columns and not df.empty:
        # Sort by time
        df = df.sort_values('datetime')
        # Take last 24 values
        return df['aqi'].tail(24).tolist()
    return [50] * 24  # Fallback


def aggregate_to_daily(hourly_preds, start_date):
    """Aggregate 72 hourly predictions to 3 Daily Averages."""
    daily_preds = []
    
    # Day 1: 0-23
    # Day 2: 24-47
    # Day 3: 48-71
    
    current_date = start_date
    chunk_size = 24
    
    for i in range(0, len(hourly_preds), chunk_size):
        chunk = hourly_preds[i:i+chunk_size]
        if not chunk: continue
        
        avg_aqi = sum(chunk) / len(chunk)
        
        # Categorize
        info = categorize_aqi(avg_aqi)
        
        daily_preds.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "aqi_pred": round(avg_aqi, 1),
            "category": info["category"],
            "level": info["level"],
            "color": info["color"],
            "message": info["message"],
            "emoji": info["emoji"],
            "is_hazardous": info["is_hazardous"]
        })
        current_date += timedelta(days=1)
        
    return daily_preds


def main():
    logging.basicConfig(level=logging.INFO)
    
    # 1. Fetch Data
    print("🔄 Fetching forecast and history...")
    forecast_df = fetch_hourly_forecast(LAT, LON)
    if forecast_df.empty:
        print("❌ Failed to fetch forecast.")
        sys.exit(1)
        
    history = load_recent_history()
    if len(history) < 24:
        print("⚠️ Warning: Insufficient history for lags. Padding...")
        if len(history) == 0:
            history = [50] * 24  # Use default moderate AQI
        else:
            padding_value = history[-1]
            history = history + [padding_value] * (24 - len(history))
    # 2. Determine Best Model
    metrics_path = PROJECT_ROOT / "data" / "model_metrics.json"
    best_model = "SVR"
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                best_model = json.load(f).get("best_model", "SVR")
        except: pass

    # 3. Run Predictions (Recursive)
    all_daily_preds = {}
    
    for m_name in MODELS_TO_RUN:
        print(f"🚀 Predicting with {m_name}...")
        # Construct path manually
        model_file = f"{m_name}_model.keras" if m_name == "LSTM" else f"{m_name}_model.pkl"
        path = PROJECT_ROOT / "models" / model_file
        
        if not path.exists():
            print(f"⚠️ Model {m_name} not found at {path}. Skipping.")
            continue
            
        # Creating engine with specific model path loads that model
        engine = AQIInferenceEngine(model_path=str(path))
        
        hourly = predict_recursive_72h(engine, history, forecast_df)
        
        # Start date = Tomorrow? Or Today? OpenMeteo forecast usually starts Today.
        # hourly forecast from API starts at current hour or 00:00?
        # forecast_days=3 starts from Today 00:00 usually.
        # If we run this at 10AM, the first few hours of prediction are "re-predicting" past hours of today.
        # That's fine. We aggregate to daily average.
        
        start_date = date.today()
        daily = aggregate_to_daily(hourly, start_date)
        all_daily_preds[m_name] = daily

    # 4. Save Prediction (Best Model)
    actual_model_used = best_model
    if best_model in all_daily_preds:
        final_daily = all_daily_preds[best_model]
    elif all_daily_preds:
        actual_model_used = list(all_daily_preds.keys())[0]
        final_daily = all_daily_preds[actual_model_used]
    else:
        print("❌ No predictions generated.")
        return

    # Add model name to records
    for d in final_daily:
        d["model_used"] = actual_model_used
    df_best = pd.DataFrame(final_daily)
    out_best = PROJECT_ROOT / "data" / "predictions_3day.csv"
    df_best.to_csv(out_best, index=False)
    
    # 5. Save Comparison
    # We need a structure: date, SVR_pred, LightGBM_pred, LSTM_pred
    # daily gives dicts. We extract 'aqi_pred'.
    
    dates = [d['date'] for d in final_daily]
    comp_data = {'date': dates}
    
    for m_name, d_list in all_daily_preds.items():
        # Match dates
        # d_list might have 3 days.
        vals = []
        for dt in dates:
            # find matching date
            match = next((x['aqi_pred'] for x in d_list if x['date'] == dt), 0)
            vals.append(match)
        comp_data[f"{m_name}_pred"] = vals
        
    df_comp = pd.DataFrame(comp_data)
    out_comp = PROJECT_ROOT / "data" / "predictions_comparison.csv"
    df_comp.to_csv(out_comp, index=False)
    
    print("✅ Predictions generated and saved.")
    print("Best Forecast:\n", df_best)


if __name__ == "__main__":
    main()
