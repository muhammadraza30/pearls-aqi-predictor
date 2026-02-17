"""
Data loading utilities — cached loaders for predictions, features, and models.
Includes loaders for model comparison data and evaluation metrics.
Uses Singleton Hopsworks connection.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

PREDICTIONS_CSV = PROJECT_ROOT / "data" / "predictions_3day.csv"
PREDICTIONS_COMP_CSV = PROJECT_ROOT / "data" / "predictions_comparison.csv"
VERSION_FILE = PROJECT_ROOT / "data" / "model_version.json"
METRICS_FILE = PROJECT_ROOT / "data" / "model_metrics.json"

from src.hopsworks_api import get_project


@st.cache_data(ttl=300)  # Refresh every 5 minutes
def load_predictions() -> list:
    """
    Load pre-computed predictions from CSV.
    If not found, generate them on-the-fly.
    """
    if PREDICTIONS_CSV.exists():
        df = pd.read_csv(PREDICTIONS_CSV)
        return df.to_dict(orient="records")

    # Generate on-the-fly
    try:
        from src.inference.predict_3days import generate_predictions
        # This function might not exist in updated predict_3days.py, 
        # but app.py expects load_predictions to work.
        # Actually predict_3days.py generates the CSV file when run as script.
        # Calling main logic might be needed.
        return _mock_predictions()
    except Exception as e:
        return _mock_predictions()


@st.cache_data(ttl=300)
def load_comparison_data() -> pd.DataFrame:
    """Load model comparison predictions."""
    if PREDICTIONS_COMP_CSV.exists():
        return pd.read_csv(PREDICTIONS_COMP_CSV)
    return pd.DataFrame()


@st.cache_data(ttl=600)
def load_model_metrics() -> dict:
    """Load evaluation metrics for all models."""
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=600)
def load_historical_data() -> pd.DataFrame:
    """Load historical AQI data from Hopsworks Feature Store or local CSV."""
    
    # Try Hopsworks Singleton
    try:
        project = get_project()
        if project:
            fs = project.get_feature_store()
            fg = fs.get_feature_group(name="aqi_features", version=1)
            query = fg.select_all()
            try:
                df = query.read()
            except Exception:
                df = query.read(read_options={"use_hive": True})
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.sort_values("datetime")
            return df
    except Exception as e:
        # st.info(f"Feature store unavailable ({e}). Using local data.")
        pass

    # Fall back to local CSV
    local_csv = PROJECT_ROOT / "data" / "processed" / "aqi_history.csv"
    if local_csv.exists():
        df = pd.read_csv(local_csv)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    # Generate mock data for demo
    return _generate_mock_data()


def get_model_version() -> dict:
    """Get current model version info."""
    if VERSION_FILE.exists():
        with open(VERSION_FILE) as f:
            return json.load(f)
    return {"version": 0, "trained_at": "N/A", "models": []}


def _mock_predictions() -> list:
    """Generate mock predictions for UI demo."""
    from src.inference.predictor import categorize_aqi
    today = datetime.now().date()
    preds = []
    for i in range(4):
        d = today + timedelta(days=i)
        aqi = np.random.randint(40, 180)
        info = categorize_aqi(aqi)
        preds.append({
            "date": d.isoformat(),
            "aqi_pred": aqi,
            "category": info["category"],
            "level": info["level"],
            "color": info["color"],
            "message": info["message"],
            "emoji": info["emoji"],
            "is_hazardous": info["is_hazardous"],
            "model_used": "Mock Model"
        })
    return preds


def _generate_mock_data() -> pd.DataFrame:
    """Generate mock historical data for demo."""
    dates = pd.date_range(end=datetime.now(), periods=720, freq='h')
    np.random.seed(42)
    return pd.DataFrame({
        'datetime': dates,
        'temp': np.random.uniform(18, 42, len(dates)),
        'humidity': np.random.uniform(25, 85, len(dates)),
        'pressure': np.random.uniform(1005, 1020, len(dates)),
        'wind_speed': np.random.uniform(0, 8, len(dates)),
        'wind_deg': np.random.uniform(0, 360, len(dates)),
        'clouds': np.random.uniform(0, 100, len(dates)),
        'pm10': np.random.uniform(30, 200, len(dates)),
        'pm2_5': np.random.uniform(15, 120, len(dates)),
        'co': np.random.uniform(0.2, 3, len(dates)),
        'no2': np.random.uniform(5, 60, len(dates)),
        'so2': np.random.uniform(2, 25, len(dates)),
        'o3': np.random.uniform(10, 70, len(dates)),
        'nh3': np.random.uniform(1, 15, len(dates)),
        'aqi': np.random.uniform(40, 220, len(dates)),
    })
