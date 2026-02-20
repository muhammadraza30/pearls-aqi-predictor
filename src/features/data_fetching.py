import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import openmeteo_requests
import requests_cache
from retry_requests import retry

from dotenv import load_dotenv

# Load env from project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

LAT = float(os.getenv("LATITUDE", "24.8608"))
LON = float(os.getenv("LONGITUDE", "67.0104"))

# Column rename map: API names -> our schema
RENAME_MAP = {
    "time": "datetime",
    "temperature_2m": "temp",
    "relative_humidity_2m": "humidity",
    "pressure_msl": "pressure",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_deg",
    "cloud_cover": "clouds",
    "carbon_monoxide": "co",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "ozone": "o3",
    "us_aqi": "aqi",
}

# Ordered output columns (before feature engineering)
RAW_COLUMNS = [
    "datetime", "temp", "humidity", "pressure", "wind_speed", "wind_deg",
    "clouds", "pm10", "pm2_5", "co", "no2", "so2", "o3", "aqi",
]


def _get_last_complete_hour() -> datetime:
    """
    Returns the last fully completed hour.
    e.g. if now is 2026-02-20 03:30 -> returns 2026-02-20 02:00
    """
    now = datetime.now()
    return now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)


# ─────────────────────────── CURRENT DATA ───────────────────────────


def fetch_current_data(lat: float = None, lon: float = None) -> pd.DataFrame:
    """
    Fetches the latest hour's weather + AQI data from Open-Meteo forecast API.
    Returns a single-row DataFrame matching the raw schema.
    """
    return fetch_recent_data(lat, lon, past_days=0)


def fetch_recent_data(lat: float = None, lon: float = None, past_days: int = 3) -> pd.DataFrame:
    """
    Fetches recent weather + AQI data (past N days + today) from Open-Meteo forecast API.
    Useful for incremental updates where context (lags) is needed.
    """
    lat = lat or LAT
    lon = lon or LON

    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": past_days,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "pressure_msl",
            "wind_speed_10m", "wind_direction_10m", "cloud_cover",
        ],
    }

    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aqi_params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": past_days,
        "hourly": [
            "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
            "sulphur_dioxide", "ozone", "us_aqi",
        ],
    }

    try:
        w_resp = requests.get(weather_url, params=weather_params, timeout=10)
        a_resp = requests.get(aqi_url, params=aqi_params, timeout=10)

        if w_resp.status_code != 200 or a_resp.status_code != 200:
            print(f"❌ API error: Weather={w_resp.status_code}, AQI={a_resp.status_code}")
            return pd.DataFrame()

        w_json = w_resp.json().get("hourly", {})
        a_json = a_resp.json().get("hourly", {})

        if not w_json or not a_json:
            return pd.DataFrame()

        # Create DataFrames from hourly data
        w_df = pd.DataFrame({
            "datetime": pd.to_datetime(w_json.get("time")),
            "temp": w_json.get("temperature_2m"),
            "humidity": w_json.get("relative_humidity_2m"),
            "pressure": w_json.get("pressure_msl"),
            "wind_speed": w_json.get("wind_speed_10m"),
            "wind_deg": w_json.get("wind_direction_10m"),
            "clouds": w_json.get("cloud_cover"),
        })

        a_df = pd.DataFrame({
            "datetime": pd.to_datetime(a_json.get("time")),
            "pm10": a_json.get("pm10"),
            "pm2_5": a_json.get("pm2_5"),
            "co": a_json.get("carbon_monoxide"),
            "no2": a_json.get("nitrogen_dioxide"),
            "so2": a_json.get("sulphur_dioxide"),
            "o3": a_json.get("ozone"),
            "aqi": a_json.get("us_aqi"),
        })

        # Merge on datetime
        df = pd.merge(w_df, a_df, on="datetime")

        # Truncate to last complete hour
        last_hour = _get_last_complete_hour()
        df = df[df["datetime"] <= last_hour]

        return df[RAW_COLUMNS]

    except Exception as e:
        print(f"❌ Exception fetching recent data: {e}")
        return pd.DataFrame()


# ─────────────────────────── HISTORICAL DATA ───────────────────────────


def fetch_historical_data(
    lat: float = None,
    lon: float = None,
    start_date: str = "2025-08-01",
    end_date: str = None,
) -> pd.DataFrame:
    """
    Fetches hourly historical weather + AQI data from Open-Meteo archive API.
    Uses openmeteo_requests with retry for robustness.

    Args:
        lat, lon:     Coordinates (default from .env).
        start_date:   Start date string 'YYYY-MM-DD'.
        end_date:     End date string 'YYYY-MM-DD'.
                      Defaults to today's date.

    Returns:
        DataFrame with raw columns, truncated to last complete hour.
    """
    lat = lat or LAT
    lon = lon or LON

    # Helper to ensure YYYY-MM-DD format
    def _format_date(date_str):
        try:
            # Check if it's already YYYY-MM-DD
            datetime.strptime(date_str, "%Y-%m-%d")
            return date_str
        except ValueError:
            # Try DD-MM-YYYY
            try:
                return datetime.strptime(date_str, "%d-%m-%Y").strftime("%Y-%m-%d")
            except ValueError:
                # Fallback or raise
                return date_str

    start_date = _format_date(start_date)

    if end_date is None:
        # Default to today to fetch as much recent data as possible from archive
        # Note: Archive API usually takes 'end_date' inclusive.
        end_date = datetime.now().strftime("%Y-%m-%d")
    else:
        end_date = _format_date(end_date)

    # Set up retrying cached session
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # ── Weather ──
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "pressure_msl",
            "wind_speed_10m", "wind_direction_10m", "cloud_cover",
        ],
    }

    # ── AQI ──
    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aqi_params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": [
            "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
            "sulphur_dioxide", "ozone", "us_aqi",
        ],
    }

    try:
        weather_responses = openmeteo.weather_api(weather_url, params=weather_params)
        aqi_responses = openmeteo.weather_api(aqi_url, params=aqi_params)

        # ── Process weather ──
        w_resp = weather_responses[0]
        w_hourly = w_resp.Hourly()

        dt_index = pd.date_range(
            start=pd.to_datetime(w_hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(w_hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=w_hourly.Interval()),
            inclusive="left",
        )

        weather_df = pd.DataFrame({
            "datetime": dt_index,
            "temp": w_hourly.Variables(0).ValuesAsNumpy(),
            "humidity": w_hourly.Variables(1).ValuesAsNumpy(),
            "pressure": w_hourly.Variables(2).ValuesAsNumpy(),
            "wind_speed": w_hourly.Variables(3).ValuesAsNumpy(),
            "wind_deg": w_hourly.Variables(4).ValuesAsNumpy(),
            "clouds": w_hourly.Variables(5).ValuesAsNumpy(),
        })

        # ── Process AQI ──
        a_resp = aqi_responses[0]
        a_hourly = a_resp.Hourly()

        dt_index_aqi = pd.date_range(
            start=pd.to_datetime(a_hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(a_hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=a_hourly.Interval()),
            inclusive="left",
        )

        aqi_df = pd.DataFrame({
            "datetime": dt_index_aqi,
            "pm10": a_hourly.Variables(0).ValuesAsNumpy(),
            "pm2_5": a_hourly.Variables(1).ValuesAsNumpy(),
            "co": a_hourly.Variables(2).ValuesAsNumpy(),
            "no2": a_hourly.Variables(3).ValuesAsNumpy(),
            "so2": a_hourly.Variables(4).ValuesAsNumpy(),
            "o3": a_hourly.Variables(5).ValuesAsNumpy(),
            "aqi": a_hourly.Variables(6).ValuesAsNumpy(),
        })

        # Merge on datetime
        df = pd.merge(weather_df, aqi_df, on="datetime")

        # Fill missing values
        df = df.ffill().bfill()

        # Truncate to last complete hour
        last_hour = _get_last_complete_hour()
        last_hour_utc = pd.Timestamp(last_hour).tz_localize("UTC")
        df = df[df["datetime"] <= last_hour_utc]

        print(f"✅ Fetched {len(df)} historical rows ({start_date} → {end_date}, truncated to {last_hour})")
        return df[RAW_COLUMNS]

    except Exception as e:
        print(f"❌ Exception fetching historical data: {e}")
        return pd.DataFrame()


# ─────────────────────── CONVENIENCE (backward compat) ───────────────────────

def fetch_historical_weather(lat: float = None, lon: float = None) -> pd.DataFrame:
    """
    Backward-compatible wrapper used by predict_3days.py.
    Fetches historical data with default date range and renames columns
    to match the old schema expected by callers.
    """
    return fetch_historical_data(lat=lat, lon=lon)
