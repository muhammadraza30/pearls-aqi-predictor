from src.features.data_fetching import (
    fetch_current_data as fetch_weather_data_raw,
    fetch_historical_data as fetch_historical_weather,
)
from src.features.feature_engineering import engineer_features
from src.features.data_fetching import fetch_current_data


def fetch_weather_data(lat, lon):
    import requests

    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "temperature_2m", "relative_humidity_2m", "pressure_msl",
            "wind_speed_10m", "wind_direction_10m", "cloud_cover",
        ],
    }

    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aqi_params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
            "sulphur_dioxide", "ozone", "us_aqi",
        ],
    }

    try:
        w_response = requests.get(weather_url, params=weather_params, timeout=10)
        a_response = requests.get(aqi_url, params=aqi_params, timeout=10)

        if w_response.status_code != 200 or a_response.status_code != 200:
            return {}, {}

        return w_response.json(), a_response.json()
    except Exception:
        return {}, {}


def process_data(weather_data, aqi_data):
    """
    Backward-compatible wrapper.
    New code should use data_fetching.fetch_current_data() instead.
    """
    import pandas as pd
    from datetime import datetime

    if not weather_data or not aqi_data:
        return pd.DataFrame()

    current_w = weather_data.get("current", {})
    current_a = aqi_data.get("current", {})
    now_floored = datetime.now().replace(minute=0, second=0, microsecond=0)

    data = {
        "datetime": now_floored,
        "temp": current_w.get("temperature_2m"),
        "humidity": current_w.get("relative_humidity_2m"),
        "pressure": current_w.get("pressure_msl"),
        "wind_speed": current_w.get("wind_speed_10m"),
        "wind_deg": current_w.get("wind_direction_10m"),
        "clouds": current_w.get("cloud_cover"),
        "pm10": current_a.get("pm10"),
        "pm2_5": current_a.get("pm2_5"),
        "co": current_a.get("carbon_monoxide"),
        "no2": current_a.get("nitrogen_dioxide"),
        "so2": current_a.get("sulphur_dioxide"),
        "o3": current_a.get("ozone"),
        "aqi": current_a.get("us_aqi"),
    }

    return pd.DataFrame([data])
