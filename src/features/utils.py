import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_weather_data(lat, lon):
    """
    Fetches current weather and AQI data from Open-Meteo API (JSON).
    Matches the schema used in backfill.py.
    """
    # Weather
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["temperature_2m", "relative_humidity_2m", "pressure_msl", "wind_speed_10m", "wind_direction_10m", "cloud_cover"]
    }
    
    # AQI
    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aqi_params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "ammonia", "us_aqi"]
    }

    try:
        w_response = requests.get(weather_url, params=weather_params, timeout=10)
        a_response = requests.get(aqi_url, params=aqi_params, timeout=10)
        
        if w_response.status_code != 200 or a_response.status_code != 200:
            print(f"Error calling APIs: Weather={w_response.status_code}, AQI={a_response.status_code}")
            return {}, {}
            
        return w_response.json(), a_response.json()
        
    except Exception as e:
        print(f"Exception fetching data: {e}")
        return {}, {}

def fetch_historical_weather(lat, lon, days=90):
    """Fetch historical weather and AQI data for backfill."""
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Weather Archive
    w_url = "https://archive-api.open-meteo.com/v1/archive"
    w_params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "pressure_msl", "wind_speed_10m", "wind_direction_10m", "cloud_cover"],
        "timezone": "UTC"
    }
    
    # AQI Archive (Standard API supports history)
    a_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    a_params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "ammonia", "us_aqi"],
        "timezone": "UTC"
    }

    try:
        r_w = requests.get(w_url, params=w_params, timeout=20)
        r_a = requests.get(a_url, params=a_params, timeout=20)
        
        if r_w.status_code != 200 or r_a.status_code != 200:
            print(f"Error fetching history: W={r_w.status_code}, A={r_a.status_code}")
            return pd.DataFrame()
            
        w_data = r_w.json()
        a_data = r_a.json()
        
        # Process hourly data
        df_w = pd.DataFrame(w_data.get("hourly", {}))
        df_a = pd.DataFrame(a_data.get("hourly", {}))
        
        # Align timestamps (assuming standard API response aligns them)
        # Or merge on 'time'
        if "time" in df_w.columns and "time" in df_a.columns:
            df = pd.merge(df_w, df_a, on="time", how="inner")
        else:
            return pd.DataFrame()
            
        # Rename cols to match schema
        rename_map = {
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
            "us_aqi": "aqi"
        }
        df = df.rename(columns=rename_map)
        return df

    except Exception as e:
        print(f"Exception fetching history: {e}")
        return pd.DataFrame()

def process_data(weather_data, aqi_data):
    """
    Processes Open-Meteo JSON responses into a structured DataFrame.
    """
    if not weather_data or not aqi_data:
        return pd.DataFrame()
        
    current_w = weather_data.get('current', {})
    current_a = aqi_data.get('current', {})
    
    # Map to schema columns
    data = {
        'datetime': datetime.now(),
        'temp': current_w.get('temperature_2m'),
        'humidity': current_w.get('relative_humidity_2m'),
        'pressure': current_w.get('pressure_msl'),
        'wind_speed': current_w.get('wind_speed_10m'),
        'wind_deg': current_w.get('wind_direction_10m'),
        'clouds': current_w.get('cloud_cover'),
        
        'pm10': current_a.get('pm10'),
        'pm2_5': current_a.get('pm2_5'),
        'co': current_a.get('carbon_monoxide'),
        'no2': current_a.get('nitrogen_dioxide'),
        'so2': current_a.get('sulphur_dioxide'),
        'o3': current_a.get('ozone'),
        'aqi': current_a.get('us_aqi'), # Standard 0-500 scale
    }
    
    return pd.DataFrame([data])

def engineer_features(df):
    """
    Adds derived features to the DataFrame.
    ensure 'unix_time' is in SECONDS (int).
    """
    if df.empty:
        return df

    df['datetime'] = pd.to_datetime(df['datetime'])
    # Convert ns to seconds (// 10^9)
    df['unix_time'] = (df['datetime'].astype('int64') // 10**9)
    df['unix_time'] = df['unix_time'].astype('int64')
    
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['weekday'] = df['datetime'].dt.dayofweek
    
    # Fill missing if API failed partially
    # Forward-fill time-series gaps, then fill remaining with column medians
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    # Or alternatively, drop rows with critical missing data:
    # df = df.dropna(subset=['aqi', 'pm2_5', 'pm10'])    
    return df
